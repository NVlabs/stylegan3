import os
import click
from tqdm import tqdm

try:
    import ffmpeg
except ImportError:
    raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')

try:
    import skvideo.io
except ImportError:
    raise ImportError('scikit-video not found! Install it via "pip install scikit-video"')

import PIL
from PIL import Image
import scipy.ndimage as nd

from fractions import Fraction
import numpy as np
import torch
from torchvision import transforms

from typing import Union, Tuple

import dnnlib
import legacy
from torch_utils import gen_utils

from network_features import VGG16FeaturesNVIDIA, DiscriminatorFeatures

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor


# ----------------------------------------------------------------------------


def normalize_image(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """Change dynamic range of an image from [0, 255] to [-1, 1]"""
    image = np.array(image, dtype=np.float32)
    image = image / 127.5 - 1.0
    return image


def get_video_information(mp4_filename: Union[str, os.PathLike],
                          max_length_seconds: float = None,
                          starting_second: float = 0.0) -> Tuple[int, float, int, int, int, int]:
    """Take a mp4 file and return a list containing each frame as a NumPy array"""
    metadata = skvideo.io.ffprobe(mp4_filename)
    # Get video properties
    fps = int(np.rint(float(Fraction(metadata['video']['@avg_frame_rate']))))  # Possible error here if we get 0
    total_video_num_frames = int(metadata['video']['@nb_frames'])
    video_duration = float(metadata['video']['@duration'])
    video_width = int(metadata['video']['@width'])
    video_height = int(metadata['video']['@height'])
    # Maximum number of frames to return (if not provided, return the full video)
    if max_length_seconds is None:
        print('Considering the full video...')
        max_length_seconds = video_duration
    if starting_second != 0.0:
        print('Using part of the video...')
        starting_second = min(starting_second, video_duration)
        max_length_seconds = min(video_duration - starting_second, max_length_seconds)
    max_num_frames = int(np.rint(max_length_seconds * fps))
    max_frames = min(total_video_num_frames, max_num_frames)
    returned_duration = min(video_duration, max_length_seconds)
    # Frame to start from
    starting_frame = int(np.rint(starting_second * fps))

    return fps, returned_duration, starting_frame, max_frames, video_width, video_height


def get_video_frames(mp4_filename: Union[str, os.PathLike],
                     run_dir: Union[str, os.PathLike],
                     starting_frame: int,
                     max_frames: int,
                     center_crop: bool = False,
                     save_selected_frames: bool = False) -> np.ndarray:
    """Get all the frames of a video as a np.ndarray"""
    # DEPRECATED
    print('Getting video frames...')
    frames = skvideo.io.vread(mp4_filename)  # TODO: crazy things with scikit-video
    frames = frames[starting_frame:min(starting_frame + max_frames, len(frames)), :, :, :]
    frames = np.transpose(frames, (0, 3, 2, 1))  # NHWC => NCWH
    if center_crop:
        frame_width, frame_height = frames.shape[2], frames.shape[3]
        min_side = min(frame_width, frame_height)
        frames = frames[:, :, (frame_width - min_side) // 2:(frame_width + min_side) // 2, (frame_height - min_side) // 2:(frame_height + min_side) // 2]

    if save_selected_frames:
        skvideo.io.vwrite(os.path.join(run_dir, 'selected_frames.mp4'), np.transpose(frames, (0, 3, 2, 1)))
    return frames


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Encoder options
@click.option('--encoder', type=click.Choice(['discriminator', 'vgg16', 'clip']), help='Choose the model to encode each frame into the latent space Z.', default='discriminator', show_default=True)
@click.option('--vgg16-layer', type=click.Choice(['conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'adavgpool', 'fc1', 'fc2']), help='Choose the layer to use from VGG16 (if used as encoder)', default='adavgpool', show_default=True)
# Source video options
@click.option('--source-video', '-video', 'video_file', type=click.Path(exists=True, dir_okay=False), help='Path to video file', required=True)
@click.option('--max-video-length', type=click.FloatRange(min=0.0, min_open=True), help='How many seconds of the video to take (from the starting second)', default=None, show_default=True)
@click.option('--starting-second', type=click.FloatRange(min=0.0), help='Second to start the video from', default=0.0, show_default=True)
@click.option('--frame-transform', type=click.Choice(['none', 'center-crop', 'resize']), help='TODO: Transform to apply to the individual frame.')
@click.option('--center-crop', is_flag=True, help='Center-crop each frame of the video')
@click.option('--save-selected-frames', is_flag=True, help='Save the selected frames of the input video after the selected transform')
# Synthesis options
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a dlatent (.npy/.npz)', default=None)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Video options
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out','visual-reactive'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def visual_reactive_interpolation(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        encoder: str,
        vgg16_layer: str,
        video_file: Union[str, os.PathLike],
        max_video_length: float,
        starting_second: float,
        frame_transform: str,
        center_crop: bool,
        save_selected_frames: bool,
        truncation_psi: float,
        new_center: Tuple[str, Union[int, np.ndarray]],
        noise_mode: str,
        anchor_latent_space: bool,
        outdir: Union[str, os.PathLike],
        description: str,
        compress: bool,
        smoothing_sec: float = 0.1  # For Gaussian blur; the lower, the faster the reaction; higher leads to more generated frames being the same
):
    print(f'Loading networks from "{network_pkl}"...')

    # Define the model (load both D, G, and the features of D)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if encoder == 'discriminator':
        print('Loading Discriminator and its features...')
        with dnnlib.util.open_url(network_pkl) as f:
            D = legacy.load_network_pkl(f)['D'].eval().requires_grad_(False).to(device)  # type: ignore

        D_features = DiscriminatorFeatures(D).requires_grad_(False).to(device)
        del D
    elif encoder == 'vgg16':
        print('Loading VGG16 and its features...')
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        vgg16_features = VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)
        del vgg16

    elif encoder == 'clip':
        print('Loading CLIP model...')
        try:
            import clip
        except ImportError:
            raise ImportError('clip not installed! Install it via "pip install git+https://github.com/openai/CLIP.git"')
        model, preprocess = clip.load('ViT-B/32', device=device)
        model = model.requires_grad_(False)  # Otherwise OOM

    print('Loading Generator...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)  # type: ignore

    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    if new_center is None:
        # Stick to the tracked center of W during training
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            new_center = f'seed_{new_center}'
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value, truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Create the run dir with the given name description; add slowdown if different than the default (1)
    description = 'visual-reactive' if len(description) == 0 else description
    run_dir = gen_utils.make_run_dir(outdir, description)
    # Name of the video
    video_name, _ = os.path.splitext(video_file)
    video_name = video_name.split(os.sep)[-1]  # Get the actual name of the video
    mp4_name = f'visual-reactive_{video_name}'

    # Get all the frames of the video and its properties
    # TODO: resize the frames to the size of the network (G.img_resolution)
    fps, max_video_length, starting_frame, max_frames, width, height = get_video_information(video_file,
                                                                                             max_video_length,
                                                                                             starting_second)

    videogen = skvideo.io.vreader(video_file)
    fake_dlatents = list()
    if save_selected_frames:
        # skvideo.io.vwrite sets FPS=25, so we have to manually enter it via FFmpeg
        # TODO: use only ffmpeg-python
        writer = skvideo.io.FFmpegWriter(os.path.join(run_dir, f'selected-frames_{video_name}.mp4'),
                                         inputdict={'-r': str(fps)})

    for idx, frame in enumerate(tqdm(videogen, desc=f'Getting frames+latents of "{video_name}"', unit='frames')):
        # Only save the frames that the user has selected
        if idx < starting_frame:
            continue
        if idx > starting_frame + max_frames:
            break

        if center_crop:
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            min_side = min(frame_width, frame_height)
            frame = frame[(frame_height - min_side) // 2:(frame_height + min_side) // 2, (frame_width - min_side) // 2:(frame_width + min_side) // 2, :]

        if save_selected_frames:
            writer.writeFrame(frame)

        # Get fake latents
        if encoder == 'discriminator':
            frame = normalize_image(frame)  # [0, 255] => [-1, 1]
            frame = torch.from_numpy(np.transpose(frame, (2, 1, 0))).unsqueeze(0).to(device)  # HWC => CWH => NCWH, N=1
            fake_z = D_features.get_layers_features(frame, layers=['fc'])[0]

        elif encoder == 'vgg16':
            preprocess = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
            frame = preprocess(frame).unsqueeze(0).to(device)
            fake_z = vgg16_features.get_layers_features(frame, layers=[vgg16_layer])[0]
            fake_z = fake_z.view(1, 512, -1).mean(2)  # [1, C, H, W] => [1, C]; can be used in any layer

        elif encoder == 'clip':
            frame = Image.fromarray(frame)  # [0, 255]
            frame = preprocess(frame).unsqueeze(0).to(device)
            fake_z = model.encode_image(frame)

        # Normalize the latent so that it's ~N(0, 1), or divide by its .max()
        # fake_z = fake_z / fake_z.max()
        fake_z = (fake_z - fake_z.mean()) / fake_z.std()

        # Get dlatent
        fake_w = G.mapping(fake_z, None)
        # Truncation trick
        fake_w = w_avg + (fake_w - w_avg) * truncation_psi
        fake_dlatents.append(fake_w)

    if save_selected_frames:
        # Close the video writer
        writer.close()

    # Set the fake_dlatents as a torch tensor; we can't just do torch.tensor(fake_dlatents) as with NumPy :(
    fake_dlatents = torch.cat(fake_dlatents, 0)
    # Smooth out so larger changes in the scene are the ones that affect the generation
    fake_dlatents = torch.from_numpy(nd.gaussian_filter(fake_dlatents.cpu(),
                                                        sigma=[smoothing_sec * fps, 0, 0])).to(device)

    # Auxiliary function for moviepy
    def make_frame(t):
        # Get the frame, dlatent, and respective image
        frame_idx = int(np.clip(np.round(t * fps), 0, len(fake_dlatents) - 1))
        fake_w = fake_dlatents[frame_idx]
        image = gen_utils.w_to_img(G, fake_w, noise_mode)
        # Create grid for this timestamp
        grid = gen_utils.create_image_grid(image, (1, 1))
        # Grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=max_video_length)
    videoclip.set_duration(max_video_length)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution, if successful)
    if compress:
        gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)

    # TODO: merge the videos side by side, but we will need them be the same height
    if save_selected_frames:
        # GUIDE: https://github.com/kkroening/ffmpeg-python/issues/150
        min_height = min(height, G.img_resolution)

        input0 = ffmpeg.input(os.path.join(run_dir, f'selected-frames_{video_name}.mp4'))
        input1 = ffmpeg.input(os.path.join(run_dir, f'{mp4_name}-compressed.mp4' if compress else f'{mp4_name}.mp4'))
        out = ffmpeg.filter([input0, input1], 'hstack').output(os.path.join(run_dir, 'side-by-side.mp4'))

    # Save the configuration used
    new_center = 'w_avg' if new_center is None else new_center
    ctx.obj = {
        'network_pkl': network_pkl,
        'encoder_options': {
            'encoder': encoder,
            'vgg16_layer': vgg16_layer,
        },
        'source_video_options': {
            'source_video': video_file,
            'sorce_video_params': {
                'fps': fps,
                'height': height,
                'width': width,
                'length': max_video_length,
                'starting_frame': starting_frame,
                'total_frames': max_frames
            },
            'max_video_length': max_video_length,
            'starting_second': starting_second,
            'frame_transform': frame_transform,
            'center_crop': center_crop,
            'save_selected_frames': save_selected_frames
        },
        'synthesis_options': {
            'truncation_psi': truncation_psi,
            'new_center': new_center,
            'noise_mode': noise_mode,
            'smoothing_sec': smoothing_sec
        },
        'video_options': {
            'compress': compress
        },
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }

    gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    visual_reactive_interpolation()


# ----------------------------------------------------------------------------
