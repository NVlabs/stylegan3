import torch
from torch.autograd import Variable
from torchvision import transforms

import PIL
from PIL import Image

try:
    import ffmpeg
except ImportError:
    raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')

import scipy.ndimage as nd
import numpy as np

import os
import click
from typing import Union, Tuple, Optional, List
from tqdm import tqdm

from torch_utils import gen_utils
from network_features import DiscriminatorFeatures


# ----------------------------------------------------------------------------


@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


def get_available_layers(max_resolution: int) -> List[str]:
    """Helper function to get the available layers given a max resolution (first block in the Discriminator)"""
    max_res_log2 = int(np.log2(max_resolution))
    block_resolutions = [2**i for i in range(max_res_log2, 2, -1)]

    available_layers = ['from_rgb']
    for block_res in block_resolutions:
        # We don't add the skip layer, as it's the same as conv1 (due to in-place addition; could be changed)
        available_layers.extend([f'b{block_res}_conv0', f'b{block_res}_conv1'])
    # We also skip 'b4_mbstd', as it doesn't add any new information compared to b8_conv1
    available_layers.extend(['b4_conv', 'fc', 'out'])
    return available_layers


# ----------------------------------------------------------------------------
# DeepDream code; modified from Erik Linder-Norén's repository: https://github.com/eriklindernoren/PyTorch-Deep-Dream

def get_image(seed: int = 0,
              image_noise: str = 'random',
              starting_image: Union[str, os.PathLike] = None,
              image_size: int = 1024,
              convert_to_grayscale: bool = False,
              device: torch.device = torch.device('cpu')) -> Tuple[PIL.Image.Image, str]:
    """Set the random seed (NumPy + PyTorch), as well as get an image from a path or generate a random one with the seed"""
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    # Load image or generate a random one if none is provided
    if starting_image is not None:
        image = Image.open(starting_image).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    else:
        if image_noise == 'random':
            starting_image = f'random_image-seed_{seed:08d}.jpg'
            image = Image.fromarray(rnd.randint(0, 255, (image_size, image_size, 3), dtype='uint8'))
        elif image_noise == 'perlin':
            try:
                # Graciously using Mathieu Duchesneau's implementation: https://github.com/duchesneaumathieu/pyperlin
                from pyperlin import FractalPerlin2D
                starting_image = f'perlin_image-seed_{seed:08d}.jpg'
                shape = (3, image_size, image_size)
                resolutions = [(2**i, 2**i) for i in range(1, 6+1)]  # for lacunarity = 2.0  # TODO: set as cli variable
                factors = [0.5**i for i in range(6)]  # for persistence = 0.5 TODO: set as cli variables
                g_cuda = torch.Generator(device=device).manual_seed(seed)
                rgb = FractalPerlin2D(shape, resolutions, factors, generator=g_cuda)().cpu().numpy()
                rgb = (255 * (rgb + 1) / 2).astype(np.uint8)  # [-1.0, 1.0] => [0, 255]
                image = Image.fromarray(np.stack(rgb, axis=2), 'RGB')

            except ImportError:
                raise ImportError('pyperlin not found! Install it via "pip install pyperlin"')

    if convert_to_grayscale:
        image = image.convert('L').convert('RGB')  # We do a little trolling to Pillow (so we have a 3-channel image)

    return image, starting_image


def crop_resize_rotate(img: PIL.Image.Image,
                       crop_size: int = None,
                       new_size: int = None,
                       rotation_deg: float = None,
                       translate_x: float = 0.0,
                       translate_y: float = 0.0) -> PIL.Image.Image:
    """Center-crop the input image into a square of sides crop_size; can be resized to new_size; rotated rotation_deg counter-clockwise"""
    # Center-crop the input image
    if crop_size is not None:
        w, h = img.size                                         # Input image width and height
        img = img.crop(box=((w - crop_size) // 2,               # Left pixel coordinate
                            (h - crop_size) // 2,               # Upper pixel coordinate
                            (w + crop_size) // 2,               # Right pixel coordinate
                            (h + crop_size) // 2))              # Lower pixel coordinate
    # Resize
    if new_size is not None:
        img = img.resize(size=(new_size, new_size),             # Requested size of the image in pixels; (width, height)
                         resample=Image.LANCZOS)                # Resampling filter
    # Rotation and translation
    if rotation_deg is not None:
        img = img.rotate(angle=rotation_deg,                    # Angle to rotate image, counter-clockwise
                         resample=Image.BICUBIC,                # Resampling filter; options: Image.Resampling.{NEAREST, BILINEAR, BICUBIC}
                         expand=False,                          # If True, the whole rotated image will be shown
                         translate=(translate_x, translate_y),  # Translate the image, from top-left corner (post-rotation)
                         fillcolor=(0, 0, 0))                   # Black background
    # TODO: tile the background
    return img


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np: torch.Tensor) -> np.ndarray:
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    # image_np = (image_np + 1.0) / 2.0
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (255 * image_np).astype('uint8')
    return image_np


def clip(image_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp per channel"""
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def dream(image: PIL.Image.Image,
          model: torch.nn.Module,
          layers: List[str],
          channels: List[int] = None,
          normed: bool = False,
          sqrt_normed: bool = False,
          iterations: int = 20,
          lr: float = 1e-2) -> np.ndarray:
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model.get_layers_features(image, layers=layers, channels=channels, normed=normed, sqrt_normed=sqrt_normed)
        loss = sum(layer.norm() for layer in out)                   # More than one layer may be used
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        # image.data = torch.clamp(image.data, -1.0, 1.0)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image: PIL.Image.Image,
               model: torch.nn.Module,
               model_resolution: int,
               layers: List[str],
               channels: List[int],
               seed: Union[int, type(None)],
               normed: bool,
               sqrt_normed: bool,
               iterations: int,
               lr: float,
               octave_scale: float,
               num_octaves: int,
               unzoom_octave: bool = False,
               disable_inner_tqdm: bool = False,
               ignore_initial_transform: bool = False) -> np.ndarray:
    """ Main deep dream method """
    # Center-crop and resize
    if not ignore_initial_transform:
        image = crop_resize_rotate(img=image, crop_size=min(image.size), new_size=model_resolution)
    # Preprocess image
    image = preprocess(image)
    # image = torch.from_numpy(np.array(image)).permute(-1, 0, 1) / 127.5 - 1.0  # alternative
    image = image.unsqueeze(0).cpu().data.numpy()
    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        # Alternatively, see if we get better results with: https://www.tensorflow.org/tutorials/generative/deepdream#taking_it_up_an_octave
        octave = nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1)
        # Necessary for StyleGAN's Discriminator, as it cannot handle any image size
        if unzoom_octave:
            octave = nd.zoom(octave, np.array(octaves[-1].shape) / np.array(octave.shape), order=1)
        octaves.append(octave)

    detail = np.zeros_like(octaves[-1])
    tqdm_desc = f'Dreaming w/layers {"|".join(x for x in layers)}'
    tqdm_desc = f'Seed: {seed} - {tqdm_desc}' if seed is not None else tqdm_desc
    for octave, octave_base in enumerate(tqdm(octaves[::-1], desc=tqdm_desc, disable=disable_inner_tqdm)):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, layers, channels, normed, sqrt_normed, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


# ----------------------------------------------------------------------------


@main.command(name='style-transfer')
def style_transfer_discriminator():
    print('Coming soon!')
    # Reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


# ----------------------------------------------------------------------------


@main.command(name='dream')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='Random seeds to use. Accepted comma-separated values, ranges, or combinations: "a,b,c", "a-c", "a,b-d,e".', default=0)
@click.option('--random-image-noise', '-noise', 'image_noise', type=click.Choice(['random', 'perlin']), default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=1e-2, show_default=True)
@click.option('--iterations', '-it', type=int, help='Number of gradient ascent steps per octave', default=20, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Layers of the Discriminator to use as the features. If "all", will generate a dream image per available layer in the loaded model. If "use_all", will use all available layers.', default='b16_conv1', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=int, help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=True, show_default=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seeds: List[int],
        image_noise: str,
        starting_image: Union[str, os.PathLike],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # TODO: conditional model
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: Optional[List[int]],
        norm_model_layers: bool,
        sqrt_norm_model_layers: bool,
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get the model resolution (image resizing and getting available layers)
    model_resolution = D.img_resolution

    # TODO: do this better, as we can combine these conditions later
    layers = layers.split(',')

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    if 'all' in layers:
        # Get all the available layers in a list
        layers = get_available_layers(max_resolution=model_resolution)

        for seed in seeds:
            # Get the image and image name
            image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                              starting_image=starting_image,
                                              image_size=model_resolution,
                                              convert_to_grayscale=convert_to_grayscale)

            # Make the run dir in the specified output directory
            desc = f'discriminator-dream-all_layers-seed_{seed}'
            desc = f'{desc}-{description}' if len(description) != 0 else desc
            run_dir = gen_utils.make_run_dir(outdir, desc)

            # Save starting image
            image.save(os.path.join(run_dir, f'{os.path.basename(starting_image).split(".")[0]}.jpg'))

            # Save the configuration used
            ctx.obj = {
                'network_pkl': network_pkl,
                'synthesis_options': {
                    'seed': seed,
                    'random_image_noise': image_noise,
                    'starting_image': starting_image,
                    'class_idx': class_idx,
                    'learning_rate': learning_rate,
                    'iterations': iterations},
                'layer_options': {
                    'layer': layers,
                    'channels': channels,
                    'norm_model_layers': norm_model_layers,
                    'sqrt_norm_model_layers': sqrt_norm_model_layers},
                'octaves_options': {
                    'num_octaves': num_octaves,
                    'octave_scale': octave_scale,
                    'unzoom_octave': unzoom_octave},
                'extra_parameters': {
                    'outdir': run_dir,
                    'description': description}
            }
            # Save the run configuration
            gen_utils.save_config(ctx=ctx, run_dir=run_dir)

            # For each layer:
            for layer in layers:
                # Extract deep dream image
                dreamed_image = deep_dream(image, model, model_resolution, layers=[layer], channels=channels, seed=seed, normed=norm_model_layers,
                                           sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                           octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave)

                # Save the resulting dreamed image
                filename = f'layer-{layer}_dreamed_{os.path.basename(starting_image).split(".")[0]}.jpg'
                Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

    else:
        if 'use_all' in layers:
            # Get all available layers
            layers = get_available_layers(max_resolution=model_resolution)
        else:
            # Parse the layers given by the user and leave only those available by the model
            available_layers = get_available_layers(max_resolution=model_resolution)
            layers = [layer for layer in layers if layer in available_layers]

        # Make the run dir in the specified output directory
        desc = f'discriminator-dream-layers_{"-".join(x for x in layers)}'
        desc = f'{desc}-{description}' if len(description) != 0 else desc
        run_dir = gen_utils.make_run_dir(outdir, desc)

        for seed in seeds:
            # Get the image and image name
            image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                              starting_image=starting_image,
                                              image_size=model_resolution,
                                              convert_to_grayscale=convert_to_grayscale)

            # Extract deep dream image
            dreamed_image = deep_dream(image, model, model_resolution, layers=layers, channels=channels, seed=seed, normed=norm_model_layers,
                                       sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                       octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave)

            # Save the configuration used
            ctx.obj = {
                'network_pkl': network_pkl,
                'synthesis_options': {
                    'seed': seed,
                    'starting_image': starting_image,
                    'class_idx': class_idx,
                    'learning_rate': learning_rate,
                    'iterations': iterations},
                'layer_options': {
                    'layer': layers,
                    'channels': channels,
                    'norm_model_layers': norm_model_layers,
                    'sqrt_norm_model_layers': sqrt_norm_model_layers},
                'octaves_options': {
                    'octave_scale': octave_scale,
                    'num_octaves': num_octaves,
                    'unzoom_octave': unzoom_octave},
                'extra_parameters': {
                'outdir': run_dir,
                'description': description}
            }
            # Save the run configuration
            gen_utils.save_config(ctx=ctx, run_dir=run_dir)

            # Save the resulting image and initial image
            filename = f'dreamed_{os.path.basename(starting_image)}'
            Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))
            image.save(os.path.join(run_dir, os.path.basename(starting_image)))
            starting_image = None


# ----------------------------------------------------------------------------


@main.command(name='dream-zoom')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--random-image-noise', '-noise', 'image_noise', type=click.Choice(['random', 'perlin']), default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Comma-separated list of the layers of the Discriminator to use as the features. If "use_all", will use all available layers.', default='b16_conv0', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# Individual frame manipulation options
@click.option('--pixel-zoom', '-zoom', type=int, help='How many pixels to zoom per step (positive for zoom in, negative for zoom out, padded with black)', default=2, show_default=True)
@click.option('--rotation-deg', '-rot', type=float, help='Rotate image counter-clockwise per frame (padded with black)', default=0.0, show_default=True)
@click.option('--translate-x', '-tx', type=float, help='Translate the image in the horizontal axis per frame (from left to right, padded with black)', default=0.0, show_default=True)
@click.option('--translate-y', '-ty', type=float, help='Translate the image in the vertical axis per frame (from top to bottom, padded with black)', default=0.0, show_default=True)
# Video options
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
@click.option('--duration-sec', type=float, help='Duration length of the video', default=15.0, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Add flag to reverse the generated video')
@click.option('--include-starting-image', type=bool, help='Include the starting image in the final video', default=True, show_default=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream_zoom(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seed: int,
        image_noise: Optional[str],
        starting_image: Optional[Union[str, os.PathLike]],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # TODO: conditional model
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: List[int],
        norm_model_layers: Optional[bool],
        sqrt_norm_model_layers: Optional[bool],
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: Optional[bool],
        pixel_zoom: int,
        rotation_deg: float,
        translate_x: int,
        translate_y: int,
        fps: int,
        duration_sec: float,
        reverse_video: bool,
        include_starting_image: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get the model resolution (for resizing the starting image if needed)
    model_resolution = D.img_resolution
    zoom_size = model_resolution - 2 * pixel_zoom

    layers = layers.split(',')
    if 'use_all' in layers:
        # Get all available layers
        layers = get_available_layers(max_resolution=model_resolution)
    else:
        # Parse the layers given by the user and leave only those available by the model
        available_layers = get_available_layers(max_resolution=model_resolution)
        layers = [layer for layer in layers if layer in available_layers]

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Get the image and image name
    image, starting_image = get_image(seed=seed, image_noise=image_noise,
                                      starting_image=starting_image,
                                      image_size=model_resolution,
                                      convert_to_grayscale=convert_to_grayscale)

    # Make the run dir in the specified output directory
    desc = 'discriminator-dream-zoom'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seed': seed,
            'random_image_noise': image_noise,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations
        },
        'layer_options': {
            'layers': layers,
            'channels': channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers
        },
        'octaves_options': {
            'num_octaves': num_octaves,
            'octave_scale': octave_scale,
            'unzoom_octave': unzoom_octave
        },
        'frame_manipulation_options': {
            'pixel_zoom': pixel_zoom,
            'rotation_deg': rotation_deg,
            'translate_x': translate_x,
            'translate_y': translate_y,
        },
        'video_options': {
            'fps': fps,
            'duration_sec': duration_sec,
            'reverse_video': reverse_video,
            'include_starting_image': include_starting_image,
        },
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    num_frames = int(np.rint(duration_sec * fps))  # Number of frames for the video
    n_digits = int(np.log10(num_frames)) + 1       # Number of digits for naming each frame

    # Save the starting image
    starting_image_name = f'dreamed_{0:0{n_digits}d}.jpg' if include_starting_image else 'starting_image.jpg'
    image.save(os.path.join(run_dir, starting_image_name))

    for idx, frame in enumerate(tqdm(range(num_frames), desc='Dreaming...', unit='frame')):
        # Zoom in after the first frame
        if idx > 0:
            image = crop_resize_rotate(image, crop_size=zoom_size, new_size=model_resolution,
                                       rotation_deg=rotation_deg, translate_x=translate_x, translate_y=translate_y)
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, seed=seed, normed=norm_model_layers,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations,
                                   lr=learning_rate, octave_scale=octave_scale, num_octaves=num_octaves,
                                   unzoom_octave=unzoom_octave, disable_inner_tqdm=True)

        # Save the resulting image and initial image
        filename = f'dreamed_{idx + 1:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

        # Now, the dreamed image is the starting image
        image = Image.fromarray(dreamed_image, 'RGB')

    # Save the final video
    print('Saving video...')
    ffmpeg_command = r'/usr/bin/ffmpeg' if os.name != 'nt' else r'C:\\Ffmpeg\\bin\\ffmpeg.exe'
    stream = ffmpeg.input(os.path.join(run_dir, f'dreamed_%0{n_digits}d.jpg'), framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, 'dream-zoom.mp4'), crf=20, pix_fmt='yuv420p')
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, cmd=ffmpeg_command)

    # Save the reversed video apart from the original one, so the user can compare both
    if reverse_video:
        stream = ffmpeg.input(os.path.join(run_dir, 'dream-zoom.mp4'))
        stream = stream.video.filter('reverse')
        stream = ffmpeg.output(stream, os.path.join(run_dir, 'dream-zoom_reversed.mp4'), crf=20, pix_fmt='yuv420p')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # ibidem


# ----------------------------------------------------------------------------

@main.command(name='channel-zoom')
@click.pass_context
@click.option('--network-pkl', help='Network pickle filename', required=True, type=click.Path(exists=True))
def channel_zoom():
    """Zoom in using all the channels of a network (or a specified layer)"""
    # TODO: Implement this
    pass

# ----------------------------------------------------------------------------


@main.command(name='random-interp')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), help='Model base configuration', default=None)
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='Random seeds to generate the Perlin noise from', required=True)
@click.option('--interp-type', '-interp', type=click.Choice(['linear', 'spherical']), help='Type of interpolation in Z or W', default='spherical', show_default=True)
@click.option('--smooth', is_flag=True, help='Add flag to smooth the interpolation between the seeds')
@click.option('--random-image-noise', '-noise', 'image_noise', type=click.Choice(['random', 'perlin']), default='random', show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--convert-to-grayscale', '-grayscale', is_flag=True, help='Add flag to grayscale the initial image')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=str, help='Comma-separated list of the layers of the Discriminator to use as the features. If "use_all", will use all available layers.', default='b16_conv0', show_default=True)
@click.option('--channels', type=gen_utils.num_range, help='Comma-separated list and/or range of the channels of the Discriminator to use as the features. If "None", will use all channels in each specified layer.', default=None, show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# TODO: Individual frame manipulation options
# Video options
@click.option('--seed-sec', '-sec', type=float, help='Number of seconds between each seed transition', default=5.0, show_default=True)
@click.option('--fps', type=gen_utils.parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def random_interpolation(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: Optional[str],
        seeds: List[int],
        interp_type: Optional[str],
        smooth: Optional[bool],
        image_noise: Optional[str],
        starting_image: Optional[Union[str, os.PathLike]],
        convert_to_grayscale: bool,
        class_idx: Optional[int],  # TODO: conditional model
        learning_rate: float,
        iterations: int,
        layers: str,
        channels: List[int],
        norm_model_layers: Optional[bool],
        sqrt_norm_model_layers: Optional[bool],
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: Optional[bool],
        seed_sec: float,
        fps: int,
        outdir: Union[str, os.PathLike],
        description: str,
):
    """Do a latent walk between random Perlin images (given the seeds) and generate a video with these frames."""
    # TODO: To make this better and more stable, we generate Perlin noise animations, not interpolations
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load Discriminator
    D = gen_utils.load_network('D', network_pkl, cfg, device)

    # Get model resolution
    model_resolution = D.img_resolution
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    layers = layers.split(',')
    # Get all available layers
    if 'use_all' in layers:
        layers = get_available_layers(max_resolution=model_resolution)
    else:
        # Parse the layers given by the user and leave only those available by the model
        available_layers = get_available_layers(max_resolution=model_resolution)
        layers = [layer for layer in layers if layer in available_layers]

    # Make the run dir in the specified output directory
    desc = f'random-interp-layers_{"-".join(x for x in layers)}'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Number of steps to take between each random image
    n_steps = int(np.rint(seed_sec * fps))
    # Total number of frames
    num_frames = int(n_steps * (len(seeds) - 1))
    # Total video length in seconds
    duration_sec = num_frames / fps

    # Number of digits for naming purposes
    n_digits = int(np.log10(num_frames)) + 1

    # Create interpolation of noises
    random_images = []
    for seed in seeds:
        # Get the starting seed and image
        image, _ = get_image(seed=seed, image_noise=image_noise, starting_image=starting_image,
                             image_size=model_resolution, convert_to_grayscale=convert_to_grayscale)
        image = np.array(image) / 255.0
        random_images.append(image)
    random_images = np.stack(random_images)

    all_images = np.empty([0] + list(random_images.shape[1:]), dtype=np.float32)
    # Do interpolation
    for i in range(len(random_images) - 1):
        # Interpolate between each pair of images
        interp = gen_utils.interpolate(random_images[i], random_images[i + 1], n_steps, interp_type, smooth)
        # Append it to the list of all images
        all_images = np.append(all_images, interp, axis=0)

    # DeepDream expects a list of PIL.Image objects
    pil_images = []
    for idx in range(len(all_images)):
        im = (255 * all_images[idx]).astype(dtype=np.uint8)
        pil_images.append(Image.fromarray(im))

    for idx, image in enumerate(tqdm(pil_images, desc='Interpolating...', unit='frame', total=num_frames)):
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, channels=channels, seed=None,
                                   normed=norm_model_layers, disable_inner_tqdm=True, ignore_initial_transform=True,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                   octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave)

        # Save the resulting image and initial image
        filename = f'{image_noise}-interpolation_frame_{idx:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seeds': seeds,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations},
        'layer_options': {
            'layer': layers,
            'channels': channels,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers},
        'octaves_options': {
            'octave_scale': octave_scale,
            'num_octaves': num_octaves,
            'unzoom_octave': unzoom_octave},
        'extra_parameters': {
            'outdir': run_dir,
            'description': description}
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Generate video
    print('Saving video...')
    ffmpeg_command = r'/usr/bin/ffmpeg' if os.name != 'nt' else r'C:\\Ffmpeg\\bin\\ffmpeg.exe'
    stream = ffmpeg.input(os.path.join(run_dir, f'{image_noise}-interpolation_frame_%0{n_digits}d.jpg'), framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, f'{image_noise}-interpolation.mp4'), crf=20, pix_fmt='yuv420p')
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, cmd=ffmpeg_command)

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
