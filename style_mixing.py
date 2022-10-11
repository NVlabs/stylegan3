import os
from typing import List, Union, Optional, Tuple
import click

import dnnlib
from torch_utils import gen_utils

import numpy as np
import PIL.Image
import scipy
import torch

import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor

# ----------------------------------------------------------------------------


# TODO: this is no longer true for StyleGAN3, we have 14 layers irrespective of resolution
def parse_styles(s: str) -> List[int]:
    """
    Helper function for parsing style layers. s will be a comma-separated list of values, and these can be
    either ranges ('a-b'), ints ('a', 'b', 'c', ...), or the style layer names ('coarse', 'middle', 'fine').

    A combination of these can also be used. For example, if the user wishes to mix the 'coarse' and 'fine'
    layers, then the input can be: 'coarse,fine'. If just the 'middle' and '14-17' layers are to be used,
    then 'middle,14-17' or '14-17,middle' can be the used as input.

    The repeated styles will be deleted, as these won't add anything to our final result.
    """
    style_layers_dict = {'coarse': list(range(0, 4)), 'middle': list(range(4, 8)), 'fine': list(range(8, 18))}
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if el in style_layers_dict:
            nums.extend(style_layers_dict[el])
        else:
            nums.extend(gen_utils.num_range(el, remove_repeated=True))
    # Sanity check: delete repeating numbers and limit values between 0 and 17
    nums = list(set([max(min(x, 17), 0) for x in nums]))
    return nums


# TODO: For StyleGAN3, there's only 'coarse' and 'fine' groups, though the boundary is not 100% clear
def style_names(max_style: int, file_name: str, desc: str, col_styles: List[int]) -> Tuple[str, str]:
    """
    Add the styles if they are being used (from the StyleGAN paper)
    to both the file name and the new directory to be created.
    """
    if list(range(0, 4)) == col_styles:
        styles = 'coarse_styles'
    elif list(range(4, 8)) == col_styles:
        styles = 'middle_styles'
    elif list(range(8, max_style)) == col_styles:
        styles = 'fine_styles'
    elif list(range(0, 8)) == col_styles:
        styles = 'coarse+middle_styles'
    elif list(range(4, max_style)) == col_styles:
        styles = 'middle+fine_styles'
    elif list(range(0, 4)) + list(range(8, max_style)) == col_styles:
        styles = 'coarse+fine_styles'
    else:
        styles = 'custom_styles'

    file_name = f'{file_name}-{styles}'
    desc = f'{desc}-{styles}'

    return file_name, desc


def _parse_cols(s: str, G, device: torch.device, truncation_psi: float) -> List[torch.Tensor]:
    """s can be a path to a npy/npz file or a seed number (int)"""
    s = s.split(',')
    w = torch.Tensor().to(device)
    for el in s:
        if os.path.isfile(el):
            w_el = gen_utils.get_latent_from_file(el)  # np.ndarray
            w_el = torch.from_numpy(w_el).to(device)  # torch.tensor
            w = torch.cat((w_el, w))
        else:
            nums = gen_utils.num_range(el, remove_repeated=True)
            for n in nums:
                w = torch.cat((gen_utils.get_w_from_seed(G, device, n, truncation_psi), w))
    return w


# ----------------------------------------------------------------------------


# We group the different types of style-mixing (grid and video) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='grid')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
# Synthesis options
@click.option('--row-seeds', '-rows', 'row_seeds', type=gen_utils.num_range, help='Random seeds to use for image rows', required=True)
@click.option('--col-seeds', '-cols', 'col_seeds', type=gen_utils.num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=parse_styles, help='Style layers to use; can pass "coarse", "middle", "fine", or a list or range of ints', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'images'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def generate_style_mix(
        ctx: click.Context,
        network_pkl: str,
        cfg: Optional[str],
        device: Optional[str],
        row_seeds: List[int],
        col_seeds: List[int],
        col_styles: List[int],
        truncation_psi: float,
        noise_mode: str,
        anchor_latent_space: bool,
        outdir: str,
        description: str,
):
    """Generate style-mixing images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py grid --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    # TODO: add class_idx
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')

    # Load the network
    G = gen_utils.load_network('G_ema', network_pkl, cfg, device)

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Sanity check: loaded model and selected styles must be compatible
    max_style = G.mapping.num_ws
    if max(col_styles) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style - 1} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_styles[:] = [style for style in col_styles if style < max_style]

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))  # TODO: change this in order to use _parse_cols
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = gen_utils.w_to_img(G, all_w, noise_mode)
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = gen_utils.w_to_img(G, w, noise_mode)[0]
            image_dict[(row_seed, col_seed)] = image

    # Name of grid and run directory
    grid_name = 'grid'
    description = 'stylemix-grid' if len(description) == 0 else description
    # Add to the name the styles (from the StyleGAN paper) if they are being used
    grid_name, description = style_names(max_style, grid_name, description, col_styles)
    # Create the run dir with the given name description
    run_dir = gen_utils.make_run_dir(outdir, description)

    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new(gen_utils.channels_dict[G.synthesis.img_channels],  # Handle RGBA case
                           (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key],
                                             gen_utils.channels_dict[G.synthesis.img_channels]),
                         (W * col_idx, H * row_idx))
    canvas.save(os.path.join(run_dir, f'{grid_name}.png'))

    print('Saving individual images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image,
                            gen_utils.channels_dict[G.synthesis.img_channels]).save(os.path.join(run_dir, f'{row_seed}-{col_seed}.png'))

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'row_seeds': row_seeds,
        'col_seeds': col_seeds,
        'col_styles': col_styles,
        'truncation_psi': truncation_psi,
        'noise_mode': noise_mode,
        'run_dir': run_dir,
        'description': description,
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


@main.command(name='video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), help='Noise mode', default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--row-seed', '-row', 'row_seed', type=int, help='Random seed to use for video row', required=True)
@click.option('--columns', '-cols', 'columns', type=str, help='Path to dlatents (.npy/.npz) or seeds to use ("a", "b-c", "e,f-g,h,i", etc.), or a combination of both', required=True)
@click.option('--styles', 'col_styles', type=parse_styles, help='Style layers to use; can pass "coarse", "middle", "fine", or a list or range of ints', default='0-6', show_default=True)
@click.option('--only-stylemix', is_flag=True, help='Add flag to only show the style-mixed images in the video')
# Video options
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file via ffmpeg-python (same resolution, lower file size)')
@click.option('--duration-sec', type=float, help='Duration of the video in seconds', default=30, show_default=True)
@click.option('--fps', type=click.IntRange(min=1), help='Video FPS.', default=30, show_default=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'video'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def random_stylemix_video(
        ctx: click.Context,
        network_pkl: str,
        cfg: Optional[str],
        row_seed: int,
        columns: str,
        col_styles: List[int],
        only_stylemix: bool,
        compress: bool,
        truncation_psi: float,
        noise_mode: str,
        anchor_latent_space: bool,
        fps: int,
        duration_sec: float,
        outdir: Union[str, os.PathLike],
        description: str,
        smoothing_sec: Optional[float] = 3.0  # for Gaussian blur; won't be a parameter, change at own risk
):
    """Generate random style-mixing video using pretrained network pickle.

        Examples:

        \b
        python style_mixing.py video --row=85 --cols=55,821,1789 --fps=60 \\
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

        \b
        python style_mixing.py video --row=0 --cols=7-10 --styles=fine,1,3,5-7 --duration-sec=60 \\
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    # TODO: add class_idx
    # Calculate number of frames
    num_frames = int(np.rint(duration_sec * fps))

    device = torch.device('cuda')

    # Load the network
    G = gen_utils.load_network('G_ema', network_pkl, cfg, device)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Get the average dlatent
    w_avg = G.mapping.w_avg

    # Sanity check: loaded model and selected styles must be compatible
    max_style = G.mapping.num_ws
    if max(col_styles) > max_style:
        click.secho(f'WARNING: Maximum col-style allowed: {max_style - 1} for loaded network "{network_pkl}" '
                    f'of resolution {G.img_resolution}x{G.img_resolution}', fg='red')
        click.secho('Removing col-styles exceeding this value...', fg='blue')
        col_styles[:] = [style for style in col_styles if style < max_style]

    # First column (video) latents
    print('Generating source W vectors...')
    src_shape = [num_frames, G.z_dim]
    src_z = np.random.RandomState(row_seed).randn(*src_shape).astype(np.float32)
    src_z = scipy.ndimage.gaussian_filter(src_z, sigma=[smoothing_sec * fps, 0], mode='wrap')  # wrap to form a loop
    src_z /= np.sqrt(np.mean(np.square(src_z)))  # normalize

    # Map to W and do truncation trick
    src_w = G.mapping(torch.from_numpy(src_z).to(device), None)
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # First row (images) latents
    dst_w = _parse_cols(columns, G, device, truncation_psi)
    # dst_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in col_seeds])
    # dst_w = G.mapping(torch.from_numpy(dst_z).to(device), None)
    # dst_w = w_avg + (dst_w - w_avg) * truncation_psi

    # Width and height of the generated image
    W = G.img_resolution
    H = G.img_resolution

    # Video name
    mp4_name = f'{len(dst_w)}x1'
    # Run dir name
    description = 'stylemix-video' if len(description) == 0 else description
    # Add to the name the styles (from the StyleGAN paper) if they are being used to both file and run dir
    mp4_name, description = style_names(max_style, mp4_name, description, col_styles)
    # Create the run dir with the description
    run_dir = gen_utils.make_run_dir(outdir, description)

    # If user wishes to only show the style-transferred images (nice for 1x1 case)
    if only_stylemix:
        print('Generating style-mixing video (saving only the style-transferred images)...')
        # We generate a canvas where we will paste all the generated images
        canvas = PIL.Image.new('RGB', (W * len(dst_w), H * len([row_seed])), 'black')  # use any color you want

        def make_frame(t):
            # Get the frame number according to time t
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            # For each of the column images
            for col, _ in enumerate(dst_w):
                # Select the pertinent latent w column
                w_col = dst_w[col].unsqueeze(0)  # [18, 512] -> [1, 18, 512]
                # Replace the values defined by col_styles
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate the style-mixed images
                col_images = gen_utils.w_to_img(G, w_col, noise_mode)
                # Paste them in their respective spot in the grid
                for row, image in enumerate(list(col_images)):
                    canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * H, row * W))

            return np.array(canvas)

        mp4_name = f'{mp4_name}-only-stylemix'
    else:
        print('Generating style-mixing video (saving the whole grid)...')
        # Generate an empty canvas where we will paste all the generated images
        canvas = PIL.Image.new('RGB', (W * (len(dst_w) + 1), H * (len([row_seed]) + 1)), 'black')

        # Generate all destination images (first row; static images)
        dst_images = gen_utils.w_to_img(G, dst_w, noise_mode)
        # Paste them in the canvas
        for col, dst_image in enumerate(list(dst_images)):
            canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), ((col + 1) * H, 0))

        def make_frame(t):
            # Get the frame number according to time t
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            # Get the image at this frame (first column; video)
            src_image = gen_utils.w_to_img(G, src_w[frame_idx], noise_mode)[0]
            # Paste it to the lower left
            canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, H))

            # For each of the column images (destination images)
            for col, _ in enumerate(list(dst_images)):
                # Select pertinent latent w column
                w_col = dst_w[col].unsqueeze(0)  # [18, 512] -> [1, 18, 512]
                # Replace the values defined by col_styles
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate these style-mixed images
                col_images = gen_utils.w_to_img(G, w_col, noise_mode)
                # Paste them in their respective spot in the grid
                for row, image in enumerate(list(col_images)):
                    canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * H, (row + 1) * W))

            return np.array(canvas)

        mp4_name = f'{mp4_name}-style-mixing'

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(duration_sec)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Save the configuration used for the experiment
    ctx.obj = {
        'network_pkl': network_pkl,
        'row_seed': row_seed,
        'columns': columns,
        'col_styles': col_styles,
        'only_stylemix': only_stylemix,
        'compress': compress,
        'truncation_psi': truncation_psi,
        'noise_mode': noise_mode,
        'duration_sec': duration_sec,
        'video_fps': fps,
        'run_dir': run_dir,
        'description': description,
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Compress the video (smaller file size, same resolution; not guaranteed though)
    if compress:
        gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
