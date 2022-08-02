import os
from typing import List, Optional, Union, Tuple
import click

import dnnlib
from torch_utils import gen_utils

import scipy
import numpy as np
import PIL.Image
import torch

import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor


# ----------------------------------------------------------------------------


# We group the different types of generation (images, grid, video, wacky stuff) into a main function
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='images')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options (feed a list of seeds or give the projected w to synthesize)
@click.option('--seeds', type=gen_utils.num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--projected-w', help='Projection result file; can be either .npy or .npz files', type=click.Path(exists=True, dir_okay=False), metavar='FILE')
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
# Grid options
@click.option('--save-grid', is_flag=True, help='Use flag to save image grid')
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Grid width (number of columns)', default=None)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Grid height (number of rows)', default=None)
# Extra parameters for saving the results
@click.option('--save-dlatents', is_flag=True, help='Use flag to save individual dlatents (W) for each individual resulting image')
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'images'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='generate-images', show_default=True)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        device: Optional[str],
        cfg: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        class_idx: Optional[int],
        noise_mode: str,
        anchor_latent_space: Optional[bool],
        projected_w: Optional[Union[str, os.PathLike]],
        new_center: Tuple[str, Union[int, np.ndarray]],
        save_grid: Optional[bool],
        grid_width: int,
        grid_height: int,
        save_dlatents: Optional[bool],
        outdir: str,
        description: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py images --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py images --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py images --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py images --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    Generate class conditional StyleGAN2 WikiArt images, save each individually, and save all of them as a grid
    python generate.py images --cfg=stylegan2 --network=wikiart1024-C --class=155 \\
        --trunc=0.7 --seeds=10-50 --save-grid
    """
    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    description = 'generate-images' if len(description) == 0 else description
    # Create the run dir with the given name description
    run_dir = gen_utils.make_run_dir(outdir, description)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws, ext = gen_utils.get_latent_from_file(projected_w, return_ext=True)
        ws = torch.tensor(ws, device=device)
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        n_digits = int(np.log10(len(ws))) + 1  # number of digits for naming the images
        if ext == '.npy':
            img = gen_utils.w_to_img(G, ws, noise_mode)[0]
            PIL.Image.fromarray(img, gen_utils.channels_dict[G.synthesis.img_channels]).save(f'{run_dir}/proj.png')
        else:
            for idx, w in enumerate(ws):
                img = gen_utils.w_to_img(G, w, noise_mode)[0]
                PIL.Image.fromarray(img,
                                    gen_utils.channels_dict[G.synthesis.img_channels]).save(f'{run_dir}/proj{idx:0{n_digits}d}.png')
        return

    # Labels.
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Recenter the latent space, if specified
    if new_center is None:
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value,
                                              truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Generate images.
    images = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        dlatent = gen_utils.get_w_from_seed(G, device, seed, truncation_psi=1.0)
        # Do truncation trick with center (new or global)
        w = w_avg + (dlatent - w_avg) * truncation_psi
        img = gen_utils.w_to_img(G, w, noise_mode)[0]
        if save_grid:
            images.append(img)
        PIL.Image.fromarray(img,
                            gen_utils.channels_dict[G.synthesis.img_channels]).save(os.path.join(run_dir, f'seed{seed}.png'))
        if save_dlatents:
            np.save(os.path.join(run_dir, f'seed{seed}.npy'), w.unsqueeze(0).cpu().numpy())

    if save_grid:
        print('Saving image grid...')
        # We let the function infer the shape of the grid
        if (grid_width, grid_height) == (None, None):
            PIL.Image.fromarray(gen_utils.create_image_grid(np.array(images)),
                                gen_utils.channels_dict[G.synthesis.img_channels]).save(os.path.join(run_dir, 'grid.png'))
        # The user tells the specific shape of the grid, but one value may be None
        else:
            PIL.Image.fromarray(gen_utils.create_image_grid(np.array(images), (grid_width, grid_height)),
                                gen_utils.channels_dict[G.synthesis.img_channels]).save(os.path.join(run_dir, 'grid.png'))

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'config': cfg,
        'seeds': seeds,
        'truncation_psi': truncation_psi,
        'class_idx': class_idx,
        'noise_mode': noise_mode,
        'save_grid': save_grid,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'run_dir': run_dir,
        'description': description,
        'projected_w': projected_w
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)


# ----------------------------------------------------------------------------


@main.command(name='random-video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options
@click.option('--seeds', type=gen_utils.num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Video options
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Video grid width / number of columns', default=None, show_default=True)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Video grid height / number of rows', default=None, show_default=True)
@click.option('--slowdown', type=gen_utils.parse_slowdown, help='Slow down the video by this amount; will be approximated to the nearest power of 2', default='1', show_default=True)
@click.option('--duration-sec', '-sec', type=float, help='Duration length of the video', default=30.0, show_default=True)
@click.option('--fps', type=click.IntRange(min=1), help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'video'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results')
def random_interpolation_video(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: str,
        seeds: List[int],
        truncation_psi: float,
        new_center: Tuple[str, Union[int, np.ndarray]],
        class_idx: Optional[int],
        noise_mode: str,
        anchor_latent_space: bool,
        grid_width: int,
        grid_height: int,
        slowdown: int,
        duration_sec: float,
        fps: int,
        outdir: Union[str, os.PathLike],
        description: str,
        compress: bool,
        smoothing_sec: Optional[float] = 3.0  # for Gaussian blur; won't be a command-line parameter, change at own risk
):
    """
    Generate a random interpolation video using a pretrained network.

    Examples:

    \b
    # Generate a 30-second long, untruncated MetFaces video at 30 FPS (3 rows and 2 columns; horizontal):
    python generate.py random-video --seeds=0-5 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate a 60-second long, truncated 1x2 MetFaces video at 60 FPS (2 rows and 1 column; vertical):
    python generate.py random-video --trunc=0.7 --seeds=10,20 --grid-width=1 --grid-height=2 \\
        --fps=60 -sec=60 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    """
    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Create the run dir with the given name description; add slowdown if different from the default (1)
    desc = 'random-video'
    desc = f'random-video-{description}' if description is not None else desc
    desc = f'{desc}-{slowdown}xslowdown' if slowdown != 1 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Number of frames in the video and its total duration in seconds
    num_frames = int(np.rint(duration_sec * fps))
    total_duration = duration_sec * slowdown

    print('Generating latent vectors...')
    # TODO: let another helper function handle each case, we will use it for the grid
    # If there's more than one seed provided and the shape isn't specified by the user
    if (grid_width is None and grid_height is None) and len(seeds) >= 1:
        # TODO: this can be done by another function
        # Number of images in the grid video according to the seeds provided
        num_seeds = len(seeds)
        # Get the grid width and height according to num, giving priority to the number of columns
        grid_width = max(int(np.ceil(np.sqrt(num_seeds))), 1)
        grid_height = max((num_seeds - 1) // grid_width + 1, 1)
        grid_size = (grid_width, grid_height)
        shape = [num_frames, G.z_dim]  # This is per seed
        # Get the z latents
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)

    # If only one seed is provided, but the user specifies the grid shape:
    elif None not in (grid_width, grid_height) and len(seeds) == 1:
        grid_size = (grid_width, grid_height)
        shape = [num_frames, np.prod(grid_size), G.z_dim]
        # Since we have one seed, we use it to generate all latents
        all_latents = np.random.RandomState(*seeds).randn(*shape).astype(np.float32)

    # If one or more seeds are provided, and the user also specifies the grid shape:
    elif None not in (grid_width, grid_height) and len(seeds) >= 1:
        # Case is similar to the first one
        num_seeds = len(seeds)
        grid_size = (grid_width, grid_height)
        available_slots = np.prod(grid_size)
        if available_slots < num_seeds:
            diff = num_seeds - available_slots
            click.secho(f'More seeds were provided ({num_seeds}) than available spaces in the grid ({available_slots})',
                        fg='red')
            click.secho(f'Removing the last {diff} seeds: {seeds[-diff:]}', fg='blue')
            seeds = seeds[:available_slots]
        shape = [num_frames, G.z_dim]
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)

    else:
        ctx.fail('Error: wrong combination of arguments! Please provide either a list of seeds, one seed and the grid '
                 'width and height, or more than one seed and the grid width and height')

    # Let's smooth out the random latents so that now they form a loop (and are correctly generated in a 512-dim space)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[smoothing_sec * fps, 0, 0], mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Name of the video
    mp4_name = f'{grid_width}x{grid_height}-slerp-{slowdown}xslowdown'

    # Labels.
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Let's slowdown the video, if so desired
    while slowdown > 1:
        all_latents, duration_sec, num_frames = gen_utils.double_slowdown(latents=all_latents,
                                                                          duration=duration_sec,
                                                                          frames=num_frames)
        slowdown //= 2

    if new_center is None:
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value,
                                              truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Auxiliary function for moviepy
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        latents = torch.from_numpy(all_latents[frame_idx]).to(device)
        # Do the truncation trick (with the global centroid or the new center provided by the user)
        w = G.mapping(latents, None)
        w = w_avg + (w - w_avg) * truncation_psi
        # Get the images with the new center
        images = gen_utils.w_to_img(G, w, noise_mode)
        # RGBA -> RGB
        images = images[:, :, :, :3]
        # Generate the grid for this timestamp
        grid = gen_utils.create_image_grid(images, grid_size)
        # moviepy.editor.VideoClip expects 3 channels
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(total_duration)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Save the configuration used
    new_center = 'w_avg' if new_center is None else new_center
    ctx.obj = {
        'network_pkl': network_pkl,
        'config': cfg,
        'seeds': seeds,
        'truncation_psi': truncation_psi,
        'new_center': new_center,
        'class_idx': class_idx,
        'noise_mode': noise_mode,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'slowdown': slowdown,
        'duration_sec': duration_sec,
        'video_fps': fps,
        'run_dir': run_dir,
        'description': desc,
        'compress': compress,
        'smoothing_sec': smoothing_sec
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Compress the video (lower file size, same resolution)
    if compress:
        gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------


@main.command('circular-video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options
@click.option('--seed', type=int, help='Random seed', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
# Video options
@click.option('--grid-width', '-gw', type=click.IntRange(min=1), help='Video grid width / number of columns', required=True)
@click.option('--grid-height', '-gh', type=click.IntRange(min=1), help='Video grid height / number of rows', required=True)
@click.option('--duration-sec', '-sec', type=float, help='Duration length of the video', default=10.0, show_default=True)
@click.option('--fps', type=click.IntRange(min=1), help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'video'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results')
def circular_video(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        cfg: str,
        seed: int,
        truncation_psi: float,
        new_center: Tuple[str, Union[int, np.ndarray]],
        class_idx: Optional[int],
        noise_mode: str,
        anchor_latent_space: bool,
        grid_width: int,
        grid_height: int,
        duration_sec: float,
        fps: int,
        compress: bool,
        outdir: Union[str, os.PathLike],
        description: str
):
    """
    Generate a circular interpolation video in two random axes of Z, given a seed
    """
    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Get the labels, if the model is conditional
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Get center of the latent space (global or user-indicated)
    if new_center is None:
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value,
                                              truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Create the run dir with the given name description; add slowdown if different from the default (1)
    desc = 'circular-video'
    desc = f'circular-video-{description}' if description is not None else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Calculate the total number of frames in the video
    num_frames = int(np.rint(duration_sec * fps))

    grid_size = (grid_width, grid_height)
    # Get the latents with the random state
    random_state = np.random.RandomState(seed)
    # Choose two random dims on which to plot the circles (from 0 to G.z_dim-1),
    # one pair for each element of the grid (2*grid_width*grid_height in total)
    z1, z2 = np.split(random_state.choice(G.z_dim, 2 * np.prod(grid_size), replace=False), 2)

    # We partition the circle in equal strides w.r.t. num_frames
    get_angles = lambda num_frames: np.linspace(0, 2*np.pi, num_frames)
    angles = get_angles(num_frames=num_frames)

    # Basic Polar to Cartesian transformation
    polar_to_cartesian = lambda radius, theta: (radius * np.cos(theta), radius * np.sin(theta))
    # Using a fixed radius (this value is irrelevant), we generate the circles in each chosen grid
    Z1, Z2 = polar_to_cartesian(radius=5.0, theta=angles)

    # Our latents will be comprising mostly of zeros
    all_latents = np.zeros([num_frames, np.prod(grid_size), G.z_dim]).astype(np.float32)
    # Obtain all the frames belonging to the specific box in the grid,
    # replacing the zero values with the circle perimeter values
    for box in range(np.prod(grid_size)):
        box_frames = all_latents[:, box]
        box_frames[:, [z1[box], z2[box]]] = np.vstack((Z1, Z2)).T

    # Auxiliary function for moviepy
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        latents = torch.from_numpy(all_latents[frame_idx]).to(device)
        # Get the images with the respective label
        dlatents = gen_utils.z_to_dlatent(G, latents, label, truncation_psi=1.0)  # Get the pure dlatent
        # Do truncation trick
        w = w_avg + (dlatents - w_avg) * truncation_psi
        # Get the images
        images = gen_utils.w_to_img(G, w, noise_mode)
        # RGBA -> RGB
        images = images[:, :, :, :3]
        # Generate the grid for this timestep
        grid = gen_utils.create_image_grid(images, grid_size)
        # Grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(duration_sec)

    # Name of the video
    mp4_name = f'{grid_width}x{grid_height}-circular'

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Save the configuration used
    new_center = 'w_avg' if new_center is None else new_center
    ctx.obj = {
        'network_pkl': network_pkl,
        'config': cfg,
        'seed': seed,
        'z1, z2': [[int(i), int(j)] for i, j in zip(z1, z2)],
        'truncation_psi': truncation_psi,
        'new_center': new_center,
        'class_idx': class_idx,
        'noise_mode': noise_mode,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'duration_sec': duration_sec,
        'video_fps': fps,
        'run_dir': run_dir,
        'description': desc,
        'compress': compress
    }
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Compress the video (lower file size, same resolution)
    if compress:
        gen_utils.compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
