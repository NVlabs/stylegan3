import os
import re
import json

from typing import List, Tuple, Union, Optional
from collections import OrderedDict
from locale import atof

import click
import numpy as np
import torch

import dnnlib
import legacy


# ----------------------------------------------------------------------------


channels_dict = {1: 'L', 3: 'RGB', 4: 'RGBA'}


# ----------------------------------------------------------------------------


def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


# ----------------------------------------------------------------------------


def parse_fps(fps: Union[str, int]) -> int:
    """Return FPS for the video; at worst, video will be 1 FPS, but no lower.
    Useful if we don't have click, else simply use click.IntRange(min=1)"""
    if isinstance(fps, int):
        return max(fps, 1)
    try:
        fps = int(atof(fps))
        return max(fps, 1)
    except ValueError:
        print(f'Typo in "--fps={fps}", will use default value of 30')
        return 30


def num_range(s: str, remove_repeated: bool = False) -> List[int]:
    """
    Extended helper function from the original (original is contained here).
    Accept a comma separated list of numbers 'a,b,c', a range 'a-c', or a combination
    of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...', and return as a list of ints.
    """
    nums = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for el in s.split(','):
        match = range_re.match(el)
        if match:
            # Sanity check 1: accept ranges 'a-b' or 'b-a', with a<=b
            lower, upper = int(match.group(1)), int(match.group(2))
            if lower <= upper:
                r = list(range(lower, upper + 1))
            else:
                r = list(range(upper, lower + 1))
            # We will extend nums as r is also a list
            nums.extend(r)
        else:
            # It's a single number, so just append it (if it's an int)
            try:
                nums.append(int(atof(el)))
            except ValueError:
                continue  # we ignore bad values
    # Sanity check 2: delete repeating numbers by default, but keep order given by user
    if remove_repeated:
        nums = list(OrderedDict.fromkeys(nums))
    return nums


def float_list(s: str) -> List[float]:
    """
    Helper function for parsing a string of comma-separated floats and returning each float
    """
    str_list = s.split(',')
    nums = []
    float_re = re.compile(r'^(\d+.\d+)$')
    for el in str_list:
        match = float_re.match(el)
        if match:
            nums.append(float(match.group(1)))
        else:
            try:
                nums.append(float(el))
            except ValueError:
                continue  # Ignore bad values

    return nums


def parse_slowdown(slowdown: Union[str, int]) -> int:
    """Function to parse the 'slowdown' parameter by the user. Will approximate to the nearest power of 2."""
    # TODO: slowdown should be any int
    if not isinstance(slowdown, int):
        try:
            slowdown = atof(slowdown)
        except ValueError:
            print(f'Typo in "{slowdown}"; will use default value of 1')
            slowdown = 1
    assert slowdown > 0, '"slowdown" cannot be negative or 0!'
    # Let's approximate slowdown to the closest power of 2 (nothing happens if it's already a power of 2)
    slowdown = 2**int(np.rint(np.log2(slowdown)))
    return max(slowdown, 1)  # Guard against 0.5, 0.25, ... cases


def parse_new_center(s: str) -> Tuple[str, Union[int, Tuple[np.ndarray, Optional[str]]]]:
    """Get a new center for the W latent space (a seed or projected dlatent; to be transformed later)"""
    try:
        new_center = int(s)  # it's a seed
        return s, new_center
    except ValueError:
        new_center = get_latent_from_file(s, return_ext=False)  # it's a projected dlatent
        return s, new_center


def load_network(name: str, network_pkl: Union[str, os.PathLike], cfg: Optional[str], device: torch.device):
    """Load and return the discriminator D from a trained network."""
    # Define the model
    if cfg is not None:
        assert network_pkl in resume_specs[cfg], f'This model is not available for config {cfg}!'
        network_pkl = resume_specs[cfg][network_pkl]
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = legacy.load_network_pkl(f)[name].eval().requires_grad_(False).to(device)  # type: ignore

    return net


def parse_class(G, class_idx: int, ctx: click.Context) -> Union[int, type(None)]:
    """Parse the class_idx and return it, if it's allowed by the conditional model G"""
    if G.c_dim == 0:
        # Unconditional model
        return None
    # Conditional model, so class must be specified by user
    if class_idx is None:
        ctx.fail('Must specify class label with --class when using a conditional network!')
    if class_idx not in range(G.c_dim):
        ctx.fail(f'Your class label can be at most {G.c_dim - 1}!')
    print(f'Using class {class_idx} (available labels: range({G.c_dim - 1})...)')
    return class_idx


# ----------------------------------------------------------------------------


def compress_video(
        original_video: Union[str, os.PathLike],
        original_video_name: Union[str, os.PathLike],
        outdir: Union[str, os.PathLike],
        ctx: click.Context) -> None:
    """ Helper function to compress the original_video using ffmpeg-python. moviepy creates huge videos, so use
        ffmpeg to 'compress' it (won't be perfect, 'compression' will depend on the video dimensions). ffmpeg
        can also be used to e.g. resize the video, make a GIF, save all frames in the video to the outdir, etc.
    """
    try:
        import ffmpeg
    except (ModuleNotFoundError, ImportError):
        ctx.fail('Missing ffmpeg! Install it via "pip install ffmpeg-python"')

    print('Compressing the video...')
    resized_video_name = os.path.join(outdir, f'{original_video_name}-compressed.mp4')
    ffmpeg.input(original_video).output(resized_video_name).run(capture_stdout=True, capture_stderr=True)
    print('Success!')


# ----------------------------------------------------------------------------


def interpolation_checks(
        t: Union[float, np.ndarray],
        v0: np.ndarray,
        v1: np.ndarray) -> Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:
    """Tests for the interpolation functions"""
    # Make sure 0.0<=t<=1.0
    assert np.min(t) >= 0.0 and np.max(t) <= 1.0
    # Guard against v0 and v1 not being NumPy arrays
    if not isinstance(v0, np.ndarray):
        v0 = np.array(v0)
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    # Both should have the same shape in order to interpolate between them
    assert v0.shape == v1.shape, f'Incompatible shapes! v0: {v0.shape}, v1: {v1.shape}'
    return t, v0, v1


def lerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray]) -> np.ndarray:
    """
    Linear interpolation between v0 (starting) and v1 (final) vectors; for optimal results,
    use t as an np.ndarray to return all results at once via broadcasting
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    v2 = (1.0 - t) * v0 + t * v1
    return v2


def slerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        dot_threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical linear interpolation between v0 (starting) and v1 (final) vectors; for optimal
    results, use t as an np.ndarray to return all results at once via broadcasting.

    dot_threshold is the threshold for considering if the two vectors are collinear (not recommended to alter).

    Adapted from the Python code at: https://en.wikipedia.org/wiki/Slerp (at the time, now no longer available).
    Most likely taken from Jonathan Blow's code in C++:
            http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It
    """
    t, v0, v1 = interpolation_checks(t, v0, v1)
    # Copy vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't always use np.dot, so we use the definition)
    dot = np.sum(v0 * v1)
    # If it's ~1, vectors are ~colineal, so use lerp
    if np.abs(dot) > dot_threshold:
        return lerp(t, v0, v1)
    # Stay within domain of arccos
    dot = np.clip(dot, -1.0, 1.0)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Divide the angle into t steps
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def interpolate(
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        n_steps: int,
        interp_type: str = 'spherical',
        smooth: bool = False) -> np.ndarray:
    """
    Interpolation function between two vectors, v0 and v1. We will either do a 'linear' or 'spherical' interpolation,
    taking n_steps. The steps can be 'smooth'-ed out, so that the transition between vectors isn't too drastic.
    """
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False)
    # TODO: have a dictionary with easing functions that contains my 'smooth' one (might be useful for someone else)
    if smooth:
        # Smooth out the interpolation with a polynomial of order 3 (cubic function f)
        # Constructed f by setting f'(0) = f'(1) = 0, and f(0) = 0, f(1) = 1 => f(t) = -2t^3+3t^2 = t^2 (3-2t)
        # NOTE: I've merely rediscovered the Smoothstep function S_1(x): https://en.wikipedia.org/wiki/Smoothstep
        t_array = t_array ** 2 * (3 - 2 * t_array)  # One line thanks to NumPy arrays
    # TODO: this might be possible to optimize by using the fact they're numpy arrays, but haven't found a nice way yet
    funcs_dict = {'linear': lerp, 'spherical': slerp}
    vectors = np.array([funcs_dict[interp_type](t, v0, v1) for t in t_array], dtype=np.float32)
    return vectors


# ----------------------------------------------------------------------------


def double_slowdown(latents: np.ndarray, duration: float, frames: int) -> Tuple[np.ndarray, float, int]:
    """
    Auxiliary function to slow down the video by 2x. We return the new latents, duration, and frames of the video
    """
    # Make an empty latent vector with double the amount of frames, but keep the others the same
    z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)
    # In the even frames, populate it with the latents
    for i in range(len(latents)):
        z[2 * i] = latents[i]
    # Interpolate in the odd frames
    for i in range(1, len(z), 2):
        # slerp between (t=0.5) even frames; for the last frame, we loop to the first one (z[0])
        z[i] = slerp(0.5, z[i - 1], z[i + 1]) if i != len(z) - 1 else slerp(0.5, z[0], z[i - 1])
    # TODO: we could change this to any slowdown: slerp(1/slowdown, ...), and we return z, slowdown * duration, ...
    # Return the new latents, and the respective new duration and number of frames
    return z, 2 * duration, 2 * frames


# ----------------------------------------------------------------------------


def make_affine_transform(m: Union[torch.Tensor, np.ndarray] = None,
                          angle: float = 0.0,
                          translate_x: float = 0.0,
                          translate_y: float = 0.0,
                          scale_x: float = 1.0,
                          scale_y: float = 1.0,
                          shear_x: float = 0.0,
                          shear_y: float = 0.0,
                          mirror_x: bool = False,
                          mirror_y: bool = False) -> np.array:
    """Make affine transformation with the given parameters. If none are passed, will return the identity.
    As a guide for affine transformations: https://en.wikipedia.org/wiki/Affine_transformation"""
    # m is the starting affine transformation matrix (e.g., G.synthesis.input.transform)
    if m is None:
        m = np.eye(3, dtype=np.float64)
    elif isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    elif isinstance(m, np.ndarray):
        pass
    # Remember these are the inverse transformations!
    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                [-np.sin(angle), np.cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
    # Translation matrix
    translation_matrix = np.array([[1.0, 0.0, -translate_x],
                                   [0.0, 1.0, -translate_y],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)
    # Scale matrix (don't let it go into negative or 0)
    scale_matrix = np.array([[1. / max(scale_x, 1e-4), 0.0, 0.0],
                             [0.0, 1. / max(scale_y, 1e-4), 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    # Shear matrix
    shear_matrix = np.array([[1.0, -shear_x, 0.0],
                             [-shear_y, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
    # Mirror/reflection in x matrix
    xmirror_matrix = np.array([[1.0 - 2 * mirror_x, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
    # Mirror/reflection in y matrix
    ymirror_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0 - 2 * mirror_y, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)

    # Make the resulting affine transformation (note that these are non-commutative, so we *choose* this order)
    m = m @ rotation_matrix @ translation_matrix @ scale_matrix @ shear_matrix @ xmirror_matrix @ ymirror_matrix
    return m


def anchor_latent_space(G) -> None:
    # Thanks to @RiversHaveWings and @nshepperd1
    if hasattr(G.synthesis, 'input'):
        # Unconditional models differ by a bit
        if G.c_dim == 0:
            shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0)).squeeze(0)
        else:
            shift = G.synthesis.input.affine(G.mapping.w_avg).mean(0)
        G.synthesis.input.affine.bias.data.add_(shift)
        G.synthesis.input.affine.weight.data.zero_()


def force_fp32(G) -> None:
    """Force fp32 as in during training"""
    G.synthesis.num_fp16_res = 0
    for name, layer in G.synthesis.named_modules():
        if hasattr(layer, 'conv_clamp'):
            layer.conv_clamp = None
            layer.use_fp16 = False


def use_cpu(G) -> None:
    """Use the CPU instead of the GPU; force_fp32 must be set to True, apart from the device setting"""
    # @nurpax found this before: https://github.com/NVlabs/stylegan2-ada-pytorch/issues/54#issuecomment-793713965, but we
    # will use @JCBrouwer's solution:  https://github.com/NVlabs/stylegan2-ada-pytorch/issues/105#issuecomment-838577639
    import functools
    G.forward = functools.partial(G.forward, force_fp32=True)

# ----------------------------------------------------------------------------

resume_specs = {
        # For StyleGAN2/ADA models; --cfg=stylegan2
        'stylegan2': {
            # Official NVIDIA models
            'ffhq256':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'ffhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'celebahq256':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'lsundog256':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'afhqcat512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'afhqdog512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'afhqwild512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'brecahad512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'cifar10':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl',
            # Other configs are available at: https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/, but I will list here the config-f only
            'lsuncar512':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl',  # config-f
            'lsuncat256':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl',  # config-f
            'lsunchurch256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl',  # config-f
            'lsunhorse256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl',  # config-f
            # Community models. More can be found at: https://github.com/justinpinkney/awesome-pretrained-stylegan2 by @justinpinkney, but weren't added here
            'minecraft1024': 'https://github.com/jeffheaton/pretrained-gan-minecraft/releases/download/v1/minecraft-gan-2020-12-22.pkl',  # Thanks to @jeffheaton
            'imagenet512':   'https://battle.shawwn.com/sdc/stylegan2-imagenet-512/model.ckpt-533504.pkl',  # Thanks to @shawwn
            'wikiart1024-C': 'https://archive.org/download/wikiart-stylegan2-conditional-model/WikiArt5.pkl',  # Thanks to @pbaylies; conditional (167 classes in total: --class=0 to 166)
            'wikiart1024-U': 'https://archive.org/download/wikiart-stylegan2-conditional-model/WikiArt_Uncond2.pkl',  # Thanks to @pbaylies; unconditional
            'maps1024':      'https://archive.org/download/mapdreamer/mapdreamer.pkl',  # Thanks to @tjukanov
            'fursona512':    'https://thisfursonadoesnotexist.com/model/network-e621-r-512-3194880.pkl',  # Thanks to @arfafax
            'mlpony512':     'https://thisponydoesnotexist.net/model/network-ponies-1024-151552.pkl',  # Thanks to @arfafax
            # Deceive-D/APA models (ignoring the faces models): https://github.com/EndlessSora/DeceiveD
            'afhqcat256':    'https://drive.google.com/u/0/uc?export=download&confirm=zFoN&id=1P9ouHIK-W8JTb6bvecfBe4c_3w6gmMJK',
            'anime256':      'https://drive.google.com/u/0/uc?export=download&confirm=6Uie&id=1EWOdieqELYmd2xRxUR4gnx7G10YI5dyP',
            'cub256':        'https://drive.google.com/u/0/uc?export=download&confirm=KwZS&id=1J0qactT55ofAvzddDE_xnJEY8s3vbo1_',
            # Self-Distilled StyleGAN (full body representation of each class): https://github.com/self-distilled-stylegan/self-distilled-internet-photos
            'sddogs1024':    'https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl',
            'sdelephant512': 'https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl',
            'sdhorses512':   'https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl',
            'sdbicycles256': 'https://storage.googleapis.com/self-distilled-stylegan/bicycles_256_pytorch.pkl',
            'sdlions512':    'https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl',
            'sdgiraffes512': 'https://storage.googleapis.com/self-distilled-stylegan/giraffes_512_pytorch.pkl',
            'sdparrots512':  'https://storage.googleapis.com/self-distilled-stylegan/parrots_512_pytorch.pkl'

        },
        # For StyleGAN3 config-r models (--cfg=stylegan3-r)
        'stylegan3-r': {
            # Official NVIDIA models
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
        },
        # For StyleGAN3 config-t models (--cfg=stylegan3-t)
        'stylegan3-t': {
            # Official NVIDIA models
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',
            # Community models, found in: https://github.com/justinpinkney/awesome-pretrained-stylegan3 by @justinpinkney
            'landscapes256': 'https://drive.google.com/u/0/uc?export=download&confirm=eJHe&id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1',  # Thanks to @justinpinkney
            'wikiart1024':   'https://drive.google.com/u/0/uc?export=download&confirm=2tz5&id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj',  # Thanks to @justinpinkney
            # -> Wombo Dream-based models found in: https://github.com/edstoica/lucid_stylegan3_datasets_models by @edstoica; TODO: more to come, update the list as they are released!
            'mechfuture256': 'https://www.dropbox.com/s/v2oie53cz62ozvu/network-snapshot-000029.pkl?dl=1',  # Thanks to @edstoica; 29kimg tick
            'vivflowers256': 'https://www.dropbox.com/s/o33lhgnk91hstvx/network-snapshot-000069.pkl?dl=1',  # Thanks to @edstoica; 68kimg tick
            'alienglass256': 'https://www.dropbox.com/s/gur14k0e7kspguy/network-snapshot-000038.pkl?dl=1',  # Thanks to @edstoica; 38kimg tick
            'scificity256': 'https://www.dropbox.com/s/1kfsmlct4mriphc/network-snapshot-000210.pkl?dl=1',  # Thanks to @edstoica; 210kimg tick
            'scifiship256': 'https://www.dropbox.com/s/02br3mjkma1hubc/network-snapshot-000162.pkl?dl=1',  # Thanks to @edstoica; 168kimg tick
        }
}

# ----------------------------------------------------------------------------


# TODO: all of the following functions must work for RGBA images
def w_to_img(G, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const') -> np.ndarray:
    """
    Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
    returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        Note: this function should be used after doing the truncation trick!
    """
    assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
    if len(dlatents.shape) == 2:
        dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
    synth_image = G.synthesis(dlatents, noise_mode=noise_mode)
    synth_image = (synth_image + 1) * 255/2  # [-1.0, 1.0] -> [0.0, 255.0]
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()  # NCWH => NWHC
    return synth_image


def z_to_dlatent(G, latents: torch.Tensor, label: torch.Tensor, truncation_psi: float = 1.0) -> torch.Tensor:
    """Get the dlatent from the given latent, class label and truncation psi"""
    assert isinstance(latents, torch.Tensor), f'latents should be a torch.Tensor!: "{type(latents)}"'
    assert isinstance(label, torch.Tensor), f'label should be a torch.Tensor!: "{type(label)}"'
    if len(latents.shape) == 1:
        latents = latents.unsqueeze(0)  # An individual latent => [1, G.z_dim]
    dlatents = G.mapping(z=latents, c=label, truncation_psi=truncation_psi)

    return dlatents


def z_to_img(G, latents: torch.Tensor, label: torch.Tensor, truncation_psi: float, noise_mode: str = 'const') -> np.ndarray:
    """
    Get an image/np.ndarray from a latent Z using G, the label, truncation_psi, and noise_mode. The shape
    of the output image/np.ndarray will be [len(latents), G.img_resolution, G.img_resolution, G.img_channels]
    """
    dlatents = z_to_dlatent(G=G, latents=latents, label=label, truncation_psi=1.0)
    dlatents = G.mapping.w_avg + (G.mapping.w_avg - dlatents) * truncation_psi
    img = w_to_img(G=G, dlatents=dlatents, noise_mode=noise_mode)  # Let's not redo code
    return img


def get_w_from_seed(G, device: torch.device, seed: int, truncation_psi: float) -> torch.Tensor:
    """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
    z = np.random.RandomState(seed).randn(1, G.z_dim)
    w = G.mapping(torch.from_numpy(z).to(device), None)
    w_avg = G.mapping.w_avg
    w = w_avg + (w - w_avg) * truncation_psi

    return w


def get_latent_from_file(file: Union[str, os.PathLike],
                    return_ext: bool = False,
                    named_latent: str = 'w') -> Tuple[np.ndarray, Optional[str]]:
    """Get dlatent (w) from a .npy or .npz file"""
    filename, file_extension = os.path.splitext(file)
    assert file_extension in ['.npy', '.npz'], f'"{file}" has wrong file format! Only ".npy" or ".npz" are allowed'
    if file_extension == '.npy':
        latent = np.load(file)
        extension = '.npy'
    else:
        latent = np.load(file)[named_latent]
        extension = '.npz'
    if len(latent.shape) == 4:
        latent = latent[0]
    return (latent, extension) if return_ext else latent


# ----------------------------------------------------------------------------


def save_config(ctx: click.Context, run_dir: Union[str, os.PathLike], save_name: str = 'config.json') -> None:
    """Save the configuration stored in ctx.obj into a JSON file at the output directory."""
    with open(os.path.join(run_dir, save_name), 'w') as f:
        json.dump(ctx.obj, f, indent=4, sort_keys=True)


# ----------------------------------------------------------------------------


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir
