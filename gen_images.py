# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
from time import perf_counter

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default="ffhq1024.pkl", required=False)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default=0, required=False)
@click.option('--w_space', is_flag=True, help='Interpolate in W space instead of Z space')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir_img', help='Where to save the output images', type=str, default="out/images", required=False, metavar='DIR')
@click.option('--outdir_seeds', help='Where to save the latent vector', type=str, default="out/seeds", required=False, metavar='DIR')
@click.option('--append_latent', is_flag=True, help='If true, append the latent vector to the latent vector file (if it exists). Otherwise, overwrite the file.')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir_img: str,
    outdir_seeds: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    w_space: bool, 
    latent_vector: str,
    append_latent: bool
):
    """Generate images using pretrained network pickle.

    Examples:

    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python generate.py --outdir_img=out --trunc=1 --seeds=2 --network=ffhq1024.pkl

    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        print("Loading Generator...")
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Might use discriminator later to determine how fake the image is

    # with dnnlib.util.open_url(network_pkl) as f:
    #     print("Loading discriminator...")
    #     D = legacy.load_network_pkl(f)['D'].to(device)

    os.makedirs(outdir_img, exist_ok=True)
    os.makedirs(outdir_seeds, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    vector_seeds = []
    start_time = perf_counter()
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        time_remaining = (perf_counter() - start_time) * (len(seeds) - seed_idx) / (seed_idx + 1)
        print('Generating image for seed %d (%d/%d) ... Time Remaining: %d' % (seed, seed_idx, len(seeds), time_remaining))

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        if w_space:
            w = G.mapping(z, label)
            vector_seeds.append(w.cpu().numpy())
        else:
            vector_seeds.append(z.cpu().numpy())

        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir_img}/seed{seed:04d}.png')
    
    vector_seeds = np.concatenate(vector_seeds, axis=0)
    if(append_latent):
        print("Appending latent vector to file...")
        if(os.path.exists(f'{outdir_seeds}/seeds.npy')):
            old_vector_seeds = np.load(f'{outdir_seeds}/seeds.npy')
            vector_seeds = np.concatenate((old_vector_seeds, vector_seeds), axis=0)
    np.save(f'{outdir_seeds}/seeds.npy', vector_seeds)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
