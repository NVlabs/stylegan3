import os
from time import perf_counter
from typing import Tuple

import click
import imageio
import numpy as np
import PIL.Image
import torch


import dnnlib
from scripts import legacy

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
@click.option('--network', 'network_pkl', type=str, default="ffhq1024.pkl", help='Network pickle filename', required=False)
@click.option('--lvector_dir', help='Where to find images', type=str, default="out/seeds", required=False, metavar='DIR')
@click.option('--coef_dir', help='Where to find coeffs', type=str, default="out/vectors", required=False, metavar='DIR')
@click.option('--steps', help='Number of interpolation steps', type=int, default=100, required=False, metavar='DIR')
@click.option('--outdir_img', help='Where to save the output images', type=str, default="out/interpolation/gender", required=False, metavar='DIR')
@click.option('--save-video', help='Save an mp4 video of interpolation progress', type=bool, default=True, show_default=True)
@click.option('--fps', help='Video framerate', type=int, default=10, show_default=True)
def interpolate(
   network_pkl: str,
   lvector_dir: str,
   coef_dir: str,
   steps: int,
   outdir_img: str,
   save_video: bool,
   fps: int
):
    """Train SVM on latent vectors"""

    # Data.
    latent_vectors = torch.load(lvector_dir+'/latent_vectors.pt').cpu().numpy()

    # latent_vectors have 5000 rows and 512 columns, get the 2 row
    seed = latent_vectors[4914]

    # Labels.
    coefficients = torch.load(coef_dir+'/gender_coef.pt')
    coefficient = coefficients[0]
    #### Multiply per -1 to convert to woman
    # coefficient = np.multiply(coefficient, -1)

    # Interpolation
    interpolation = []
    interpolation.append(seed)
    nth = np.divide(coefficient, steps)
    for i in range(1, 2*steps):
        aux_coef = np.multiply(nth, i)
        aux_seed = np.add(aux_coef, seed)
        interpolation.append(aux_seed)

    # Loading network
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Video
    os.makedirs(outdir_img, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir_img}/video.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')

    # Generate images    
    os.makedirs(outdir_img+"/img", exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    start_time = perf_counter()
    for seed_idx, seed in enumerate(interpolation):
        # Feedback
        time_remaining = (perf_counter() - start_time) * (len(interpolation) - seed_idx) / (seed_idx + 1)
        print('Generating image for step %d/%d, %.1f sec remaining...' % (seed_idx, len(interpolation), time_remaining))

        # Generate image.
        z = torch.from_numpy(seed).to(device).unsqueeze(0)
        if hasattr(G.synthesis, 'input'):
            m = make_transform((0,0), 0)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        img = G(z, label, truncation_psi=1, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir_img}/img/{seed_idx:03d}.png')

        # Video
        if save_video:
            video.append_data(np.array(img[0].cpu()))
    
    if save_video:
        video.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    interpolate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
