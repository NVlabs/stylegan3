# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Compare average power spectra between real and generated images,
or between multiple generators."""

import os
import numpy as np
import torch
import torch.fft
import scipy.ndimage
import matplotlib.pyplot as plt
import click
import tqdm
import dnnlib

import legacy
from training import dataset

#----------------------------------------------------------------------------
# Setup an iterator for streaming images, in uint8 NCHW format, based on the
# respective command line options.

def stream_source_images(source, num, seed, device, data_loader_kwargs=None): # => num_images, image_size, image_iter
    ext = source.split('.')[-1].lower()
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    if ext == 'pkl':
        if num is None:
            raise click.ClickException('--num is required when --source points to network pickle')
        with dnnlib.util.open_url(source) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        def generate_image(seed):
            rnd = np.random.RandomState(seed)
            z = torch.from_numpy(rnd.randn(1, G.z_dim)).to(device)
            c = torch.zeros([1, G.c_dim], device=device)
            if G.c_dim > 0:
                c[:, rnd.randint(G.c_dim)] = 1
            return (G(z=z, c=c) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        _ = generate_image(seed) # warm up
        image_iter = (generate_image(seed + idx) for idx in range(num))
        return num, G.img_resolution, image_iter

    elif ext == 'zip' or os.path.isdir(source):
        dataset_obj = dataset.ImageFolderDataset(path=source, max_size=num, random_seed=seed)
        if num is not None and num != len(dataset_obj):
            raise click.ClickException(f'--source contains fewer than {num} images')
        data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=1, **data_loader_kwargs)
        image_iter = (image.to(device) for image, _label in data_loader)
        return len(dataset_obj), dataset_obj.resolution, image_iter

    else:
        raise click.ClickException('--source must point to network pickle, dataset zip, or directory')

#----------------------------------------------------------------------------
# Load average power spectrum from the specified .npz file and construct
# the corresponding heatmap for visualization.

def construct_heatmap(npz_file, smooth):
    npz_data = np.load(npz_file)
    spectrum = npz_data['spectrum']
    image_size = npz_data['image_size']
    hmap = np.log10(spectrum) * 10 # dB
    hmap = np.fft.fftshift(hmap)
    hmap = np.concatenate([hmap, hmap[:1, :]], axis=0)
    hmap = np.concatenate([hmap, hmap[:, :1]], axis=1)
    if smooth > 0:
        sigma = spectrum.shape[0] / image_size * smooth
        hmap = scipy.ndimage.gaussian_filter(hmap, sigma=sigma, mode='nearest')
    return hmap, image_size

#----------------------------------------------------------------------------

@click.group()
def main():
    """Compare average power spectra between real and generated images,
    or between multiple generators.

    Example:

    \b
    # Calculate dataset mean and std, needed in subsequent steps.
    python avg_spectra.py stats --source=~/datasets/ffhq-1024x1024.zip

    \b
    # Calculate average spectrum for the training data.
    python avg_spectra.py calc --source=~/datasets/ffhq-1024x1024.zip \\
        --dest=tmp/training-data.npz --mean=112.684 --std=69.509

    \b
    # Calculate average spectrum for a pre-trained generator.
    python avg_spectra.py calc \\
        --source=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl \\
        --dest=tmp/stylegan3-r.npz --mean=112.684 --std=69.509 --num=70000

    \b
    # Display results.
    python avg_spectra.py heatmap tmp/training-data.npz
    python avg_spectra.py heatmap tmp/stylegan3-r.npz
    python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz

    \b
    # Save as PNG.
    python avg_spectra.py heatmap tmp/training-data.npz --save=tmp/training-data.png --dpi=300
    python avg_spectra.py heatmap tmp/stylegan3-r.npz --save=tmp/stylegan3-r.png --dpi=300
    python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz --save=tmp/slices.png --dpi=300
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
def stats(source, num, seed, device=torch.device('cuda')):
    """Calculate dataset mean and standard deviation needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, _image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)

    # Accumulate moments.
    moments = torch.zeros([3], dtype=torch.float64, device=device)
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = image.to(torch.float64)
        moments += torch.stack([torch.ones_like(image).sum(), image.sum(), image.square().sum()])
    moments = moments / moments[0]

    # Compute mean and standard deviation.
    mean = moments[1]
    std = (moments[2] - moments[1].square()).sqrt()
    print(f'--mean={mean:g} --std={std:g}')

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--mean', help='Dataset mean for whitening', metavar='FLOAT', type=float, required=True)
@click.option('--std', help='Dataset standard deviation for whitening', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
def calc(source, dest, mean, std, num, seed, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)
    spectrum_size = image_size * interp
    padding = spectrum_size - image_size

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)

    # Accumulate power spectrum.
    spectrum = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = (image.to(torch.float64) - mean) / std
        image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
        spectrum += torch.fft.fftn(image, dim=[2,3]).abs().square().mean(dim=[0,1])
    spectrum /= num_images

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, spectrum=spectrum.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-file', nargs=1)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=1.25, show_default=True)
def heatmap(npz_file, save, smooth, dpi):
    """Visualize 2D heatmap based on the given .npz file."""
    hmap, image_size = construct_heatmap(npz_file=npz_file, smooth=smooth)

    # Setup plot.
    plt.figure(figsize=[6, 4.8], dpi=dpi, tight_layout=True)
    freqs = np.linspace(-0.5, 0.5, num=hmap.shape[0], endpoint=True) * image_size
    ticks = np.linspace(freqs[0], freqs[-1], num=5, endpoint=True)
    levels = np.linspace(-40, 20, num=13, endpoint=True)

    # Draw heatmap.
    plt.xlim(ticks[0], ticks[-1])
    plt.ylim(ticks[0], ticks[-1])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.contourf(freqs, freqs, hmap, levels=levels, extend='both', cmap='Blues')
    plt.gca().set_aspect('equal')
    plt.colorbar(ticks=levels)
    plt.contour(freqs, freqs, hmap, levels=levels, extend='both', linestyles='solid', linewidths=1, colors='midnightblue', alpha=0.2)

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-files', nargs=-1, required=True)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
def slices(npz_files, save, dpi, smooth):
    """Visualize 1D slices based on the given .npz files."""
    cases = [dnnlib.EasyDict(npz_file=npz_file) for npz_file in npz_files]
    for c in cases:
        c.hmap, c.image_size = construct_heatmap(npz_file=c.npz_file, smooth=smooth)
        c.label = os.path.splitext(os.path.basename(c.npz_file))[0]

    # Check consistency.
    image_size = cases[0].image_size
    hmap_size = cases[0].hmap.shape[0]
    if any(c.image_size != image_size or c.hmap.shape[0] != hmap_size for c in cases):
        raise click.ClickException('All .npz must have the same resolution')

    # Setup plot.
    plt.figure(figsize=[12, 4.6], dpi=dpi, tight_layout=True)
    hmap_center = hmap_size // 2
    hmap_range = np.arange(hmap_center, hmap_size)
    freqs0 = np.linspace(0, image_size / 2, num=(hmap_size // 2 + 1), endpoint=True)
    freqs45 = np.linspace(0, image_size / np.sqrt(2), num=(hmap_size // 2 + 1), endpoint=True)
    xticks0 = np.linspace(freqs0[0], freqs0[-1], num=9, endpoint=True)
    xticks45 = np.round(np.linspace(freqs45[0], freqs45[-1], num=9, endpoint=True))
    yticks = np.linspace(-50, 30, num=9, endpoint=True)

    # Draw 0 degree slice.
    plt.subplot(1, 2, 1)
    plt.title('0\u00b0 slice')
    plt.xlim(xticks0[0], xticks0[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks0)
    plt.yticks(yticks)
    for c in cases:
        plt.plot(freqs0, c.hmap[hmap_center, hmap_range], label=c.label)
    plt.grid()
    plt.legend(loc='upper right')

    # Draw 45 degree slice.
    plt.subplot(1, 2, 2)
    plt.title('45\u00b0 slice')
    plt.xlim(xticks45[0], xticks45[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks45)
    plt.yticks(yticks)
    for c in cases:
        plt.plot(freqs45, c.hmap[hmap_range, hmap_range], label=c.label)
    plt.grid()
    plt.legend(loc='upper right')

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
