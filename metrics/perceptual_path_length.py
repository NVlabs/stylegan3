# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Perceptual Path Length (PPL) from the paper "A Style-Based Generator
Architecture for Generative Adversarial Networks". Matches the original
implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py"""

import copy
import numpy as np
import torch
from . import metric_utils

#----------------------------------------------------------------------------

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d

#----------------------------------------------------------------------------

class PPLSampler(torch.nn.Module):
    def __init__(self, G, G_kwargs, epsilon, space, sampling, crop, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self, c):
        # Generate random latents and interpolation t-values.
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)

        # Interpolate in W or Z.
        if self.space == 'w':
            w0, w1 = self.G.mapping(z=torch.cat([z0,z1]), c=torch.cat([c,c])).chunk(2)
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else: # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0, wt1 = self.G.mapping(z=torch.cat([zt0,zt1]), c=torch.cat([c,c])).chunk(2)

        # Randomize noise buffers.
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))

        # Generate images.
        img = self.G.synthesis(ws=torch.cat([wt0,wt1]), noise_mode='const', force_fp32=True, **self.G_kwargs)

        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample to 256x256.
        factor = self.G.img_resolution // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)
        if self.G.img_channels == 1:
            img = img.repeat([1, 3, 1, 1])

        # Evaluate differential LPIPS.
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

#----------------------------------------------------------------------------

def compute_ppl(opts, num_samples, epsilon, space, sampling, crop, batch_size):
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=opts.num_gpus, rank=opts.rank, verbose=opts.progress.verbose)

    # Setup sampler and labels.
    sampler = PPLSampler(G=opts.G, G_kwargs=opts.G_kwargs, epsilon=epsilon, space=space, sampling=sampling, crop=crop, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to(opts.device)
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_size)

    # Sampling loop.
    dist = []
    progress = opts.progress.sub(tag='ppl sampling', num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size * opts.num_gpus):
        progress.update(batch_start)
        x = sampler(next(c_iter))
        for src in range(opts.num_gpus):
            y = x.clone()
            if opts.num_gpus > 1:
                torch.distributed.broadcast(y, src=src)
            dist.append(y)
    progress.update(num_samples)

    # Compute PPL.
    if opts.rank != 0:
        return float('nan')
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)

#----------------------------------------------------------------------------
