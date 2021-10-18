# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Equivariance metrics (EQ-T, EQ-T_frac, and EQ-R) from the paper
"Alias-Free Generative Adversarial Networks"."""

import copy
import numpy as np
import torch
import torch.fft
from torch_utils.ops import upfirdn2d
from . import metric_utils

#----------------------------------------------------------------------------
# Utilities.

def sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, sinc(x), torch.zeros_like(x))

def rotation_matrix(angle):
    angle = torch.as_tensor(angle).to(torch.float32)
    mat = torch.eye(3, device=angle.device)
    mat[0, 0] = angle.cos()
    mat[0, 1] = angle.sin()
    mat[1, 0] = -angle.sin()
    mat[1, 1] = angle.cos()
    return mat

#----------------------------------------------------------------------------
# Apply integer translation to a batch of 2D images. Corresponds to the
# operator T_x in Appendix E.1.

def apply_integer_translation(x, tx, ty):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.round().to(torch.int64)
    iy = ty.round().to(torch.int64)

    z = torch.zeros_like(x)
    m = torch.zeros_like(x)
    if abs(ix) < W and abs(iy) < H:
        y = x[:, :, max(-iy,0) : H+min(-iy,0), max(-ix,0) : W+min(-ix,0)]
        z[:, :, max(iy,0) : H+min(iy,0), max(ix,0) : W+min(ix,0)] = y
        m[:, :, max(iy,0) : H+min(iy,0), max(ix,0) : W+min(ix,0)] = 1
    return z, m

#----------------------------------------------------------------------------
# Apply integer translation to a batch of 2D images. Corresponds to the
# operator T_x in Appendix E.2.

def apply_fractional_translation(x, tx, ty, a=3):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.floor().to(torch.int64)
    iy = ty.floor().to(torch.int64)
    fx = tx - ix
    fy = ty - iy
    b = a - 1

    z = torch.zeros_like(x)
    zx0 = max(ix - b, 0)
    zy0 = max(iy - b, 0)
    zx1 = min(ix + a, 0) + W
    zy1 = min(iy + a, 0) + H
    if zx0 < zx1 and zy0 < zy1:
        taps = torch.arange(a * 2, device=x.device) - b
        filter_x = (sinc(taps - fx) * sinc((taps - fx) / a)).unsqueeze(0)
        filter_y = (sinc(taps - fy) * sinc((taps - fy) / a)).unsqueeze(1)
        y = x
        y = upfirdn2d.filter2d(y, filter_x / filter_x.sum(), padding=[b,a,0,0])
        y = upfirdn2d.filter2d(y, filter_y / filter_y.sum(), padding=[0,0,b,a])
        y = y[:, :, max(b-iy,0) : H+b+a+min(-iy-a,0), max(b-ix,0) : W+b+a+min(-ix-a,0)]
        z[:, :, zy0:zy1, zx0:zx1] = y

    m = torch.zeros_like(x)
    mx0 = max(ix + a, 0)
    my0 = max(iy + a, 0)
    mx1 = min(ix - b, 0) + W
    my1 = min(iy - b, 0) + H
    if mx0 < mx1 and my0 < my1:
        m[:, :, my0:my1, mx0:mx1] = 1
    return z, m

#----------------------------------------------------------------------------
# Construct an oriented low-pass filter that applies the appropriate
# bandlimit with respect to the input and output of the given affine 2D
# image transformation.

def construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = sinc(xi * cutoff_in) * sinc(yi * cutoff_in)
    fo = sinc(xo * cutoff_out) * sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = lanczos_window(xi, a) * lanczos_window(yi, a)
    wo = lanczos_window(xo, a) * lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------
# Apply the given affine transformation to a batch of 2D images.

def apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------
# Apply fractional rotation to a batch of 2D images. Corresponds to the
# operator R_\alpha in Appendix E.3.

def apply_fractional_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(angle)
    return apply_affine_transformation(x, mat, a=a, amax=a*2, **filter_kwargs)

#----------------------------------------------------------------------------
# Modify the frequency content of a batch of 2D images as if they had undergo
# fractional rotation -- but without actually rotating them. Corresponds to
# the operator R^*_\alpha in Appendix E.3.

def apply_fractional_pseudo_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(-angle)
    f = construct_affine_bandlimit_filter(mat, a=a, amax=a*2, up=1, **filter_kwargs)
    y = upfirdn2d.filter2d(x=x, f=f)
    m = torch.zeros_like(y)
    c = f.shape[0] // 2
    m[:, :, c:-c, c:-c] = 1
    return y, m

#----------------------------------------------------------------------------
# Compute the selected equivariance metrics for the given generator.

def compute_equivariance_metrics(opts, num_samples, batch_size, translate_max=0.125, rotate_max=1, compute_eqt_int=False, compute_eqt_frac=False, compute_eqr=False):
    assert compute_eqt_int or compute_eqt_frac or compute_eqr

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    I = torch.eye(3, device=opts.device)
    M = getattr(getattr(getattr(G, 'synthesis', None), 'input', None), 'transform', None)
    if M is None:
        raise ValueError('Cannot compute equivariance metrics; the given generator does not support user-specified image transformations')
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_size)

    # Sampling loop.
    sums = None
    progress = opts.progress.sub(tag='eq sampling', num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size * opts.num_gpus):
        progress.update(batch_start)
        s = []

        # Randomize noise buffers, if any.
        for name, buf in G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))

        # Run mapping network.
        z = torch.randn([batch_size, G.z_dim], device=opts.device)
        c = next(c_iter)
        ws = G.mapping(z=z, c=c)

        # Generate reference image.
        M[:] = I
        orig = G.synthesis(ws=ws, noise_mode='const', **opts.G_kwargs)

        # Integer translation (EQ-T).
        if compute_eqt_int:
            t = (torch.rand(2, device=opts.device) * 2 - 1) * translate_max
            t = (t * G.img_resolution).round() / G.img_resolution
            M[:] = I
            M[:2, 2] = -t
            img = G.synthesis(ws=ws, noise_mode='const', **opts.G_kwargs)
            ref, mask = apply_integer_translation(orig, t[0], t[1])
            s += [(ref - img).square() * mask, mask]

        # Fractional translation (EQ-T_frac).
        if compute_eqt_frac:
            t = (torch.rand(2, device=opts.device) * 2 - 1) * translate_max
            M[:] = I
            M[:2, 2] = -t
            img = G.synthesis(ws=ws, noise_mode='const', **opts.G_kwargs)
            ref, mask = apply_fractional_translation(orig, t[0], t[1])
            s += [(ref - img).square() * mask, mask]

        # Rotation (EQ-R).
        if compute_eqr:
            angle = (torch.rand([], device=opts.device) * 2 - 1) * (rotate_max * np.pi)
            M[:] = rotation_matrix(-angle)
            img = G.synthesis(ws=ws, noise_mode='const', **opts.G_kwargs)
            ref, ref_mask = apply_fractional_rotation(orig, angle)
            pseudo, pseudo_mask = apply_fractional_pseudo_rotation(img, angle)
            mask = ref_mask * pseudo_mask
            s += [(ref - pseudo).square() * mask, mask]

        # Accumulate results.
        s = torch.stack([x.to(torch.float64).sum() for x in s])
        sums = sums + s if sums is not None else s
    progress.update(num_samples)

    # Compute PSNRs.
    if opts.num_gpus > 1:
        torch.distributed.all_reduce(sums)
    sums = sums.cpu()
    mses = sums[0::2] / sums[1::2]
    psnrs = np.log10(2) * 20 - mses.log10() * 10
    psnrs = tuple(psnrs.numpy())
    return psnrs[0] if len(psnrs) == 1 else psnrs

#----------------------------------------------------------------------------
