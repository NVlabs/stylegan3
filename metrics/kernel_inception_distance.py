# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID) from the paper "Demystifying MMD
GANs". Matches the original implementation by Binkowski et al. at
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py"""

import numpy as np
from . import metric_utils

#----------------------------------------------------------------------------

def compute_kid(opts, max_real, num_gen, num_subsets, max_subset_size):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()

    if opts.rank != 0:
        return float('nan')

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

#----------------------------------------------------------------------------
