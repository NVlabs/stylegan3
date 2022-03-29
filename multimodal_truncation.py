import os
# from typing import List, Optional, Union, Tuple
# import click

import dnnlib
from torch_utils import gen_utils

# import scipy
import numpy as np
import PIL.Image
import torch

import legacy

# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
# import moviepy.editor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------

# TODO: make this nice! For now it's just meant for a quick test/defining parameters to use in general
# # We group the different types of generation (images, grid, video, wacky stuff) into a main function
# @click.group()
# def main():
#     pass


# ----------------------------------------------------------------------------
network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl'
description = 'afhq512-t'
outdir = './out/multimodal/sgan3'
seed = 0

num_latents = 60000  # keep fixed
num_clusters = 64  # 32, 64, 128

print(f'Loading networks from "{network_pkl}"...')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

desc = f'multimodal-truncation-{num_clusters}clusters'
desc = f'{desc}-{description}' if len(description) != 0 else desc
# Create the run dir with the given name description
run_dir = gen_utils.make_run_dir(outdir, desc)

print('Generating all the latents...')
z = torch.from_numpy(np.random.RandomState(seed).randn(num_latents, G.z_dim)).to(device)
w = G.mapping(z, None)[:, 0, :]

scaler = StandardScaler()
scaler.fit(w.cpu())

w_scaled = scaler.transform(w.cpu())

kmeans = KMeans(n_clusters=num_clusters, random_state=0, init='random', verbose=1).fit(w_scaled)

w_avg_multi = torch.from_numpy(scaler.inverse_transform(kmeans.cluster_centers_)).to(device)  # a NumPy array :D

fixed_w = gen_utils.get_w_from_seed(G, device, 0, 1.0)
fixed_img = gen_utils.w_to_img(G, fixed_w)[0]
# PIL.Image.fromarray(fixed_img, 'RGB').save(os.path.join(run_dir, 'pure_dlatent.jpg'))
np.save(os.path.join(run_dir, 'pure_dlatent.npy'), fixed_w.unsqueeze(0).cpu().numpy())

fixed_w_truncated = G.mapping.w_avg + (fixed_w - G.mapping.w_avg) * 0.5
fixed_img_truncated = gen_utils.w_to_img(G, fixed_w_truncated)[0]
# PIL.Image.fromarray(fixed_img_truncated, 'RGB').save(os.path.join(run_dir, 'truncated_dlatent.jpg'))
np.save(os.path.join(run_dir, 'truncated_dlatent.npy'), fixed_w_truncated.unsqueeze(0).cpu().numpy())

global_w_avg = gen_utils.get_w_from_seed(G, device, 0, 0.0)
global_avg_img = gen_utils.w_to_img(G, global_w_avg)[0]
# PIL.Image.fromarray(global_avg_img, 'RGB').save(os.path.join(run_dir, 'global_w_avg.jpg'))
np.save(os.path.join(run_dir, 'global_w_avg.npy'), global_w_avg.unsqueeze(0).cpu().numpy())

PIL.Image.fromarray(gen_utils.create_image_grid(np.array([fixed_img, global_avg_img, fixed_img_truncated]), (3, 1)), 'RGB').save(os.path.join(run_dir, 'comparison_global_mean.jpg'))

truncated_centroids_imgs = []
pure_centroids_imgs = []

for idx, w_avg in enumerate(w_avg_multi):
    new_w = w_avg + (fixed_w - w_avg) * 0.5
    new_w_img = gen_utils.w_to_img(G, new_w)[0]
    truncated_centroids_imgs.append(new_w_img)
    # PIL.Image.fromarray(new_w_img, 'RGB').save(os.path.join(run_dir, f'centroid_{idx+1:03d}-{num_clusters}clusters.jpg'))

    w_avg = torch.tile(w_avg, (1, G.mapping.num_ws, 1))
    img = gen_utils.w_to_img(G, w_avg)[0]
    pure_centroids_imgs.append(img)
    # PIL.Image.fromarray(img, 'RGB').save(os.path.join(run_dir, f'pure_centroid_{idx+1:03d}-{num_clusters}clusters.jpg'))
    np.save(os.path.join(run_dir, f'centroid_{idx+1:03d}.npy'), w_avg.unsqueeze(0).cpu().numpy())

# Save grids
PIL.Image.fromarray(gen_utils.create_image_grid(np.array(truncated_centroids_imgs)), 'RGB').save(os.path.join(run_dir, 'truncated_centroids.jpg'))
PIL.Image.fromarray(gen_utils.create_image_grid(np.array(pure_centroids_imgs)), 'RGB').save(os.path.join(run_dir, 'pure_centroids.jpg'))

if False:
    # Try only with the GPU-based KMeans
    inertia = []

    print('Getting wonky...')
    for k in range(1, 128):
        k_means = KMeans(n_clusters=k, init='random', random_state=0).fit(w_scaled)
        inertia.append(k_means.inertia_)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 9))
    plt.plot(range(1, 128), inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(run_dir, 'inertia.jpg'), bbox_inches='tight')
    plt.close()
