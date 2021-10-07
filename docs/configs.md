# Training configurations

This document provides guidelines for selecting appropriate training options for various scenarios, as well as an extensive list of recommended configurations.

#### Example

In the remainder of this document, we summarize each configuration as follows:

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :------------: | :--
| <sub>StyleGAN3&#8209;T</sub> | <sub>18.47</sub> | <sub>12.29</sub> | <sub>4.3</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=8.2 --mirror=1`</sub>

This corresponds to the following command line:

```.bash
# Train StyleGAN3-T for AFHQv2 using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \
  --gpus=8 --batch=32 --gamma=8.2 --mirror=1
```

Explanation of the columns:
- **Config**: StyleGAN3-T (translation equiv.), StyleGAN3-R (translation and rotation equiv.), or StyleGAN2. Reflects the value of `--cfg`.
- **s/kimg**: Raw training speed, measured separately on Tesla V100 and A100 using our recommended Docker image. The number indicates how many seconds, on average, it takes to process 1000 images from the training set. The number tends to vary slightly over the course of training; typically by no more than &plusmn;20%.
- **GPU mem**: Maximum GPU memory usage observed during training, reported in gigabytes per GPU. The above example uses 8 GPUs, which means that the total GPU memory usage is around 34.4 GB.
- **Options**: Command line options for `train.py`, excluding `--outdir` and `--data`.

#### Total training time

In addition the raw s/kimg number, the training time also depends on the `--kimg` and `--metric` options. `--kimg` controls the total number of training iterations and is set to 25000 by default. This is long enough to reach convergence in typical cases, but in practice the results should already look quite reasonable around 5000 kimg. `--metrics` determines which quality metrics are computed periodically during training. The default is `fid50k_full`, which increases the training time slightly; typically by no more than 5%. The automatic computation can be disabled by specifying `--metrics=none`.

In the above example, the total training time on V100 is approximately 18.47 s/kimg * 25000 kimg * 1.05 &thickapprox; 485,000 seconds &thickapprox; 5 days and 14 hours. Disabling metric computation (`--metrics=none`) reduces this to approximately 5 days and 8 hours.

## General guidelines

The most important hyperparameter that needs to be tuned on a per-dataset basis is the R<sub>1</sub> regularization weight, `--gamma`, that must be specified explicitly for `train.py`. As a rule of thumb, the value of `--gamma` scales quadratically with respect to the training set resolution: doubling the resolution (e.g., 256x256 &rarr; 512x512) means that `--gamma` should be multiplied by 4 (e.g., 2 &rarr; 8). The optimal value is usually the same for `--cfg=stylegan3-t` and `--cfg=stylegan3-r`, but considerably lower for `--cfg=stylegan2`.

In practice, we recommend selecting the value of `--gamma` as follows:
- Find the closest match for your specific case in this document (config, resolution, and GPU count).
- Try training with the same `--gamma` first.
- Then, try increasing the value by 2x and 4x, and also decreasing it by 2x and 4x.
- Pick the value that yields the lowest FID.

The results may also be improved by adjusting `--mirror` and `--aug`, depending on the training data. Specifying `--mirror=1` augments the dataset with random *x*-flips, which effectively doubles the number of images. This is generally beneficial with datasets that are horizontally symmetric (e.g., FFHQ), but it can be harmful if the images contain noticeable asymmetric features (e.g., text or letters). Specifying `--aug=noaug` disables adaptive discriminator augmentation (ADA), which may improve the results slightly if the training set is large enough (at least 100k images when accounting for *x*-flips). With small datasets (less than 30k images), it is generally a good idea to leave the augmentations enabled.

It is possible to speed up the training by decreasing network capacity, i.e., `--cbase=16384`. This typically leads to lower quality results, but the difference is less pronounced with low-resolution datasets (e.g., 256x256).

#### Scaling to different number of GPUs

You can select the number of GPUs by changing the value of `--gpu`; this does not affect the convergence curves or training dynamics in any way. By default, the total batch size (`--batch`) is divided evenly among the GPUs, which means that decreasing the number of GPUs yields higher per-GPU memory usage. To avoid running out of memory, you can decrease the per-GPU batch size by specifying `--batch-gpu`, which performs the same computation in multiple passes using gradient accumulation.

By default, `train.py` exports network snapshots once every 200 kimg, i.e., the product of `--snap=50` and `--tick=4`. When using few GPUs (e.g., 1&ndash;2), this means that it may take a very long time for the first snapshot to appear. We recommend increasing the snapshot frequency in such cases by specifying `--snap=20`, `--snap=10`, or `--snap=5`.

Note that the configurations listed in this document have been specifically tuned for 8 GPUs. The safest way to scale them to different GPU counts is to adjust `--gpu`, `--batch-gpu`, and `--snap` as described above, but it may be possible to reach faster convergence by adjusting some of the other hyperparameters as well. Note, however, that adjusting the total batch size (`--batch`) requires some experimentation; decreasing `--batch` usually necessitates increasing regularization (`--gamma`) and/or decreasing the learning rates (most importantly `--dlr`).

#### Transfer learning

Transfer learning makes it possible to reach very good results very quickly, especially when the training set is small and/or the images resemble the ones produced by a pre-trained model. To enable transfer learning, you can point `--resume` to one of the pre-trained models that we provide for [StyleGAN3](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3) and [StyleGAN2](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan2). For example:

```.bash
# Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl
```

The pre-trained model should be selected to match the specified config, resolution, and architecture-related hyperparameters (e.g., `--cbase`, `--map-depth`, and `--mbstd-group`). You check this by looking at the `fakes_init.png` exported by `train.py` at the beginning; if the configuration is correct, the images should look reasonable.

With transfer learning, the results may be improved slightly by adjusting `--freezed`, in addition to the above guidelines for `--gamma`, `--mirror`, and `--aug`. In our experience, `--freezed=10` and `--freezed=13` tend to work reasonably well.

## Recommended configurations

This section lists recommended settings for StyleGAN3-T and StyleGAN3-R for different resolutions and GPU counts, selected according to the above guidelines. These are intended to provide a good starting point when experimenting with a new dataset. Please note that many of the options (e.g., `--gamma`, `--mirror`, and `--aug`) are still worth adjusting on a case-by-case basis.

#### 128x128 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :--------------: | :--------------: | :------------: | :--
| <sub>StyleGAN3&#8209;T</sub> | <sub>1</sub> | <sub>73.68</sub> | <sub>27.20</sub> | <sub>7.2</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>2</sub> | <sub>37.30</sub> | <sub>13.74</sub> | <sub>7.1</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=0.5 --snap=20`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>4</sub> | <sub>20.66</sub> | <sub>7.52</sub>  | <sub>4.1</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=0.5`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>8</sub> | <sub>11.31</sub> | <sub>4.40</sub>  | <sub>2.6</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=0.5`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>1</sub> | <sub>58.44</sub> | <sub>34.23</sub> | <sub>8.3</sub> | <sub>`--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>2</sub> | <sub>29.92</sub> | <sub>17.29</sub> | <sub>8.2</sub> | <sub>`--cfg=stylegan3-r --gpus=2 --batch=32 --gamma=0.5 --snap=20`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>4</sub> | <sub>15.49</sub> | <sub>9.53</sub>  | <sub>4.5</sub> | <sub>`--cfg=stylegan3-r --gpus=4 --batch=32 --gamma=0.5`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>8</sub> | <sub>8.43</sub>  | <sub>5.69</sub>  | <sub>2.7</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=0.5`</sub>

#### 256x256 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :--------------: | :--------------: | :------------: | :--
| <sub>StyleGAN3&#8209;T</sub> | <sub>1</sub> | <sub>89.15</sub> | <sub>49.81</sub> | <sub>9.5</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=2 --batch-gpu=16 --snap=10`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>2</sub> | <sub>45.45</sub> | <sub>25.05</sub> | <sub>9.3</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=2 --snap=20`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>4</sub> | <sub>23.94</sub> | <sub>13.26</sub> | <sub>5.2</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=2`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>8</sub> | <sub>13.04</sub> | <sub>7.32</sub>  | <sub>3.1</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=2`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>1</sub> | <sub>87.37</sub> | <sub>56.73</sub> | <sub>6.7</sub> | <sub>`--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=2 --batch-gpu=8 --snap=10`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>2</sub> | <sub>44.12</sub> | <sub>28.60</sub> | <sub>6.7</sub> | <sub>`--cfg=stylegan3-r --gpus=2 --batch=32 --gamma=2 --batch-gpu=8 --snap=20`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>4</sub> | <sub>22.42</sub> | <sub>14.39</sub> | <sub>6.6</sub> | <sub>`--cfg=stylegan3-r --gpus=4 --batch=32 --gamma=2`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>8</sub> | <sub>11.88</sub> | <sub>8.03</sub>  | <sub>3.7</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=2`</sub>

#### 512x512 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :---------------: | :---------------: | :------------: | :--
| <sub>StyleGAN3&#8209;T</sub> | <sub>1</sub> | <sub>137.33</sub> | <sub>90.25</sub>  | <sub>7.8</sub> | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=8 --batch-gpu=8 --snap=10`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>2</sub> | <sub>69.65</sub>  | <sub>45.42</sub>  | <sub>7.7</sub> | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=8 --batch-gpu=8 --snap=20`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>4</sub> | <sub>34.88</sub>  | <sub>22.81</sub>  | <sub>7.6</sub> | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=8`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>8</sub> | <sub>18.47</sub>  | <sub>12.29</sub>  | <sub>4.3</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=8`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>1</sub> | <sub>158.91</sub> | <sub>110.13</sub> | <sub>6.0</sub> | <sub>`--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=8 --batch-gpu=4 --snap=10`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>2</sub> | <sub>79.96</sub>  | <sub>55.18</sub>  | <sub>6.0</sub> | <sub>`--cfg=stylegan3-r --gpus=2 --batch=32 --gamma=8 --batch-gpu=4 --snap=20`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>4</sub> | <sub>40.86</sub>  | <sub>27.99</sub>  | <sub>5.9</sub> | <sub>`--cfg=stylegan3-r --gpus=4 --batch=32 --gamma=8 --batch-gpu=4`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>8</sub> | <sub>20.44</sub>  | <sub>14.04</sub>  | <sub>5.9</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=8`</sub>

#### 1024x1024 resolution

| <sub>Config</sub><br><br>    | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :----------: | :---------------: | :---------------: | :-------------: | :--
| <sub>StyleGAN3&#8209;T</sub> | <sub>1</sub> | <sub>221.85</sub> | <sub>156.91</sub> | <sub>7.0</sub>  | <sub>`--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>2</sub> | <sub>113.44</sub> | <sub>79.16</sub>  | <sub>6.8</sub>  | <sub>`--cfg=stylegan3-t --gpus=2 --batch=32 --gamma=32 --batch-gpu=4 --snap=10`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>4</sub> | <sub>57.04</sub>  | <sub>39.62</sub>  | <sub>6.7</sub>  | <sub>`--cfg=stylegan3-t --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --snap=20`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>8</sub> | <sub>28.71</sub>  | <sub>20.01</sub>  | <sub>6.6</sub>  | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=32`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>1</sub> | <sub>263.44</sub> | <sub>184.81</sub> | <sub>10.2</sub> | <sub>`--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>2</sub> | <sub>134.22</sub> | <sub>92.58</sub>  | <sub>10.1</sub> | <sub>`--cfg=stylegan3-r --gpus=2 --batch=32 --gamma=32 --batch-gpu=4 --snap=10`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>4</sub> | <sub>67.33</sub>  | <sub>46.53</sub>  | <sub>10.0</sub> | <sub>`--cfg=stylegan3-r --gpus=4 --batch=32 --gamma=32 --batch-gpu=4 --snap=20`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>8</sub> | <sub>34.12</sub>  | <sub>23.42</sub>  | <sub>9.9</sub>  | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=32`</sub>

## Configurations used in StyleGAN3 paper

This section lists the exact settings that we used in the "Alias-Free Generative Adversarial Networks" paper.

#### FFHQ-U and FFHQ at 1024x1024 resolution

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :------------: | :--
| <sub>StyleGAN2</sub>         | <sub>17.55</sub> | <sub>14.57</sub> | <sub>6.2</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>28.71</sub> | <sub>20.01</sub> | <sub>6.6</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=32.8 --mirror=1 --aug=noaug`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>34.12</sub> | <sub>23.42</sub> | <sub>9.9</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=32.8 --mirror=1 --aug=noaug`</sub>

#### MetFaces-U at 1024x1024 resolution

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :-------------: | :--
| <sub>StyleGAN2</sub>         | <sub>18.74</sub> | <sub>11.80</sub> | <sub>7.4</sub>  | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=10 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>29.84</sub> | <sub>21.06</sub> | <sub>7.7</sub>  | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=16.4 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>35.10</sub> | <sub>24.32</sub> | <sub>10.9</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl`</sub>

#### MetFaces at 1024x1024 resolution

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :-------------: | :--
| <sub>StyleGAN2</sub>         | <sub>18.74</sub> | <sub>11.80</sub> | <sub>7.4</sub>  | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=5 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>29.84</sub> | <sub>21.06</sub> | <sub>7.7</sub>  | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>35.10</sub> | <sub>24.32</sub> | <sub>10.9</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=3.3 --mirror=1 --kimg=5000 --snap=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl`</sub>

#### AFHQv2 at 512x512 resolution

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :--------------: | :--------------: | :------------: | :--
| <sub>StyleGAN2</sub>         | <sub>10.90</sub> | <sub>6.60</sub>  | <sub>3.9</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=5 --mirror=1`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>18.47</sub> | <sub>12.29</sub> | <sub>4.3</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=32 --gamma=8.2 --mirror=1`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>20.44</sub> | <sub>14.04</sub> | <sub>5.9</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=32 --gamma=16.4 --mirror=1`</sub>

#### FFHQ-U ablations at 256x256 resolution

| <sub>Config</sub><br><br>    | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :--------------------------- | :-------------: | :-------------: | :------------: | :--
| <sub>StyleGAN2</sub>         | <sub>3.61</sub> | <sub>2.19</sub> | <sub>2.7</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=64 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --glr=0.0025 --dlr=0.0025 --mbstd-group=8`</sub>
| <sub>StyleGAN3&#8209;T</sub> | <sub>7.40</sub> | <sub>3.74</sub> | <sub>3.5</sub> | <sub>`--cfg=stylegan3-t --gpus=8 --batch=64 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025`</sub>
| <sub>StyleGAN3&#8209;R</sub> | <sub>6.71</sub> | <sub>4.81</sub> | <sub>4.2</sub> | <sub>`--cfg=stylegan3-r --gpus=8 --batch=64 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025`</sub>

## Old StyleGAN2-ADA configurations

This section lists command lines that can be used to match the configurations provided by our previous [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) codebase. The first table corresponds to `--cfg=auto` (default) for different resolutions and GPU counts, while the second table lists the remaining alternatives.

#### Default configuration

| <sub>Res.</sub><br><br> | <sub>GPUs</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :---------------------- | :----------: | :---------------: | :--------------: | :------------: | :--
| <sub>128&sup2;</sub>    | <sub>1</sub> | <sub>12.51</sub>  | <sub>6.79</sub>  | <sub>6.2</sub> | <sub>`--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>128&sup2;</sub>    | <sub>2</sub> | <sub>6.43</sub>   | <sub>3.45</sub>  | <sub>6.2</sub> | <sub>`--cfg=stylegan2 --gpus=2 --batch=64 --gamma=0.0512 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>128&sup2;</sub>    | <sub>4</sub> | <sub>3.82</sub>   | <sub>2.23</sub>  | <sub>3.5</sub> | <sub>`--cfg=stylegan2 --gpus=4 --batch=64 --gamma=0.0512 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>256&sup2;</sub>    | <sub>1</sub> | <sub>20.84</sub>  | <sub>12.53</sub> | <sub>4.5</sub> | <sub>`--cfg=stylegan2 --gpus=1 --batch=16 --gamma=0.8192 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>256&sup2;</sub>    | <sub>2</sub> | <sub>10.93</sub>  | <sub>6.36</sub>  | <sub>4.5</sub> | <sub>`--cfg=stylegan2 --gpus=2 --batch=32 --gamma=0.4096 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>256&sup2;</sub>    | <sub>4</sub> | <sub>5.39</sub>   | <sub>3.20</sub>  | <sub>4.5</sub> | <sub>`--cfg=stylegan2 --gpus=4 --batch=64 --gamma=0.2048 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>256&sup2;</sub>    | <sub>8</sub> | <sub>3.89</sub>   | <sub>2.38</sub>  | <sub>2.6</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=64 --gamma=0.2048 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384`</sub>
| <sub>512&sup2;</sub>    | <sub>1</sub> | <sub>71.59</sub>  | <sub>41.06</sub> | <sub>6.8</sub> | <sub>`--cfg=stylegan2 --gpus=1 --batch=8 --gamma=6.5536 --map-depth=2 --glr=0.0025 --dlr=0.0025`</sub>
| <sub>512&sup2;</sub>    | <sub>2</sub> | <sub>36.79</sub>  | <sub>20.83</sub> | <sub>6.8</sub> | <sub>`--cfg=stylegan2 --gpus=2 --batch=16 --gamma=3.2768 --map-depth=2 --glr=0.0025 --dlr=0.0025`</sub>
| <sub>512&sup2;</sub>    | <sub>4</sub> | <sub>18.12</sub>  | <sub>10.45</sub> | <sub>6.7</sub> | <sub>`--cfg=stylegan2 --gpus=4 --batch=32 --gamma=1.6384 --map-depth=2 --glr=0.0025 --dlr=0.0025`</sub>
| <sub>512&sup2;</sub>    | <sub>8</sub> | <sub>9.09</sub>   | <sub>5.24</sub>  | <sub>6.8</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=64 --gamma=0.8192 --map-depth=2 --glr=0.0025 --dlr=0.0025`</sub>
| <sub>1024&sup2;</sub>   | <sub>1</sub> | <sub>141.83</sub> | <sub>90.39</sub> | <sub>7.2</sub> | <sub>`--cfg=stylegan2 --gpus=1 --batch=4 --gamma=52.4288 --map-depth=2`</sub>
| <sub>1024&sup2;</sub>   | <sub>2</sub> | <sub>73.13</sub>  | <sub>46.04</sub> | <sub>7.2</sub> | <sub>`--cfg=stylegan2 --gpus=2 --batch=8 --gamma=26.2144 --map-depth=2`</sub>
| <sub>1024&sup2;</sub>   | <sub>4</sub> | <sub>36.95</sub>  | <sub>23.15</sub> | <sub>7.0</sub> | <sub>`--cfg=stylegan2 --gpus=4 --batch=16 --gamma=13.1072 --map-depth=2`</sub>
| <sub>1024&sup2;</sub>   | <sub>8</sub> | <sub>18.47</sub>  | <sub>11.66</sub> | <sub>7.3</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=6.5536 --map-depth=2`</sub>

#### Repro configurations

| <sub>Name</sub><br><br> | <sub>s/kimg</sub><br><sup>(V100)</sup> | <sub>s/kimg</sub><br><sup>(A100)</sup> | <sub>GPU</sub><br><sup>mem</sup> | <sub>Options</sub><br><br>
| :---------------------- | :--------------: | :--------------: | :------------: | :--
| <sub>`stylegan2`</sub>  | <sub>17.55</sub> | <sub>14.57</sub> | <sub>6.2</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=10`</sub>
| <sub>`paper256`</sub>   | <sub>4.01</sub>  | <sub>2.47</sub>  | <sub>2.7</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=64 --gamma=1 --cbase=16384 --glr=0.0025 --dlr=0.0025 --mbstd-group=8`</sub>
| <sub>`paper512`</sub>   | <sub>9.11</sub>  | <sub>5.28</sub>  | <sub>6.7</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=64 --gamma=0.5 --glr=0.0025 --dlr=0.0025 --mbstd-group=8`</sub>
| <sub>`paper1024`</sub>  | <sub>18.56</sub> | <sub>11.75</sub> | <sub>6.9</sub> | <sub>`--cfg=stylegan2 --gpus=8 --batch=32 --gamma=2`</sub>
