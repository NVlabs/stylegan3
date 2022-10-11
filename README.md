# StyleGAN3-Fun<br><sub>Let's have fun with StyleGAN2/ADA/3!</sub>

SOTA GANs are hard to train and to explore, and StyleGAN2/ADA/3 are no different. The point of this repository is to allow
the user to both easily train and explore the trained models without unnecessary headaches. 

As [before](https://github.com/PDillis/stylegan2-fun), we will build upon the official repository, which has the advantage
of being backwards-compatible. As such, we can use our previously-trained models from StyleGAN2 and StyleGAN2-ADA. Please
get acquainted with the official repository and its codebase, as we will be building upon it and as such, increase its
capabilities (but hopefully not its complexity!). 

## Additions

This repository adds/has the following changes (not yet the complete list):

* ***Dataset tool***
  * **Add RGBA support**, so revert saving images to `.png` ([Issue #156](https://github.com/NVlabs/stylegan3/issues/156) by @1378dm). Training can use RGBA and images can be generated.
    * ***TODO:*** ~~Check that training code is correct for normalizing the alpha channel~~, as well as making the 
      interpolation code work with this new format (look into [`moviepy.editor.VideoClip`](https://zulko.github.io/moviepy/getting_started/videoclips.html?highlight=mask#mask-clips))
    * For now, interpolations will only be saved in RGB format.
  * **Add `--center-crop-tall`**: add vertical black bars to the sides instead, in the same vein as the horizontal bars in
    `--center-crop-wide`.
  * Grayscale images in the dataset are converted to `RGB`
    * If you want to turn this off, remove the respective line in `dataset_tool.py`, e.g., if your dataset is made of images in a folder, then the function to be used is
    `open_image_folder` in `dataset_tool.py`, and the line to be removed is `img = img.convert('RGB')` in the `iterate_images` inner function.
  * The dataset can be forced to be of a specific number of channels, that is, grayscale, RGB or RGBA.
    * To use this, set `--force-channels=1` for grayscale, `--force-channels=3` for RGB, and `--force-channels=4` for RGBA.
  * If the dataset tool encounters an error, print it along the offending image, but continue with the rest of the dataset 
    ([PR #39](https://github.com/NVlabs/stylegan3/pull/39) from [Andreas Jansson](https://github.com/andreasjansson)). 
  * ***TODO:*** Add multi-crop, as used in [Earth View](https://github.com/PDillis/earthview#multi-crop---data_augmentpy).
* ***Training***
  * `--mirrory`: Added vertical mirroring for doubling the dataset size (quadrupling if `--mirror` is used; make sure your dataset has either or both 
    of these symmetries in order for it to make sense to use them)
  * `--gamma`: If no R1 regularization is provided, the heuristic formula from [StyleGAN2](https://github.com/NVlabs/stylegan2) will be used.
  * `--aug`: ***TODO:*** add [Deceive-D/APA](https://github.com/EndlessSora/DeceiveD) as an option.
  * `--augpipe`: Now available to use is [StyleGAN2-ADA's](https://github.com/NVlabs/stylegan2-ada-pytorch) full list of augpipe, i.e., individual augmentations (`blit`, `geom`, `color`, `filter`, `noise`, `cutout`) or their combinations (`bg`, `bgc`, `bgcf`, `bgcfn`, `bgcfnc`).
  * `--img-snap`: Set when to save snapshot images, so now it's independent of when the model is saved (e.g., save image snapshots more often to know how the model is training without saving the model itself, to save space).
  * `--snap-res`: The resolution of the snapshots, depending on how many images you wish to see per snapshot. Available resolutions: `1080p`, `4k`, and `8k`.
  * `--resume-kimg`: Starting number of `kimg`, useful when continuing training a previous run
  * `--outdir`: Automatically set as `training-runs`, so no need to set beforehand (in general this is true throughout the repository)
  * `--metrics`: Now set by default to `None`, so there's no need to worry about this one
  * `--freezeD`: Renamed `--freezed` for better readability
  * `--freezeM`: Freeze the first layers of the Mapping Network Gm (`G.mapping`)
  * `--freezeE`: Freeze the embedding layer of the Generator (for class-conditional models)
  * `--freezeG`: ***TODO:*** Freeze the first layers of the Synthesis Network (`G.synthesis`; less cost to transfer learn, focus on high layers?)
  * `--resume`: All available pre-trained models from NVIDIA (and more) can be used with a simple dictionary, depending on the `--cfg` used.
  For example, if you wish to use StyleGAN3's `config-r`, then set `--cfg=stylegan3-r`. In addition, if you wish to transfer learn from FFHQU at 1024 resolution, set `--resume=ffhqu1024`.
    * The full list of currently available models to transfer learn from (or synthesize new images with) is the following (***TODO:*** add small description of each model, 
    so the user can better know which to use for their particular usecase; proper citation to original authors as well):
    
        <details>
        <summary>StyleGAN2 models</summary>
    
      1. Majority, if not all, are `config-f`: set `--cfg=stylegan2`
         * `ffhq256`
         * `ffhqu256`
         * `ffhq512`
         * `ffhq1024`
         * `ffhqu1024`
         * `celebahq256`
         * `lsundog256`
         * `afhqcat512`
         * `afhqdog512`
         * `afhqwild512`
         * `afhq512`
         * `brecahad512`
         * `cifar10` (conditional, 10 classes)
         * `metfaces1024`
         * `metfacesu1024`
         * `lsuncar512` (config-f)
         * `lsuncat256` (config-f)
         * `lsunchurch256` (config-f)
         * `lsunhorse256` (config-f)
         * `minecraft1024` (thanks to @jeffheaton)
         * `imagenet512` (thanks to @shawwn)
         * `wikiart1024-C` (conditional, 167 classes; thanks to @pbaylies)
         * `wikiart1024-U` (thanks to @pbaylies)
         * `maps1024` (thanks to @tjukanov)
         * `fursona512` (thanks to @arfafax)
         * `mlpony512` (thanks to @arfafax)
         * `afhqcat256` (Deceive-D/APA models)
         * `anime256` (Deceive-D/APA models)
         * `cub256` (Deceive-D/APA models)
         * `sddogs1024` (Self-Distilled StyleGAN models)
         * `sdelephant512` (Self-Distilled StyleGAN models)
         * `sdhorses512` (Self-Distilled StyleGAN models)
         * `sdbicycles256` (Self-Distilled StyleGAN models)
         * `sdlions512` (Self-Distilled StyleGAN models)
         * `sdgiraffes512` (Self-Distilled StyleGAN models)
         * `sdparrots512` (Self-Distilled StyleGAN models)
        </details>
    
        <details>
        <summary>StyleGAN3 models</summary>
    
      1. `config-t`: set `--cfg=stylegan3-t`
         * `afhq512`
         * `ffhqu256`
         * `ffhq1024`
         * `ffhqu1024`
         * `metfaces1024`
         * `metfacesu1024`
         * `landscapes256` (thanks to @justinpinkney)
         * `wikiart1024` (thanks to @justinpinkney)
         * `mechfuture256` (thanks to @edstoica; 29 kimg tick)
         * `vivflowers256` (thanks to @edstoica; 68 kimg tick)
         * `alienglass256` (thanks to @edstoica; 38 kimg tick)
         * `scificity256` (thanks to @edstoica; 210 kimg tick)
         * `scifiship256` (thanks to @edstoica; 168 kimg tick)
      2. `config-r`: set `--cfg=stylegan3-r`
         * `afhq512`
         * `ffhq1024`
         * `ffhqu1024`
         * `ffhqu256`
         * `metfaces1024`
         * `metfacesu1024`
        </details>

    * The main sources of these pretrained models are both the [official NVIDIA repository](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3),
      as well as other community repositories, such as [Justin Pinkney](https://github.com/justinpinkney) 's [Awesome Pretrained StyleGAN2](https://github.com/justinpinkney/awesome-pretrained-stylegan2)
      and [Awesome Pretrained StyleGAN3](https://github.com/justinpinkney/awesome-pretrained-stylegan3), [Deceive-D/APA](https://github.com/EndlessSora/DeceiveD),
      [Self-Distilled StyleGAN/Internet Photos](https://github.com/self-distilled-stylegan/self-distilled-internet-photos), and [edstoica](https://github.com/edstoica) 's 
      [Wombo Dream](https://www.wombo.art/) [-based models](https://github.com/edstoica/lucid_stylegan3_datasets_models). Others can be found around the net and are properly credited in this repository,
      so long as they can be easily downloaded with [`dnnlib.util.open_url`](https://github.com/PDillis/stylegan3-fun/blob/4ce9d6f7601641ba1e2906ed97f2739a63fb96e2/dnnlib/util.py#L396).

* ***Interpolation videos***
    * [Random interpolation](https://youtu.be/DNfocO1IOUE)
      * [Generate images/interpolations with the internal representations of the model](https://nvlabs-fi-cdn.nvidia.com/_web/stylegan3/videos/video_8_internal_activations.mp4)
        * Usage: Add `--layer=<layer_name>` to specify which layer to use for interpolation.
        * If you don't know the names of the layers available for your model, add the flag `--available-layers` and the 
        layers will be printed to the console, along their names, number of channels, and sizes. 
        * Use one of `--grayscale` or `--rgb` to specify whether to save the images as grayscale or RGB during the interpolation.
        * For `--rgb`, three consecutive channels (starting at `--starting-channel=0`) will be used to create the RGB image. For `--grayscale`, only the first channel will be used.
    * Style-mixing
    * [Sightseeding](https://twitter.com/PDillis/status/1270341433401249793?s=20&t=yLueNkagqsidZFqZ2jNPAw) (jumpiness has been fixed)
    * [Circular interpolation](https://youtu.be/4nktYGjSVHg)
    * [Visual-reactive interpolation](https://youtu.be/KoEAkPnE-zA) (Beta)
    * Audiovisual-reactive interpolation (TODO)
    * ***TODO:*** Give support to RGBA models!
* ***Projection into the latent space***
    * [Project into $\mathcal{W}+$](https://arxiv.org/abs/1904.03189)
    * Additional losses to use for better projection (e.g., using VGG16 or [CLIP](https://github.com/openai/CLIP))
* ***[Discriminator Synthesis](https://arxiv.org/abs/2111.02175)*** (official code)
    * Generate a static image or a [video](https://youtu.be/hEJKWL2VQTE) with a feedback loop
    * Start from a random image (`random` or `perlin`, using [Mathieu Duchesneau's implementation](https://github.com/duchesneaumathieu/pyperlin)) or from an existing one
* ***Expansion on GUI/`visualizer.py`***
    * Added the rest of the affine transformations
    * Added widget for class-conditional models (***TODO:*** mix classes with continuous values for `cls`!)
* ***General model and code additions***
    * [Multi-modal truncation trick](https://arxiv.org/abs/2202.12211): find the different clusters in your model and use the closest one to your dlatent, in order to increase the fidelity (TODO: finish skeleton implementation)
    * StyleGAN3: anchor the latent space for easier to follow interpolations (thanks to [Rivers Have Wings](https://github.com/crowsonkb) and [nshepperd](https://github.com/nshepperd)).
    * Use CPU instead of GPU if desired (not recommended, but perfectly fine for generating images, whenever the custom CUDA kernels fail to compile).
    * Add missing dependencies and channels so that the [`conda`](https://docs.conda.io/en/latest/) environment is correctly setup in Windows 
      (PR's [#111](https://github.com/NVlabs/stylegan3/pull/111)/[#125](https://github.com/NVlabs/stylegan3/pull/125) and [#80](https://github.com/NVlabs/stylegan3/pull/80) /[#143](https://github.com/NVlabs/stylegan3/pull/143) from the base, respectively)
    * Use [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada) models with any part of the code (Issue [#9](https://github.com/PDillis/stylegan3-fun/issues/9))
      * The StyleGAN-NADA models must first be converted via [Vadim Epstein](https://github.com/eps696) 's conversion code found [here](https://github.com/eps696/stylegan2ada#tweaking-models).
    * Add PR [#173](https://github.com/NVlabs/stylegan3/pull/173) for adding the last remaining unknown kwarg for using StyleGAN2 models using TF 1.15.
* ***TODO*** list (this is a long one with more to come, so any help is appreciated):
  * Multi-modal truncation trick: finish skeleton code, add automatic selection of centroid (w.r.t. L2 or LPIPS; user-selected)
  * [PTI](https://github.com/danielroich/PTI) for better inversion
  * [Better sampling](https://arxiv.org/abs/2110.08009)
  * [Progressive growing modules for StyleGAN-XL](https://github.com/autonomousvision/stylegan_xl) to be able to use the pretrained models
  * [Add cross-model interpolation](https://twitter.com/arfafax/status/1297681537337446400?s=20&t=xspnTaLFTvd7y4krg8tkxA)
  * Generate class labels automatically with dataset structure (subfolders and such)
  * Make it easy to download pretrained models from Drive, otherwise a lot of models can't be used with `dnnlib.util.open_url`
    (e.g., [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human) models)
  * Finish documentation for better user experience, add videos/images, code samples, visuals...
  * Add [Ensembling Off-the-shelf Models for GAN Training](https://arxiv.org/abs/2112.09130) and [Any-resolution Training for High-resolution Image Synthesis](https://chail.github.io/anyres-gan/)

## Notebooks (Coming soon!)

## Sponsors ![GitHub Sponsor](https://img.shields.io/github/sponsors/PDillis?label=Sponsor&logo=GitHub)

This repository is sponsored by:

[isosceles](https://www.jasonfletcher.info/vjloops/)

Thank you so much! 

If you wish to sponsor me, click here: [![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/PDillis)


---

## Alias-Free Generative Adversarial Networks (StyleGAN3)<br><sub>Official PyTorch implementation of the NeurIPS 2021 paper</sub>

![Teaser image](./docs/stylegan3-teaser-1920x1006.png)

**Alias-Free Generative Adversarial Networks**<br>
Tero Karras, Miika Aittala, Samuli Laine, Erik H&auml;rk&ouml;nen, Janne Hellsten, Jaakko Lehtinen, Timo Aila<br>
https://nvlabs.github.io/stylegan3<br>

Abstract: *We observe that despite their hierarchical convolutional nature, the synthesis process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to be glued to image coordinates instead of the surfaces of depicted objects. We trace the root cause to careless signal processing that causes aliasing in the generator network. Interpreting all signals in the network as continuous, we derive generally applicable, small architectural changes that guarantee that unwanted information cannot leak into the hierarchical synthesis process. The resulting networks match the FID of StyleGAN2 but differ dramatically in their internal representations, and they are fully equivariant to translation and rotation even at subpixel scales. Our results pave the way for generative models better suited for video and animation.*

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## Release notes

This repository is an updated version of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), with several new features:
- Alias-free generator architecture and training configurations (`stylegan3-t`, `stylegan3-r`).
- Tools for interactive visualization (`visualizer.py`), spectral analysis (`avg_spectra.py`), and video generation (`gen_video.py`).
- Equivariance metrics (`eqt50k_int`, `eqt50k_frac`, `eqr50k`).
- General improvements: reduced memory usage, slightly faster training, bug fixes.

Compatibility:
- Compatible with old network pickles created using [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada) and [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).  (Note: running old StyleGAN2 models on StyleGAN3 code will produce the same results as running them on stylegan2-ada/stylegan2-ada-pytorch.  To benefit from the StyleGAN3 architecture, you need to retrain.)
- Supports old StyleGAN2 training configurations, including ADA and transfer learning. See [Training configurations](./docs/configs.md) for details.
- Improved compatibility with Ampere GPUs and newer versions of PyTorch, CuDNN, etc.

## Synthetic image detection

While new generator approaches enable new media synthesis capabilities, they may also present a new challenge for AI forensics algorithms for detection and attribution of synthetic media. In collaboration with digital forensic researchers participating in DARPA's SemaFor program, we curated a synthetic image dataset that allowed the researchers to test and validate the performance of their image detectors in advance of the public release. Please see [here](https://github.com/NVlabs/stylegan3-detector) for more details.

## Additional material

- [Result videos](https://nvlabs-fi-cdn.nvidia.com/stylegan3/videos/)
- [Curated example images](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/)
- [StyleGAN3 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3) for config T (translation equiv.) and config R (translation and rotation equiv.)
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan3-t-ffhq-1024x1024.pkl`, `stylegan3-t-ffhqu-1024x1024.pkl`, `stylegan3-t-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-r-ffhq-1024x1024.pkl`, `stylegan3-r-ffhqu-1024x1024.pkl`, `stylegan3-r-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-t-metfaces-1024x1024.pkl`, `stylegan3-t-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-r-metfaces-1024x1024.pkl`, `stylegan3-r-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-t-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan3-r-afhqv2-512x512.pkl`</sub><br>
- [StyleGAN2 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan2) compatible with this codebase
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan2-ffhq-1024x1024.pkl`, `stylegan2-ffhq-512x512.pkl`, `stylegan2-ffhq-256x256.pkl`</sub><br>
  > <sub>`stylegan2-ffhqu-1024x1024.pkl`, `stylegan2-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan2-metfaces-1024x1024.pkl`, `stylegan2-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan2-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan2-afhqcat-512x512.pkl`, `stylegan2-afhqdog-512x512.pkl`, `stylegan2-afhqwild-512x512.pkl`</sub><br>
  > <sub>`stylegan2-brecahad-512x512.pkl`, `stylegan2-cifar10-32x32.pkl`</sub><br>
  > <sub>`stylegan2-celebahq-256x256.pkl`, `stylegan2-lsundog-256x256.pkl`</sub><br>

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using Tesla V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate stylegan3`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

See [Troubleshooting](./docs/troubleshooting.md) for help on common installation and run-time problems.

## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```bash
# Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

# Render a 4x2 grid of interpolations for seeds 0 through 31.
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.

**Docker**: You can run the above curated image example using Docker as follows:

```bash
# Build the stylegan3:latest image
docker build --tag stylegan3 .

# Run the gen_images.py script using Docker:
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    stylegan3 \
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
         --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

Note: The Docker image requires NVIDIA driver release `r470` or later.

The `docker run` invocation may look daunting, so let's unpack its contents here:

- `--gpus all -it --rm --user $(id -u):$(id -g)`: with all GPUs enabled, run an interactive session with current user's UID/GID to avoid Docker writing files as root.
- ``-v `pwd`:/scratch --workdir /scratch``: mount current running dir (e.g., the top of this git repo on your host machine) to `/scratch` in the container and use that as the current working dir.
- `-e HOME=/scratch`: let PyTorch and StyleGAN3 code know where to cache temporary files such as pre-trained models and custom PyTorch extension build results. Note: if you want more fine-grained control, you can instead set `TORCH_EXTENSIONS_DIR` (for custom extensions build dir) and `DNNLIB_CACHE_DIR` (for pre-trained model download cache). You want these cache dirs to reside on persistent volumes so that their contents are retained across multiple `docker run` invocations.

## Interactive visualization

This release contains an interactive model visualization tool that can be used to explore various characteristics of a trained model.  To start it, run:

```bash
python visualizer.py
```

<a href="./docs/visualizer_screen0.png"><img alt="Visualizer screenshot" src="./docs/visualizer_screen0_half.png"></img></a>

## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.

The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options:

```python
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
```

Please refer to [`gen_images.py`](./gen_images.py) for complete code example.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

**FFHQ**: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and create a zip archive using `dataset_tool.py`:

```bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-1024x1024.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
```

See the [FFHQ README](https://github.com/NVlabs/ffhq-dataset) for information on how to obtain the unaligned FFHQ dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create a ZIP archive:

```bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces-1024x1024.zip
```

See the [MetFaces README](https://github.com/NVlabs/metfaces-dataset) for information on how to obtain the unaligned MetFaces dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**AFHQv2**: Download the [AFHQv2 dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create a ZIP archive:

```bash
python dataset_tool.py --source=~/downloads/afhqv2 --dest=~/datasets/afhqv2-512x512.zip
```

Note that the above command creates a single combined dataset using all images of all three classes (cats, dogs, and wild animals), matching the setup used in the StyleGAN3 paper. Alternatively, you can also create a separate dataset for each class:

```bash
python dataset_tool.py --source=~/downloads/afhqv2/train/cat --dest=~/datasets/afhqv2cat-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/dog --dest=~/datasets/afhqv2dog-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/wild --dest=~/datasets/afhqv2wild-512x512.zip
```

## Training

You can train new networks using `train.py`. For example:

```bash
# Train StyleGAN3-T for AFHQv2 using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1

# Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

# Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
```

Note that the result quality and training time depend heavily on the exact set of options. The most important ones (`--gpus`, `--batch`, and `--gamma`) must be specified explicitly, and they should be selected with care. See [`python train.py --help`](./docs/train-help.txt) for the full list of options and [Training configurations](./docs/configs.md) for general guidelines &amp; recommendations, along with the expected training speed &amp; memory usage in different scenarios.

The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-stylegan3-t-afhqv2-512x512-gpus8-batch32-gamma8.2`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.

## Quality metrics

By default, `train.py` automatically computes FID for each network pickle exported during training. We recommend inspecting `metric-fid50k_full.jsonl` (or TensorBoard) at regular intervals to monitor the training progress. When desired, the automatic computation can be disabled with `--metrics=none` to speed up the training slightly.

Additional quality metrics can also be computed after the training:

```bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=eqt50k_int,eqr50k \
    --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl
```

The first example looks up the training configuration and performs the same operation as if `--metrics=eqt50k_int,eqr50k` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--data` and `--mirror` must be specified explicitly.

Note that the metrics can be quite expensive to compute (up to 1h), and many of them have an additional one-off cost for each new dataset (up to 30min). Also note that the evaluation is done using a different random seed each time, so the results will vary if the same metric is computed multiple times.

Recommended metrics:
* `fid50k_full`: Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset.
* `kid50k_full`: Kernel inception distance<sup>[2]</sup> against the full dataset.
* `pr50k3_full`: Precision and recall<sup>[3]</sup> againt the full dataset.
* `ppl2_wend`: Perceptual path length<sup>[4]</sup> in W, endpoints, full image.
* `eqt50k_int`: Equivariance<sup>[5]</sup> w.r.t. integer translation (EQ-T).
* `eqt50k_frac`: Equivariance w.r.t. fractional translation (EQ-T<sub>frac</sub>).
* `eqr50k`: Equivariance w.r.t. rotation (EQ-R).

Legacy metrics:
* `fid50k`: Fr&eacute;chet inception distance against 50k real images.
* `kid50k`: Kernel inception distance against 50k real images.
* `pr50k3`: Precision and recall against 50k real images.
* `is50k`: Inception score<sup>[6]</sup> for CIFAR-10.

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
5. [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3), Karras et al. 2021
6. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016

## Spectral analysis

The easiest way to inspect the spectral properties of a given generator is to use the built-in FFT mode in `visualizer.py`. In addition, you can visualize average 2D power spectra (Appendix A, Figure 15) as follows:

```bash
# Calculate dataset mean and std, needed in subsequent steps.
python avg_spectra.py stats --source=~/datasets/ffhq-1024x1024.zip

# Calculate average spectrum for the training data.
python avg_spectra.py calc --source=~/datasets/ffhq-1024x1024.zip \
    --dest=tmp/training-data.npz --mean=112.684 --std=69.509

# Calculate average spectrum for a pre-trained generator.
python avg_spectra.py calc \
    --source=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl \
    --dest=tmp/stylegan3-r.npz --mean=112.684 --std=69.509 --num=70000

# Display results.
python avg_spectra.py heatmap tmp/training-data.npz
python avg_spectra.py heatmap tmp/stylegan3-r.npz
python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz
```

<a href="./docs/avg_spectra_screen0.png"><img alt="Average spectra screenshot" src="./docs/avg_spectra_screen0_half.png"></img></a>

## License

Copyright &copy; 2021, NVIDIA Corporation & affiliates. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

## Citation

```
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

We thank David Luebke, Ming-Yu Liu, Koki Nagano, Tuomas Kynk&auml;&auml;nniemi, and Timo Viitanen for reviewing early drafts and helpful suggestions. Fr&eacute;do Durand for early discussions. Tero Kuosmanen for maintaining our compute infrastructure. AFHQ authors for an updated version of their dataset. Getty Images for the training images in the Beaches dataset. We did not receive external funding or additional revenues for this project.
