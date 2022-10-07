import torch
import torch.nn as nn
from torchvision import models

import numpy as np

from typing import List, Tuple
from collections import OrderedDict
import operator


# ----------------------------------------------------------------------------


class VGG16Features(torch.nn.Module):
    """
    Use pre-trained VGG16 provided by PyTorch. Code modified from lainwired/pacifinapacific
    https://github.com/pacifinapacific/StyleGAN_LatentEditor. My modification is that we can use
    the ReLU activation if we want, or the pure conv1_1, conv1_2, conv3_2, and conv4_2 activations.

    My conclusions are that it's best to have one model of VGG, so I will use the one provided by NVIDIA
    as it is both easier to slice and it can return LPIPS if so desired.
    """
    # Image2StyleGAN: How to Embed Images into the StyleGAN latent space? https://arxiv.org/abs/1904.03189,
    #                   layers = [0, 2, 12, 19]
    # Image2StyleGAN++: How to Edit the Embedded Images? https://arxiv.org/abs/1911.11544,
    #                   layers = [0, 2, 7, 14], but make sure to return conv3_3 twice for the Style Loss
    def __init__(self, device, use_relu=False):
        super(VGG16Features, self).__init__()
        # Load and partition the model
        vgg16 = models.vgg16(pretrained=True).to(device)
        self.vgg16_features = vgg16.features
        self.avgpool = vgg16.avgpool  # TODO: more work can be done to partition any part of the model, but not my jam
        self.classifier = vgg16.classifier

        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()

        layers = [0, 2, 12, 19]
        if use_relu:
            layers = [layer + 1 for layer in layers]

        for i in range(layers[0] + 1):
            self.conv1_1.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[0] + 1, layers[1] + 1):
            self.conv1_2.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[1] + 1, layers[2] + 1):
            self.conv3_2.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[2] + 1, layers[3] + 1):
            self.conv4_2.add_module(str(i), self.vgg16_features[i])

        # We're not optimizing VGG16
        for param in self.parameters():
            param.requires_grad = False

    def get_feature_layers(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        conv3_2 = self.conv3_2(conv1_2)
        conv4_2 = self.conv4_2(conv3_2)

        conv1_1 = conv1_1 / torch.numel(conv1_1)
        conv1_2 = conv1_2 / torch.numel(conv1_2)
        conv3_2 = conv3_2 / torch.numel(conv3_2)
        conv4_2 = conv4_2 / torch.numel(conv4_2)

        return conv1_1, conv1_2, conv3_2, conv4_2


class VGG16FeaturesNVIDIA(torch.nn.Module):
    def __init__(self, vgg16):
        super(VGG16FeaturesNVIDIA, self).__init__()
        # NOTE: ReLU is already included in the output of every conv output
        self.conv1_1 = vgg16.layers.conv1
        self.conv1_2 = vgg16.layers.conv2
        self.pool1 = vgg16.layers.pool1

        self.conv2_1 = vgg16.layers.conv3
        self.conv2_2 = vgg16.layers.conv4
        self.pool2 = vgg16.layers.pool2

        self.conv3_1 = vgg16.layers.conv5
        self.conv3_2 = vgg16.layers.conv6
        self.conv3_3 = vgg16.layers.conv7
        self.pool3 = vgg16.layers.pool3

        self.conv4_1 = vgg16.layers.conv8
        self.conv4_2 = vgg16.layers.conv9
        self.conv4_3 = vgg16.layers.conv10
        self.pool4 = vgg16.layers.pool4

        self.conv5_1 = vgg16.layers.conv11
        self.conv5_2 = vgg16.layers.conv12
        self.conv5_3 = vgg16.layers.conv13
        self.pool5 = vgg16.layers.pool5
        self.adavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))  # We need this for 256x256 images (> 224x224)

        self.fc1 = vgg16.layers.fc1
        self.fc2 = vgg16.layers.fc2
        self.fc3 = vgg16.layers.fc3
        self.softmax = vgg16.layers.softmax

    def get_layers_features(self, x: torch.Tensor, layers: List[str], normed: bool = False, sqrt_normed: bool = False):
        """
        x is an image/tensor of shape [1, 3, 256, 256], and layers is a list of the names of the layers you wish
        to return in order to compare the activations/features with another image.

        Example:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            img1 = torch.randn(1, 3, 256, 256, device=device)
            img2 = torch.randn(1, 3, 256, 256, device=device)
            layers = ['conv1_1', 'conv1_2', 'conv3_3', 'conv3_3', 'fc3']  # Indeed, return twice conv3_3

            # Load the VGG16 feature detector.
            url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            with dnnlib.util.open_url(url) as f:
                vgg16 = torch.jit.load(f).eval().to(device)

            vgg16 = VGG16FeaturesNVIDIA(vgg16)

            # Get the desired features from the layers list
            features1 = vgg16.get_layers_features(img1, layers)
            features2 = vgg16.get_layers_features(img2, layers)

            # Get, e.g., the MSE loss between the two features
            mse = torch.nn.MSELoss(reduction='mean')
            loss = sum(map(lambda x, y: mse(x, y), features1, features2))
        """
        # Legend: => conv2d, -> max pool 2d, ~> adaptive average pool 2d, ->> fc layer; shapes of input/output are shown
        assert layers is not None

        features_dict = OrderedDict()
        features_dict['conv1_1'] = self.conv1_1(x)                            # [1, 3, 256, 256] => [1, 64, 256, 256]
        features_dict['conv1_2'] = self.conv1_2(features_dict['conv1_1'])     # [1, 64, 256, 256] => [1, 64, 256, 256]
        features_dict['pool1'] = self.pool1(features_dict['conv1_2'])         # [1, 64, 256, 256] -> [1, 64, 128, 128]

        features_dict['conv2_1'] = self.conv2_1(features_dict['pool1'])       # [1, 64, 128, 128] => [1, 128, 128, 128]
        features_dict['conv2_2'] = self.conv2_2(features_dict['conv2_1'])     # [1, 128, 128, 128] => [1, 128, 128, 128]
        features_dict['pool2'] = self.pool2(features_dict['conv2_2'])         # [1, 128, 128, 128] -> [1, 128, 64, 64]

        features_dict['conv3_1'] = self.conv3_1(features_dict['pool2'])       # [1, 128, 64, 64] => [1, 256, 64, 64]
        features_dict['conv3_2'] = self.conv3_2(features_dict['conv3_1'])     # [1, 256, 64, 64] => [1, 256, 64, 64]
        features_dict['conv3_3'] = self.conv3_3(features_dict['conv3_2'])     # [1, 256, 64, 64] => [1, 256, 64, 64]
        features_dict['pool3'] = self.pool3(features_dict['conv3_3'])         # [1, 256, 64, 64] -> [1, 256, 32, 32]

        features_dict['conv4_1'] = self.conv4_1(features_dict['pool3'])       # [1, 256, 32, 32] => [1, 512, 32, 32]
        features_dict['conv4_2'] = self.conv4_2(features_dict['conv4_1'])     # [1, 512, 32, 32] => [1, 512, 32, 32]
        features_dict['conv4_3'] = self.conv4_3(features_dict['conv4_2'])     # [1, 512, 32, 32] => [1, 512, 32, 32]
        features_dict['pool4'] = self.pool4(features_dict['conv4_3'])         # [1, 512, 32, 32] -> [1, 512, 16, 16]

        features_dict['conv5_1'] = self.conv5_1(features_dict['pool4'])       # [1, 512, 16, 16] => [1, 512, 16, 16]
        features_dict['conv5_2'] = self.conv5_2(features_dict['conv5_1'])     # [1, 512, 16, 16] => [1, 512, 16, 16]
        features_dict['conv5_3'] = self.conv5_3(features_dict['conv5_2'])     # [1, 512, 16, 16] => [1, 512, 16, 16]
        features_dict['pool5'] = self.pool5(features_dict['conv5_3'])         # [1, 512, 16, 16] -> [1, 512, 8, 8]

        features_dict['adavgpool'] = self.adavgpool(features_dict['pool5'])   # [1, 512, 8, 8] ~> [1, 512, 7, 7]
        features_dict['fc1'] = self.fc1(features_dict['adavgpool'])           # [1, 512, 7, 7] ->> [1, 4096]; w/ReLU
        features_dict['fc2'] = self.fc2(features_dict['fc1'])                 # [1, 4096] ->> [1, 4096]; w/ReLU
        features_dict['fc3'] = self.softmax(self.fc3(features_dict['fc2']))   # [1, 4096] ->> [1, 1000]; w/o ReLU; apply softmax

        result_list = list()
        for layer in layers:
            if normed:
                # Divide each layer by the number of elements in it
                result_list.append(features_dict[layer] / torch.numel(features_dict[layer]))
            elif sqrt_normed:
                # Divide each layer by the square root of the number of elements in it
                result_list.append(features_dict[layer] / torch.tensor(torch.numel(features_dict[layer]),
                                                                       dtype=torch.float).sqrt())
            else:
                result_list.append(features_dict[layer])
        return result_list


# ----------------------------------------------------------------------------


class DiscriminatorFeatures(torch.nn.Module):
    def __init__(self, D):
        super(DiscriminatorFeatures, self).__init__()

        # assert D.init_kwargs.architecture == 'resnet'  # removed as some resnet models don't have this attribute
        self.block_resolutions = D.block_resolutions

        # For loop to get all the inner features of the trained Discriminator with a resnet architecture
        for res in self.block_resolutions:
            if res == D.img_resolution:
                setattr(self, 'from_rgb', operator.attrgetter(f'b{res}.fromrgb')(D))
            setattr(self, f'b{res}_skip', operator.attrgetter(f'b{res}.skip')(D))
            setattr(self, f'b{res}_conv0', operator.attrgetter(f'b{res}.conv0')(D))
            setattr(self, f'b{res}_conv1', operator.attrgetter(f'b{res}.conv1')(D))

        # Unique, last block with a fc/out, so we can extract features in a regular fashion
        setattr(self, 'b4_mbstd', D.b4.mbstd)
        setattr(self, 'b4_conv', D.b4.conv)
        setattr(self, 'adavgpool', nn.AdaptiveAvgPool2d(4))  # Necessary if images are of different resolution than D.img_resolution
        setattr(self, 'fc', D.b4.fc)
        setattr(self, 'out', D.b4.out)

    def get_block_resolutions(self):
        """Get the block resolutions available for the current Discriminator. Remove?"""
        return self.block_resolutions

    def get_layers_features(self,
                            x: torch.Tensor,            # Input image
                            layers: List[str] = None,
                            channels: List[int] = None,
                            normed: bool = False,
                            sqrt_normed: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Get the feature of a specific layer of the Discriminator (with resnet architecture). The following shows the
        shapes of an image, x, as it flows through the different blocks that compose the Discriminator.

        *** Legend: => conv2d, -> flatten, ->> fc layer, ~> mbstd layer, +> adaptive average pool ***

        # First block / DiscriminatorBlock
        from_rgb = self.from_rgb(x)                                         # [1, 3, 1024, 1024] => [1, 32, 1024, 1024]
        b1024_skip = self.b1024_skip(from_rgb, gain=np.sqrt(0.5))           # [1, 32, 1024, 1024] => [1, 64, 512, 512]
        b1024_conv0 = self.b1024_conv0(from_rgb)                            # [1, 32, 1024, 1024] => [1, 32, 1024, 1024]
        b1024_conv1 = self.b1024_conv1(b1024_conv0, gain=np.sqrt(0.5))      # [1, 32, 1024, 1024] => [1, 64, 512, 512]
        b1024_conv1 = b1024_skip.add_(b1024_conv1)                          # [1, 64, 512, 512]

        # Second block / DiscriminatorBlock
        b512_skip = self.b512_skip(b1024_conv1, gain=np.sqrt(0.5))          # [1, 64, 512, 512] => [1, 128, 256, 256]
        b512_conv0 = self.b512_conv0(b1024_conv1)                           # [1, 64, 512, 512] => [1, 64, 512, 512]
        b512_conv1 = self.b512_conv1(b512_conv0, gain=np.sqrt(0.5))         # [1, 64, 512, 512] => [1, 128, 256, 256]
        b512_conv1 = b512_skip.add_(b512_conv1)                             # [1, 128, 256, 256]

        # Third block / DiscriminatorBlock
        b256_skip = self.b256_skip(b512_conv1, gain=np.sqrt(0.5))           # [1, 128, 256, 256] => [1, 256, 128, 128]
        b256_conv0 = self.b256_conv0(b512_conv1)                            # [1, 128, 256, 256] => [1, 128, 256, 256]
        b256_conv1 = self.b256_conv1(b256_conv0, gain=np.sqrt(0.5))         # [1, 128, 256, 256] => [1, 256, 128, 128]
        b256_conv1 = b256_skip.add_(b256_conv1)                             # [1, 256, 128, 128]

        # Fourth block / DiscriminatorBlock
        b128_skip = self.b128_skip(b256_conv1, gain=np.sqrt(0.5))           # [1, 256, 128, 128] => [1, 512, 64 ,64]
        b128_conv0 = self.b128_conv0(b256_conv1)                            # [1, 256, 128, 128] => [1, 256, 128, 128]
        b128_conv1 = self.b128_conv1(b128_conv0, gain=np.sqrt(0.5))         # [1, 256, 128, 128] => [1, 512, 64, 64]
        b128_conv1 = b128_skip.add_(b128_conv1)                             # [1, 512, 64, 64]

        # Fifth block / DiscriminatorBlock
        b64_skip = self.b64_skip(b128_conv1, gain=np.sqrt(0.5))             # [1, 512, 64, 64] => [1, 512, 32, 32]
        b64_conv0 = self.b64_conv0(b128_conv1)                              # [1, 512, 64, 64] => [1, 512, 64, 64]
        b64_conv1 = self.b64_conv1(b64_conv0, gain=np.sqrt(0.5))            # [1, 512, 64, 64] => [1, 512, 32, 32]
        b64_conv1 = b64_skip.add_(b64_conv1)                                # [1, 512, 32, 32]

        # Sixth block / DiscriminatorBlock
        b32_skip = self.b32_skip(b64_conv1, gain=np.sqrt(0.5))              # [1, 512, 32, 32] => [1, 512, 16, 16]
        b32_conv0 = self.b32_conv0(b64_conv1)                               # [1, 512, 32, 32] => [1, 512, 32, 32]
        b32_conv1 = self.b32_conv1(b32_conv0, gain=np.sqrt(0.5))            # [1, 512, 32, 32] => [1, 512, 16, 16]
        b32_conv1 = b32_skip.add_(b32_conv1)                                # [1, 512, 16, 16]

        # Seventh block / DiscriminatorBlock
        b16_skip = self.b16_skip(b32_conv1, gain=np.sqrt(0.5))              # [1, 512, 16, 16] => [1, 512, 8, 8]
        b16_conv0 = self.b16_conv0(b32_conv1)                               # [1, 512, 16, 16] => [1, 512, 16, 16]
        b16_conv1 = self.b16_conv1(b16_conv0, gain=np.sqrt(0.5))            # [1, 512, 16, 16] => [1, 512, 8, 8]
        b16_conv1 = b16_skip.add_(b16_conv1)                                # [1, 512, 8, 8]

        # Eighth block / DiscriminatorBlock
        b8_skip = self.b8_skip(b16_conv1, gain=np.sqrt(0.5))                # [1, 512, 8, 8] => [1, 512, 4, 4]
        b8_conv0 = self.b8_conv0(b16_conv1)                                 # [1, 512, 8, 8] => [1, 512, 8, 8]
        b8_conv1 = self.b8_conv1(b8_conv0, gain=np.sqrt(0.5))               # [1, 512, 8, 8] => [1, 512, 4, 4]
        b8_conv1 = b8_skip.add_(b8_conv1)                                   # [1, 512, 4, 4]

        # Ninth block / DiscriminatorEpilogue
        b4_mbstd = self.b4_mbstd(b8_conv1)                                  # [1, 512, 4, 4] ~> [1, 513, 4, 4]
        b4_conv = self.adavgpool(self.b4_conv(b4_mbstd))                    # [1, 513, 4, 4] => [1, 512, 4, 4] +> [1, 512, 4, 4]
        fc = self.fc(b4_conv.flatten(1))                                    # [1, 512, 4, 4] -> [1, 8192] ->> [1, 512]
        out = self.out(fc)                                                  # [1, 512] ->> [1, 1]
        """
        assert not (normed and sqrt_normed), 'Choose one of the normalizations!'

        # Return the full output if no layers are indicated
        if layers is None:
            layers = ['out']

        features_dict = OrderedDict()  # Can just be a dictionary, but I plan to use the order of the features later on
        features_dict['from_rgb'] = getattr(self, 'from_rgb')(x)    # [1, 3, D.img_resolution, D.img_resolution] =>
        #                                                                => [1, 32, D.img_resolution, D.img_resolution]

        for idx, res in enumerate(self.block_resolutions):

            # conv0 and skip from the first block use from_rgb
            if idx == 0:
                features_dict[f'b{res}_skip'] = getattr(self, f'b{res}_skip')(
                    features_dict['from_rgb'], gain=np.sqrt(0.5))
                features_dict[f'b{res}_conv0'] = getattr(self, f'b{res}_conv0')(features_dict['from_rgb'])

            # The rest use the previous block's conv1
            else:
                features_dict[f'b{res}_skip'] = getattr(self, f'b{res}_skip')(
                    features_dict[f'b{self.block_resolutions[idx - 1]}_conv1'], gain=np.sqrt(0.5)
                )
                features_dict[f'b{res}_conv0'] = getattr(self, f'b{res}_conv0')(
                    features_dict[f'b{self.block_resolutions[idx - 1]}_conv1']
                )
            # Finally, pass the current block's conv0 and do the skip connection addition
            features_dict[f'b{res}_conv1'] = getattr(self, f'b{res}_conv1')(features_dict[f'b{res}_conv0'],
                                                                            gain=np.sqrt(0.5))
            features_dict[f'b{res}_conv1'] = features_dict[f'b{res}_skip'].add_(features_dict[f'b{res}_conv1'])

        # Irrespective of the image size/model size, the last block will be the same:
        features_dict['b4_mbstd'] = getattr(self, 'b4_mbstd')(features_dict['b8_conv1'])  # [1, 512, 4, 4] ~> [1, 513, 4, 4]
        features_dict['b4_conv'] = getattr(self, 'b4_conv')(features_dict['b4_mbstd'])    # [1, 513, 4, 4] => [1, 512, 4, 4]
        features_dict['b4_conv'] = getattr(self, 'adavgpool')(features_dict['b4_conv'])   # [1, 512, 4, 4] +> [1, 512, 4, 4]  (Needed if x's resolution is not D.img_resolution)
        features_dict['fc'] = getattr(self, 'fc')(features_dict['b4_conv'].flatten(1))    # [1, 512, 4, 4] -> [1, 8192] ->> [1, 512]
        features_dict['out'] = getattr(self, 'out')(features_dict['fc'])                  # [1, 512] ->> [1, 1]

        result_list = list()
        for layer in layers:
            if channels is not None:
                max_channels = features_dict[layer].shape[1]  # The number of channels in the layer
                channels = [c for c in channels if c < max_channels]  # Remove channels that are too high
                channels = [c for c in channels if c >= 0]  # Remove channels that are too low
                channels = list(set(channels))  # Remove duplicates
                if layer not in ['fc', 'out']:
                    features_dict[layer] = features_dict[layer][:, channels, :, :]  # [1, max_channels, size, size] => [1, len(channels), size, size]
                else:
                    features_dict[layer] = features_dict[layer][:, channels]  # [1, max_channels] => [1, len(channels)]
            # Two options to normalize, otherwise we only add the unmodified output; recommended if using more than one layer
            if normed:
                result_list.append(features_dict[layer] / torch.numel(features_dict[layer]))
            elif sqrt_normed:
                result_list.append(features_dict[layer] / torch.tensor(torch.numel(features_dict[layer]),
                                                                        dtype=torch.float).sqrt())
            else:
                result_list.append(features_dict[layer])

        return tuple(result_list)
