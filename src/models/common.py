from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange


def create_mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity, dropout=0.):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Dropout(dropout), act()]
    return nn.Sequential(*layers)


def conv_block(dim_in, dim_out, dropout=0., downsample=1, use_batch_norm=True, kernel_size=5,
               activation: Type[nn.Module] = nn.LeakyReLU):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(dim_out) if use_batch_norm else nn.Identity(),
        activation(),
        nn.MaxPool2d(kernel_size=downsample, stride=downsample) if downsample > 1 else nn.Identity(),
        nn.Dropout2d(dropout)
    )


def deconv_block(dim_in, dim_out, dropout=0., upsample=1, use_batch_norm=True, kernel_size=5, output_padding=0,
                 activation: Type[nn.Module] = nn.LeakyReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, output_padding=0, bias=False),
        nn.BatchNorm2d(dim_out) if use_batch_norm else nn.Identity(),
        activation(),
        nn.ConvTranspose2d(dim_out, dim_out, kernel_size=upsample, padding=0, stride=upsample,
                           output_padding=output_padding, bias=False) if upsample > 1 else nn.Identity(),
        nn.Dropout2d(dropout)
    )


def convnet(dims=(3, 64, 128, 256, 512), downsamples=(2, 2, 2, 1), kernel_sizes=(5, 5, 5, 5), dropout=0.0, use_batch_norm=True):
    n_layers = len(downsamples)
    assert len(dims) - 1 == n_layers == len(kernel_sizes)
    is_lasts = [False] * n_layers
    is_lasts[-1] = True
    return nn.Sequential(
        *(conv_block(dim_in, dim_out, dropout=0.0 if is_last else dropout, kernel_size=kernel,
                     downsample=downsample, use_batch_norm=use_batch_norm)
          for dim_in, dim_out, downsample, kernel, is_last in zip(dims[:-1], dims[1:], downsamples, kernel_sizes, is_lasts))
    )


def deconvnet(dims=(512, 256, 128, 64, 3), upsamples=(1, 2, 2, 2), kernel_sizes=(5, 5, 5, 5), dropout=0.0, use_batch_norm=True, output_paddings=(0, 0, 0, 0)):
    n_layers = len(upsamples)
    assert len(dims) - 1 == n_layers == len(kernel_sizes) == len(output_paddings)
    is_lasts = [False] * n_layers
    is_lasts[-1] = True
    return nn.Sequential(
        *(deconv_block(dim_in, dim_out, dropout, upsample=upsample, kernel_size=kernel,
                       use_batch_norm=use_batch_norm, output_padding=out_pad, activation=nn.Identity if is_last else nn.LeakyReLU)
          for dim_in, dim_out, upsample, out_pad, kernel, is_last in zip(dims[:-1], dims[1:], upsamples, output_paddings, kernel_sizes, is_lasts)),
    )


class ImageEncoder(nn.Module):
    def __init__(self, dims=(3, 64, 128, 256, 512), downsamples=(2, 2, 2, 1), kernel_sizes=(5, 5, 5, 5), dropout=0.0, use_batch_norm=True, input_size=None):
        super(ImageEncoder, self).__init__()
        self.convnet = convnet(dims, downsamples, kernel_sizes, dropout, use_batch_norm)
        if input_size:
            with torch.no_grad():
                _, C, H, W = self.convnet(torch.randn((1, dims[0], *input_size))).size()
            self.flatten = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(C * H * W, C) if (H, W) != (1, 1) else nn.Identity()
            )
        else:
            self.flatten = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            )

    def forward(self, x):
        x = self.convnet(x)
        return self.flatten(x)


class ImageDecoder(nn.Module):
    def __init__(self, dims=(512, 256, 128, 64, 3), upsamples=(1, 2, 2, 2), kernel_sizes=(5, 5, 5, 5), dropout=0.0, use_batch_norm=True, target_size=None):
        super(ImageDecoder, self).__init__()
        self.scale = np.prod(upsamples)
        self.target_size = target_size
        if target_size:
            output_paddings = []
            H, W = target_size
            for scale in reversed(upsamples):
                H, h = H // scale, H % scale
                W, w = W // scale, W % scale
                output_paddings.append((h, w))
            output_paddings = list(reversed(output_paddings))
            self.deconv = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=(H, W), stride=(H, W), bias=False)
        else:
            self.deconv = None
            output_paddings = [0] * len(upsamples)
        self.deconvnet = deconvnet(dims, upsamples, kernel_sizes, dropout, use_batch_norm, output_paddings=output_paddings)

    def forward(self, x, target_size=None):
        target_size = target_size or self.target_size
        H, W = (target_size, target_size) if isinstance(target_size, int) else (target_size[-2], target_size[-1])
        if len(x.size()) == 2:
            x = rearrange(x, "b c -> b c () ()")
        if self.deconv:
            x = self.deconv(x)
        else:
            x = x.repeat(1, 1, H // self.scale, W // self.scale)
        x = self.deconvnet(x)
        if (H, W) != (x.size(-2), x.size(-1)):
            if self.target_size:
                raise Exception(
                    f"Target size set but deconvolution produced different size from original: {(H, W)} != {(x.size(-2), x.size(-1))}")
            x = F.interpolate(x, (H, W))
        return x
