"""
Adapted from
https://github.com/facebookresearch/ConvNeXt

Copyright (c) Meta Platforms, Inc. and affiliates.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from src.models.cnn.utils import ChannelNorm


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),  # depthwise conv
            ChannelNorm(dim),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.layers(x) + self.drop_path(x)
        return x


class ConvNeXtTrunk(nn.Module):
    r""" ConvNeXt Trunk
    Args:
        in_channels (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(self, in_channels=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.):
        super().__init__()

        assert len(depths) == len(dims)
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            ChannelNorm(dims[0])
        )
        self.downsample_layers.append(stem)
        for i_dim, o_dim in zip(dims[:-1], dims[1:]):
            downsample_layer = nn.Sequential(
                ChannelNorm(i_dim),
                nn.Conv2d(i_dim, o_dim, kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j].item()) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_channels=3, num_classes=1000,
                 depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.):
        super().__init__()
        self.trunk = ConvNeXtTrunk(in_channels, depths, dims, drop_path_rate)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.trunk(x)
        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x
