import warnings
from typing import Tuple, Iterable

import numpy as np
import torch
from gym import spaces
import torch.nn as nn
from gym3 import types
from gym3.types import DictType

from src.models.common import ImageEncoder, create_mlp
from src.utils.gym_spaces_utils import OneHotEncoding, is_image_space_channels_first, is_image_space


class TupleEncode(nn.Module):
    def __init__(self, space: spaces.Tuple, conv_config):
        super().__init__()
        self.nets = nn.ModuleList()
        self.dims = []
        for s in space:
            m, d = create_encoder(s, conv_config)
            self.nets.append(m)
            self.dims.append(d)

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=-1)


class DictEncode(nn.Module):
    def __init__(self, space: spaces.Dict, conv_config):
        super().__init__()
        self.nets = nn.ModuleDict()
        self.dims = []
        for k in space:
            m, d = create_encoder(space[k], conv_config)
            self.nets[k] = m
            self.dims.append(d)

    def forward(self, x):
        return torch.cat([net(x[k]) for k, net in self.nets.items()], dim=-1)


def create_encoder(
        space: spaces.Space,
        conv_config: dict = None,
) -> Tuple[nn.Module, int]:
    if isinstance(space, spaces.Box):
        if is_image_space(space):
            assert is_image_space_channels_first(space)
            C, H, W = space.shape
            encoder = ImageEncoder(dims=(C, *conv_config["hidden"]), downsamples=conv_config["scale"], kernel_sizes=conv_config["kernel_sizes"],
                                   dropout=conv_config["dropout"], use_batch_norm=conv_config["use_batch_norm"], input_size=(H, W))
            return encoder, conv_config["hidden"][-1]
        if len(space.shape) > 1:
            warnings.warn(f"Flattening spaces.Box of size {space.shape}")
        return nn.Flatten(), spaces.flatdim(space)

    elif isinstance(space, (OneHotEncoding, spaces.MultiBinary)):
        return nn.Identity(), space.n

    elif isinstance(space, spaces.Tuple):
        model = TupleEncode(space, conv_config)
        return model, sum(model.dims)

    elif isinstance(space, (spaces.Dict, DictType)):
        model = DictEncode(space, conv_config)
        return model, sum(model.dims)

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {space}")


class Encoder(nn.Module):
    def __init__(
            self,
            space: spaces.Space,
            mlp_config: dict = None,
            conv_config: dict = None
    ):
        super().__init__()

        model, dim = create_encoder(space, conv_config)
        self.model = nn.Sequential(
            model,
            create_mlp(sizes=(dim, *mlp_config["hidden"]), activation=nn.ReLU)
        )

        self.output_dim = mlp_config["hidden"][-1]

    def forward(self, x):
        return self.model(x)
