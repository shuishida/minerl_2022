import warnings

import numpy as np
from gym import spaces
import torch.nn as nn

from src.models.common import create_mlp, ImageDecoder
from src.utils.gym_spaces_utils import OneHotEncoding, is_image_space_channels_first, is_image_space
from src.utils.nested_utils import nested_rearrange, nested_einsum


class MLPReshape(nn.Module):
    def __init__(self, mlp_config, dim_in, output_shape):
        super(MLPReshape, self).__init__()
        self.mlp = create_mlp(sizes=(dim_in, *mlp_config["hidden"], np.prod(output_shape)), activation=nn.ReLU)
        self.output_shape = output_shape

    def forward(self, x):
        return self.mlp(x).view(-1, *self.output_shape)


class TupleDecode(nn.Module):
    def __init__(self, space: spaces.Tuple, *args, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList([create_decoder(s, *args, **kwargs) for s in space])

    def forward(self, x):
        return (m(x) for m in self.nets)


class DictDecode(nn.Module):
    def __init__(self, space: spaces.Dict, *args, **kwargs):
        super().__init__()
        self.nets = nn.ModuleDict({k: create_decoder(space[k], *args, **kwargs) for k in space})

    def forward(self, x):
        return {k: net(x) for k, net in self.nets.items()}


def create_decoder(
        space: spaces.Space,
        dim_in: int,
        mlp_config: dict = None,
        deconv_config: dict = None,
        n_options: int = 1,
) -> nn.Module:
    if isinstance(space, spaces.Box):
        if is_image_space(space):
            assert is_image_space_channels_first(space)
            C, H, W = space.shape
            decoder = ImageDecoder(dims=(dim_in, *deconv_config["hidden"], C * n_options), upsamples=(*deconv_config["scale"], 1),
                                   kernel_sizes=(*deconv_config["kernel_sizes"], 1), dropout=deconv_config["dropout"],
                                   use_batch_norm=deconv_config["use_batch_norm"], target_size=(H, W))
            return decoder
        if len(space.shape) > 1:
            warnings.warn(f"Flattening spaces.Box of size {space.shape}")
        return MLPReshape(mlp_config, dim_in, (space.shape[0] * n_options, *space.shape[1:]))

    elif isinstance(space, (OneHotEncoding, spaces.MultiBinary)):
        return create_mlp(sizes=(dim_in, *mlp_config["hidden"], space.n * n_options))

    elif isinstance(space, spaces.Tuple):
        return TupleDecode(space, dim_in, mlp_config=mlp_config, deconv_config=deconv_config, n_options=n_options)

    elif isinstance(space, spaces.Dict):
        return DictDecode(space, dim_in, mlp_config=mlp_config, deconv_config=deconv_config, n_options=n_options)

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {space}")


class Decoder(nn.Module):
    def __init__(
            self,
            space: spaces.Space,
            dim_in: int,
            mlp_config: dict = None,
            deconv_config: dict = None,
            n_options: int = 1
    ):
        super().__init__()
        self.n_options = n_options
        self.model = create_decoder(space, dim_in, mlp_config, deconv_config, n_options)

    def forward(self, x, options=None):
        x = self.model(x)
        if options is not None:
            x = nested_rearrange(x, "b (o c) ... -> b o c ...", o=self.n_options)
            x = nested_einsum("b o ..., b o -> b ...", x, options)
        return x

