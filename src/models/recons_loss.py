import warnings
from typing import Tuple, Iterable, Callable, Any

import numpy as np
import torch
from gym import spaces
import torch.nn as nn
from torch import Tensor

from src.utils.gym_spaces_utils import OneHotEncoding


class GaussianReconsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return -log_pxz.sum() / len(x)


class TupleLoss(nn.Module):
    def __init__(self, space: spaces.Tuple):
        super().__init__()
        self.nets = nn.ModuleList([get_recons_loss(s) for s in space])

    def forward(self, x_hat, x):
        return (m(x_hat, x) for m in self.nets)


class DictLoss(nn.Module):
    def __init__(self, space: spaces.Dict):
        super().__init__()
        self.nets = nn.ModuleDict({k: get_recons_loss(space[k]) for k in space})

    def forward(self, x_hat, x):
        return {k: net(x_hat[k], x[k]) for k, net in self.nets.items()}


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target.argmax(dim=-1))


def get_recons_loss(
        space: spaces.Space
) -> nn.Module:
    if isinstance(space, spaces.Box):
        return nn.MSELoss()
    elif isinstance(space, OneHotEncoding):
        return CrossEntropyLoss()
    elif isinstance(space, spaces.MultiBinary):
        return nn.BCELoss()
    elif isinstance(space, spaces.Tuple):
        return TupleLoss(space)
    elif isinstance(space, spaces.Dict):
        return DictLoss(space)
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {space}")
