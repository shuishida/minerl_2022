import pickle

import torch
import torch.nn as nn

from typing import Iterable, Union, Tuple, List

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def polyak_update(params: Iterable[torch.Tensor], target_params: Iterable[torch.Tensor], tau: float):
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class TrackStats1d:
    def __init__(self, dim, momentum=0.01):
        self.dim = dim
        self.mean = 0
        self.std = 1
        self.momentum = momentum

    def __call__(self, x):
        self.mean = (1 - self.momentum) * self.mean + self.momentum * x.mean(dim=self.dim)


class NoGrad:
    def __init__(self, *models: nn.Module):
        self._models = models
        self.training = []

    def __enter__(self):
        for model in self._models:
            self.training.append(model.training)
            if model.training:
                model.eval()
            model.requires_grad_(requires_grad=False)

    def __exit__(self, *args):
        for model, training in zip(self._models, self.training):
            if training:
                model.train()
            model.requires_grad_(requires_grad=True)


class Rearrange(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Rearrange, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, *self.args, **self.kwargs)


def gumbel_softmax(logits: Tensor, tau: float = 1, eps: float = 1e-10, dim: int = -1) -> Tuple[Tensor, Tensor]:
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, eps=eps, dim=dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, y_soft


def binned(x, resolution: int):
    samples = torch.floor((x + 1) / 2 * resolution + 0.5) * 2 / resolution - 1
    return samples.detach() + x - x.detach()


def binned_tanh(x, resolution: int):
    x = torch.tanh(x)
    return binned(x, resolution)


def state_activation(enc, name="binned_tanh", resolution=16, **kwargs):
    if name == "binned_tanh":
        return binned_tanh(enc, resolution=resolution), enc
    elif name == "tanh":
        return torch.tanh(enc), enc
    elif name == "identity":
        return enc, enc
    else:
        raise NotImplementedError


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs
