from typing import List

import torch
from einops import rearrange
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F, Parameter

from src.models.common import create_mlp
from src.models.decoders import Decoder
from src.utils.nested_utils import nested_slice, nested_reduce, tree_batch_collate


def bernoulli(logit):
    probs = torch.sigmoid(logit)
    sample = torch.bernoulli(probs)
    return sample.detach() + probs - probs.detach(), probs


def binary(logit):
    probs = torch.sigmoid(logit)
    sample = probs > 0.5
    return sample + probs - probs.detach(), probs


def binned(x, resolution: int):
    samples = torch.floor((x + 1) / 2 * resolution + 0.5) * 2 / resolution - 1
    return samples.detach() + x - x.detach()


def binned_tanh(x, resolution: int):
    x = torch.tanh(x)
    return binned(x, resolution)


def tanh_bernoulli(logit, temp=1):
    probs = torch.tanh(logit)
    samples = torch.bernoulli(probs / 2 + 0.5) * 2 - 1
    return samples.detach() + (probs - probs.detach()) * temp, probs


def multinomial(logit):
    probs = torch.softmax(logit, dim=-1)
    samples = torch.multinomial(probs, 1)
    return samples.detach() + probs - probs.detach(), probs


def state_activation(enc, name="binned_tanh", resolution=None, **kwargs):
    if name == "binned_tanh":
        return binned_tanh(enc, resolution=resolution), enc
    elif name == "tanh":
        return torch.tanh(enc), enc
    elif name == "binary":
        return binary(enc)
    elif name == "bernoulli":
        return bernoulli(enc)
    elif name == "tanh_bernoulli":
        return tanh_bernoulli(enc)
    elif name == "identity":
        return enc, enc
    else:
        raise NotImplementedError


def lookahead(state_like_items: List[torch.Tensor], action_like_items: List[torch.Tensor], rewards, firsts, dones, max_steps=None):
    if isinstance(action_like_items, torch.Tensor):
        action_like_items = [action_like_items]
    if not max_steps: max_steps = len(firsts) - 1
    _actions, _rewards, _firsts, _dones = [], [], [], []
    _state_like_items = [([], []) for _ in range(len(state_like_items))]
    _action_like_items = [[] for _ in range(len(action_like_items))]
    rewards = rearrange(rewards[1:], "l b -> b () l")
    for k in range(1, max_steps + 1):
        _valid = (~firsts[k:]) & (~dones[:-k])
        for i, action_like in enumerate(action_like_items):
            _action_like_items[i].append(nested_slice(nested_slice(action_like, slice(0, -k)), _valid))
        _r = F.conv1d(rewards, torch.ones((1, 1, k), dtype=torch.float, device=rewards.device))
        _rewards.append(rearrange(_r, "b () l -> l b")[_valid])
        _firsts.append(firsts[:-k][_valid])
        _dones.append(dones[k:][_valid])
        for i, state_like in enumerate(state_like_items):
            _state_like_items[i][0].append(state_like[:-k][_valid])
            _state_like_items[i][1].append(state_like[k:][_valid])
    _state_like_items = [(torch.cat(_curr), torch.cat(_next)) for (_curr, _next) in _state_like_items]
    _action_like_items = [tree_batch_collate(_action_like, "cat") for _action_like in _action_like_items]
    return _state_like_items, _action_like_items, torch.cat(_rewards), torch.cat(_firsts), torch.cat(_dones)


class Reward(nn.Module):
    def __init__(self,
                 state_dim: int,
                 mlp_config: dict
                 ):
        super().__init__()
        self.mlp = create_mlp(sizes=(state_dim * 3, *mlp_config["hidden"], 1), activation=nn.ReLU)

    def forward(self, states, next_states):
        x = torch.cat([states, next_states, next_states - states], dim=-1)
        x = self.mlp(x)
        return x


class ActionDecoder(nn.Module):
    def __init__(self,
                 action_space: spaces.Space,
                 dim_in: int,
                 mlp_config: dict
                 ):
        super().__init__()
        self.decoder = Decoder(action_space, dim_in * 3, mlp_config)

    def forward(self, state, subgoal_state):
        x = torch.cat([state, subgoal_state, subgoal_state - state], dim=-1)
        x = self.decoder(x)
        return x


def tanh_similarity(input, target, eps=1e-6, reduction="mean"):
    input = input / (1 + eps)
    target = target / (1 + eps)
    return F.l1_loss(torch.arctanh(input), torch.arctanh(target), reduction=reduction)
