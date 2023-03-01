from copy import deepcopy
from typing import Callable, Any, Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gym import spaces

import numpy as np
from openai_vpt.agent import MineRLAgent
from openai_vpt.lib.util import FanInInitReLULayer
from src.models.common import create_mlp
from src.models.decoders import Decoder
from src.models.encoders import Encoder
from src.models.recons_loss import get_recons_loss
from src.algo.utils import Reward, state_activation, ActionDecoder, lookahead, tanh_similarity
from src.utils.model_utils import gumbel_softmax
from src.utils.nested_utils import nested_rearrange, nested_sum, inverse_collate, nested_slice, nested_einsum, nested_shape, \
    nested_lambda


def kl_div_log(logits_p, logits_q):
    return (torch.exp(logits_q) * (logits_q - logits_p)).sum(dim=-1).mean()


def entropy(logits):
    return - (logits * logits.exp()).sum(dim=-1).mean()


def reweight(discrete, minlength):
    weights = torch.bincount(discrete, minlength=minlength)
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    # weights /= weights.sum()  # normalize
    weights = weights[discrete]
    return weights


def binary_reweight(binary):
    weights = torch.where(binary, torch.ones_like(binary) / (binary.sum() or np.inf), torch.ones_like(binary) / ((~binary).sum() or np.inf))
    return weights #/ weights.sum()


class BCModel(nn.Module):
    def __init__(self,
                 agent: MineRLAgent,
                 action_space,
                 action_inv_tf,
                 model_config: dict = None,
                 agent_config: dict = None,
                 verbose: bool = False
                 ):
        super().__init__()
        self.verbose = verbose
        self.model_config = model_config
        self.gamma = model_config["gamma"]
        self.agent = agent

        self.action_space = action_space

        self.state_dim = model_config["latent"]["state_dim"]
        self.weights = model_config["weights"]

        self.action_sampler = action_inv_tf

        # self.policy = create_mlp(sizes=(self.state_dim, *model_config["trans_mlp"]["hidden"], self.n_options), activation=nn.ReLU)

        self.action_recons_loss = get_recons_loss(action_space)

        # self.value_head = self.agent.policy.make_value_head(self.agent.policy.state_dim)
        self.pi_head = self.agent.policy.make_action_head(self.agent.policy.state_dim, **self.agent.policy.pi_head_kwargs)
        self.pi_head.load_state_dict(self.agent.policy.pi_head.state_dict())

        hidsize = self.agent.policy.net.hidsize
        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.agent.policy.net.dense_init_norm_kwargs)
        self.lastlayer.load_state_dict(self.agent.policy.net.lastlayer.state_dict())
        self.final_ln = torch.nn.LayerNorm(hidsize)
        self.final_ln.load_state_dict(self.agent.policy.net.final_ln.state_dict())
        for param in self.final_ln.parameters():
            param.requires_grad = False
        self.d_net = create_mlp(sizes=(self.state_dim, 1), activation=nn.ReLU)

    def get_pi(self, states):
        x = self.lastlayer(states)
        x = self.final_ln(x)

        pi_logits = self.pi_head(x)
        # vpred = self.value_head(x)

        return pi_logits

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, states, actions, rewards, firsts, dones):
        pi_logits = self.get_pi(states)
        pi_logits = nested_rearrange(pi_logits, "l b () d -> l b d")

        with torch.no_grad():
            pi_base_logits, vpred_base = self.agent.policy.pi_v(states)
            pi_base_logits = nested_rearrange(pi_base_logits, "l b () d -> l b d")

        ((_states, _next_states),), (_actions, _pi_logits, _pi_base_logits), _rewards, _firsts, _dones \
            = lookahead([states], [actions, pi_logits, pi_base_logits], rewards, firsts, dones, max_steps=1)

        # print(_pi_logits["buttons"][:7].max(dim=-1), _actions["buttons"][:7].argmax(dim=-1))

        dones_loss = F.binary_cross_entropy_with_logits(self.d_net(states).squeeze(-1), dones.float(), reduction="none")

        # print(_pi_logits["buttons"][:5, :7], _pi_base_logits["buttons"][:5, :7], _actions["buttons"][:5].argmax(dim=-1),
        #       _pi_logits["buttons"][:5].max(dim=-1), _pi_base_logits["buttons"][:5, :7].max(dim=-1))

        losses = dict(
            l_nll=nested_sum(nested_lambda(lambda p, a, h: (F.nll_loss(p, a.argmax(dim=-1), reduction="none") * reweight(a.argmax(dim=-1), h.num_actions)).mean(), _pi_logits, _actions, self.pi_head)) * self.weights["nll"],
            l_kl=nested_sum(nested_lambda(kl_div_log, _pi_logits, _pi_base_logits)) * self.weights["kl"],
            l_ent=nested_sum(nested_lambda(entropy, _pi_logits)) * self.weights["ent"],
            l_diverse=nested_sum(nested_lambda(lambda p: - entropy(p.exp().mean(dim=0).log()), _pi_logits)) * self.weights["diverse"],
            l_d=(dones_loss * binary_reweight(dones)).sum() * self.weights["done"],
        )

        return sum(losses.values()), losses

    def sample_actions(self, states, deterministic=False):
        with torch.no_grad():
            actions = self.get_pi(states)
        # print(self.d_net(states).sigmoid())
        return self.action_sampler(actions, deterministic), self.d_net(states).sigmoid().squeeze(0).squeeze(0) > 0.9
