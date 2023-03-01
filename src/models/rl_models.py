import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.common import create_mlp
from src.utils.model_utils import state_activation


class Transition(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_options: int,
                 activation: str,
                 coeff: float = 1.0
                 ):
        super(Transition, self).__init__()
        self.state_dim = state_dim
        self.activation = activation
        self.n_options = n_options
        self.coeff = coeff
        self.mlp = create_mlp(sizes=(self.state_dim, 1024, 1024, self.state_dim * n_options), activation=nn.ReLU)

    def forward(self, states, options, next_states_enc):
        enc = self.mlp(states)
        enc = rearrange(enc, "b (o s) -> b o s", s=self.state_dim)
        next_states_enc = next_states_enc.detach().unsqueeze(1).expand_as(enc)
        recon_loss = F.smooth_l1_loss(enc, next_states_enc, reduction="none")
        return torch.einsum("b o s, b o -> b s", recon_loss, options.detach()).mean() + self.coeff * recon_loss.mean()

    def next(self, states, options_index=None):
        enc = self.mlp(states)
        enc = rearrange(enc, "b (o s) -> b o s", s=self.state_dim)
        if options_index is not None:
            index = torch.arange(len(options_index), device=enc.device)
            enc = enc[index, options_index]
        return state_activation(enc, name=self.activation)
