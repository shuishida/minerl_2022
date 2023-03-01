import torch
import torch.nn as nn


class ChannelNorm(nn.Module):
    def __init__(self, n_channels, eps=1e-3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, n_channels, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, n_channels, 1, 1)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = x.std(1, keepdim=True)
        x = (x - u) / (s + self.eps)
        x = self.weight * x + self.bias
        return x
