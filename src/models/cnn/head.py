import torch
import torch.nn as nn


class CNNPoolHead(nn.Module):
    def __init__(self, trunk, input_dim, output_dim, *, layer_norm=False, use_softmax=False):
        super().__init__()
        self.trunk = trunk

        self.head = nn.Sequential(
            nn.LayerNorm(input_dim, eps=1e-6) if layer_norm else nn.Identity(),
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=-1) if use_softmax else nn.Identity()
        )

    def forward(self, x):
        x = self.trunk(x)
        x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x


class CNNFlatHead(nn.Module):
    def __init__(self, trunk, input_shape, output_dim, *, layer_norm=False, use_softmax=False):
        super().__init__()
        self.trunk = trunk

        with torch.no_grad():
            dim = trunk(torch.zeros(2, *input_shape)).view(2, -1).shape[-1]

        self.head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6) if layer_norm else nn.Identity(),
            nn.Linear(dim, output_dim),
            nn.Softmax(dim=-1) if use_softmax else nn.Identity()
        )

    def forward(self, x):
        x = self.trunk(x)
        x = x.view(x.size(0), -1)  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x
