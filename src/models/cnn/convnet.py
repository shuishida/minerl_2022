import torch.nn as nn

from src.models.cnn.head import CNNFlatHead, CNNPoolHead


class ConvNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0., downsample=False, use_batch_norm=True):
        super(ConvNetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity(),
            nn.BatchNorm2d(dim_out) if use_batch_norm else nn.Identity()
        )

        # Alternative implementation similar to ClusterNet by Xu Ji
        # self.layers = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity(),
            # nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2, bias=False),
            # nn.BatchNorm2d(dim_out) if use_batch_norm else nn.Identity(),
            # nn.ReLU()
        # )

    def forward(self, x):
        return self.layers(x)


class ConvNetTrunk(nn.Module):
    def __init__(self, in_channels, dims=(64, 128, 256, 512), downsamples=(True, True, True, False),
                 dropout=0.0, use_batch_norm=True):
        super(ConvNetTrunk, self).__init__()
        dims = (in_channels,) + dims
        self.layers = nn.Sequential(
            *(ConvNetBlock(dim_in, dim_out, dropout, downsample, use_batch_norm)
              for dim_in, dim_out, downsample in zip(dims[:-1], dims[1:], downsamples))
        )
        self.dims = dims

    def forward(self, x):
        return self.layers(x)


class ConvFlatNet(nn.Module):
    def __init__(self, input_shape, output_dim, head_layer_norm=False, use_softmax=False, **kwargs):
        super(ConvFlatNet, self).__init__()
        trunk = ConvNetTrunk(input_shape[0], **kwargs)
        self.net = CNNFlatHead(trunk, input_shape, output_dim, layer_norm=head_layer_norm, use_softmax=use_softmax)

    def forward(self, x):
        return self.net(x)


class ConvPoolNet(nn.Module):
    def __init__(self, input_dim, output_dim, head_layer_norm=False, use_softmax=False, **kwargs):
        super(ConvPoolNet, self).__init__()
        trunk = ConvNetTrunk(input_dim, **kwargs)
        self.net = CNNPoolHead(trunk, trunk.dims[-1], output_dim, layer_norm=head_layer_norm, use_softmax=use_softmax)

    def forward(self, x):
        return self.net(x)
