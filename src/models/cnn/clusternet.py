import torch.nn as nn


class ClusterNet6cTrunk(nn.Module):
    def __init__(self, in_channels=1):
        super(ClusterNet6cTrunk, self).__init__()

        self.batchnorm_track = False

        self.conv_size = 5
        self.pad = 2
        self.in_channels = in_channels

        self.cfg = [(64, 1), ('M', None), (128, 1), ('M', None),
                    (256, 1), ('M', None), (512, 1)]

        # self.features = self._make_layers()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return x
    #
    # def _make_layers(self, batch_norm=True):
    #     layers = []
    #     in_channels = self.in_channels
    #     for tup in self.cfg:
    #         assert (len(tup) == 2)
    #
    #         out, dilation = tup
    #         sz = self.conv_size
    #         stride = 1
    #         pad = self.pad  # to avoid shrinking
    #
    #         if out == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         elif out == 'A':
    #             layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
    #         else:
    #             conv2d = nn.Conv2d(in_channels, out, kernel_size=sz,
    #                                stride=stride, padding=pad,
    #                                dilation=dilation, bias=False)
    #             print(batch_norm)
    #             if batch_norm:
    #                 layers += [conv2d, nn.BatchNorm2d(out,
    #                                                   track_running_stats=False),
    #                            nn.ReLU(inplace=True)]
    #             else:
    #                 layers += [conv2d, nn.ReLU(inplace=True)]
    #             in_channels = out
    #
    #     return nn.Sequential(*layers)


class ClusterNet6cHead(nn.Module):
    def __init__(self, n_clusters):
        super(ClusterNet6cHead, self).__init__()

        num_features = 512
        features_sp_size = 3

        self.head = nn.Sequential(
            nn.Linear(num_features * features_sp_size * features_sp_size, n_clusters),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.head(x)


class ClusterNet6c(nn.Module):
    def __init__(self, n_clusters, in_channels=1):
        super(ClusterNet6c, self).__init__()

        self.trunk = ClusterNet6cTrunk(in_channels)
        self.head = ClusterNet6cHead(n_clusters)

    def forward(self, x, kmeans_use_features=False):
        x = self.trunk(x)
        x = self.head(x)
        return x
