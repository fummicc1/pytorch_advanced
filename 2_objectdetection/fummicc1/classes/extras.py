import torch.nn as nn


def make_extras() -> nn.ModuleList:
    layers = []
    in_channels = 1024

    config = [
        256,
        512,
        128,
        256,
        128,
        256,
        128,
        256
    ]
    layers += [
        nn.Conv2d(in_channels, config[0], kernel_size=1),
        nn.Conv2d(config[0], config[1], kernel_size=3, stride=2, padding=1),
        nn.Conv2d(config[1], config[2], kernel_size=1),
        nn.Conv2d(config[2], config[3], kernel_size=3, stride=2, padding=1),
        nn.Conv2d(config[3], config[4], kernel_size=1),
        nn.Conv2d(config[4], config[5], kernel_size=3),
        nn.Conv2d(config[5], config[6], kernel_size=1),
        nn.Conv2d(config[6], config[7], kernel_size=3),
    ]
    return nn.ModuleList(layers)
