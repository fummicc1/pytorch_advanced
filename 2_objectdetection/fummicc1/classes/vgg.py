import torch.nn as nn


def make_vgg() -> nn.ModuleList:
    layers = []
    in_channels = 3  # 色チャネル数

    config = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "MC",
        512,
        512,
        512,
        "M",
        512,
        512,
        512
    ]

    for v in config:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5,
        conv6,
        nn.ReLU(inplace=True),
        conv7,
        nn.ReLU(inplace=True)
    ]
    return nn.ModuleList(layers)
