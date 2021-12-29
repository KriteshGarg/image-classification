import torch
from torch import nn
from torch import Tensor

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def conv_with_padding(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False,
                     dilation=dilation)


class Shortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(Shortcut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        return out


class ResnetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, identity_net: nn.Module = None):
        super(ResnetBasicBlock, self).__init__()
        self.conv_1 = conv_with_padding(in_channels, out_channels, stride=stride)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv_with_padding(out_channels, out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.shortcut = identity_net
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()

        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer_1_basic_block_1 = ResnetBasicBlock(in_channels=32, out_channels=32)
        self.layer_1_basic_block_2 = ResnetBasicBlock(in_channels=32, out_channels=32)

        shortcut_1 = Shortcut(in_channels=32, out_channels=64, stride=2)
        self.layer_2_basic_block_1 = ResnetBasicBlock(in_channels=32, out_channels=64, stride=2,
                                                      identity_net=shortcut_1)
        self.layer_2_basis_block_2 = ResnetBasicBlock(in_channels=64, out_channels=64)

        shortcut_2 = Shortcut(in_channels=64, out_channels=128, stride=2)
        self.layer_3_basic_block_1 = ResnetBasicBlock(in_channels=64, out_channels=128, stride=2,
                                                      identity_net=shortcut_2)
        self.layer_3_basis_block_2 = ResnetBasicBlock(in_channels=128, out_channels=128)

        shortcut_3 = Shortcut(in_channels=128, out_channels=256, stride=2)
        self.layer_4_basic_block_1 = ResnetBasicBlock(in_channels=128, out_channels=256, stride=2,
                                                      identity_net=shortcut_3)
        self.layer_4_basis_block_2 = ResnetBasicBlock(in_channels=256, out_channels=256)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        logger.info(f"Initiating Model : {type(self).__name__}")

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer_1_basic_block_1(out)
        out = self.layer_1_basic_block_2(out)
        out = self.layer_2_basic_block_1(out)
        out = self.layer_2_basis_block_2(out)
        out = self.layer_3_basic_block_1(out)
        out = self.layer_3_basis_block_2(out)
        out = self.layer_4_basic_block_1(out)
        out = self.layer_4_basis_block_2(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    resnet = Resnet18(num_classes=10)
    pytorch_total_params = sum(p.numel() for p in resnet.parameters())
    print(pytorch_total_params)
    print(resnet)
