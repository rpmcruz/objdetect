'''
We implement or wrap some backbones here so that the API is the same (they all
return 3 outputs of different sizes), that way, models can use them
interchangeably.
'''

import torchvision
import torch

# YOLO3 uses Darknet53 as the backbone. This implementation is based on
# https://github.com/developer0hye/PyTorch-Darknet53

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_num),
        torch.nn.LeakyReLU())

def rep_blocks(in_channels, num_blocks):
    layers = []
    for i in range(0, num_blocks):
        layers.append(ResBlock(in_channels))
    return torch.nn.Sequential(*layers)

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, 1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x
        return out

class Darknet53(torch.nn.Module):
    channels = [256, 512, 1024]

    def __init__(self):
        super().__init__()
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.block1 = rep_blocks(64, 1)
        self.p1 = conv_batch(64, 128, stride=2)
        self.block2 = rep_blocks(128, 2)
        self.p2 = conv_batch(128, 256, stride=2)
        self.block3 = rep_blocks(256, 8)
        self.p3 = conv_batch(256, 512, stride=2)
        self.block4 = rep_blocks(512, 8)
        self.p4 = conv_batch(512, 1024, stride=2)

    def forward(self, x):
        p1 = self.p1(self.block1(self.conv2(self.conv1(x))))
        p2 = self.p2(self.block2(p1))
        p3 = self.p3(self.block3(p2))
        p4 = self.p4(self.block4(p3))
        return p2, p3, p4

# Most networks use resnet50 or some other resnet.

class Resnet50(torch.nn.Module):
    channels = [512, 1024, 2048]

    def __init__(self, pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        layers = list(resnet.children())[:-2]
        self.main = torch.nn.Sequential(*layers[:-3])
        self.final = torch.nn.ModuleList(layers[-3:])

    def forward(self, x):
        x = self.main(x)
        out = []
        for l in self.final:
            x = l(x)
            out.append(x)
        return out
