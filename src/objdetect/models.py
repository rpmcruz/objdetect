'''
These are just simple implementations of possible backbones and heads. You might want to use a more complex pre-trained backbone. The models must produce grids that are then contrasted against the true grids.
'''

from torch.nn import functional as F
from torch import nn
import torch
import numpy as np

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.batch_norm(self.conv(x))

class SimpleBackbone(nn.Module):
    '''Very simple backbone. Applies convolutions with stride-2 for how many times as length of the channels list. This is a simple architecture, you may want to use a pre-trained architecture instead.'''
    def __init__(self, channels_levels, use_batchnorm):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = 3
        for next in channels_levels:
            layer = ConvBatchNorm(prev, next, 2, 1) if use_batchnorm else nn.Conv2d(prev, next, 3, 2, 1)
            self.layers.append(layer)
            prev = next

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
            x = F.relu(x)
        return outs

class HeadScores(nn.Module):
    '''Produces a 1xHxW output. If in eval mode, then a sigmoid is applied.'''
    def __init__(self, ninputs):
        super().__init__()
        self.conv = nn.Conv2d(ninputs, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        if not self.training:
            return torch.sigmoid(x)
        return x

class HeadExpBboxes(nn.Module):
    '''Produces a 4xHxW output with exp applied to force the values to be positive.'''
    def __init__(self, ninputs):
        super().__init__()
        self.conv = nn.Conv2d(ninputs, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        return torch.exp(x)

class HeadCenterSizeBboxes(nn.Module):
    '''Produces a 4xHxW output with sigmoids applied to the first two values (that is, the center of the bounding box).'''
    def __init__(self, ninputs):
        super().__init__()
        self.conv = nn.Conv2d(ninputs, 4, 1)

    def forward(self, x):
        bb = self.conv(x)
        bb[:, 0] = torch.sigmoid(bb[:, 0])
        bb[:, 1] = torch.sigmoid(bb[:, 1])
        return bb

class HeadClasses(nn.Module):
    '''Produces a KxHxW, where K is the number of classes. If in eval mode, then a softmax is applied.'''
    def __init__(self, ninputs, nclasses):
        super().__init__()
        self.conv = nn.Conv2d(ninputs, nclasses, 1)

    def forward(self, x):
        x = self.conv(x)
        if not self.training:
            x = F.softmax(x, 1)
        return x

class Model(nn.Module):
    '''Combines a backbone with a list of dictionaries of heads. The reason why a list is used is because the backbone may produce multiple grids.''' 
    def __init__(self, backbone, multi_heads):
        super().__init__()
        self.backbone = backbone
        self.multi_heads = nn.ModuleList([nn.ModuleDict(heads) for heads in multi_heads])

    def forward(self, x):
        xs = self.backbone(x)
        return {name: h(x) for x, heads in zip(xs, self.multi_heads) for name, h in heads.items()}
