import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

'''
These are just simple implementations of possible backbones and heads. You might
want to use a more complex pre-trained backbone.

The neural network must produce a grid with, at least, the confidence of there
being an object at that cell (confs_grid) and the respective bounding box
(bboxes_grid) in the form of a dictionary.

The output shapes must be (grid_height, grid_width, n_anchors, n_something)
where n_something is what you want to predict. If you do not use anchors, just
specify n_anchors=1.

In the default models, if the model is in train-mode, then we do not apply any
sigmoid or softmax activation function, because logits are more numerical
stable.
'''

class Backbone(nn.Module):
    def __init__(self, img_size, grid_size):
        super().__init__()
        assert img_size[0] % grid_size[0] == 0, f'Image size ({img_size[0]}) must be multiple of grid size ({grid_size[0]})'
        assert img_size[0] // grid_size[0] == img_size[1] // grid_size[1], f'Proportion between image size ({img_size[0]}x{img_size[1]}) must be the same relative to grid size ({grid_size[0]}x{grid_size[1]})'
        n = int(np.log2(img_size[0] // grid_size[0]))
        self.layers = nn.ModuleList()
        prev, next = img_size[2], 32
        for _ in range(n):
            self.layers.append(nn.Conv2d(prev, next, 3, 2, 1))
            prev, next = next, next*2
        self.n_outputs = prev

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class Head(nn.Module):
    def __init__(self, n_inputs, n_anchors):
        super().__init__()
        assert n_anchors >= 1, 'n_anchors ({n_anchors}) must be at least 1'
        self.has_conv = nn.Conv2d(n_inputs, n_anchors*1, 1)
        self.bbox_conv = nn.Conv2d(n_inputs, n_anchors*4, 1)
        self.n_anchors = n_anchors

    def forward(self, x):
        h = self.has_conv(x)
        h = h.view(h.shape[0], 1, self.n_anchors, h.shape[2], h.shape[3])
        if not self.training:
            h = torch.sigmoid(h)
        b = self.bbox_conv(x)
        b = b.view(b.shape[0], 4, self.n_anchors, b.shape[2], b.shape[3])
        b[:, 0] = torch.sigmoid(b[:, 0])
        b[:, 1] = torch.sigmoid(b[:, 1])
        return {'confs_grid': h, 'bboxes_grid': b}

class HeadWithClasses(Head):
    def __init__(self, n_inputs, n_anchors, n_classes):
        super().__init__(n_inputs, n_anchors)
        self.classes_conv = nn.Conv2d(n_inputs, n_anchors*n_classes, 1)
        self.n_classes = n_classes

    def forward(self, x):
        outputs = super().forward(x)
        y = self.classes_conv(x)
        y = y.view(y.shape[0], self.n_classes, self.n_anchors, y.shape[2], y.shape[3])
        if not self.training:
            y = F.softmax(y, 2)
        outputs['classes_grid'] = y
        return outputs

class Model(nn.Module):  # same as before
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
