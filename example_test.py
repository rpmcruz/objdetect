import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--imagesize', type=int, default=256)
parser.add_argument('--gridsize', type=int, default=8)
parser.add_argument('--batchsize', type=int, default=32)
args = parser.parse_args()

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import albumentations as A
import objdetect as od

d = torch.load(args.model)
model = d['model']
anchors = d['anchors']

# you may use albumentations or your own function
transforms = A.Compose([
    A.Resize(args.imagesize, args.imagesize),
    A.Normalize(0, 1),
], bbox_params=A.BboxParams('albumentations', ['classes']))

grid_transform = lambda datum: od.grid.bboxes_to_grids(datum, (args.gridsize, args.gridsize), anchors)
ts = od.datasets.VOCDetection('data', 'val', False, transforms, grid_transform)
labels = od.datasets.VOCDetection.labels
ts = DataLoader(ts, args.batchsize)

inv_grid_transform = lambda x: od.grid.batch_grids_to_bboxes(x, anchors)
inputs, preds = od.loop.evaluate(model, ts, inv_grid_transform)

import matplotlib.pyplot as plt
for input, pred in zip(inputs, preds):
    plt.imshow(input['image'])
    od.plot.bboxes_with_classes(input['bboxes'], input['classes'], (args.imagesize, args.imagesize), labels, 'g', '--')
    od.plot.bboxes_with_classes(pred['bboxes'], pred['classes'], (args.imagesize, args.imagesize), labels, 'r', '-')
    plt.show()
