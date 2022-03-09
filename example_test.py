from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import albumentations as A
import objdetect as od

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256, 3)
GRID_SIZE = (8, 8)

d = torch.load('model.pth')
model = d['model']
anchors = d['anchors']

# you may use albumentations or your own function
transforms = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(0, 1),
], bbox_params=A.BboxParams('albumentations', ['classes']))

grid_transform = lambda datum: od.grid.bboxes_to_grids(datum, GRID_SIZE, anchors)
ts = od.datasets.VOCDetection('data', 'val', False, transforms, grid_transform)
labels = od.datasets.VOCDetection.labels
ts = DataLoader(ts, BATCH_SIZE)

inv_grid_transform = lambda x: od.grid.batch_grids_to_bboxes(x, anchors)
inputs, preds = od.loop.evaluate(model, ts, inv_grid_transform)

import matplotlib.pyplot as plt
for input, pred in zip(inputs, preds):
    plt.imshow(input['image'])
    od.plot.bboxes_with_classes(input['bboxes'], input['classes'], IMAGE_SIZE, labels, 'g', '--')
    od.plot.bboxes_with_classes(pred['bboxes'], pred['classes'], IMAGE_SIZE, labels, 'r', '-')
    plt.show()
