import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--download', action='store_true')
parser.add_argument('--nanchors', type=int, default=9)
parser.add_argument('--imagesize', type=int, default=256)
parser.add_argument('--gridsize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=32)
args = parser.parse_args()

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import albumentations as A
import objdetect as od

# you may use albumentations or your own function
transforms = A.Compose([
    A.Resize(int(args.imagesize[0]*1.1), int(args.imagesize[1]*1.1)),
    A.RandomCrop(args.imagesize[0], args.imagesize[1]),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(0, 1),
], bbox_params=A.BboxParams('albumentations', ['classes']))

# first, find anchors without augmentation
tr = od.datasets.VOCDetection('data', 'train', args.download, None, None)
anchors = od.anchors.compute_clusters(tr, args.nanchors)
# then re-load dataset, now with the anchors
grid_transform = lambda datum: od.grid.bboxes_to_grids(datum, (args.gridsize, args.gridsize), anchors)
tr = od.datasets.VOCDetection('data', 'train', False, transforms, grid_transform)
labels = od.datasets.VOCDetection.labels

# create model
backbone = od.models.Backbone((args.imagesize, args.imagesize, 3), (args.gridsize, args.gridsize))
head = od.models.HeadWithClasses(backbone.n_outputs, args.nanchors, len(labels))
model = od.models.Model(backbone, head).cuda()

outputs = model(torch.rand(10, 3, args.imagesize[1], args.imagesize[0], device='cuda'))
print(summary(model, (10, 3, args.imagesize[1], args.imagesize[0])))
print('outputs:', [f'{k}: {v.shape}' for k, v in outputs.items()])

# train
tr = DataLoader(tr, args.batchsize, True)
opt = torch.optim.Adam(model.parameters())
losses = {
    'confs_grid': nn.BCEWithLogitsLoss(),
    'bboxes_grid': nn.MSELoss(),
    'classes_grid': nn.CrossEntropyLoss(),
}
od.loop.train(model, tr, opt, losses, args.epochs)

# save things
torch.save({'model': model, 'anchors': anchors}, 'model.pth')
