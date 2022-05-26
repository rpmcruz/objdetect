'''
This should be seen as a demonstration of how a model such as FCOS *could* be implemented, but this lacks quite a few things to be a perfect duplication.
Something our implementation lacks is, when there is more than one possible object for a given location, choose the one with the minimum area. This could possibly be implemented at the data loader level by ordering objects by descending order of area.
FCOS paper: https://arxiv.org/abs/1904.01355
'''

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 12)
import objdetect as od

NCLASSES = len(od.data.VOCDetection.labels)
GRID_SIZES = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
NLEVELS = len(GRID_SIZES)

######################## GRID & AUG ########################

grid_transforms = [od.grid.Transform(
    grid_size,
    od.grid.SizeFilter(0 if i == 0 else 4, np.inf if i == NLEVELS-1 else 8),
    od.grid.SliceAcrossCeilBbox(),
    {f'hasobjs{i}': od.grid.NewHasObj(), f'bboxes{i}': od.grid.NewBboxes(), f'classes{i}': od.grid.NewClasses(), f'centerness{i}': od.grid.NewCenterness()},
    {f'hasobjs{i}': od.grid.SetHasObj(), f'bboxes{i}': od.grid.SetRelBboxes(), f'classes{i}': od.grid.SetClasses(), f'centerness{i}': od.grid.SetCenterness()},
) for i, grid_size in enumerate(GRID_SIZES)]

transforms = od.aug.Compose(
    od.aug.ResizeAndNormalize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
    od.aug.RandomHflip(),
    *grid_transforms,
    od.grid.RemoveKeys(['bboxes', 'classes'])
)

inv_transforms = od.inv_grid.MultiLevelInvTransform(
    [lambda datum, i=i: datum[f'hasobjs{i}'][0] >= 0.5 for i in range(NLEVELS)],
    {'hasobjs': [f'hasobjs{i}' for i in range(NLEVELS)], 'bboxes': [f'bboxes{i}' for i in range(NLEVELS)], 'classes': [f'classes{i}' for i in range(NLEVELS)]},
    {'hasobjs': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvRelBboxes(), 'classes': od.inv_grid.InvClasses()}
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True)

ts = od.data.VOCDetection('/data', 'val', transforms)
ts = torch.utils.data.DataLoader(ts, 32)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([True]*5, [32, 64, 128, 256, 512], False)
heads = [{
    f'hasobjs{i}': od.models.HeadHasObjs(32*2**i),
    f'bboxes{i}': od.models.HeadExpBboxes(32*2**i),
    f'classes{i}': od.models.HeadClasses(32*2**i, NCLASSES),
    f'centerness{i}': od.models.HeadHasObjs(32*2**i),
} for i in range(NLEVELS)]
model = od.models.Model(backbone, heads)
model = model.cuda()
print(model)
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = od.grid.merge_dicts([{
    f'hasobjs{i}': lambda data: 1,
    f'bboxes{i}': lambda data, i=i: data[f'hasobjs{i}'],
    f'classes{i}': lambda data, i=i: data[f'hasobjs{i}'],
    f'centerness{i}': lambda data, i=i: data[f'hasobjs{i}'],
} for i in range(NLEVELS)])
loss_fns = od.grid.merge_dicts([{
    # we could use sigmoid_focal_loss like RetinaNet
    f'hasobjs{i}': nn.BCEWithLogitsLoss(reduction='none'),
    f'bboxes{i}': od.losses.ConvertRel2Abs(od.losses.GIoU(False)),
    f'classes{i}': nn.CrossEntropyLoss(reduction='none'),
    f'centerness{i}': nn.BCEWithLogitsLoss(reduction='none'),
} for i in range(NLEVELS)])

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 1000, od.loop.StopPatience())

######################## EVALUATE ########################

# We are going to validate using the training data
inputs, outputs = od.loop.eval(ts, model)

LAMBDA_NMS = 0.5
for i in range(4):
    for j in range(2):
        plt.subplot(2, 4, j*4+i+1)
        plt.imshow(inputs[i]['image'])
        od.plot.hasobjs(outputs[i][f'hasobjs{j}'])
        inv_outputs = inv_transforms(outputs[i])
        inv_bboxes, inv_classes = od.post.NMS(inv_outputs['hasobjs'], inv_outputs['bboxes'], inv_outputs['classes'], lambda_nms=LAMBDA_NMS)
        od.plot.bboxes(inputs[i]['image'], inv_bboxes)
        od.plot.classes(inputs[i]['image'], inv_bboxes, inv_classes, od.data.VOCDetection.labels)
plt.tight_layout()
plt.savefig('fcos.png')
