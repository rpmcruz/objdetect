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
import objdetect as od

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

NCLASSES = len(od.data.VOCDetection.labels)
GRID_SIZES = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
NLEVELS = len(GRID_SIZES)

######################## GRID & AUG ########################

grid_transforms = [od.grid.Transform(
    grid_size,
    od.grid.SizeFilter(0 if i == 0 else 4, np.inf if i == NLEVELS-1 else 8),
    od.grid.SliceAcrossCeilBbox(),
    {f'scores{i}': od.grid.NewScore(), f'bboxes{i}': od.grid.NewBboxes(), f'classes{i}': od.grid.NewClasses(), f'centerness{i}': od.grid.NewCenterness()},
    {f'scores{i}': od.grid.SetScore(), f'bboxes{i}': od.grid.SetRelBboxes(), f'classes{i}': od.grid.SetClasses(), f'centerness{i}': od.grid.SetCenterness()},
) for i, grid_size in enumerate(GRID_SIZES)]

dict_transform = od.aug.Compose([
    od.aug.Resize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
    od.aug.RandomHflip(),
    *grid_transforms,
    od.grid.RemoveKeys(['bboxes', 'classes'])
])

val_dict_transform = od.aug.Compose([
    od.aug.Resize(256, 256),
    *grid_transforms,
    od.grid.RemoveKeys(['bboxes', 'classes'])
])

inv_transforms = od.inv_grid.MultiLevelInvTransform(
    [lambda datum, i=i: datum[f'scores{i}'][0] >= 0.5 for i in range(NLEVELS)],
    {'scores': [f'scores{i}' for i in range(NLEVELS)], 'bboxes': [f'bboxes{i}' for i in range(NLEVELS)], 'classes': [f'classes{i}' for i in range(NLEVELS)]},
    {'scores': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvRelBboxes(), 'classes': od.inv_grid.InvClasses()},
    True
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', None, dict_transform)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6, pin_memory=True)

ts = od.data.VOCDetection('/data', 'val', None, val_dict_transform)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6, pin_memory=True)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([32, 64, 128, 256, 512], False)
heads = [{
    f'scores{i}': od.models.HeadScores(32*2**i),
    f'bboxes{i}': od.models.HeadExpBboxes(32*2**i),
    f'classes{i}': od.models.HeadClasses(32*2**i, NCLASSES),
    f'centerness{i}': od.models.HeadScores(32*2**i),
} for i in range(NLEVELS)]
model = od.models.Model(backbone, heads)
model = model.to(device)
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = od.grid.merge_dicts([{
    f'scores{i}': lambda data: 1,
    f'bboxes{i}': lambda data, i=i: data[f'scores{i}'],
    f'classes{i}': lambda data, i=i: data[f'scores{i}'],
    f'centerness{i}': lambda data, i=i: data[f'scores{i}'],
} for i in range(NLEVELS)])
loss_fns = od.grid.merge_dicts([{
    # we could use sigmoid_focal_loss like RetinaNet
    f'scores{i}': nn.BCEWithLogitsLoss(reduction='none'),
    f'bboxes{i}': od.losses.ConvertRel2Abs(od.losses.GIoU(False)),
    f'classes{i}': nn.CrossEntropyLoss(reduction='none'),
    f'centerness{i}': nn.BCEWithLogitsLoss(reduction='none'),
} for i in range(NLEVELS)])

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 100, od.loop.StopPatience())

######################## EVALUATE ########################

# We are going to validate using the training data
inputs, outputs = od.loop.eval(ts, model, inv_transforms)
outputs = od.post.NMS(outputs, lambda_nms=0.5)

LAMBDA_NMS = 0.5
for i in range(3*4):
    plt.subplot(3, 4, i+1)
    od.plot.image(inputs[i]['image'])
    od.plot.bboxes(inputs[i]['image'], outputs[i]['bboxes'])
    od.plot.classes(inputs[i]['image'], outputs[i]['bboxes'], outputs[i]['classes'], od.data.VOCDetection.labels)
plt.tight_layout()
plt.savefig('fcos.png')

print('AP:', od.metrics.AP(outputs, inputs, 0.5))
print('mAP:', od.metrics.mAP(outputs, inputs, 0.5))
