'''
This should be seen as a demonstration of how a model such as YOLOv3 *could* be implemented, but it *should* not be seen as a perfect duplication.

YOLOv3 paper: https://arxiv.org/abs/1804.02767
Also see YOLO9000 (aka YOLOv2): https://arxiv.org/pdf/1612.08242.pdf
'''

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 12)
import objdetect as od

NCLASSES = len(od.data.VOCDetection.labels)
NANCHORS = 5
GRID_SIZES = [(8, 8)]*NANCHORS

######################## ANCHORS ########################

tr = od.data.VOCDetection('/data', 'train', None)
anchors = od.grid.compute_clusters(tr, NANCHORS)

######################## GRID ########################

grid_transforms = [od.grid.Transform(
    grid_size,
    od.grid.AnchorFilter(anchor, 0.5),
    od.grid.SliceOnlyCenterBbox(),
    {f'hasobjs{i}': od.grid.NewHasObj(), f'bboxes{i}': od.grid.NewBboxes(), f'classes{i}': od.grid.NewClasses()},
    {f'hasobjs{i}': od.grid.SetHasObj(), f'bboxes{i}': od.grid.SetOffsetSizeBboxesAnchor(anchor), f'classes{i}': od.grid.SetClasses()},
) for i, (anchor, grid_size) in enumerate(zip(anchors, GRID_SIZES))]

transforms = od.aug.Compose(
    od.aug.ResizeAndNormalize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
    od.aug.RandomHflip(),
    *grid_transforms,
    od.grid.RemoveKeys(['bboxes', 'classes'])
)

inv_transforms = od.inv_grid.MultiLevelInvTransform(
    [lambda datum, i=i: datum[f'hasobjs{i}'][0] >= 0.5 for i in range(NANCHORS)],
    {'hasobjs': [f'hasobjs{i}' for i in range(NANCHORS)], 'bboxes': [f'bboxes{i}' for i in range(NANCHORS)], 'classes': [f'classes{i}' for i in range(NANCHORS)]},
    {'hasobjs': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvOffsetSizeBboxesAnchor(anchors), 'classes': od.inv_grid.InvClasses()}
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6)

ts = od.data.VOCDetection('/data', 'val', transforms)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([False]*4 + [True], [32, 64, 128, 256, 512], True)
heads = [od.grid.merge_dicts([{f'hasobjs{i}': od.models.HeadHasObjs(512), f'bboxes{i}': od.models.HeadExpBboxes(512), f'classes{i}': od.models.HeadClasses(512, NCLASSES)} for i in range(NANCHORS)])]
model = od.models.Model(backbone, heads)
model = model.cuda()
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = od.grid.merge_dicts([{
    f'hasobjs{i}': lambda data: 1,
    f'bboxes{i}': lambda data, i=i: data[f'hasobjs{i}'],
    f'classes{i}': lambda data, i=i: data[f'hasobjs{i}'],
} for i in range(NANCHORS)])
loss_fns = od.grid.merge_dicts([{
    # we could use sigmoid_focal_loss like RetinaNet
    f'hasobjs{i}': nn.BCEWithLogitsLoss(reduction='none'),
    f'bboxes{i}': nn.MSELoss(reduction='none'),
    f'classes{i}': nn.CrossEntropyLoss(reduction='none'),
} for i in range(NANCHORS)])

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 1000, od.loop.StopPatience())

######################## EVALUATE ########################

# We are going to validate using the training data
inputs, outputs = od.loop.eval(ts, model)

for i in range(3*4):
    plt.subplot(3, 4, i+1)
    plt.imshow(inputs[i]['image'])
    od.plot.hasobjs(outputs[i]['hasobjs'])
    inv_outputs = inv_transforms(outputs[i])
    inv_bboxes, inv_classes = od.post.NMS(inv_outputs['hasobjs'], inv_outputs['bboxes'], inv_outputs['classes'], lambda_nms=0.5)
    od.plot.bboxes(inputs[i]['image'], inv_bboxes)
    od.plot.classes(inputs[i]['image'], inv_bboxes, inv_classes, od.data.VOCDetection.labels)
plt.tight_layout()
plt.savefig('yolo3.png')
