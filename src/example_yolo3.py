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
import objdetect as od

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

NCLASSES = len(od.data.VOCDetection.labels)
NANCHORS = 5
GRID_SIZES = [(8, 8)]*NANCHORS

######################## ANCHORS ########################

tr = od.data.VOCDetection('/data', 'train', None, None)
anchors = od.grid.compute_clusters(tr, NANCHORS)

######################## GRID & AUG ########################

grid_transforms = [od.grid.Transform(
    grid_size,
    od.grid.AnchorFilter(anchor, 0.5),
    od.grid.SliceOnlyCenterBbox(),
    {f'scores{i}': od.grid.NewScore(), f'bboxes{i}': od.grid.NewBboxes(), f'classes{i}': od.grid.NewClasses()},
    {f'scores{i}': od.grid.SetScore(), f'bboxes{i}': od.grid.SetOffsetSizeBboxesAnchor(anchor), f'classes{i}': od.grid.SetClasses()},
) for i, (anchor, grid_size) in enumerate(zip(anchors, GRID_SIZES))]

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
    [lambda datum, i=i: datum[f'scores{i}'][0] >= 0.5 for i in range(NANCHORS)],
    {'scores': [f'scores{i}' for i in range(NANCHORS)], 'bboxes': [f'bboxes{i}' for i in range(NANCHORS)], 'classes': [f'classes{i}' for i in range(NANCHORS)]},
    {'scores': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvOffsetSizeBboxesAnchor(anchors), 'classes': od.inv_grid.InvClasses()},
    True
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', None, dict_transform)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6, pin_memory=True)

ts = od.data.VOCDetection('/data', 'val', None, val_dict_transform)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6, pin_memory=True)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([32, 64, 128, 256, 512], True)
heads = [{}]*4 + [od.grid.merge_dicts([{f'scores{i}': od.models.HeadScores(512), f'bboxes{i}': od.models.HeadExpBboxes(512), f'classes{i}': od.models.HeadClasses(512, NCLASSES)} for i in range(NANCHORS)])]
model = od.models.Model(backbone, heads)
model = model.to(device)
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = od.grid.merge_dicts([{
    f'scores{i}': lambda data: 1,
    f'bboxes{i}': lambda data, i=i: data[f'scores{i}'],
    f'classes{i}': lambda data, i=i: data[f'scores{i}'],
} for i in range(NANCHORS)])
loss_fns = od.grid.merge_dicts([{
    # we could use sigmoid_focal_loss like RetinaNet
    f'scores{i}': nn.BCEWithLogitsLoss(reduction='none'),
    f'bboxes{i}': nn.MSELoss(reduction='none'),
    f'classes{i}': nn.CrossEntropyLoss(reduction='none'),
} for i in range(NANCHORS)])

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 100, od.loop.StopPatience())

######################## EVALUATE ########################

# We are going to validate using the training data
inputs, outputs = od.loop.eval(ts, model, inv_transforms)
outputs = od.post.NMS(outputs, lambda_nms=0.5)

for i in range(3*4):
    plt.subplot(3, 4, i+1)
    od.plot.image(inputs[i]['image'])
    od.plot.bboxes(inputs[i]['image'], outputs[i]['bboxes'])
    od.plot.classes(inputs[i]['image'], outputs[i]['bboxes'], outputs[i]['classes'], od.data.VOCDetection.labels)
plt.tight_layout()
plt.savefig('yolo3.png')

print('AP:', od.metrics.AP(outputs, inputs, 0.5))
print('mAP:', od.metrics.mAP(outputs, inputs, 0.5))
