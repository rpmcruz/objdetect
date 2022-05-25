'''
This an example more akin to Pixor in the sense that there are no anchors (like YOLOv3) or multi-level predictions (like FCOS).
Pixor paper: https://arxiv.org/abs/1902.06326
'''

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 12)
import objdetect as od

NCLASSES = len(od.data.VOCDetection.labels)

######################## GRID ########################

grid_transform = od.grid.Transform(
    (8, 8),
    None,
    od.grid.SliceOnlyCenterBbox(),
    {'hasobjs': od.grid.NewHasObj(), 'bboxes': od.grid.NewBboxes(), 'classes': od.grid.NewClasses()},
    {'hasobjs': od.grid.SetHasObj(), 'bboxes': od.grid.SetOffsetSizeBboxes(), 'classes': od.grid.SetClasses()}
)

transforms = od.aug.Compose(
    od.aug.ResizeAndNormalize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
    od.aug.RandomHflip(),
    grid_transform,
)

inv_transforms = od.inv_grid.InvTransform(
    lambda datum: datum['hasobjs'][0] >= 0.5,
    {'hasobjs': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvOffsetSizeBboxes(), 'classes': od.inv_grid.InvClasses()}
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6)

ts = od.data.VOCDetection('/data', 'val', transforms)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([32, 64, 128, 256, 512], False)
heads = [{}]*4 + [{'hasobjs': od.models.HeadHasObjs(512), 'bboxes': od.models.HeadExpBboxes(512), 'classes': od.models.HeadClasses(512, NCLASSES)}]
model = od.models.Model(backbone, heads)
model = model.cuda()
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = {
    'hasobjs': lambda data: 1,
    'bboxes': lambda data: data['hasobjs'],
    'classes': lambda data: data['hasobjs'],
}
loss_fns = {
    # we could use sigmoid_focal_loss like RetinaNet
    'hasobjs': nn.BCEWithLogitsLoss(reduction='none'),
    'bboxes': nn.MSELoss(reduction='none'),
    'classes': nn.CrossEntropyLoss(reduction='none'),
}

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 100)

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
plt.savefig('simple.png')
