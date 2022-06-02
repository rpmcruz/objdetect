'''
This an example more akin to Pixor in the sense that there are no anchors (like YOLOv3) or multi-level predictions (like FCOS).
Pixor paper: https://arxiv.org/abs/1902.06326
'''

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchinfo import summary
from torch import nn
import torch
import matplotlib.pyplot as plt
import objdetect as od

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

NCLASSES = len(od.data.VOCDetection.labels)

######################## GRID & AUG ########################

grid_transform = od.grid.Transform(
    (8, 8),
    None,
    od.grid.SliceOnlyCenterBbox(),
    {'scores': od.grid.NewScore(), 'bboxes': od.grid.NewBboxes(), 'classes': od.grid.NewClasses()},
    {'scores': od.grid.SetScore(), 'bboxes': od.grid.SetOffsetSizeBboxes(), 'classes': od.grid.SetClasses()}
)

dict_transform = od.aug.Compose([
    od.aug.Resize(int(256*1.05), int(256*1.05)),
    od.aug.RandomCrop(256, 256),
    od.aug.RandomBrightnessContrast(0.1, 0.05),
    od.aug.RandomHflip(),
    grid_transform
])

val_dict_transform = od.aug.Compose([
    od.aug.Resize(256, 256),
    grid_transform
])

inv_transforms = od.inv_grid.InvTransform(
    lambda datum: datum['scores'][0] >= 0.5,
    {'scores': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvOffsetSizeBboxes(), 'classes': od.inv_grid.InvClasses()}
)

######################## DATA ########################

tr = od.data.VOCDetection('/data', 'train', None, dict_transform)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6, pin_memory=True)

ts = od.data.VOCDetection('/data', 'val', None, val_dict_transform)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6, pin_memory=True)

######################## MODEL ########################

backbone = od.models.SimpleBackbone([32, 64, 128, 256, 512], False)
heads = [{}]*4 + [{'scores': od.models.HeadScores(512), 'bboxes': od.models.HeadExpBboxes(512), 'classes': od.models.HeadClasses(512, NCLASSES)}]
model = od.models.Model(backbone, heads)
model = model.to(device)
print(summary(model, (10, 3, 256, 256)))

######################## TRAIN ########################

opt = torch.optim.Adam(model.parameters())

weight_loss_fns = {
    'scores': lambda data: 1,
    'bboxes': lambda data: data['scores'],
    'classes': lambda data: data['scores'],
}
loss_fns = {
    # we could use sigmoid_focal_loss like RetinaNet
    'scores': nn.BCEWithLogitsLoss(reduction='none'),
    'bboxes': nn.MSELoss(reduction='none'),
    'classes': nn.CrossEntropyLoss(reduction='none'),
}

od.loop.train(tr, model, opt, weight_loss_fns, loss_fns, 100)

######################## EVALUATE ########################

# We are going to validate using the training data
inputs, outputs = od.loop.eval(ts, model)

for i in range(3*4):
    inv_outputs = inv_transforms(outputs[i])
    inv_bboxes, inv_classes = od.post.NMS(inv_outputs['scores'], inv_outputs['bboxes'], inv_outputs['classes'], lambda_nms=0.5)

    plt.subplot(3, 4, i+1)
    od.plot.image(inputs[i]['image'])
    od.plot.grid_bools(inputs[i]['image'], outputs[i]['scores'][0])
    od.plot.grid_lines(inputs[i]['image'], 8, 8)
    od.plot.bboxes(inputs[i]['image'], inv_bboxes)
    od.plot.classes(inputs[i]['image'], inv_bboxes, inv_classes, od.data.VOCDetection.labels)
plt.tight_layout()
plt.savefig('simple.png')
