'''
This should be seen as a demonstration of how a model such as YOLOv3 *could* be implemented, but it *should* not be seen as a perfect duplication. Most evidently, this lacks anchors.
YOLOv3 paper: https://arxiv.org/abs/1804.02767
'''

from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 12)
import objdetect as od

NCLASSES = len(od.data.VOCDetection.labels)

transforms = od.aug.Compose(
    od.aug.ResizeAndNormalize((256, 256)),
    od.grid.Transform(
        (8, 8),
        None,
        od.grid.SliceOnlyCenterBbox(),
        {'hasobjs': od.grid.NewHasObj(), 'bboxes': od.grid.NewBboxes(), 'classes': od.grid.NewClasses()},
        {'hasobjs': od.grid.SetHasObj(), 'bboxes': od.grid.SetCenterSizeBboxesOnce(), 'classes': od.grid.SetClasses()}
    ),
    od.grid.RemoveKeys(['classes']),
)

inv_transforms = od.inv_grid.InvTransform(
    lambda datum: datum['hasobjs'][0] >= 0.5,
    {'hasobjs': od.inv_grid.InvScores(), 'bboxes': od.inv_grid.InvCenterSizeBboxes(), 'classes': od.inv_grid.InvClasses()}
)

tr = od.data.VOCDetection('/data', 'train', transforms)
tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=6)

ts = od.data.VOCDetection('/data', 'train', transforms)
ts = torch.utils.data.DataLoader(ts, 32, num_workers=6)

backbone = od.models.SimpleBackbone([False]*4 + [True])
heads = [{'hasobjs': od.models.HeadHasObjs(512), 'bboxes': od.models.HeadExpBboxes(512), 'classes': od.models.HeadClasses(512, NCLASSES)}]
model = od.models.Model(backbone, heads)
model = model.cuda()
print(summary(model, (10, 3, 256, 256)))

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
