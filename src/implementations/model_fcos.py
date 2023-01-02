'''
This is a simplified version of FCOS using a single grid.
https://arxiv.org/abs/1904.01355
'''

import torchvision
import torch
import objdetect as od

bboxes_loss = torchvision.ops.generalized_box_iou_loss
centerness_loss = torch.nn.BCEWithLogitsLoss()
labels_loss = torchvision.ops.sigmoid_focal_loss

class Heads(torch.nn.Module):
    # FCOS heads are shared between grids
    def __init__(self, in_channels, nclasses):
        # like FCOS, we do not have a dedicated 'scores' prediction. it's just
        # the argmax of the classes.
        self.classes = torch.nn.Conv2d(in_channels, nclasses, 1)
        self.bboxes = torch.nn.Conv2d(in_channels, 4, 1)
        self.centerness = torch.nn.Conv2d(in_channels, 1, 1)

class Grid(torch.nn.Module):
    def __init__(self, heads, img_size):
        super().__init__()
        self.img_size = img_size
        self.heads = heads
        # "a trainable scalar si to automatically adjust the base of the
        # exponential function"
        self.s = torch.nn.parameter.Parameter(torch.ones(()))

    def forward(self, x):
        # like FCOS, the network is predicting bboxes in relative terms, we need
        # to convert to absolute bboxes because the loss requires so.
        bboxes = torch.exp(self.s * self.heads.bboxes(x))
        bboxes = od.transforms.rel_bboxes(bboxes, self.img_size)
        return {'labels': self.heads.classes(x), 'bboxes': bboxes,
            'centerness': self.heads.centerness(x)}

    def post_process(self, preds, threshold=0.05):
        scores, labels = torch.sigmoid(preds['labels']).max(1, keepdim=True)
        bboxes = preds['bboxes']
        centerness = torch.sigmoid(preds['centerness'])
        mask = scores[:, 0] >= threshold
        # like FCOS, centerness will help NMS choose the best bbox.
        scores = scores * centerness
        return {
            'scores': od.grid.mask_select(mask, scores, True),
            'bboxes': od.grid.mask_select(mask, bboxes, True),
            'labels': od.grid.mask_select(mask, labels, True),
        }

    def compute_loss(self, preds, targets):
        grid_size = preds['bboxes'].shape[2:]
        mask, indices = od.grid.where(od.grid.slice_all_center, targets['bboxes'], grid_size, self.img_size)
        # preds grid -> list
        pred_bboxes = od.grid.mask_select(mask, preds['bboxes'])
        pred_labels = od.grid.mask_select(mask, preds['labels'])
        pred_centerness = od.grid.mask_select(mask, preds['centerness'])
        # targets list -> list
        target_bboxes = od.grid.indices_select(indices, targets['bboxes'])
        target_labels = od.grid.indices_select(indices, targets['labels'])
        # labels: must be one-hot since we use independent classifiers
        target_labels = torch.nn.functional.one_hot(target_labels.long(),
            preds['labels'].shape[1]).float()
        # compute centerness: requires doing the transformation in grid-space
        target_bboxes_grid = od.grid.to_grid(mask, indices, targets['bboxes'])
        target_rel_bboxes = od.transforms.rel_bboxes(target_bboxes_grid, self.img_size)
        target_centerness = od.transforms.centerness(target_rel_bboxes)
        target_centerness = od.grid.mask_select(mask, target_centerness)
        # compute losses
        return bboxes_loss(pred_bboxes, target_bboxes).mean() + \
            labels_loss(pred_labels, target_labels).mean() + \
            centerness_loss(pred_centerness, target_centerness)

class Model(torch.nn.Module):
    def __init__(self, nclasses, img_size):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        layers = list(resnet.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers[:-3])
        self.resnet_convs = layers[-3:]
        self.updown_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(2048, ),
        ])
        heads = Heads(2048, nclasses)
        self.grids = torch.nn.ModuleList([Grid(heads, img_size) for _ in range(5)])

    def forward(self, x):
        x = self.backbone(x)
        return self.grid(x)

    def post_process(self, x):
        return self.grid.post_process(x)

    def compute_loss(self, preds, targets):
        return self.grid.compute_loss(preds, targets)
