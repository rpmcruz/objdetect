'''
This should be a fairly faithful implementation of the FCOS paper.
https://arxiv.org/abs/1904.01355
'''

import torchvision
import torch
import backbones
import objdetect as od

bboxes_loss = torchvision.ops.generalized_box_iou_loss
centerness_loss = torch.nn.BCEWithLogitsLoss()
labels_loss = torchvision.ops.sigmoid_focal_loss

class Heads(torch.nn.Module):
    # FCOS heads are shared between grids
    def __init__(self, in_channels, nclasses):
        # like FCOS, we do not have a dedicated 'scores' prediction. it's just
        # the argmax of the classes.
        super().__init__()
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
        # when using multi-scale, it's possible that some scales have nothing
        # to predict. return now to avoid nan losses.
        if len(pred_bboxes) == 0: return 0
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

class TopDown(torch.nn.Module):
    # FCOS uses the same TopDown as "Feature Pyramid Networks for Object
    # Detection".
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 256, 1)
        self.conv3 = torch.nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, left, top):
        up = torch.nn.functional.interpolate
        top = up(top, scale_factor=2, mode='nearest-exact')
        return self.conv3(self.conv1(left) + top)

class Model(torch.nn.Module):
    def __init__(self, nclasses, img_size):
        super().__init__()
        self.backbone = backbones.Resnet50()
        channels = self.backbone.channels
        self.P3 = TopDown(channels[0])
        self.P4 = TopDown(channels[1])
        self.P5 = torch.nn.Conv2d(channels[2], 256, 3, padding=1)
        self.P6 = torch.nn.Conv2d(channels[2], 256, 1, 2, 1)
        self.P7 = torch.nn.Conv2d(256, 256, 1, 2, 1)
        heads = Heads(256, nclasses)
        self.grids = torch.nn.ModuleList([Grid(heads, img_size) for _ in range(5)])
        self.ms = [0, 64, 128, 256, 512, float('inf')]

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p5 = self.P5(c5)
        p4 = self.P4(c4, p5)
        p3 = self.P3(c3, p4)
        p6 = self.P6(c5)
        p7 = self.P7(p6)
        return [grid(p) for grid, p in zip(self.grids, [p3, p4, p5, p6, p7])]

    def post_process(self, xs):
        xs = [grid.post_process(x) for grid, x in zip(self.grids, xs)]
        return {key: sum(x[key] for x in xs) for key in ['scores', 'bboxes', 'labels']}

    def compute_loss(self, preds, targets):
        return sum(
            grid.compute_loss(pred, od.utils.filter_grid(targets, m_min, m_max))
            for grid, pred, m_min, m_max in zip(self.grids, preds, self.ms, self.ms[1:]))
