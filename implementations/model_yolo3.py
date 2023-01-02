'''
This should be a pretty faithful implementation of the YOLO v3 paper, which was
the last version of YOLO by Joseph Redmon. (Other authors have called their
papers "YOLO", but those papers are unrelated to the original author.)
https://arxiv.org/abs/1804.02767
'''

import torchvision
import torch
import objdetect as od

scores_loss = torch.nn.BCEWithLogitsLoss()
labels_loss = torch.nn.BCEWithLogitsLoss()
bboxes_loss = torch.nn.MSELoss()

# YOLO3 uses Darknet53 as the backbone. This implementation is based on
# https://github.com/developer0hye/PyTorch-Darknet53

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        torch.nn.BatchNorm2d(out_num),
        torch.nn.LeakyReLU())

def rep_blocks(self, in_channels, num_blocks):
    layers = []
    for i in range(0, num_blocks):
        layers.append(ResBlock(in_channels))
    return torch.nn.Sequential(*layers)

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, 1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x
        return out

class Darknet53(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.block1 = rep_blocks(64, 1)
        self.p1 = conv_batch(64, 128, stride=2)
        self.block2 = rep_blocks(128, 2)
        self.p2 = conv_batch(128, 256, stride=2)
        self.block3 = rep_blocks(256, 8)
        self.p3 = conv_batch(256, 512, stride=2)
        self.block4 = rep_blocks(512, 8)
        self.p4 = conv_batch(512, 1024, stride=2)

    def forward(self, x):
        p1 = self.p1(self.block1(self.conv2(self.conv1(x))))
        p2 = self.p2(self.block2(p1))
        p3 = self.p3(self.block2(p2))
        p4 = self.p4(self.block4(p3))
        return p1, p2, p3, p4

# YOLO3

class Grid(torch.nn.Module):
    def __init__(self, in_channels, nclasses, img_size, anchor, scale):
        super().__init__()
        self.img_size = img_size
        self.ph, self.pw = anchor
        self.scale = scale
        self.scores = torch.nn.Conv2d(in_channels, 1, 1)
        self.classes = torch.nn.Conv2d(in_channels, nclasses, 1)
        self.bboxes = torch.nn.Conv2d(in_channels, 4, 1)

    def forward(self, x):
        xx = torch.arange(0, gw, dtype=torch.float32, device=x.device)[None, :]
        yy = torch.arange(0, gh, dtype=torch.float32, device=x.device)[:, None]
        bboxes = self.bboxes(x)
        bboxes[:, 0] = torch.sigmoid(xx * bboxes[:, 0])
        bboxes[:, 1] = torch.sigmoid(yy * bboxes[:, 1])
        bboxes[:, 2] = self.pw * torch.exp(bboxes[:, 2])
        bboxes[:, 3] = self.ph * torch.exp(bboxes[:, 3])
        return {'scores': self.scores(x), 'labels': self.classes(x),
            'bboxes': bboxes}

    def post_process(self, preds, threshold=0.5):
        # YOLO3 uses independent classifiers (not softmax)
        labels = torch.sigmoid(preds['labels'], 1).argmax(1, keepdim=True)
        scores = torch.sigmoid(preds['scores'])
        boxes = preds['bboxes']
        mask = scores[:, 0] >= threshold
        return {
            'scores': od.grid.mask_select(mask, scores, True),
            'bboxes': od.grid.mask_select(mask, bboxes, True),
            'labels': od.grid.mask_select(mask, labels, True),
        }

    def compute_loss(self, preds, targets):
        grid_size = preds['bboxes'].shape[2:]
        mask, indices = od.grid.where(od.grid.slice_center, targets['bboxes'], grid_size, self.img_size)
        # preds grid -> list
        pred_scores = od.grid.mask_select(mask, preds['scores'])
        pred_bboxes = od.grid.mask_select(mask, preds['bboxes'])
        pred_labels = od.grid.mask_select(mask, preds['labels'])
        # targets list -> list
        target_scores = mask[:, None]
        target_bboxes = od.grid.indices_select(indices, targets['bboxes'])
        target_labels = od.grid.indices_select(indices, targets['labels'])
        # labels: must be one-hot since we use independent classifiers
        target_labels = torch.nn.functional.one_hot(target_labels.long(),
            preds['labels'].shape[1]).float()
        # compute losses
        return bboxes_loss(pred_bboxes, target_bboxes) + \
            labels_loss(pred_labels, target_labels) + \
            scores_loss(pred_scores, target_scores)

class Model(torch.nn.Module):
    def __init__(self, nclasses, img_size, anchors_per_scale):
        super().__init__()
        self.backbone = Darknet53()
        channels = [256, 512, 1024]
        self.grids = torch.nn.Module([
            Grid(channels[scale], anchor, scale+1)
            for scale, anchors in enumerate(anchors_per_scale)
            for anchor in anchors
        ])
        self.anchors = [anchor for anchors in anchors_per_scale for anchor in anchors]

    def forward(self, x):
        ps = self.backbone(x)
        return [grid(ps[grid.scale]) for grid in grids]

    def post_process(self, x):
        xs = [grid.post_process(x) for grid, x in zip(self.grids, xs)]
        return {key: sum(x[key] for x in xs) for key in ['scores', 'bboxes', 'labels']}

    def compute_loss(self, preds, targets):
        # "our system only assigns one bounding box prior [anchor] for each
        # ground truth object"
        return sum([grid.compute_loss(pred, {
            'bboxes': [bboxes[ix] for bboxes, ix in zip(targets['bboxes'], targets_ix)],
            'labels': [labels[ix] for labels, ix in zip(targets['labels'], targets_ix)],
        }) for grid, pred in zip(self.grids, preds)])
