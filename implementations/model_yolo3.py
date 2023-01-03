'''
This should be a pretty faithful implementation of the YOLO v3 paper, which was
the last version of YOLO by Joseph Redmon. (Other authors have called their
papers "YOLO", but those papers are unrelated to the original author.)
https://arxiv.org/abs/1804.02767
'''

import torchvision
import torch
import backbones
import objdetect as od

scores_loss = torch.nn.BCEWithLogitsLoss()
labels_loss = torch.nn.BCEWithLogitsLoss()
bboxes_loss = torch.nn.MSELoss()

def compute_anchors(ds):
    # like in the paper, we compute anchors "evenly across scales"
    bboxes = od.anchors.flatten_sizes([d['bboxes'] for d in ds])
    areas = [bb[0]*bb[1] for bb in bboxes]
    ix = sorted(range(len(areas)), key=areas.__getitem__)
    q1, q2 = int(len(ix)*(1/3)), int(len(ix)*(2/3))
    return [od.anchors.compute_anchors(bboxes[:q1], 3),
        od.anchors.compute_anchors(bboxes[q1:q2], 3),
        od.anchors.compute_anchors(bboxes[q2:], 3)]

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
        bboxes = self.bboxes(x)
        gh, gw = bboxes.shape[2:]
        xx = torch.arange(0, gw, dtype=torch.float32, device=x.device)[None, :]
        yy = torch.arange(0, gh, dtype=torch.float32, device=x.device)[:, None]
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
        self.backbone = backbones.Darknet53()
        channels = self.backbone.channels
        self.grids = torch.nn.ModuleList([
            Grid(channels[scale], nclasses, img_size, anchor, scale)
            for scale, anchors in enumerate(anchors_per_scale)
            for anchor in anchors
        ])
        self.anchors = torch.tensor([anchor for anchors in anchors_per_scale for anchor in anchors])

    def forward(self, x):
        ps = self.backbone(x)
        return [grid(ps[grid.scale]) for grid in self.grids]

    def post_process(self, x):
        xs = [grid.post_process(x) for grid, x in zip(self.grids, xs)]
        return {key: sum(x[key] for x in xs) for key in ['scores', 'bboxes', 'labels']}

    def compute_loss(self, preds, targets):
        # "our system only assigns one bounding box prior [anchor] for each
        # ground truth object"
        targets_ix = [[od.anchors.anchors_ious(bbox, self.anchors).argmax()
            for bbox in bboxes] for bboxes in targets['bboxes']]
        return sum([grid.compute_loss(pred, {
            'bboxes': [bboxes[ix] for bboxes, ix in zip(targets['bboxes'], targets_ix)],
            'labels': [labels[ix] for labels, ix in zip(targets['labels'], targets_ix)],
        }) for grid, pred in zip(self.grids, preds)])
