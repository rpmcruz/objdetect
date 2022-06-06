'''
PyTorch already comes with most useful losses, such as MSE, BCE and sigmoid_focal_loss, but this provides other losses such as UnitBox losses.
'''

import torch
from objdetect.post import valid_bboxes

def convert_rel2abs(bboxes):
    '''Similar to `inv_grid.InvRelBboxes`. Converts relative bounding boxes to absolute bounding boxes.'''
    _, _, h, w = bboxes.shape
    yy = torch.linspace(0, h, 1, device=bboxes.device)
    xx = torch.linspace(0, w, 1, device=bboxes.device)
    yy, xx = torch.meshgrid(yy, xx, indexing='xy')
    bboxes_offset = torch.stack((
        xx/w-bboxes[:, 0], yy/h-bboxes[:, 1],
        bboxes[:, 2]+xx/w, bboxes[:, 3]+yy/h
    ), 1)
    return bboxes_offset

def ConvertRel2Abs(loss_fn):
    '''This is useful because some losses require the bounding boxes to be absolute.'''
    def f(bboxes1, bboxes2):
        return loss_fn(convert_rel2abs(bboxes1), convert_rel2abs(bboxes2))
    return f

def IoU(do_validation, smooth=1):
    '''Implements loss 1-IoU (Intersection over Union).'''
    def f(bboxes1, bboxes2):
        # ensure bounding boxes make sense
        if do_validation:
            bboxes1 = valid_bboxes(bboxes1)
            bboxes2 = valid_bboxes(bboxes2)
        # area of each
        A1 = (bboxes1[:, 2]-bboxes1[:, 0])*(bboxes1[:, 3]-bboxes1[:, 1])
        A2 = (bboxes2[:, 2]-bboxes2[:, 0])*(bboxes2[:, 3]-bboxes2[:, 1])
        # intersection
        Iw = torch.clamp(torch.minimum(bboxes1[:, 2], bboxes2[:, 2])-torch.maximum(bboxes1[:, 0], bboxes2[:, 0]), min=0)
        Ih = torch.clamp(torch.minimum(bboxes1[:, 3], bboxes2[:, 3])-torch.maximum(bboxes1[:, 1], bboxes2[:, 1]), min=0)
        I = Iw*Ih
        # union
        U = A1 + A2 - I
        # result
        IoU = (I+smooth) / (U+smooth)
        return 1-IoU
    return f

def GIoU(do_validation, smooth=1):
    '''Implements the loss from [Generalized Intersection over Union paper](https://giou.stanford.edu/), used by papers such as FCOS.'''
    def f(bboxes1, bboxes2):
        # ensure bounding boxes make sense
        if do_validation:
            bboxes1 = valid_bboxes(bboxes1)
            bboxes2 = valid_bboxes(bboxes2)
        # area of each
        A1 = (bboxes1[:, 2]-bboxes1[:, 0])*(bboxes1[:, 3]-bboxes1[:, 1])
        A2 = (bboxes2[:, 2]-bboxes2[:, 0])*(bboxes2[:, 3]-bboxes2[:, 1])
        # intersection
        Iw = torch.clamp(torch.minimum(bboxes1[:, 2], bboxes2[:, 2])-torch.maximum(bboxes1[:, 0], bboxes2[:, 0]), min=0)
        Ih = torch.clamp(torch.minimum(bboxes1[:, 3], bboxes2[:, 3])-torch.maximum(bboxes1[:, 1], bboxes2[:, 1]), min=0)
        I = Iw*Ih
        # smallest enclosing box
        Ew = torch.maximum(bboxes1[:, 2], bboxes2[:, 2])-torch.minimum(bboxes1[:, 0], bboxes2[:, 0])
        Eh = torch.maximum(bboxes1[:, 3], bboxes2[:, 3])-torch.minimum(bboxes1[:, 1], bboxes2[:, 1])
        E = Ew*Eh
        # union
        U = A1 + A2 - I
        # result
        IoU = (I+smooth) / (U+smooth)
        GIoU = IoU - (E-U+smooth)/(E+smooth)
        return 1-GIoU
    return f
