'''
Post-processing techniques to reduce the amount of superfluous bounding boxes.
'''

import torch

def valid_bboxes(bboxes):
    '''Converts the given bounding boxes into valid ones. That is, x2/y2 is swapped by x1/y1 if x2<x1 or y2<y1.'''
    return torch.stack((
        torch.clamp(torch.minimum(bboxes[:, 0], bboxes[:, 2]), min=0),
        torch.clamp(torch.minimum(bboxes[:, 1], bboxes[:, 3]), min=0),
        torch.clamp(torch.maximum(bboxes[:, 0], bboxes[:, 2]), max=1),
        torch.clamp(torch.maximum(bboxes[:, 1], bboxes[:, 3]), max=1),
    ), 1)

def same(bi, bj):
    '''Intersection over union utility.'''
    intersection = (min(bi[2], bj[2])-max(bi[0], bj[0])) * (min(bi[3], bj[3])-max(bi[1], bj[1]))
    union = (bi[2]-bi[0])*(bi[3]-bi[1]) + (bj[2]-bj[0])*(bj[3]-bj[1]) - intersection
    return intersection / union

def NMS(data, lambda_nms=0.5):
    '''Non-Maximum Suppression (NMS) algorithm. It is a popular post-processing algorithm to clean-up similar bounding boxes. The `data` parameter is a list of dictionaries (i.e. `[{'scores': [0.1, 0.9], 'bboxes': [...]}])`. Notice this modifies your data in-place.'''
    ret = [None] * len(data)
    for di, datum in enumerate(data):
        ix = [i for i in range(len(datum['scores']))
            if not any(  # discard if all conditions met
                i != j and
                datum['scores'][j] > datum['scores'][i] and
                same(datum['bboxes'][i], datum['bboxes'][j]) >= lambda_nms
            for j in range(len(datum['scores'])))]
        ret[di] = {k: v[ix] for k, v in datum.items()}
    return ret
