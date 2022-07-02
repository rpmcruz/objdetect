'''
Implementation of Precision-Recall and AP metrics.
'''

import torch

def IoU(bbox1, bbox2):
    '''Intersection over union between two bounding boxes.'''
    x0 = torch.maximum(bbox1[0], bbox2[0])
    y0 = torch.maximum(bbox1[1], bbox2[1])
    x1 = torch.minimum(bbox1[2], bbox2[2])
    y1 = torch.minimum(bbox1[3], bbox2[3])
    A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    A2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    I = torch.clamp(x1-x0, min=0) * torch.clamp(y1-y0, min=0)
    U = A1 + A2 - I
    return I / U

def IoUs(bbox1, bboxes2):
    '''Intersection over union between one bounding box against a list of others.'''
    x0 = torch.maximum(bbox1[0], bboxes2[:, 0])
    y0 = torch.maximum(bbox1[1], bboxes2[:, 1])
    x1 = torch.minimum(bbox1[2], bboxes2[:, 2])
    y1 = torch.minimum(bbox1[3], bboxes2[:, 3])
    A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    A2 = (bboxes2[:, 2]-bboxes2[:, 0]) * (bboxes2[:, 3]-bboxes2[:, 1])
    I = torch.clamp(x1-x0, min=0) * torch.clamp(y1-y0, min=0)
    U = A1 + A2 - I
    return I / U

def which_correct(preds, true, iou_threshold):
    '''For each bounding box in all image, computes if it was correctly predicted (that is, IoU is over the given threshold). For each true bounding box, it returns a boolean list of the same size indicating whether there is a matching prediction or not.'''
    return [
        [len(t['bboxes']) > 0 and torch.any(IoUs(bb, t['bboxes']) >= iou_threshold)
            for bb in p['bboxes']]
        for p, t in zip(preds, true)
    ]

def precision_recall_curve(preds, true, iou_threshold):
    '''Produces a precision-recall curve, given the has-object probabilities and respective bounding boxes. `preds` and `true` are lists of dictionaries containing at least: scores and bboxes. A good explanation of this metric: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173.'''
    assert len(preds) == len(true), f'number of preds ({len(preds)}) must match number of ground-truth ({len(true)})'
    # match bounding by whether they are correct
    correct = which_correct(preds, true, iou_threshold)
    # flatten
    scores = torch.tensor([s for p in preds for s in p['scores']])
    correct = torch.tensor([c for cs in correct for c in cs])
    # order 'correct' based on confidence probability
    ix = torch.argsort(scores, descending=True)
    correct = correct[ix]
    # compute curve
    npredictions = sum(len(p['bboxes']) for p in preds)
    cum_correct = torch.cumsum(correct, 0)
    precision = cum_correct / torch.arange(1, len(correct)+1)
    recall = cum_correct / npredictions
    # smooth zigzag
    for i in range(len(precision)-1, 0, -1):
        if precision[i-1] < precision[i]:
            precision[i-1] = precision[i]
    precision = torch.cat((torch.tensor([1]), precision, torch.tensor([0])))
    recall = torch.cat((torch.tensor([0]), recall, torch.tensor([1])))
    return precision, recall

def AP(preds, true, iou_threshold):
    '''Produces an AP-score based on the precision-recall curve.'''
    precision, recall = precision_recall_curve(preds, true, iou_threshold)
    return torch.sum(torch.diff(recall) * precision[1:])

def filter_class(klass, preds, true, keys):
    '''Utility to filter a given class.'''
    preds = [{k: p[k][p['classes'] == klass] for k in keys} for p in preds]
    true = [{k: t[k][t['classes'] == klass] for k in keys} for t in true]
    return preds, true

def mAP(preds, true, iou_threshold):
    '''mAP = average AP for all classes.'''
    assert 'classes' in preds[0] and 'classes' in true[0]
    nclasses = 1+max([max(t['classes']) if len(t['classes']) else 0 for t in true])
    return sum(AP(*filter_class(klass, preds, true, ['scores', 'bboxes']), iou_threshold) for klass in range(nclasses)) / nclasses

def mAP_ious(preds, true, iou_thresholds=torch.arange(0.05, 1, 0.05)):
    '''mAP = average AP for all classes and for a list of IoU thresholds. This is the metric used by MS-COCO. By default, we evaluate IoU@[0.05,0.95,0.05].'''
    return torch.mean([mAP(preds, true, th) for th in iou_thresholds])

if __name__ == '__main__':  # DEBUG
    true_bbox = torch.tensor([0.4, 0, 0.6, 1])
    pred_bboxes = torch.tensor([
        [0.0, 0, 0.3, 1],  # no intersection
        [0.5, 0, 0.6, 1],  # half intersection
        [0.4, 0, 0.6, 1],  # full intersection
    ])
    print('IoUs:', IoUs(true_bbox, pred_bboxes))
    true = [{'bboxes': true_bbox[None]}]
    preds = [{'bboxes': pred_bboxes, 'scores': [0.5, 0.4, 0.8]}]
    for th in [0, 0.5, 1]:
        print(f'which_correct {th}:', which_correct(preds, true, th))
    for th in [0, 0.5, 1]:
        print(f'precision_recall_curve {th}:', precision_recall_curve(preds, true, th))
    for th in [0, 0.5, 1]:
        print(f'AP {th}:', AP(preds, true, th))
