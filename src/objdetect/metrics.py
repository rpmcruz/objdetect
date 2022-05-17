'''
Implementation of Precision-Recall and AP metrics.
'''

import numpy as np

def IoU(bbox1, bbox2):
    '''Intersection over union between two bounding boxes.'''
    x0 = max(bbox1[0], bbox2[0])
    x1 = min(bbox1[2], bbox2[2])
    y0 = max(bbox1[1], bbox2[1])
    y1 = min(bbox1[3], bbox2[3])
    A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    A2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    I = (x1-x0) * (y1-y0)
    U = A1 + A2 - I
    return I / U

def which_correct(BB_true, BB_pred, iou_threshold):
    '''For each bounding box in all image, computes if it was correctly predicted (that is, IoU is over the given threshold).'''
    correct = []
    for bb_true, bb_pred in zip(BB_true, BB_pred):
        c = [False] * len(bb_pred)
        for b_true in bb_true:
            for i, b_pred in enumerate(bb_pred):
                # if this bbox was hit, skip it
                if c[i]: break
                iou = IoU(b_true, b_pred)
                c[i] = iou >= iou_threshold
        correct += c
    return correct

def precision_recall_curve(CC_pred, BB_true, BB_pred, iou_threshold):
    '''Produces a precision-recall curve, given the has-object probabilities and respective bounding boxes.'''
    # flatten
    CC_pred = [c for cc in CC_pred for c in cc]
    # order 'correct' based on confidence probability
    correct = which_correct(BB_true, BB_pred, iou_threshold)
    ix = np.argsort(CC_pred)[::-1]
    correct = [correct[i] for i in ix]
    # compute curve
    true_bboxes = sum(len(bb_pred) for bb_pred in BB_pred)
    precision = np.cumsum(correct) / np.arange(1, len(correct)+1)
    recall = np.cumsum(correct) / true_bboxes
    # smooth zigzag
    for i in range(len(precision)-1, 0, -1):
        if precision[i-1] < precision[i]:
            precision[i-1] = precision[i]
    return precision, recall

def AP(CC_pred, BB, BB_pred, iou_threshold):
    '''Produces an AP-score based on the precision-recall curve.'''
    precision, recall = precision_recall_curve(CC_pred, BB, BB_pred, iou_threshold)
    return np.sum(np.diff(recall) * precision[1:])

def filter_class(k, YY, CC_pred, BB, BB_pred):
    '''Utility to filter a given class.'''
    return ([cc[yy == k] for yy, cc in zip(YY, CC_pred)],
        [bb[yy == k] for yy, bb in zip(YY, BB)],
        [bb[yy == k] for yy, bb in zip(YY, BB_pred)])

def mAP(YY, CC_pred, BB, BB_pred, iou_threshold):
    '''mAP = average AP for all classes.'''
    classes = np.unique(YY)
    return np.mean([AP(*filter_class(k, YY, CC_pred, BB, BB_pred), iou_threshold) for k in classes])

def mAP_ious(YY, CC_pred, BB, BB_pred, iou_thresholds):
    '''mAP = average AP for all classes and for a list of IoU thresholds.'''
    return np.mean(mAP(YY, CC_pred, BB, BB_pred, th) for th in iou_thresholds)
