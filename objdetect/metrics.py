import numpy as np

def IoU(b_true, b_pred):
    left_x = max(b_true[0]-b_true[2], b_pred[0]-b_pred[2])
    right_x = min(b_true[0]+b_true[2], b_pred[0]+b_pred[2])
    top_y = max(b_true[1]-b_true[3], b_pred[1]-b_pred[3])
    bottom_y = min(b_true[1]+b_true[3], b_pred[1]+b_pred[3])
    intersection = (right_x-left_x) * (bottom_y-top_y)
    union = b_true[2]*b_true[3]*4 + b_pred[2]*b_pred[3]*4 - intersection**2
    return intersection / union

def which_correct(BB_true, BB_pred, iou_threshold):
    # for each bbox in all image, compute if found or not
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
    precision, recall = precision_recall_curve(CC_pred, BB, BB_pred, iou_threshold)
    return np.sum(np.diff(recall) * precision[1:])

def filter_class(k, YY, CC_pred, BB, BB_pred):
    return ([cc[yy == k] for yy, cc in zip(YY, CC_pred)],
        [bb[yy == k] for yy, bb in zip(YY, BB)],
        [bb[yy == k] for yy, bb in zip(YY, BB_pred)])

def mAP(YY, CC_pred, BB, BB_pred, iou_threshold):
    # mAP = average for the classes
    classes = np.unique(YY)
    return np.mean([AP(*filter_class(k, YY, CC_pred, BB, BB_pred), iou_threshold) for k in classes])

def mAP_ious(YY, CC_pred, BB, BB_pred, iou_thresholds):
    return np.mean(mAP(YY, CC_pred, BB, BB_pred, th) for th in iou_thresholds)
