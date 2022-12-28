'''
Post-processing techniques to reduce the amount of superfluous bounding boxes, namely non-maximum suppression.
'''

import torch

def IoU(b1, b2):
    ''' Intersection over union utility. '''
    I = (min(b1[2], b2[2])-max(b1[0], b2[0])) * (min(b1[3], b2[3])-max(b1[1], b2[1]))
    U = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - I
    return I / U

def NMS(preds, lambda_nms=0.5):
    ''' Non-Maximum Suppression (NMS) algorithm. It is a popular post-processing algorithm to clean-up similar bounding boxes. '''
    scores = preds['scores']
    bboxes = preds['bboxes']
    ix = [[not any(  # discard if all conditions met
        bi != bj and scores[i][bj] > scores[i][bi] and
        IoU(bboxes[i][bi], bboxes[i][bj]) >= lambda_nms
        for bj in range(len(bboxes[i]))
    ) for bi in range(len(bboxes[i]))] for i in range(len(bboxes))]
    return {k: [vv[ii] for ii, vv in zip(ix, v)] for k, v in preds.items()}
