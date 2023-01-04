'''
Utilities to create and filter objects based on anchors.
'''

import torch

def flatten_sizes(bboxes):
    ''' Flatten and return the sizes of each bounding box. '''
    return [(b[2], b[3]) for bb in bboxes for b in bb]

def compute_anchors(flatten_bboxes, n):
    ''' Uses K-Means to produce the top-`n` sizes for the given flatten bboxes. '''
    from sklearn.cluster import KMeans
    return KMeans(n, n_init='auto').fit(flatten_bboxes).cluster_centers_

def anchors_ious(bbox, anchors):
    ''' Same as ordinary IoU but only uses width & height. '''
    I = torch.minimum(anchors[:, 0], bbox[2]) * torch.minimum(anchors[:, 1], bbox[3])
    U = anchors[:, 0]*anchors[:, 1] + bbox[2]*bbox[3] - I
    return I / U
