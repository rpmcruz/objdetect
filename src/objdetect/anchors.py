'''
Utilities to create and filter objects based on anchors.
'''

def compute_anchors(ds, n):
    '''Uses K-Means to produce the top-`n` sizes for the given dataset `ds`.'''
    from sklearn.cluster import KMeans
    BB = [d['bboxes'] for d in ds]
    BB = [(b[2], b[3]) for bb in BB for b in bb]
    return KMeans(n).fit(BB).cluster_centers_

def fits_anchor(anchor, min_iou, bboxes):
    '''Returns true to any bboxes that fit the anchor (i.e., with iou ≥ min_iou).'''
    from . import metrics
    def f(bbox):
        xc = (bbox[0]+bbox[2]) / 2
        yc = (bbox[1]+bbox[3]) / 2
        anchor_box = (xc-anchor[0]/2, yc-anchor[1]/2,
            xc+anchor[0]/2, yc+anchor[1]/2)
        return metrics.IoU(bbox, anchor_box) >= min_iou
    return [f(bbox) for bbox in bboxes]
