'''
In one-shot object detection, we need to transform the input onto a grid to compare against the neural network output which also produces a grid.
'''

import numpy as np

def SliceAcrossCeilBbox():
    '''Choose all grid locations that contain the entirety of the object (similar to FCOS).'''
    def f(h, w, bbox):
        yy = slice(int(np.ceil(bbox[1]*h)), int(np.ceil(bbox[3]*h)))
        xx = slice(int(np.ceil(bbox[0]*w)), int(np.ceil(bbox[2]*w)))
        return yy, xx
    return f

def SliceOnlyCenterBbox():
    '''Place the object only on the center location of the object (similar to YOLOv3).'''
    def f(h, w, bbox):
        xc = (bbox[0]+bbox[2])/2
        yc = (bbox[1]+bbox[3])/2
        x = int(xc*w)
        y = int(yc*h)
        return slice(y, y+1), slice(x, x+1)
    return f

###########################################################

def SizeFilter(min_cells, max_cells):
    '''Filter objects whose size is within [`min_cells`, `max_cells`] (the FCOS paper recommends [4, 8]).'''
    def f(h, w, bbox):
        xcells = (bbox[2]-bbox[0]) * w
        ycells = (bbox[3]-bbox[1]) * h
        return min_cells <= xcells <= max_cells
    return f

def compute_clusters(ds, n):
    '''Uses K-Means to produce the top-`n` sizes for the given dataset `ds`.'''
    from sklearn.cluster import KMeans
    BB = [d['bboxes'] for d in ds]
    BB = [(b[2], b[3]) for bb in BB for b in bb]
    return KMeans(n).fit(BB).cluster_centers_

def AnchorFilter(anchor, min_iou):
    '''Filter only objects with ≥IoU of a given `anchor`.'''
    from . import metrics
    def f(h, w, bbox):
        xc = (bbox[0]+bbox[2]) / 2
        yc = (bbox[1]+bbox[3]) / 2
        anchor_box = (xc-anchor[0]/2, yc-anchor[1]/2,
            xc+anchor[0]/2, yc+anchor[1]/2)
        return metrics.IoU(bbox, anchor_box) >= min_iou
    return f

###########################################################

def NewHasObj():
    '''Grid 1xhxw.'''
    def f(h, w):
        return np.zeros((1, h, w), np.float32)
    return f

def NewBboxes():
    '''Grid 4xhxw.'''
    def f(h, w):
        return np.zeros((4, h, w), np.float32)
    return f

def NewClasses():
    '''Grid hxw.'''
    def f(h, w):  # CrossEntropyLoss needs (h, w), not (1, h, w)
        return np.zeros((h, w), np.int64)
    return f

NewCenterness = NewHasObj

###########################################################

def SetHasObj():
    '''Sets 1 wherever the object is, according to the given slicing.'''
    def f(grid, yy, xx, datum, i):
        grid[:, yy, xx] = 1
    return f

def SetOffsetSizeBboxes():
    '''Sets 1 on the object center location. This is similar to YOLOv3: https://arxiv.org/abs/1804.02767. Please notice this only makes sense if slicing only one location per bbox (e.g. `SliceOnlyCenterBbox()`), because the offset is relative to the center of the current location.'''
    def f(grid, yy, xx, datum, i):
        bbox = datum['bboxes'][i]
        _, h, w = grid.shape
        xc = (bbox[0]+bbox[2])/2
        yc = (bbox[1]+bbox[3])/2
        grid[0, yy, xx] = (xc % (1/w))*w
        grid[1, yy, xx] = (yc % (1/h))*h
        grid[2, yy, xx] = np.log(bbox[2] - bbox[0])
        grid[3, yy, xx] = np.log(bbox[3] - bbox[1])
    return f

def SetOffsetSizeBboxesAnchor(anchor):
    '''Sets 1 on the object center location relative to the given anchor. See also `SetOffsetSizeBboxes()` for more details.'''
    ph, pw = anchor
    def f(grid, yy, xx, datum, i):
        bbox = datum['bboxes'][i]
        _, h, w = grid.shape
        xc = (bbox[0]+bbox[2])/2
        yc = (bbox[1]+bbox[3])/2
        grid[0, yy, xx] = (xc % (1/w))*w
        grid[1, yy, xx] = (yc % (1/h))*h
        grid[2, yy, xx] = np.log((bbox[2] - bbox[0])/pw)
        grid[3, yy, xx] = np.log((bbox[3] - bbox[1])/ph)
    return f

def SetRelBboxes():
    '''For each location, sets the distance between each size of the bounding box and each location (see the [FCOS paper](https://arxiv.org/abs/1904.01355)).'''
    def f(grid, yy, xx, datum, i):
        bbox = datum['bboxes'][i]
        _, h, w = grid.shape
        _yy, _xx = np.mgrid[yy.start:yy.stop, xx.start:xx.stop]
        grid[0, yy, xx] = (_xx/w) - bbox[0]
        grid[1, yy, xx] = (_yy/h) - bbox[1]
        grid[2, yy, xx] = bbox[2] - (_xx/w)
        grid[3, yy, xx] = bbox[3] - (_yy/h)
    return f

def SetClasses():
    '''Sets the respective class wherever the object is, according to the given slicing.'''
    def f(grid, yy, xx, datum, i):
        klass = datum['classes'][i]
        grid[yy, xx] = klass
    return f

def SetCenterness():
    '''Sets the distance to the object for each location (see the [FCOS paper](https://arxiv.org/abs/1904.01355)).'''
    def f(grid, yy, xx, datum, i):
        bbox = datum['bboxes'][i]
        _, h, w = grid.shape
        _yy, _xx = np.mgrid[yy.start:yy.stop, xx.start:xx.stop]
        L = (_xx/w) - bbox[0]
        T = (_yy/h) - bbox[1]
        R = bbox[2] - (_xx/w)
        B = bbox[3] - (_yy/h)
        factor1 = np.minimum(L, R)/np.maximum(L, R)
        factor2 = np.minimum(T, B)/np.maximum(T, B)
        return np.sqrt(factor1 * factor2)
    return f

###########################################################

def RemoveKeys(keys):
    '''Removing keys in the transform is useful whenever you want to reject some keys to go to the data loader.'''
    keys = frozenset(keys)
    def f(**datum):
        return {k: v for k, v in datum.items() if k not in keys}
    return f

def Transform(grid_size, filter_fn, slice_fn, new_grid_dict, set_grid_dict):
    '''Applies the others methods to build the respective grids.'''
    def f(**datum):
        h, w = grid_size
        grids = {name: f(h, w) for name, f in new_grid_dict.items()}
        for i, bbox in enumerate(datum['bboxes']):
            if filter_fn != None and not filter_fn(h, w, bbox):
                continue
            yy, xx = slice_fn(h, w, bbox)
            for name, f in set_grid_dict.items():
                f(grids[name], yy, xx, datum, i)
        return {**datum, **grids}
    return f

def merge_dicts(l):
    '''Utility function that concatenates a list of dictionaries into a single dictionary, useful for multi-level dictionaries.'''
    return {k: v for d in l for k, v in d.items()}
