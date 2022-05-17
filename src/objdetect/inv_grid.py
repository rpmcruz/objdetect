'''
Methods to convert our grid transformations back to the list format, used by the metrics and most plotting methods.
'''

import numpy as np

def InvScores():
    '''Inverts hasobjs grid to a list of scores.'''
    def f(ix, key, datum):
        return datum[key][0, ix]
    return f

def InvScoresWithClasses(classes_key):
    '''Inverts hasobjs grid, multiplies by the class probability, in order to produce a posterior probability.'''
    def f(ix, key, datum):
        return datum[key][0, ix] * datum[classes_key].max(0)[ix]
    return f

def InvRelBboxes():
    '''Inverts relative bounding boxes grid to a list of absolute bounding boxes.'''
    def f(ix, key, datum):
        bboxes = datum[key]
        _, h, w = bboxes.shape
        yy, xx = np.mgrid[0:h, 0:w]
        bboxes_offset = np.stack((
            xx/w-bboxes[0], yy/h-bboxes[1],
            bboxes[2]+xx/w, bboxes[3]+yy/h
        ), -1)
        return bboxes_offset[ix]
    return f

def InvCenterSizeBboxes():
    '''Inverts center and sizes grid to a list of bounding boxes.'''
    def f(ix, key, datum):
        bboxes = datum[key]
        _, h, w = bboxes.shape
        yy, xx = np.mgrid[0:h, 0:w]
        xc = (xx+bboxes[0])/w
        yc = (yy+bboxes[1])/h
        bw = np.exp(bboxes[2])
        bh = np.exp(bboxes[3])
        bboxes_offset = np.stack((
            xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2
        ), -1)
        return bboxes_offset[ix]
    return f

def InvClasses():
    '''Inverts the classes grid to a list of classes.'''
    def f(ix, key, datum):
        return datum[key].argmax(0)[ix]
    return f

def InvTransform(threshold_fn, inv_grid_dict):
    '''Applies the others methods to convert the grids into lists.'''
    def f(datum):
        ix = threshold_fn(datum)
        return {name: f(ix, name, datum) for name, f in inv_grid_dict.items()}
    return f

def MultiLevelInvTransform(threshold_fns, dependencies, inv_grid_dict):
    '''Same as `InvTransform()`, but useful for multi-level grids, where `dependencies` may be provided to specify how a final grid depends on each grid.'''
    def f(datum):
        ret = {}
        for i in range(len(threshold_fns)):
            ix = threshold_fns[i](datum)
            for name, f in inv_grid_dict.items():
                ret[name] = ret.get(name, []) + list(f(ix, dependencies[name][i], datum))
        return ret
    return f
