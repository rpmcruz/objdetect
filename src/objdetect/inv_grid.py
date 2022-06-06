'''
Methods to convert our grid transformations back to the list format, used by the metrics and most plotting methods.
'''

import torch

def InvScores():
    '''Inverts scores grid to a list of scores.'''
    def f(ix, key, datum):
        return datum[key][0, ix]
    return f

def InvScoresWithClasses(classes_key):
    '''Inverts scores grid, multiplies by the class probability, in order to produce a posterior probability.'''
    def f(ix, key, datum):
        return datum[key][0, ix] * datum[classes_key].max(0)[ix]
    return f

def InvRelBboxes():
    '''Inverts relative bounding boxes grid to a list of absolute bounding boxes.'''
    def f(ix, key, datum):
        bboxes = datum[key]
        _, h, w = bboxes.shape
        xx = torch.arange(0, w, dtype=torch.float32)[None, :]
        yy = torch.arange(0, h, dtype=torch.float32)[:, None]
        bboxes_offset = torch.stack((
            xx/w-bboxes[0], yy/h-bboxes[1],
            bboxes[2]+xx/w, bboxes[3]+yy/h
        ), -1)
        return bboxes_offset[ix]
    return f

def InvOffsetSizeBboxes():
    '''Inverts center and sizes grid to a list of bounding boxes.'''
    def f(ix, key, datum):
        bboxes = datum[key]
        _, h, w = bboxes.shape
        xx = torch.arange(0, w, dtype=torch.float32)[None, :]
        yy = torch.arange(0, h, dtype=torch.float32)[:, None]
        xc = (xx+bboxes[0])/w
        yc = (yy+bboxes[1])/h
        bw = torch.exp(bboxes[2])
        bh = torch.exp(bboxes[3])
        bboxes_offset = torch.stack((
            xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2
        ), -1)
        return bboxes_offset[ix]
    return f

def InvOffsetSizeBboxesAnchor(anchors):
    '''Similar to `InvOffsetSizeBboxes()` but supporting anchors.'''
    import re
    pattern = re.compile(r'(\d+)$')
    def f(ix, key, datum):
        ph, pw = anchors[int(pattern.search(key).group(1))]  # a bit ugly
        bboxes = datum[key]
        _, h, w = bboxes.shape
        xx = torch.arange(0, w, dtype=torch.float32)[None, :]
        yy = torch.arange(0, h, dtype=torch.float32)[:, None]
        xc = (xx+bboxes[0])/w
        yc = (yy+bboxes[1])/h
        bw = pw * torch.exp(bboxes[2])
        bh = ph * torch.exp(bboxes[3])
        bboxes_offset = torch.stack((
            xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2
        ), -1)
        return bboxes_offset[ix]
    return f

def InvClasses():
    '''Inverts the classes grid to a list of classes.'''
    def f(ix, key, datum):
        if ix.shape == datum[key].shape:
            # special case when classes are not probabilities (e.g. inputs)
            return datum[key][ix]
        return datum[key].argmax(0)[ix]
    return f

def InvTransform(threshold_fn, inv_grid_dict):
    '''Applies the others methods to convert the grids into lists.'''
    def f(data):
        return [{name: f(threshold_fn(d), name, d) for name, f in inv_grid_dict.items()} for d in data]
    return f

def inv_collate(data):
    '''Reverses torch's collate. Converts back to list of dictionaries.'''
    n = len(data[next(iter(data))])
    return [{k: v[i] for k, v in data.items()} for i in range(n)]

def MultiLevelInvTransform(threshold_fns, dependencies, inv_grid_dict):
    '''Same as `InvTransform()`, but useful for multi-level grids, where `dependencies` may be provided to specify how a final grid depends on each grid.'''
    def f(data):
        ret = [None] * len(data)
        for di, datum in enumerate(data):
            ret[di] = ret_dict = {}
            for i, th in enumerate(threshold_fns):
                ix = th(datum)
                for name, f in inv_grid_dict.items():
                    ret_dict[name] = ret_dict.get(name, []) + list(f(ix, dependencies[name][i], datum))
        return ret
    return f
