'''
In one-shot object detection, we need to transform the input onto a grid to compare against the neural network output which also produces a grid.
'''

import torch

def slices_center_locations(h, w, batch_bboxes):
    '''Place the object only on the center location of the object (similar to YOLOv3).'''
    batch_yy = ((int(((bbox[1]+bbox[3])/2)*h) for bbox in bboxes) for bboxes in batch_bboxes)
    batch_xx = ((int(((bbox[0]+bbox[2])/2)*w) for bbox in bboxes) for bboxes in batch_bboxes)
    return [[(slice(y, y+1), slice(x, x+1)) for y, x in zip(yy, xx)] for yy, xx in zip(batch_yy, batch_xx)]

def slices_all_locations(h, w, batch_bboxes):
    '''Choose all grid locations that contain the entirety of the object, even if partially.'''
    return [[(
        slice(int(torch.floor(bbox[1]*h)), int(torch.ceil(bbox[3]*h))),
        slice(int(torch.floor(bbox[0]*w)), int(torch.ceil(bbox[2]*w))),
    ) for bbox in bboxes] for bboxes in batch_bboxes]

def scores(h, w, batch_slices):
    '''Grid with 1 wherever the object is, 0 otherwise, according to the chosen slice strategy.'''
    n = len(batch_slices)
    grid = torch.zeros((n, 1, h, w), dtype=torch.float32)
    for i, slices in enumerate(batch_slices):
        for yy, xx in slices:
            grid[i, :, yy, xx] = 1
    return grid

def inv_scores(hasobjs, scores):
    '''Invert the grid created by the function with the same name.'''
    assert hasobjs.dtype is torch.bool, 'Hasobjs must be a boolean grid'
    return scores[hasobjs]

def offset_logsize_bboxes(h, w, batch_slices, batch_bboxes):
    '''Similar to [YOLOv3](https://arxiv.org/abs/1804.02767). Please notice this only makes sense if slices=slice_center_locations.'''
    n = len(batch_slices)
    grid = torch.zeros((n, 4, h, w), dtype=torch.float32)
    for i, (slices, bboxes) in enumerate(zip(batch_slices, batch_bboxes)):
        for (yy, xx), bbox in zip(slices, bboxes):
            xc = (bbox[0]+bbox[2])/2
            yc = (bbox[1]+bbox[3])/2
            grid[i, 0, yy, xx] = (xc % (1/w))*w
            grid[i, 1, yy, xx] = (yc % (1/h))*h
            grid[i, 2, yy, xx] = torch.log(torch.as_tensor(bbox[2] - bbox[0]))
            grid[i, 3, yy, xx] = torch.log(torch.as_tensor(bbox[3] - bbox[1]))
    return grid

def inv_offset_logsize_bboxes(hasobjs, bboxes):
    '''Invert the grid created by the function with the same name.'''
    assert hasobjs.dtype is torch.bool, 'Hasobjs must be a boolean grid'
    n, _, h, w = hasobjs.shape
    xx = torch.arange(0, w, dtype=torch.float32)[None, :]
    yy = torch.arange(0, h, dtype=torch.float32)[:, None]
    xc = (xx+bboxes[:, 0])/w
    yc = (yy+bboxes[:, 1])/h
    bw = torch.exp(bboxes[:, 2])
    bh = torch.exp(bboxes[:, 3])
    bboxes_offset = torch.stack((
        xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2
    ), -1)
    return [bb[h[0]] for h, bb in zip(hasobjs, bboxes_offset)]

def relative_bboxes(h, w, batch_slices, batch_bboxes):
    '''For each location, sets the distance between each size of the bounding box and each location (see the [FCOS paper](https://arxiv.org/abs/1904.01355)).'''
    n = len(batch_slices)
    grid = torch.zeros((n, 4, h, w), dtype=torch.float32)
    for i, (slices, bboxes) in enumerate(zip(batch_slices, batch_bboxes)):
        for (yy, xx), bbox in zip(slices, bboxes):
            _xx = torch.arange(xx.start, xx.stop, dtype=torch.float32)[None, :]
            _yy = torch.arange(yy.start, yy.stop, dtype=torch.float32)[:, None]
            grid[i, 0, yy, xx] = (_xx/w) - bbox[0]
            grid[i, 1, yy, xx] = (_yy/h) - bbox[1]
            grid[i, 2, yy, xx] = bbox[2] - (_xx/w)
            grid[i, 3, yy, xx] = bbox[3] - (_yy/h)
    return grid

def inv_relative_bboxes(hasobjs, bboxes):
    '''Invert the grid created by the function with the same name.'''
    assert hasobjs.dtype is torch.bool, 'Hasobjs must be a boolean grid'
    _, _, h, w = hasobjs.shape
    xx = torch.arange(0, w, dtype=torch.float32)[None, :]
    yy = torch.arange(0, h, dtype=torch.float32)[:, None]
    bboxes_offset = torch.stack((
        xx/w-bboxes[:, 0], yy/h-bboxes[:, 1],
        bboxes[:, 2]+xx/w, bboxes[:, 3]+yy/h
    ), -1)
    return [bb[h[0]] for h, bb in zip(hasobjs, bboxes_offset)]

def classes(h, w, batch_slices, batch_classes):
    '''Sets the respective class wherever the object is, according to the given slicing.'''
    n = len(batch_slices)
    grid = torch.zeros((n, h, w), dtype=torch.int64)
    for i, (slices, classes) in enumerate(zip(batch_slices, batch_classes)):
        for (yy, xx), klass in zip(slices, classes):
            grid[i, yy, xx] = klass
    return grid

def inv_classes(hasobjs, classes):
    '''Invert the grid created by the function with the same name.'''
    assert hasobjs.dtype is torch.bool, 'Hasobjs must be a boolean grid'
    return [kk[h[0]] for h, kk in zip(hasobjs, classes)]

if __name__ == '__main__':  # DEBUG
    import matplotlib.pyplot as plt
    import data, aug, plot
    ds = data.VOCDetection('/data', 'train', aug.Resize(256, 256))
    imgs, targets = data.collate_fn([ds[i] for i in range(5)])
    # debug list => grid
    slices = slices_center_locations(8, 8, targets['bboxes'])
    scores_grid = scores(8, 8, slices)
    bboxes_grid = offset_logsize_bboxes(8, 8, slices, targets['bboxes'])
    classes_grid = classes(8, 8, slices, targets['classes'])
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plot.image(imgs[i])
        plot.grid_bools(imgs[i], scores_grid[i, 0])
        plot.grid_text(imgs[i], classes_grid[i]+scores_grid[i, 0], int)
        plot.bboxes(imgs[i], ds[i]['bboxes'])
    plt.suptitle('debug list => grid')
    plt.show()
    # debug list => grid => list
    hasobjs_grid = scores_grid >= 0.5
    scores_list = inv_scores(hasobjs_grid, scores_grid)
    bboxes_list = inv_offset_logsize_bboxes(hasobjs_grid, bboxes_grid)
    classes_list = inv_classes(hasobjs_grid, classes_grid)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plot.image(imgs[i])
        plot.bboxes(imgs[i], bboxes_list[i])
        plot.bboxes(imgs[i], ds[i]['bboxes'], ec='green')
        plot.classes(imgs[i], bboxes_list[i], classes_list[i])
    plt.suptitle('debug list => grid => list')
    plt.show()
