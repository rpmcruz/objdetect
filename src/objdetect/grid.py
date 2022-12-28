'''
One-shot object detection require utilities to transform the input onto a grid to compare against the neural network output which also produces a grid.
'''

import torch

def slice_center(bbox):
    ''' Place the object only on the center location of the object (similar to YOLOv3). If the bounding boxes are normalized then you can use img_size=(1,1). '''
    cx = (bbox[0]+bbox[2]) // 2
    cy = (bbox[1]+bbox[3]) // 2
    return (cx, cy, cx+1, cy+1)

def slice_all(bbox):
    ''' Choose all grid locations that contain the object, even if only one pixel. '''
    return (
        torch.floor(bbox[0]), torch.floor(bbox[1]),
        torch.ceil(bbox[2]), torch.ceil(bbox[3])
    )

def slice_all_center(bbox):
    ''' Choose all grid locations where the center contains the object, as FCOS does, so that the bounding boxes offsets relative to the center are always positive. If the bounding boxes are normalized then you can use img_size=(1,1). '''
    return torch.round(bbox)

def slices(slicing, batch_bboxes, grid_size, img_size):
    ''' Grid with true on the locations where the object is according to the slicing `criterium`. If you use 0-1 normalized bboxes, then give `img_size=(1,1)`. '''
    device = batch_bboxes[0].device
    scale = torch.tensor([grid_size[1]/img_size[1], grid_size[0]/img_size[0]]*2, device=device)
    to_slice = lambda s: (slice(int(s[1]), int(s[3])), slice(int(s[0]), int(s[2])))
    return [[to_slice(slicing(bbox*scale)) for bbox in bboxes]
        for bboxes in batch_bboxes]

def where(batch_slices, grid_size, device=None):
    ''' Grid with 1 wherever the object is, 0 otherwise, according to the chosen slice strategy. '''
    n = len(batch_slices)
    grid = torch.zeros((n, *grid_size), dtype=torch.bool, device=device)
    for i, slices in enumerate(batch_slices):
        for yy, xx in slices:
            grid[i, yy, xx] = True
    return grid

def to_grid(batch_data, channels, batch_slices, grid_size, device=None):
    ''' Maps the `batch_data` onto a grid of size `grid_size`, according to the `batch_slices`. '''
    n = len(batch_slices)
    grid = torch.zeros((n, channels, *grid_size), dtype=torch.float32, device=device)
    for i, (slices, data) in enumerate(zip(batch_slices, batch_data)):
        for (yy, xx), d in zip(slices, data):
            grid[i, :, yy, xx] = d[:, None, None]
    return grid

def select(where, grid, keep_batches):
    ''' Select the grid components where `where` is true. If `keep_batches=True`, then the result will be a list of tensors. '''
    grid = grid.permute(0, 2, 3, 1)  # NCHW => NHWC
    if keep_batches:
        return [g[w, :] for w, g in zip(where, grid)]
    return grid[where, :]
