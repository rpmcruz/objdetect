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

def where(slicing, batch_bboxes, grid_size, img_size):
    ''' Following the given `slicing` function strategy, returns two things: a boolean grid mask where objects are located, and a list of indices of which bounding box was used. The mask can be used with `mask_select()` to convert a grid to a list, and the indices can be used with `indices_select()` to convert a list to another list matching the elements selected from the grid. '''
    device = batch_bboxes[0].device
    scale = torch.tensor([grid_size[1]/img_size[1],
        grid_size[0]/img_size[0]]*2, device=device)
    n = len(batch_bboxes)
    mask = torch.zeros((n, *grid_size), dtype=torch.bool, device=device)
    indices = torch.zeros((n, *grid_size), dtype=torch.int64, device=device)
    for i, bboxes in enumerate(batch_bboxes):
        for j, bbox in enumerate(bboxes):
            s = slicing(bbox*scale)
            yy = slice(int(s[1]), int(s[3]))
            xx = slice(int(s[0]), int(s[2]))
            mask[i, yy, xx] = True
            indices[i, yy, xx] = j
    indices = [i[m] for i, m in zip(indices, mask)]
    return mask, indices

def mask_select(mask, grid, keep_batches=False):
    ''' Select the grid components where `mask` is true. If `keep_batches=True`, then the result will be a list of tensors. '''
    grid = grid.permute(0, 2, 3, 1)  # NCHW => NHWC
    if keep_batches:
        return [g[m, :] for g, m in zip(grid, mask)]
    return grid[mask, :]

def indices_select(indices, data):
    ''' Returns a tensor with the data in the same form as the indices. '''
    return torch.cat([d[i] for d, i in zip(data, indices)])

def to_grid(mask, indices, batch_data):
    ''' Converts the given `batch_data` onto a grid, according to the `mask` and `indices` selected. Usually, you can avoid this function and work entirely in list space. '''
    device = mask.device
    channels = batch_data[0].shape[1]
    dtype = batch_data[0].dtype
    grid = torch.zeros((mask.shape[0], channels, *mask.shape[1:]), dtype=dtype, device=device)
    for g, m, d, i in zip(grid, mask, batch_data, indices):
        if len(i):
            g[:, m] = d[i].T
    return grid
