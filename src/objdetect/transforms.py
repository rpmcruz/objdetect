'''
Converts lists or grids into a different space more appropriate to the task. Notice that some of these transformations require the input to be a grid, while others are more flexible.
'''

import torch

def offset_logsize_bboxes(data, grid_size, img_size, prior_size=(1, 1)):
    ''' Similar to [YOLOv3](https://arxiv.org/abs/1804.02767). Please notice this only makes sense if slices=slice_center_locations. '''
    gh, gw = grid_size
    ih, iw = img_size
    ph, pw = prior_size
    xc = (data[:, 0] + data[:, 2]) / 2
    yc = (data[:, 1] + data[:, 3]) / 2
    xo = (xc % (iw/gw)) * (gw/iw)
    yo = (yc % (ih/gh)) * (gh/ih)
    bw = torch.log((data[:, 2] - data[:, 0]) / pw)
    bh = torch.log((data[:, 3] - data[:, 1]) / ph)
    return torch.stack((xo, yo, bw, bh), 1)

def inv_offset_logsize_bboxes(grid, img_size, prior_size=(1, 1)):
    ''' Invert the grid created by the function with the same name. '''
    device = grid.device
    gh, gw = grid.shape[2:]
    ih, iw = img_size
    ph, pw = prior_size
    xx = torch.arange(0, gw, dtype=torch.float32, device=device)[None, :]
    yy = torch.arange(0, gh, dtype=torch.float32, device=device)[:, None]
    xc = (xx + grid[:, 0]) * (iw/gw)
    yc = (yy + grid[:, 1]) * (ih/gh)
    bw = pw * torch.exp(grid[:, 2])
    bh = ph * torch.exp(grid[:, 3])
    return torch.stack((xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2), 1)

def rel_bboxes(grid, img_size, corner_offset=0.5):
    ''' Converts the grid of bounding boxes to relative bounding boxes (see FCOS l*,t*,r*,b*), and vice-versa. If you use 0-1 normalized bounding boxes, then specify `img_size=(1,1)`. Notice that this function can be called to transform in both directions (absolute <=> relative). '''
    device = grid.device
    gh, gw = grid.shape[2:]
    ih, iw = img_size
    yy = (torch.arange(gh, dtype=torch.float32, device=device) + corner_offset) * (ih/gh)
    xx = (torch.arange(gw, dtype=torch.float32, device=device) + corner_offset) * (iw/gw)
    xx, yy = torch.meshgrid(yy, xx, indexing='xy')
    mesh_grid = torch.stack((xx, yy, xx, yy))[None]
    sign = torch.tensor((-1, -1, 1, 1), dtype=torch.float32, device=device)[None, :, None, None]
    return mesh_grid + sign*grid

def centerness(rel_bboxes):
    ''' Applies `centerness` as in the FCOS paper. Notice that the given `rel_bboxes` must already be in `rel_bboxes()` space. This function works if `rel_bboxes` is already the selected list or a grid. '''
    h = torch.minimum(rel_bboxes[:, 0], rel_bboxes[:, 2]) / torch.maximum(rel_bboxes[:, 0], rel_bboxes[:, 2])
    v = torch.minimum(rel_bboxes[:, 1], rel_bboxes[:, 3]) / torch.maximum(rel_bboxes[:, 1], rel_bboxes[:, 3])
    return torch.sqrt(h*v)[:, None]
