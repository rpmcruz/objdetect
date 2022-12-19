'''
Miscelaneous utilities, for plotting or data handling.
'''

def plot(bboxes, size=(1, 1), labels=None, color='blue', linestyle='-', grid=None):
    '''Draws the given bounding boxes and (if provided) labels. Size is used to convert relative bboxes to absolute bboxes (size=(1, 1) if your bboxes are already absolute).'''
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    h, w = size
    if grid:
        plt.vlines(np.linspace(0, w, grid[1]+1), 0, h, color='gray', lw=1)
        plt.hlines(np.linspace(0, h, grid[0]+1), 0, w, color='gray', lw=1)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        plt.gca().add_patch(matplotlib.patches.Rectangle(
            (xmin*w, ymin*h), (xmax-xmin)*w, (ymax-ymin)*h,
            lw=2, ls=linestyle, ec=color, fc='none'))
        plt.text(xmin*w, ymin*h, str(labels[i]), c=color)

def collate_fn(batch):
    '''The number of bounding boxes varies for each image, therefore the default PyTorch `collate` function (which creates the batches) must be replaced so that only images are turned into tensors.'''
    import torch
    imgs = torch.stack([torch.as_tensor(d['image']) for d in batch])
    targets = {key: [torch.as_tensor(d[key]) for d in batch] for key in batch[0].keys()}
    return imgs, targets

def filter_grid(data, grid_min, grid_max):
    '''Useful to filter objects when working with grids. The filter selects only objects when grid_min <= max(width, height) < grid_max.'''
    ix = [[grid_min <= max(bb[2]-bb[0], bb[3]-bb[1]) < grid_max
        for bb in bbs] for bbs in data['bboxes']]
    return {key: [[e for i, e in zip(x, l) if i] for x, l in zip(ix, ll)] for key, ll in data.items()}
