'''
Miscelaneous utilities for data handling.
'''

import torch

def collate_fn(batch):
    ''' The number of bounding boxes varies for each image, therefore the default PyTorch `collate` function (which creates the batches) must be replaced so that only images are turned into tensors. '''
    images = torch.stack([torch.as_tensor(d['image']) for d in batch])
    targets = {key: [torch.as_tensor(d[key]) for d in batch] for key in batch[0].keys()}
    return images, targets

def filter_grid(data, grid_min, grid_max):
    ''' Useful to filter objects when working with grids. The filter selects only objects when grid_min <= max(width, height) < grid_max. '''
    ix = [[grid_min <= max(bb[2]-bb[0], bb[3]-bb[1]) < grid_max
        for bb in bbs] for bbs in data['bboxes']]
    return {key: [[e for i, e in zip(x, l) if i] for x, l in zip(ix, ll)] for key, ll in data.items()}
