'''
Miscelaneous utilities for data handling.
'''

import torch

def collate_fn(batch):
    ''' The number of bounding boxes varies for each image, therefore the default PyTorch `collate` function (which creates the batches) must be replaced so that only images are turned into tensors. '''
    images = torch.stack([torch.as_tensor(d['image']) for d in batch])
    targets = {key: [torch.as_tensor(d[key]) for d in batch] for key in batch[0].keys()}
    # a bit of a hack: albumentations shapes empty bboxes as (0,), but it's more
    # convenient for them to be shape (0, 4).
    targets['bboxes'] = [bboxes.reshape(len(bboxes), 4) for bboxes in targets['bboxes']]
    return images, targets

def filter_grid(batch, grid_min, grid_max, keys=['bboxes', 'labels']):
    ''' Useful to filter objects when working with grids. The filter selects only objects when grid_min <= max(width, height) < grid_max. '''
    device = batch['bboxes'][0].device
    batch_sizes = [torch.maximum(bb[:, 2]-bb[:, 0], bb[:, 3]-bb[:, 1])
        if len(bb) else torch.tensor((), device=device) for bb in batch['bboxes']]
    ix = [(grid_min <= sizes) & (sizes < grid_max) for sizes in batch_sizes]
    return {key: [t[i] for t, i in zip(batch[key], ix)] for key in keys}
