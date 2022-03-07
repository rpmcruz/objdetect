import numpy as np

'''
In one-shot object detection, we need to transform the input onto a grid to
compare against the neural network output which also produces a grid.

These utilities transform the given datum (which is a dictionary, as defined in
datasets.py) and adds 'confs_grid' and 'bboxes_grid', possibly other keys too,
to the dictionary. The same dictionary is then returned. An inverse function is
also provided to convert the network outputs back to the original domain.

If you do not use anchors, just specify anchors=None. For generability, the
shapes of the returned arrays will always contain one dimension of size 1, which corresponds to the anchors, even if none exists.
'''

def bboxes_to_grids(datum, grid_size, anchors):
    n_anchors = 1 if anchors is None else len(anchors)
    H_grid = np.zeros((n_anchors, 1, *grid_size), bool)
    B_grid = np.zeros((n_anchors, 4, *grid_size), np.float32)
    datum['confs_grid'] = H_grid
    datum['bboxes_grid'] = B_grid
    if 'classes' in datum:
        C_grid = np.zeros((n_anchors, 1, *grid_size), np.float32)
        datum['classes_grid'] = C_grid
    for bi, b in enumerate(datum['bboxes']):
        xc = (b[0]+b[2]) / 2
        yc = (b[1]+b[3]) / 2
        if anchors is None:
            ai = 0
            size = (1, 1)
        else:  # which anchor?
            ai = ((anchors[:, 0]-xc)**2 + (anchors[:, 1]-yc)**2).mean(0).argmin()
            size = anchors[ai]
        i = int(xc // (1/grid_size[0]))
        j = int(yc // (1/grid_size[1]))
        offset_x = (xc % (1/grid_size[0])) * grid_size[0]
        offset_y = (yc % (1/grid_size[1])) * grid_size[1]
        H_grid[ai, 0, j, i] = True
        B_grid[ai, 0, j, i] = offset_x
        B_grid[ai, 1, j, i] = offset_y
        B_grid[ai, 2, j, i] = np.log((b[2]-b[0]) / size[0])
        B_grid[ai, 3, j, i] = np.log((b[3]-b[1]) / size[1])
        if 'classes' in datum:
            C_grid[ai, j, i] = datum['classes'][bi]
    return datum

def batch_grids_to_bboxes(data, image_size, anchors):
    assert 'confs_grid' in data, 'Must contain at least one grid'
    if anchors is None:
        anchors = [(1, 1)]
    grid_size = confs_grid.shape[-2:]
    for i in range(len(data['image'])):
        for gi in range(grid_size[1]):
            for gj in range(grid_size[0]):
                for ai in range(len(anchors)):
                    if grids['confs'][i, ai, gj, gi] >= 0.5:
                        offset_x, offset_y, log_w, log_h = grids['bboxes'][i, ai, :, gj, gi]
                        plt.scatter((gi+offset_x)*32, (gj+offset_y)*32, color='black')
                        xc, yc = (gi+offset_x)*32, (gj+offset_y)*32
                        w, h = anchors[ai][0]*np.exp(log_w)*image_size[0], anchors[ai][1]*np.exp(log_h)*image_size[1]
                        plt.gca().add_patch(patches.Rectangle((xc-w/2, yc-h/2), w, h, edgecolor='g', facecolor='none'))
