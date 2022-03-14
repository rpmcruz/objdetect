import numpy as np

'''
In one-shot object detection, we need to transform the input onto a grid to
compare against the neural network output which also produces a grid. Our grid
shapes follow the dimensions: (N, Nf, Na, H, W) where Nf=number of features
(e.g., Nf=4 for bboxes) and Na=number of anchors (for consistency, even if there
are no anchors, this dimension exists, but has the value of 1).

These utilities transform the given datum (which is a dictionary, as defined in
datasets.py) with another dictionary containing 'confs_grid' and 'bboxes_grid',
possibly other keys too, to the dictionary. The original keys are not retained
to make it possible to use the default dataloader collate which converts the
datum onto tensors. An inverse function is also provided to convert the network
outputs back to the original domain.

If you do not use anchors, just specify anchors=None. For generability, the
shapes of the returned arrays will always contain one dimension of size 1, which corresponds to the anchors, even if none exists.
'''

class Transform:
    def __init__(self, grid_size, anchors, extra_keys=[]):
        self.grid_size = grid_size
        self.anchors = anchors
        self.extra_keys = extra_keys

    def __call__(self, datum):
        n_anchors = 1 if self.anchors is None else len(self.anchors)
        H_grid = np.zeros((1, n_anchors, *self.grid_size), np.float32)
        B_grid = np.zeros((4, n_anchors, *self.grid_size), np.float32)
        ret_datum = {'image': datum['image'], 'confs_grid': H_grid,
            'bboxes_grid': B_grid}

        # convert also any key that is in extra_keys. other keys, just pass them
        for key in datum.keys() - {'image', 'confs', 'bboxes'}:
            if key in self.extra_keys:
                ret_datum[key + '_grid'] = np.zeros((1, n_anchors, *self.grid_size), datum[key].dtype)
            else:
                ret_datum[key] = datum[key]

        for bi, b in enumerate(datum['bboxes']):
            xc = (b[0]+b[2]) / 2
            yc = (b[1]+b[3]) / 2
            if self.anchors is None:
                ai = 0
                size = (1, 1)
            else:  # which anchor?
                ai = ((self.anchors[:, 0]-xc)**2 + (self.anchors[:, 1]-yc)**2).mean(0).argmin()
                size = self.anchors[ai]
            gx = int(xc // (1/self.grid_size[0]))
            gy = int(yc // (1/self.grid_size[1]))
            offset_x = (xc % (1/self.grid_size[0])) * self.grid_size[0]
            offset_y = (yc % (1/self.grid_size[1])) * self.grid_size[1]
            H_grid[0, ai, gy, gx] = 1
            B_grid[0, ai, gy, gx] = offset_x
            B_grid[1, ai, gy, gx] = offset_y
            B_grid[2, ai, gy, gx] = np.log((b[2]-b[0]) / size[0])
            B_grid[3, ai, gy, gx] = np.log((b[3]-b[1]) / size[1])
            for key in self.extra_keys:
                ret_datum[key + '_grid'][0, ai, gy, gx] = datum[key][bi]
        return ret_datum

    def inv(self, data, confidence_threshold=0.5):
        # unlike the previous method, this receives a batch, not a single datum
        assert 'confs_grid' in data, 'Must contain at least one grid'
        if self.anchors is None:
            self.anchors = [(1, 1)]
        cell_size = (1 / self.grid_size[0], 1 / self.grid_size[1])
        ret = []
        for i in range(len(data['confs_grid'])):
            bboxes = []
            confs = []
            ret_datum = {'bboxes': bboxes, 'confs': confs}

            # convert also any key that is in extra_keys. other keys, just pass them
            for key in data.keys() - {'bboxes_grid', 'confs_grid'}:
                if key in self.extra_keys:
                    ret_datum[key[:-5]] = []
                else:
                    ret_datum[key] = data[key][i]

            ret.append(ret_datum)
            for gx in range(self.grid_size[0]):
                for gy in range(self.grid_size[1]):
                    for ai, anchor in enumerate(self.anchors):
                        # maybe we should multiply by class confidence here too
                        conf = data['confs_grid'][i, 0, ai, gy, gx]
                        if conf >= confidence_threshold:
                            offset_x, offset_y, log_w, log_h = data['bboxes_grid'][i, :, ai, gy, gx]
                            xc = (gx+offset_x)*cell_size[0]
                            yc = (gy+offset_y)*cell_size[1]
                            w = anchor[0]*np.exp(log_w)
                            h = anchor[1]*np.exp(log_h)
                            xmin = xc - w/2
                            ymin = yc - h/2
                            xmax = xmin + w
                            ymax = ymin + h
                            bboxes.append((xmin, ymin, xmax, ymax))
                            if 'classes_grid' in data:
                                # classes conversion is a special case
                                if data['classes_grid'].shape[1] == 1:
                                    _class = data['classes_grid'][i, 0, ai, gy, gx]
                                else:
                                    pclass = data['classes_grid'][i, :, ai, gy, gx].max()
                                    conf *= pclass
                                    _class = data['classes_grid'][i, :, ai, gy, gx].argmax()
                                ret_datum['classes'].append(int(_class))
                            for extra_key in extra_keys - {'classes_grid'}:
                                ret_datum[k[:-5]].append(data[k][i, 0, ai, gy, gx])
                            confs.append(conf)
        return ret
