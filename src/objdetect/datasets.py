from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

'''
Wrapper to common datasets. Since there can be several variables associated to
an image (the image itself, the bounding boxes, the classes, etc) we use a
dictionary to contain them -- we call one such dictionary a 'datum', and several
'data'. Each datum has at least have 'image' and 'bboxes'. Each bbox is a tuple
of type (xmin, ymin, xmax, ymax) with each value normalized.
'''

class VOCDetection(datasets.VOCDetection):
    labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root, fold, download, transforms, grid_transform):
        super().__init__(root, image_set=fold, download=download)
        self.od_transforms = transforms
        self.grid_transform = grid_transform

    def __getitem__(self, i):
        img, xml = super().__getitem__(i)
        objs = xml['annotation']['object']
        datum = {
            'image': np.array(img),
            'bboxes': [(
                float(o['bndbox']['xmin']) / img.size[0],
                float(o['bndbox']['ymin']) / img.size[1],
                float(o['bndbox']['xmax']) / img.size[0],
                float(o['bndbox']['ymax']) / img.size[1],
            ) for o in objs],
            'classes': np.array([self.labels.index(o['name']) for o in objs], np.int64),
        }
        if self.od_transforms:
            datum = self.od_transforms(**datum)
        if self.grid_transform:
            datum = self.grid_transform(datum)
        return datum

class DebugDataset(Dataset):  # used to debug models (loads only N images)
    def __init__(self, ds, N):
        super().__init__()
        self.ds = ds
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.ds[i]

if __name__ == '__main__':  # debug a dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import plot
    ds = globals()[args.dataset]('data', 'train', args.download, None, None)
    datum = ds[0]
    plt.imshow(datum['image'])
    plot.bboxes(datum['image'], datum['bboxes'])
    plt.show()
