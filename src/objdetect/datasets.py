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

class CocoDetection(data.Dataset):
    def __init__(self, images_dir, ann_file, transforms, grid_transform):
        self.od_transforms = transforms
        self.grid_transform = grid_transform
        self.images_dir = images_dir
        self.bboxes = {}
        self.classes = {}
        anns = json.load(open(ann_file))
        sizes = {img['id']: (img['width'], img['height']) for img in anns['images']}
        # the original ids have some holes, so convert them to 0..K classes
        orig_labels = {c['id']: c['name'] for c in anns['categories']}
        self.labels = list(orig_labels.values())
        for ann in anns['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']
            class_ = self.labels.index(orig_labels[ann['category_id']])
            size = sizes[img_id]
            self.bboxes.setdefault(img_id, []).append((
                bbox[0]/size[0], bbox[1]/size[1],
                (bbox[0]+bbox[2])/size[0], (bbox[1]+bbox[3])/size[1]
            ))
            self.classes.setdefault(img_id, []).append(class_)
        self.filenames = {img['id']: img['file_name'] for img in anns['images']}
        self.image_ids = list(self.bboxes.keys())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        img = imread(os.path.join(self.images_dir, self.filenames[img_id]))
        datum = {
            'image': img,
            'bboxes': self.bboxes[img_id],
            'classes': self.classes[img_id]
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
