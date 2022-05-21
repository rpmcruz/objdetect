'''
Some common datasets. Each dataset produces a dictionary with the properties relative to each image. Each bbox is a tuple of type (xmin, ymin, xmax, ymax) with each value [0, 1] normalized.
'''

from torch.utils.data import Dataset
from torchvision import datasets
from skimage.io import imread
import numpy as np
import os

class VOCDetection(datasets.VOCDetection):
    '''The popular [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset, which contains 20 classes.'''

    labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root, fold, my_transform, download=False):
        super().__init__(root, image_set=fold, download=download)
        self.my_transform = my_transform

    def __getitem__(self, i):
        image, xml = super().__getitem__(i)
        objs = xml['annotation']['object']
        image = np.array(image)
        bboxes = np.array([(
            float(o['bndbox']['xmin']) / image.shape[1],
            float(o['bndbox']['ymin']) / image.shape[0],
            float(o['bndbox']['xmax']) / image.shape[1],
            float(o['bndbox']['ymax']) / image.shape[0],
        ) for o in objs], np.float32)
        classes = np.array([self.labels.index(o['name']) for o in objs], np.int64)
        datum = {'image': image, 'bboxes': bboxes, 'classes': classes}
        if self.my_transform:
            datum = self.my_transform(**datum)
        return datum


class CocoDetection(Dataset):
    '''The popular [COCO](https://cocodataset.org/) dataset from Microsoft, which contains 80 classes.'''

    def __init__(self, images_dir, ann_file, my_transform):
        self.my_transform = my_transform
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
            self.bboxes.setdefault(img_id, []).append(np.array((
                bbox[0]/size[0], bbox[1]/size[1],
                (bbox[0]+bbox[2])/size[0], (bbox[1]+bbox[3])/size[1]
            ), np.float32))
            self.classes.setdefault(img_id, []).append(class_)
        self.filenames = {img['id']: img['file_name'] for img in anns['images']}
        self.image_ids = list(self.bboxes.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        img = imread(os.path.join(self.images_dir, self.filenames[img_id]))
        if len(img.shape) == 2:
            img = gray2rgb(img)
        datum = {
            'image': img,
            'bboxes': self.bboxes[img_id],
            'classes': np.array(self.classes[img_id], np.int64)
        }
        if self.my_transform: 
            datum = self.my_transform(**datum)
        return datum

class KITTIDetection(Dataset):
    '''The [KITTI](http://www.cvlibs.net/datasets/kitti/) self-driving dataset.'''

    labels = ['Car', 'Cyclist', 'Pedestrian', 'Person_sitting', 'Tram', 'Truck', 'Van', 'Misc', 'DontCare']

    def __init__(self, root, fold, my_transform, exclude_labels={'Misc', 'DontCare'}):
        assert fold in ('train',)
        self.labels = [l for l in self.labels if l not in exclude_labels]
        self.root = os.path.join(root, 'kitti', 'object', 'training')
        self.files = os.listdir(os.path.join(self.root, 'image_2'))
        self.my_transform = my_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        image = imread(os.path.join(self.root, 'image_2', filename))
        lines = [l.split() for l in open(os.path.join(self.root, 'label_2', filename[:-3] + 'txt')).readlines()]
        lines = [l for l in lines if l[0] in self.labels]
        bboxes = np.array([(
            float(l[4])/image.shape[1], float(l[5])/image.shape[0],
            float(l[6])/image.shape[1], float(l[7])/image.shape[0]
        ) for l in lines], np.float32)
        classes = np.array([self.labels.index(l[0]) for l in lines], np.int64)
        datum = {'image': image, 'bboxes': bboxes, 'classes': classes}
        if self.my_transform:
            datum = self.my_transform(**datum)
        return datum

class FilterClass(Dataset):
    '''A convenience class that filters a class from the provided dataset.'''

    def __init__(self, ds, klass):
        self.ds = ds
        self.ix = [i for i in range(len(ds)) if klass in ds[i]['classes']]

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        return self.ds[self.ix[i]]
