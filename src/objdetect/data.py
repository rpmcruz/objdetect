'''
Some common datasets. Each dataset produces a dictionary with the properties relative to each image. Each bbox is a tuple of type (xmin, ymin, xmax, ymax) with each value [0, 1] normalized. Images are also returned as normalized tensors CxHxW.

Use `dict_transform` for transformations that could affect both images and bboxes or possibly other things. We provide `aug` for such dict transformations, but you may also use the albumentations package.
'''

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.io import read_image, ImageReadMode
import torch
import os

class VOCDetection(datasets.VOCDetection):
    '''The popular [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset, which contains 20 classes.'''

    labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root, fold, dict_transform=None, download=False):
        super().__init__(root, image_set=fold, download=download)
        self.dict_transform = dict_transform

    def __getitem__(self, i):
        image, xml = super().__getitem__(i)
        objs = xml['annotation']['object']
        image = TF.pil_to_tensor(image)/255
        bboxes = torch.tensor([(
            float(o['bndbox']['xmin']) / image.shape[2],
            float(o['bndbox']['ymin']) / image.shape[1],
            float(o['bndbox']['xmax']) / image.shape[2],
            float(o['bndbox']['ymax']) / image.shape[1],
        ) for o in objs])
        classes = torch.tensor([self.labels.index(o['name']) for o in objs])
        datum = {'image': image, 'bboxes': bboxes, 'classes': classes}
        if self.dict_transform:
            datum = self.dict_transform(**datum)
        return datum

class CocoDetection(Dataset):
    '''The popular [COCO](https://cocodataset.org/) dataset from Microsoft, which contains 80 classes.'''

    def __init__(self, images_dir, ann_file, dict_transform=None):
        self.images_dir = images_dir
        self.dict_transform = dict_transform
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
        return len(self.image_ids)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        img = read_image(os.path.join(self.images_dir, self.filenames[img_id]), ImageReadMode.RGB)/255
        datum = {
            'image': img,
            'bboxes': torch.tensor(self.bboxes[img_id]),
            'classes': torch.tensor(self.classes[img_id])
        }
        if self.dict_transform:
            datum = self.dict_transform(**datum)
        return datum

class KITTIDetection(Dataset):
    '''The [KITTI](http://www.cvlibs.net/datasets/kitti/) self-driving dataset.'''

    labels = ['Car', 'Cyclist', 'Pedestrian', 'Person_sitting', 'Tram', 'Truck', 'Van', 'Misc', 'DontCare']

    def __init__(self, root, fold, dict_transform=None, exclude_labels={'Misc', 'DontCare'}):
        assert fold in ('train',)
        self.dict_transform = dict_transform
        self.labels = [l for l in self.labels if l not in exclude_labels]
        self.root = os.path.join(root, 'kitti', 'object', 'training')
        self.files = os.listdir(os.path.join(self.root, 'image_2'))
        self.my_transform = my_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        image = read_image(os.path.join(self.root, 'image_2', filename))/255
        lines = [l.split() for l in open(os.path.join(self.root, 'label_2', filename[:-3] + 'txt')).readlines()]
        lines = [l for l in lines if l[0] in self.labels]
        bboxes = torch.tensor([(
            float(l[4])/image.shape[2], float(l[5])/image.shape[1],
            float(l[6])/image.shape[2], float(l[7])/image.shape[1]
        ) for l in lines])
        classes = torch.tensor([self.labels.index(l[0]) for l in lines])
        datum = {'image': image, 'bboxes': bboxes, 'classes': classes}
        if self.dict_transform:
            datum = self.dict_transform(**datum)
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

def collate_fn(batch):
    imgs = torch.stack([d['image'] for d in batch])
    targets = {key: [d[key] for d in batch] for key in batch[0].keys()}
    return imgs, targets
