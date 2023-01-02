import torchvision
import torch
import numpy as np

class VOC(torch.utils.data.Dataset):
    def __init__(self, root, fold, transform=None, download=True):
        super().__init__()
        fold = 'test' if fold == 'val' else fold
        self.ds = torchvision.datasets.VOCDetection(root, image_set=fold, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        image, xml = self.ds[i]
        image = np.array(image)
        objs = xml['annotation']['object']
        labels = [o['name'] for o in objs]
        bboxes = [(
            float(o['bndbox']['xmin']), float(o['bndbox']['ymin']),
            float(o['bndbox']['xmax']), float(o['bndbox']['ymax']),
            ) for o in objs]
        d = {'image': image, 'bboxes': bboxes, 'labels': labels}
        if self.transform:
            d = self.transform(**d)
        return d

class FilterClass(torch.utils.data.Dataset):
    def __init__(self, ds, whitelist):
        super().__init__()
        self.ds = ds
        self.ix = [i for i in range(len(ds)) if any(label in whitelist for label in ds[i]['labels'])]
        self.whitelist = whitelist

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        d = self.ds[self.ix[i]]
        d['bboxes'] = [bbox for label, bbox in zip(d['labels'], d['bboxes']) if label in self.whitelist]
        d['labels'] = [self.whitelist.index(label) for label in d['labels'] if label in self.whitelist]
        return d

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import objdetect as od
    animals = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    voc = VOC(data_path, 'train', download=download)
    voc = FilterClass(voc, animals)
    data = voc[0]
    plt.imshow(data['image'])
    od.draw.bboxes(data['bboxes'], labels=data['labels'])
    plt.show()
