import torchvision
import torch
import numpy as np

class VOC(torch.utils.data.Dataset):
    nclasses = 20
    classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

    def __init__(self, root, fold, transform=None, download=False):
        super().__init__()
        fold = 'val' if fold == 'test' else fold
        self.ds = torchvision.datasets.VOCDetection(root, image_set=fold, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        image, xml = self.ds[i]
        image = np.array(image)
        objs = xml['annotation']['object']
        labels = [self.classes.index(o['name']) for o in objs]
        bboxes = torch.tensor([(
            float(o['bndbox']['xmin']), float(o['bndbox']['ymin']),
            float(o['bndbox']['xmax']), float(o['bndbox']['ymax']),
            ) for o in objs], dtype=torch.float32)
        bboxes = bboxes.reshape(len(bboxes), 4)
        d = {'image': image, 'bboxes': bboxes, 'labels': labels}
        if self.transform:
            d = self.transform(**d)
        return d

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import objdetect as od
    animals = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    voc = VOC(data_path, 'train', download=download)
    data = voc[0]
    plt.imshow(data['image'])
    od.draw.bboxes(data['bboxes'], labels=data['labels'])
    plt.show()
