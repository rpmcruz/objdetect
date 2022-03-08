from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import objdetect as od

DOWNLOAD = False  # use True on first usage
EPOCHS = 50
BATCH_SIZE = 32
N_ANCHORS = 9
IMAGE_SIZE = (256, 256, 3)
GRID_SIZE = (8, 8)

# you may use albumentations or your own function
transforms = A.Compose([
    A.Resize(int(IMAGE_SIZE[0]*1.1), int(IMAGE_SIZE[1]*1.1)),
    A.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(0, 1),
], bbox_params=A.BboxParams('albumentations', ['classes']))

# first, find anchors without augmentation
tr = od.datasets.VOCDetection('data', 'train', DOWNLOAD, None, None)
anchors = od.anchors.compute_clusters(tr, N_ANCHORS)
# then re-load dataset, now with the anchors
grid_transform = lambda datum: od.grid.bboxes_to_grids(datum, GRID_SIZE, anchors)
tr = od.datasets.VOCDetection('data', 'train', False, transforms, grid_transform)
labels = tr.labels

# create model
backbone = od.models.Backbone(IMAGE_SIZE, GRID_SIZE)
head = od.models.HeadWithClasses(backbone.n_outputs, N_ANCHORS, len(labels))
model = od.models.Model(backbone, head).cuda()

outputs = model(torch.rand(10, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1], device='cuda'))
print(summary(model, (10, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1])))
print('outputs:', [f'{k}: {v.shape}' for k, v in outputs.items()])

# train
tr = DataLoader(tr, BATCH_SIZE, True)
opt = torch.optim.Adam(model.parameters())
losses = {
    'confs_grid': nn.BCEWithLogitsLoss(),
    'bboxes_grid': nn.MSELoss(),
    'classes_grid': nn.CrossEntropyLoss(),
}
od.loop.train(model, tr, opt, losses, 1)

# save things
torch.save({'model': model, 'anchors': anchors}, 'model.pth')
