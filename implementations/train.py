import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['simple', 'fcos', 'yolo3'])
parser.add_argument('output')
parser.add_argument('--data', default='/data')
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

import torchvision
import torch
import objdetect as od
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
import importlib
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA AUG #############################

img_size = (256, 256)
transform = A.Compose([
    A.Resize(int(img_size[0]*1.1), int(img_size[1]*1.1)),
    A.RandomCrop(*img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

############################# DATA LOAD #############################

ds = data.VOC(args.data, 'train', transform)
tr = torch.utils.data.DataLoader(ds, 8, True, collate_fn=od.utils.collate_fn, num_workers=2, pin_memory=True)

############################# MODEL #############################

params = {'nclasses': ds.nclasses, 'img_size': img_size}
models = importlib.import_module(f'model_{args.model}')
if args.model == 'yolo3':
    params['anchors_per_scale'] = models.compute_anchors(ds)
model = models.Model(**params).to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for images, targets in tr:
        targets['bboxes'] = [bb.float() for bb in targets['bboxes']]
        targets = {k: [v.to(device) for v in l] for k, l in targets.items()}
        preds = model(images.to(device))
        loss_value = model.compute_loss(preds, targets)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss}')

torch.save(model.cpu(), args.output)
