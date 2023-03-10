import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--data', default='/data')
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

import torchvision
import torch
import objdetect as od
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
import data
from torchmetrics.detection.mean_ap import MeanAveragePrecision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA AUG #############################

img_size = (256, 256)
transform = A.Compose([
    A.Resize(*img_size),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

############################# DATA LOAD #############################

ds = data.VOC(args.data, 'val', transform)
ts = torch.utils.data.DataLoader(ds, 4, True, collate_fn=od.utils.collate_fn, num_workers=2, pin_memory=True)

########################## MODEL PARAMS ##########################

params = {'nclasses': ds.nclasses, 'img_size': img_size}

############################# MODEL #############################

model = torch.load(args.model, map_location=device)

############################# LOOP #############################

mean_ap = MeanAveragePrecision()

model.eval()
for images, targets in ts:
    with torch.no_grad():
        preds_grid = model(images.to(device))
        preds = model.post_process(preds_grid)
        preds = od.post.NMS(preds)
    # torchmetrics uses 'boxes' instead of 'bboxes'
    preds['boxes'] = preds.pop('bboxes')
    targets['boxes'] = targets.pop('bboxes')
    # torchmetrics requires the format list(dict(tensor)) instead of
    # dict(list(tensor)). that is, it requires batches of dictionaries.
    preds = [{k: v[i] for k, v in preds.items()} for i in range(len(preds['boxes']))]
    targets = [{k: v[i] for k, v in targets.items()} for i in range(len(targets['boxes']))]
    mean_ap.update(preds, targets)

print(args.model, mean_ap.compute())
