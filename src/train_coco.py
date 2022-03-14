import argparse
parser = argparse.ArgumentParser()
parser.add_argument('coco_imgs_path')
parser.add_argument('coco_ann_filename')
args = parser.parse_args()

import objdetect as od
import torch
from torch.utils.data import DataLoader

# anchors
print('anchors...')
tr = od.datasets.CocoDetection(args.coco_imgs_path, args.coco_ann_filename, None, None)
anchors = od.anchors.compute_clusters(tr, 9)

# grid
print('grid...')
transforms = od.aug.Compose(
    od.aug.Resize((282, 282)), od.aug.RandomCrop((256, 256)),
    od.aug.RandomHflip(), od.aug.RandomBrightnessContrast(0.1, 0)
)
grid_transform = od.grid.Transform((8, 8), anchors, ['classes'])
tr = od.datasets.CocoDetection(args.coco_imgs_path, args.coco_ann_filename, transforms, grid_transform)

# define model
print('model...')
backbone = od.models.Backbone((256, 256, 3), (8, 8))
head = od.models.HeadWithClasses(backbone.n_outputs, len(anchors), len(tr.labels))
model = od.models.Model(backbone, head).cuda()

# train
print('train...')
tr = DataLoader(tr, 256, True, num_workers=2)
opt = torch.optim.Adam(model.parameters())
losses = {
    'confs_grid': torch.nn.BCEWithLogitsLoss(reduction='none'),
    'bboxes_grid': torch.nn.MSELoss(reduction='none'),
    'classes_grid': torch.nn.CrossEntropyLoss(reduction='none'),
}
scheduler = od.loop.ConvergeStop()
od.loop.train(model, tr, opt, losses, 100000, scheduler)
torch.save({'anchors': anchors, 'model': model}, 'model-coco.pth')
