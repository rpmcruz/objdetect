from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import torch
import albumentations as A
import objdetect as od

# Each dataset must return a single "datum", which is a dictionary containing
# at least image and bboxes. Optionally, it can also contain classes and other
# attributes. In this package, we heavily rely on dictionaries to bind the data
# inputs, the model outputs and the loss functions.
# Each bounding box uses the format (xmin, ymin, xmax, ymax) with each value
# normalized [0,1].
download = False
tr = od.datasets.VOCDetection('data', 'train', download, None, None)
labels = od.datasets.VOCDetection.labels
datum = tr[4]
print(datum.keys())
# dict_keys(['image', 'bboxes', 'classes'])

# Plot the datum to see if everything looks good.
import matplotlib.pyplot as plt
plt.imshow(datum['image'])
od.plot.bboxes_with_classes(datum['image'], datum['bboxes'], datum['classes'], labels)
plt.show()

# In one-shot detection, the model receives the bounding boxes in the form of a
# grid. We provide routines to do this transformation and its inverse. Here we
# are going for a 8x8 grid and we are not going to use anchors. Please see
# example_train.py on how to use anchors. The shape of our grids are:
# (N, Nf, Na, H, W), where Nf are the features of the grid (for example, 4 for
# the bounding box) and Na the number of anchors (for consistency, even if no
# anchors are used, this value is 1).

grid_transform = od.grid.ToGridTransform((8, 8), None)
datum = grid_transform(datum)
print(datum.keys())
# dict_keys(['image', 'confs_grid', 'bboxes_grid', 'classes_grid'])

# We recommend applying grid transformations in the Dataset class itself. We
# provide some data augmentation routines, but you can use albumentations (like
# in our exampe_train.py) or any other package.
transform = od.aug.Resize((256, 256))
tr = od.datasets.VOCDetection('data', 'train', download, transform, grid_transform)

# The grid transformation is used for training and then you invert the grid
# back before plotting or applying metrics. The inversion function receives
# images in batch format since it is typically used on the network outputs, and
# returns a list with each datum.

inv_grid_transform = od.grid.BatchFromGridTransform(None)
batch = {k: v[None] for k, v in datum.items()}
inv_datum = inv_grid_transform(batch)[0]

import matplotlib.pyplot as plt
plt.imshow(datum['image'])
od.plot.grid_without_anchors(datum['image'], datum['confs_grid'], datum['bboxes_grid'])
plt.show()

import matplotlib.pyplot as plt
plt.imshow(datum['image'])
od.plot.bboxes_with_classes(datum['image'], inv_datum['bboxes'], inv_datum['classes'], labels, 'blue')
plt.show()

# To build the model we recommend defining the backbone and head separately. The
# head must output keys that match those from the grid transform (confs_grid,
# bboxes_grid, and possibly others, such as classes_grid).
# We provide simple models that you may use.

backbone = od.models.Backbone((256, 256, 3), (8, 8))
head = od.models.HeadWithClasses(backbone.n_outputs, 1, len(labels))
model = od.models.Model(backbone, head).cuda()

# We provide convenience functions for training. Losses must be a dictionary
# with the same keys as the previous ones.

transform = od.aug.Combine(
    od.aug.Resize((282, 282)), od.aug.RandomCrop((256, 256)),
    od.aug.RandomHflip(), od.aug.RandomBrightnessContrast()
)
tr = od.datasets.VOCDetection('data', 'train', download, transform, grid_transform)

tr = DataLoader(tr, 128, True)
opt = torch.optim.Adam(model.parameters())
losses = {
    'confs_grid': nn.BCEWithLogitsLoss(),
    'bboxes_grid': nn.MSELoss(),
    'classes_grid': nn.CrossEntropyLoss(),
}
od.loop.train(model, tr, opt, losses, 100)
torch.save(model, 'model.pth')

inv_grid_transform = od.grid.BatchFromGridTransform(None, 0.1)
inputs, preds = od.loop.evaluate(model, tr, inv_grid_transform)

import matplotlib.pyplot as plt
for i in range(12):
    plt.subplot(2, 6, i+1)
    plt.imshow(inputs[i]['image'])
    od.plot.bboxes_with_classes(inputs[i]['image'], inputs[i]['bboxes'], inputs[i]['classes'], labels, 'blue')
    od.plot.bboxes_with_classes(inputs[i]['image'], preds[i]['bboxes'], preds[i]['classes'], labels, 'green', '--')
plt.show()

# The framework also supports anchors. To compute the anchors, you may use our
# utility which uses KMeans. For example, if you want to find the best 9
# anchors.

tr = od.datasets.VOCDetection('data', 'train', False, None, None)
anchors = od.anchors.compute_clusters(tr, 9)
od.plot.anchors(anchors)
plt.show()

# Our framework also has common the common AP metric based on precision-recall,
# but it is not well tested.

precision, recall = od.metrics.precision_recall_curve(preds['confs'], inputs['bboxes'], preds['bboxes'], 0.5)
plt.plot(precision, recall)
plt.show()

print('AP:', od.metrics.AP(preds['confs'], inputs['bboxes'], preds['bboxes'], 0.5))
