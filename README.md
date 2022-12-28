# objdetect
Lightweight and versatile one-stage object detection toolkit.

## Introduction

I am a post-doc at FEUP (University of Porto) working on perception for autonomous driving ([THEIA project](https://noticias.up.pt/u-porto-bosch-projeto-de-investigacao-28-milhoes-de-euros/)). I developed this one-stage object detection toolkit because existing frameworks, such as [detectron2](https://github.com/facebookresearch/detectron2), are either for two-stage models or are not versatile and simple enough to adapt for new models. At the very least, I hope this package is educational for someone learning object detection. Contact: [Ricardo Cruz](mailto:rpcruz@fe.up.pt).

Functionality:
* Grid and feature transformations.
* Support for anchors and multiple grids.
* Utilities such as non-maximum suppression, plotting, evaluation metrics.

## Install

```
pip3 install git+https://github.com/rpmcruz/objdetect.git
```

## API

The package is divided into the following components:

* [`anchors`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/anchors.html): Utilities to create and filter objects based on anchors.
* [`draw`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/draw.html): Simple primitives to draw the bounding boxes.
* [`grid`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/grid.html): One-shot object detection require utilities to transform the input onto a grid to compare against the neural network output which also produces a grid.
* [`metrics`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/metrics.html): Implementation of Precision-Recall and AP metrics. This module is not yet fully tested, we recommend using torchmetrics.
* [`post`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/post.html): Post-processing techniques to reduce the amount of superfluous bounding boxes, namely non-maximum suppression.
* [`transforms`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/transforms.html): Converts lists or grids into a different space more appropriate to the task.
* [`utils`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/utils.html): Miscelaneous utilities for data handling.

## Getting Started

A [notebook example](https://github.com/rpmcruz/objdetect/blob/main/src/example.ipynb) is provided in the `src` folder which provides boiler-plate code to get you started. The notebook is mainly inspired by [FCOS](https://arxiv.org/abs/1904.01355). Following the PyTorch tradition, this package does not try to do too much behind the scenes.

```python
import objdetect as od
```

**Data:** For augmentation and data loading, we recommend using existing packages. [Albumentations](https://albumentations.ai/) provides data augmentation methods for object detection. TorchVision provides [dataset code](https://pytorch.org/vision/stable/datasets.html#image-detection-or-segmentation) for some popular object detection datasets. See the notebook for an example.

You may any format you wish for the bounding boxes. In the notebook, we use absolute x1y1x2y2 like pascalvoc. If you use 0-1 normalization (albumentations format) then you should specify `img_size=(1,1)` to the functions that require it.

Since the number of bounding boxes varies for each image, the normal PyTorch code that converts data into batches does not work. We need to specify a `collate` function for how the batches should be created.

```python
tr = torch.utils.data.DataLoader(ds, 16, True, collate_fn=od.utils.collate_fn)
```

**Grid:** When working with one-stage detection, we first need to represent the objects inside a given image as a grid (or multiple grids). You can do so during the data augmentation pipeline (which takes advantage of the DataLoader parallelization), but it might be simpler to do so inside the training loop (as done below in the `Grid` class).

Notice that slicing and how bounding boxes are setup changes greatly between models. Models like [YOLOv3](https://arxiv.org/abs/1804.02767) use a grid where each object occupies a single location (the center). Other models such as [FCOS](https://arxiv.org/abs/1904.01355) place each object on all locations as long as the center is contained. For that reason, we have slicing functions that convert the original bounding box onto grid-coordinates of where the object is contained.

The two important set of functions are:

* **Convert to grid [(N,4)] => (N,C,H,W)**
    * `od.grid.slices()` returns a list of slices for each bounding box.
    * `od.grid.to_grid()` converts the slices and some data (this data can be bounding boxes, classes, etc) to a grid.
* **Convert back to lists (N,C,H,W) => (N,4)**
    * `od.grid.where()` converts the slices onto a boolean grid indicating where the object is there. This function is used to help convert the grid back to lists, in conjunction with the following function.
    * `od.grid.select()` converts the grid to the original list. This list can contain batches when doing evaluation (`keep_batches=True`) or a single list during training (`keep_batches=False`).

**Transforms:** Such as slicing varies according to the model, so do the features required by the model. Some transformation routines are provided to convert grids and compute things such as offsets (YOLOv3), relative coordinates and centerness (FCOS), etc.

**Model:** We use the PyTorch philosophy of having the training loop done by the programmer. Here we provide some boiler-plate code of how to do so. We will create the model in the following picture.

![](src/model.svg)

The model is inspired by [FCOS](https://arxiv.org/abs/1904.01355), but without multi-grid support. However, we separate the `Grid` and `Model` classes to make it intuitive for you to add multiple grids if you wish to do so. (In such a case, you may use `od.utils.filter_grid()` to filter the bounding boxes associated to each grid.)

Notice that unlike the object detection models bundled with torchvision (e.g. [FCOS](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn)), the behavior of this boiler-plate code does **not** changes if in `train` or `eval` mode because we prefer [Python principle](https://peps.python.org/pep-0020/) "explicit is better than implicit". We recommend using an extra `post_process()` function in your model for that. But feel free to change that.

```python
bboxes_loss = torchvision.ops.generalized_box_iou_loss
centerness_loss = torch.nn.BCEWithLogitsLoss()
labels_loss = torchvision.ops.sigmoid_focal_loss

class Grid(torch.nn.Module):
    def __init__(self, in_channels, nclasses, img_size):
        super().__init__()
        self.img_size = img_size
        # like FCOS, we do not have a dedicated 'scores' prediction. it's just
        # the argmax of the classes.
        self.classes = torch.nn.Conv2d(in_channels, nclasses, 1)
        self.bboxes = torch.nn.Conv2d(in_channels, 4, 1)
        self.centerness = torch.nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        bboxes = torch.exp(self.bboxes(x))
        bboxes = od.transforms.rel_bboxes(bboxes, self.img_size)
        return {'labels': self.classes(x), 'bboxes': bboxes,
            'centerness': self.centerness(x)}

    def post_process(self, preds, threshold=0.05):
        scores, labels = torch.sigmoid(preds['labels']).max(1, keepdim=True)
        centerness = torch.sigmoid(preds['centerness'])
        ix = scores[:, 0] >= threshold
        # as in FCOS, centerness will help NMS choose the best bbox.
        scores = scores * centerness
        bboxes = preds['bboxes']
        return {
            'scores': od.grid.select(ix, scores, True),
            'bboxes': od.grid.select(ix, bboxes, True),
            'labels': od.grid.select(ix, labels, True),
        }

    def compute_loss(self, preds, targets):
        device = preds['bboxes'].device
        grid_size = preds['bboxes'].shape[2:]
        # convert everything to a grid
        slices = od.grid.slices(od.grid.slice_all_center, targets['bboxes'],
            grid_size, self.img_size)
        ix = od.grid.where(slices, grid_size, device)
        target_bboxes = od.grid.to_grid(targets['bboxes'], 4, slices, grid_size, device)
        target_bboxes_rel = od.transforms.rel_bboxes(target_bboxes, self.img_size)
        target_labels = [l[:, None] for l in targets['labels']]
        target_labels = od.grid.to_grid(target_labels, 1, slices, grid_size, device)
        # select what we want from the grid
        target_bboxes = od.grid.select(ix, target_bboxes, False)
        target_bboxes_rel = od.grid.select(ix, target_bboxes_rel, False)
        target_labels = od.grid.select(ix, target_labels, False)
        pred_bboxes = od.grid.select(ix, preds['bboxes'], False)
        pred_labels = od.grid.select(ix, preds['labels'], False)
        pred_centerness = od.grid.select(ix, preds['centerness'], False)
        # compute losses (either using the grid directly or the selection)
        # like FCOS, we treat each class as an independent classifier
        target_labels = torch.nn.functional.one_hot(target_labels[:, 0].long(),
            preds['labels'].shape[1]).float()
        target_centerness = od.transforms.centerness(target_bboxes_rel)
        return bboxes_loss(pred_bboxes, target_bboxes).mean() + \
            centerness_loss(pred_centerness, target_centerness) + \
            labels_loss(pred_labels, target_labels).mean()

class Model(torch.nn.Module):
    def __init__(self, nclasses, img_size):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.grid = Grid(2048, nclasses, img_size)

    def forward(self, x):
        x = self.backbone(x)
        return self.grid(x)

    def post_process(self, x):
        return self.grid.post_process(x)

    def compute_loss(self, preds, targets):
        return self.grid.compute_loss(preds, targets)
```

**Losses:** We currently do not deploy any losses since they are currently implemented in [torchvision](https://pytorch.org/vision/stable/ops.html#losses). We recommend using those losses. Notice that those losses receive the inputs in the format (N,4), not as a grid; that's why in the `Grid` code, we use `od.grid.select()` to convert the grid back to lists.

**Training:** Again, here is some boiler-plate code for creating your own training loop.

```python
model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for images, targets in tr:
        preds = model(images.to(device))
        loss_value = model.compute_loss(preds, targets)
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss}')
```

**Evaluation:** For evaluation purposes, we provide several metrics: the precision-recall curve, AP, mAP. These are mainly provided for educational purposes. We recommend that you use [TorchMetrics](https://torchmetrics.readthedocs.io/) for evaluation purposes.

For evaluation, you just need to do:

```python
preds = model(images.to(device))
preds = model.post_process(preds)
preds = od.post.NMS(preds)
```

## Citation

```bib
@misc{objdetect,
  author = {Ricardo Cruz},
  title = {{ObjDetect package}},
  howpublished = {\url{https://github.com/rpmcruz/objdetect}},
  year = {2022}
}
```
