# Implementations using objdetect

A couple of implementations using the [objdetect](https://github.com/rpmcruz/objdetect) package.

* [FCOS](https://arxiv.org/abs/1904.01355)
* [YOLO3](https://arxiv.org/abs/1804.02767)
* Futhermore, there is a "simple" model which is an example using a simplified FCOS without multi-scale.

The models are pretty self-contained and should be easy to understand. By default, the backbone for FCOS is ResNet50 and for YOLO3 is DarkNet53, but you can easily change it.

Evaluation when training Pascal VOC (animals only) for 100 epochs.

| Model  | mAP   |
|--------|-------|
| Simple |       |
| FCOS   |       |
| YOLO3  |       |

(These values should be taken with a grain of salt, since they depend on a lot of small choices, from the optimizer to the image sizes.)
