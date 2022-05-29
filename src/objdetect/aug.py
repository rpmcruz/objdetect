'''
Data augmentations methods that function similarly to the [albumentations package](https://albumentations.ai/), and, in fact, should be compatible with it so that you may use albumentations if you wish. We make heavy use of torchvision functional API which in our experience is much faster than other packages such as skimage.
'''

from torchvision.transforms import functional as TF
import torch

def Compose(transformations):
    '''Applies the given `transformations` in succession.'''
    def f(**data):
        for t in transformations:
            data = t(**data)
        return data
    return f

def RandomBrightnessContrast(brightness_value, contrast_value):
    '''Randomly applies brightness (product) or contrast (addition) to the image. A random value is sampled from [-v/2, v/2].'''
    def f(image, **data):
        brightness = 1 - (torch.rand(())*brightness_value - brightness_value/2)
        contrast = torch.rand(())*contrast_value - contrast_value/2
        image = torch.clamp(image*brightness + contrast, 0, 1)
        return {'image': image, **data}
    return f

def Resize(H, W):
    '''Resizes to the given size `H`x`W`.'''
    def f(image, **data):
        image = TF.resize(image, (H, W))
        return {'image': image, **data}
    return f

def RandomCrop(crop_H, crop_W):
    '''Randomly crops the image so that the final image has size `crop_H`x`crop_W`.'''
    def f(image, bboxes, **data):
        _, orig_H, orig_W = image.shape
        i = torch.randint(0, orig_H-crop_H, ())
        j = torch.randint(0, orig_W-crop_W, ())
        image = image[:, i:i+crop_H, j:j+crop_W]
        bboxes = torch.stack((
            torch.clamp((bboxes[:, 0]*orig_W - j)/crop_W, min=0),
            torch.clamp((bboxes[:, 1]*orig_H - i)/crop_H, min=0),
            torch.clamp((bboxes[:, 2]*orig_W - j)/crop_W, max=1),
            torch.clamp((bboxes[:, 3]*orig_H - i)/crop_H, max=1)
        ), -1)
        # filter only bounding boxes inside the view
        bboxes = bboxes[(bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])]
        return {'image': image, 'bboxes': bboxes, **data}
    return f

def RandomHflip():
    '''Random horizontal flips.'''
    def f(image, bboxes, **data):
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
            bboxes = torch.stack((
                1-bboxes[:, 2], bboxes[:, 1],
                1-bboxes[:, 0], bboxes[:, 3],
            ), -1)
        return {'image': image, 'bboxes': bboxes, **data}
    return f

def SortBboxesByArea(descending=True):
    '''Some papers like FCOS assign the smaller bbox on location ambiguities. We can reproduce that by simply ordering the bboxes.'''
    def f(bboxes, **data):
        areas = (bboxes[:, 2]-bboxes[:, 0])*(bboxes[:, 3]-bboxes[:, 1])
        ix = torch.argsort(areas, descending=descending)
        bboxes = bboxes[ix]
        return {'bboxes': bboxes, **data}
    return f
