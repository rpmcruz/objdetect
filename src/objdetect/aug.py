'''
Data augmentations methods that function similarly to the [albumentations package](https://albumentations.ai/), and, in fact, should be compatible with it so that you may use albumentations if you wish.
'''

from skimage import transform
import numpy as np

def ResizeAndNormalize(H, W):
    '''Uses skimage resize function which both resizes (to the given size `H` x `W`) and normalizes to [0, 1].'''
    def f(image, **data):
        image = transform.resize(image, (H, W)).astype(np.float32)
        return {'image': image, **data}
    return f

def RandomHflip():
    '''Random horizontal flips.'''
    def f(image, bboxes, **data):
        if np.random.rand() < 0.5:
            image = np.flip(image, 1).copy()
            bboxes = bboxes.copy()
            bboxes[:, 0] = 1 - bboxes[:, 0]
            bboxes[:, 2] = 1 - bboxes[:, 2]
        return {'image': image, 'bboxes': bboxes, **data}
    return f

def RandomBrightnessContrast(brightness_value, contrast_value):
    '''Randomly applies brightness (product) or contrast (addition) to the image. A random value is sampled from [-v/2, v/2].'''
    def f(image, **data):
        brightness = 1 - (np.random.rand()*brightness_value - brightness_value/2)
        contrast = np.random.rand()*contrast_value - contrast_value/2
        image = np.clip(image*brightness + contrast, 0, 1)
        return {'image': image, **data}
    return f

def RandomCrop(crop_H, crop_W):
    '''Randomly crops the image so that the final image has size `crop_H` x `crop_W`.'''
    def f(image, bboxes, **data):
        orig_H, orig_W, _ = image.shape
        i = np.random.randint(0, orig_H-crop_H)
        j = np.random.randint(0, orig_W-crop_W)
        image = image[j:j+crop_H, i:i+crop_W]
        bboxes = np.stack((
            np.maximum((bboxes[:, 0]*orig_W - i)/crop_W, 0),
            np.maximum((bboxes[:, 1]*orig_H - j)/crop_H, 0),
            np.minimum((bboxes[:, 2]*orig_W - i)/crop_W, 1),
            np.minimum((bboxes[:, 3]*orig_H - j)/crop_H, 1)
        ), -1)
        # filter only bounding boxes inside the view
        bboxes = bboxes[np.logical_and(bboxes[:, 0] < bboxes[:, 2], bboxes[:, 1] < bboxes[:, 3])]
        return {'image': image, 'bboxes': bboxes, **data}
    return f

def Compose(*transformations):
    '''Applies the given `transformations` in succession.'''
    def f(**data):
        for t in transformations:
            data = t(**data)
        return data
    return f
