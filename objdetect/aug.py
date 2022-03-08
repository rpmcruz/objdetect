from skimage import transform
import numpy as np

def resize(img_size):
    def f(x, bb):
        x = transform.resize(x, img_size)
        return x, bb
    return f

def random_resize_crop(img_size1, img_size2):
    crop_size = (img_size1[0]-img_size2[0], img_size1[1]-img_size2[1])
    assert crop_size[0] > 0, 'crop width must be positive'
    assert crop_size[1] > 0, 'crop height must be positive'
    def f(x, bb):
        i = np.random.randint(0, crop_size[0])
        j = np.random.randint(0, crop_size[1])
        x = transform.resize(x, img_size1)
        x = x[j:j-crop_size[0], i:i-crop_size[1]]
        bb[:, 0] = np.maximum(bb[:, 0]-i/img_size1[0], 0)
        bb[:, 1] = np.maximum(bb[:, 1]-j/img_size1[1], 0)
        return x, bb
    return f

def random_hflip(x, bb):
    if np.random.rand() < 0.5:
        x = np.flip(x, 1)
        bb[:, 0] = 1-bb[:, 0]
    return x, bb

def random_contrast_brightness(x, bb):
    alpha = np.random.rand()*0.2
    beta = np.random.rand()*0.2
    x = x*alpha + beta
    return x, bb
