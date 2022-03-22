from skimage import transform
import numpy as np

'''
You do not need to use our augmentation methods. You can easily use
Albumentations or any other package. Our API should be similar in use to
Albumentations, but simpler. Look at the bottom for an usage example.
'''

def Compose(*l):
    def f(**datum):
        for t in l:
            datum = t(**datum)
        return datum
    return f

def Resize(img_size):
    def f(image, **args):
        image = transform.resize(image, img_size).astype(np.float32)
        return {**args, 'image': image}
    return f

def RandomCrop(crop_size):
    def f(image, bboxes, **args):
        orig_shape = image.shape
        i = np.random.randint(0, orig_shape[1]-crop_size[0])
        j = np.random.randint(0, orig_shape[0]-crop_size[1])
        image = image[j:j+crop_size[1], i:i+crop_size[0]]
        bboxes = [(
            max((b[0]*orig_shape[1] - i)/crop_size[0], 0),
            max((b[1]*orig_shape[0] - j)/crop_size[1], 0),
            min((b[2]*orig_shape[1] - i)/crop_size[0], 1),
            min((b[3]*orig_shape[0] - j)/crop_size[1], 1),
        ) for b in bboxes]
        # filter bounding boxes outside the view
        bboxes = [b for b in bboxes if b[0] < b[2] and b[1] < b[3]]
        return {**args, 'image': image, 'bboxes': bboxes}
    return f

def RandomHflip():
    def f(image, bboxes, **args):
        if np.random.rand() < 0.5:
            image = np.flip(image, 1)
            bboxes = [(1-b[2], b[1], 1-b[0], b[3]) for b in bboxes]
        return {**args, 'image': image, 'bboxes': bboxes}
    return f

def RandomBrightnessContrast(brightness_value, contrast_value):
    def f(image, **args):
        brightness = 1 - (np.random.rand()*brightness_value - brightness_value/2)
        contrast = np.random.rand()*contrast_value - contrast_value/2
        image = np.clip(image*brightness + contrast, 0, 1)
        return  {**args, 'image': image}
    return f

def VggNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # some pre-trained models use a normalization different than z-std [0,1]
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    def f(image, **args):
        image = (image - mean) / std
        return {**args, 'image': image}
    return f

def VggReverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    def f(image, **args):
        image = image*std + mean
        return {**args, 'image': image}
    return f

if __name__ == '__main__':  # debug
    import matplotlib.pyplot as plt
    import datasets, plot
    aug = Combine(Resize((282, 282)), RandomCrop((256, 256)), RandomHflip(), RandomBrightnessContrast())
    ds = datasets.VOCDetection('data', 'train', False, aug, None)
    datum = ds[0]
    plt.imshow(datum['image'])
    plot.bboxes(datum['image'], datum['bboxes'])
    plt.show()
