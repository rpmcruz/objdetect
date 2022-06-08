'''
Useful plotting methods.
'''

import matplotlib.pyplot as plt
from matplotlib import lines, patches, colors
import numpy as np
from torchvision.transforms.functional import resize, InterpolationMode

def grid_lines(image, h, w):
    '''Draws grid lines.'''
    _, H, W = image.shape
    plt.vlines(np.linspace(0, W, w+1), 0, H, color='gray', linewidth=2)
    plt.hlines(np.linspace(0, H, h+1), 0, W, color='gray', linewidth=2)

def grid_bools(image, grid):
    '''Draws a grid boolean matrix, such as the scores grid.'''
    assert len(grid.shape) == 2
    _, H, W = image.shape
    plt.imshow(np.ones((H, W)), alpha=0.2*resize(grid[None], (H, W), InterpolationMode.NEAREST)[0], cmap=colors.ListedColormap(['green']))

def grid_text(image, grid):
    '''Draws a grid matrix of values, such as the classes grid.'''
    _, H, W = image.shape
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            plt.text((x+0.5)*W/w, (y+0.5)*H/h, str(grid[y, x]), c='blue')

def image(image):
    '''Calls `plt.imshow()` with channels permutted CxHxW => HxWxC.'''
    plt.imshow(image.permute((1, 2, 0)))

def bboxes(image, bboxes, ec='blue', ls='-'):
    '''Draws a list of bounding boxes.'''
    _, H, W = image.shape
    for xmin, ymin, xmax, ymax in bboxes:
        plt.gca().add_patch(patches.Rectangle(
            (xmin*W, ymin*H), (xmax-xmin)*W, (ymax-ymin)*H,
            lw=2, ls=ls, ec=ec, fc='none'))

def classes(image, bboxes, classes, labels=None):
    '''Draws a list of classes captions above the bounding box. If a `labels` list is provided then the respective label will be drawn, rather than the integer.'''
    _, H, W = image.shape
    for (xmin, ymin, xmax, ymax), klass in zip(bboxes, classes):
        s = labels[klass] if labels else klass
        plt.text(xmin*W, ymin*H, s)

def show():
    '''Convenience method. Just call `plt.show()`.'''
    plt.show()
