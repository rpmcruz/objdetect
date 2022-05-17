'''
Useful plotting methods.
'''

import matplotlib.pyplot as plt
from matplotlib import lines, patches, colors
import numpy as np
from skimage.transform import resize

def grid_lines(image, h, w):
    '''Draws grid lines.'''
    H, W, _ = image.shape
    plt.vlines(np.linspace(0, W, w+1), 0, H, color='gray', linewidth=0.5)
    plt.hlines(np.linspace(0, H, h+1), 0, W, color='gray', linewidth=0.5)

def grid_bools(image, grid):
    '''Draws a grid boolean matrix, such as the hasobjs grid.'''
    H, W, _ = image.shape
    plt.imshow(np.ones((H, W)), alpha=0.2*resize(grid, (H, W), 0), cmap=colors.ListedColormap(['green']))

def grid_text(image, grid):
    '''Draws a grid matrix of values, such as the classes grid.'''
    H, W, _ = image.shape
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            plt.text((x+0.5)*W/w, (y+0.5)*H/h, str(grid[y, x]), c='blue')

def bboxes(image, bboxes):
    '''Draws a list of bounding boxes.'''
    H, W, _ = image.shape
    for xmin, ymin, xmax, ymax in bboxes:
        plt.gca().add_patch(patches.Rectangle(
            (xmin*W, ymin*H), (xmax-xmin)*W, (ymax-ymin)*H,
            edgecolor='blue', facecolor='none'))

def classes(image, bboxes, classes, labels=None):
    '''Draws a list of classes captions above the bounding box. If a `labels` list is provided then the respective label will be drawn, rather than the integer.'''
    H, W, _ = image.shape
    for (xmin, ymin, xmax, ymax), klass in zip(bboxes, classes):
        s = labels[klass] if labels else klass
        plt.text(xmin*W, ymin*H, s)
