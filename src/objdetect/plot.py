import matplotlib.pyplot as plt
from matplotlib import colors, patches
from skimage.transform import resize
import numpy as np

def grid_without_anchors(image, confs_grid, bboxes_grid):
    # provided for debug purposes only
    # to debug bboxes_grid, just revert it and use plot.bboxes
    image_size = image.shape[1::-1]
    grid_size = confs_grid.shape[::-1]
    plt.vlines(np.linspace(0, image_size[0], grid_size[0]+1), 0, image_size[1], color='g', linewidth=0.5)
    plt.hlines(np.linspace(0, image_size[1], grid_size[1]+1), 0, image_size[0], color='g', linewidth=0.5)
    resized_confs_grid = resize(confs_grid[0, 0], (image_size[1], image_size[0]), 0)
    plt.imshow(np.ones((image_size[1], image_size[0])), alpha=0.5*resized_confs_grid, cmap=colors.ListedColormap(['red']))
    grid_size = confs_grid.shape[::-1]
    cell_size = (1 / grid_size[0], 1 / grid_size[1])
    for gx in range(grid_size[0]):
        for gy in range(grid_size[1]):
            if confs_grid[0, 0, gy, gx] >= 0.5:
                offset_x, offset_y, log_w, log_h = bboxes_grid[:, 0, gy, gx]
                xc, yc = (gx+offset_x)*cell_size[0], (gy+offset_y)*cell_size[1]
                w, h = np.exp(log_w), np.exp(log_h)
                plt.scatter(xc*image_size[0], yc*image_size[1], color='white')
                plt.gca().add_patch(patches.Rectangle(
                    ((xc-w/2)*image_size[0], (yc-h/2)*image_size[1]),
                    w*image_size[0], h*image_size[1],
                    edgecolor='r', facecolor='none'))

def anchors(anchors, ncols=6):  # debug purposes
    nrows = int(np.ceil(len(anchors) / ncols))
    for i, anchor in enumerate(anchors):
        plt.subplot(nrows, ncols, i+1)
        w, h = anchor
        plt.gca().add_patch(patches.Rectangle(((1-w)/2, (1-h)/2), w, h, edgecolor='g', facecolor='none'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)

def reverse_normalization(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.asarray(mean, np.float32)
    std = np.asarray(std, np.float32)
    image = image*std + mean
    return image

def bboxes(image, bboxes, color='r', ls='-'):
    image_size = image.shape[1::-1]
    for xmin, ymin, xmax, ymax in bboxes:
        plt.gca().add_patch(patches.Rectangle(
            (xmin*image_size[0], ymin*image_size[1]),
            (xmax-xmin)*image_size[0], (ymax-ymin)*image_size[1],
            edgecolor=color, facecolor='none', ls=ls))

def bboxes_with_classes(image, bboxes, classes, labels, color='r', ls='-'):
    image_size = image.shape[1::-1]
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, classes):
        plt.gca().add_patch(patches.Rectangle(
            (xmin*image_size[0], ymin*image_size[1]),
            (xmax-xmin)*image_size[0], (ymax-ymin)*image_size[1],
            edgecolor=color, facecolor='none', ls=ls))
        s = labels[label] if labels else str(label)
        plt.text(xmin*image_size[0], ymin*image_size[1], s, c=color)

def bboxes_with_angles(image, bboxes, angles, color='r', ls='-'):
    image_size = image.shape[1::-1]
    for (xmin, ymin, xmax, ymax), angle in zip(bboxes, angles):
        plt.gca().add_patch(patches.Rectangle(
            (xmin*image_size[0], ymin*image_size[1]),
            (xmax-xmin)*image_size[0], (ymax-ymin)*image_size[1],
            angle, edgecolor=color, facecolor='none', ls=ls))
