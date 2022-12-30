'''
Simple primitives to draw the bounding boxes.
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def bboxes(bboxes, scale=(1, 1), labels=None, color='blue', linestyle='-'):
    ''' Draws the given bounding boxes and (if provided) labels. If you use 0-1 normalized bboxes, then give scale=image_size to convert them. '''
    h, w = scale
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        plt.gca().add_patch(Rectangle(
            (xmin*w, ymin*h), (xmax-xmin)*w, (ymax-ymin)*h,
            lw=2, ls=linestyle, ec=color, fc='none'))
        if labels is not None:
            plt.text(xmin*w, ymin*h, str(labels[i]), c=color)

def grid(grid_size, img_size, color='gray'):
    ''' Draws the horizontal and vertical grid lines. Useful when choosing the grid sizes. '''
    gh, gw = grid_size
    ih, iw = img_size
    plt.vlines(np.linspace(0, gw, iw+1), 0, gh, color=color, lw=1)
    plt.hlines(np.linspace(0, gh, ih+1), 0, gw, color=color, lw=1)

def grid_scores(scores, img_size, lt_color='gray', gt_color='red', threshold=0.5):
    ''' Draw the output scores. Useful for debugging if the model is learning to predict the scores correctly. '''
    assert len(scores.shape) == 2
    grid_size = scores.shape
    sh = img_size[0]/grid_size[0]
    sw = img_size[1]/grid_size[1]
    for y in range(scores.shape[0]):
        for x in range(scores.shape[1]):
            score = scores[y, x]
            color = gt_color if score >= threshold else lt_color
            plt.text((x+0.5)*sw, (y+0.5)*sh, f'{score*100:.0f}', color=color, ha='center', fontsize='x-small')
