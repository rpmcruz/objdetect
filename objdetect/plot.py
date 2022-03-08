import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def bboxes(bboxes, image_size, color='r', ls='-'):
    for xmin, ymin, xmax, ymax in bboxes:
        plt.gca().add_patch(Rectangle((xmin*image_size[0], ymin*image_size[1]),
            xmax*image_size[0], ymax*image_size[1], edgecolor=color, facecolor='none', ls=ls))

def bboxes_with_classes(bboxes, classes, image_size, color='r', ls='-'):
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, classes):
        plt.gca().add_patch(Rectangle((xmin*image_size[0], ymin*image_size[1]),
            xmax*image_size[0], ymax*image_size[1], edgecolor=color, facecolor='none', ls=ls))
        plt.text(xmin*image_size[0], ymin*image_size[1], str(label), c=color)
