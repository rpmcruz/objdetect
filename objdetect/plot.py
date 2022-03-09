import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def grid(confs_grid, bboxes_grid, image_size):
    plt.imshow(np.ones((image_size[0], image_size[1])), alpha=0.25*resize(confs_grid, (image_size[0], image_size[1]), 0), cmap=colors.ListedColormap(['red']))
    grid_size = bboxes_grid.shape[-2:]
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            plt.text((x+0.5)*image_size[1]/grid_size[1], (y+0.5)*image_size[0]/grid_size[0], int(confs_grid[y, x]), c='blue')

def bboxes(bboxes, image_size, color='r', ls='-'):
    for xmin, ymin, xmax, ymax in bboxes:
        plt.gca().add_patch(Rectangle((xmin*image_size[0], ymin*image_size[1]),
            xmax*image_size[0], ymax*image_size[1], edgecolor=color, facecolor='none', ls=ls))

def bboxes_with_classes(bboxes, classes, image_size, labels, color='r', ls='-'):
    print('number of bboxes:', len(bboxes), 'number classes:', len(classes))
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, classes):
        print('position:', (xmin*image_size[0], ymin*image_size[1]), xmax*image_size[0], ymax*image_size[1])
        plt.gca().add_patch(Rectangle((xmin*image_size[0], ymin*image_size[1]),
            xmax*image_size[0], ymax*image_size[1], edgecolor=color, facecolor='none', ls=ls))
        s = labels[label] if labels else str(label)
        plt.text(xmin*image_size[0], ymin*image_size[1], s, c=color)
