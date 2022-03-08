import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot(x, yy, bb):
    w, h = x.shape[1], x.shape[0]
    for y, (cx, cy, rx, ry) in zip(yy, bb):
        plt.gca().add_patch(Rectangle(((cx-rx)*w, (cy-ry)*h), rx*2*w, ry*2*h, edgecolor='r', facecolor='none'))
        plt.text(cx*w, cy*h, str(y), c='r')
