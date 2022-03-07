from sklearn.cluster import KMeans
import numpy as np

def compute_clusters(ds, n):
    BB = [d['bboxes'] for d in ds]
    BB = [(b[2], b[3]) for bb in BB for b in bb]
    return KMeans(n).fit(BB).cluster_centers_

def visualize_anchors(anchors):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    for i, anchor in enumerate(anchors):
        plt.subplot(int(np.ceil(len(anchors)/6)), 6, i+1)
        w, h = anchor
        plt.gca().add_patch(patches.Rectangle(((1-w)/2, (1-h)/2), w, h, edgecolor='g', facecolor='none'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.show()
