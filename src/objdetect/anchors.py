from sklearn.cluster import KMeans
import numpy as np

def compute_clusters(ds, n):
    BB = [d['bboxes'] for d in ds]
    BB = [(b[2], b[3]) for bb in BB for b in bb]
    return KMeans(n).fit(BB).cluster_centers_
