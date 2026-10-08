from celllift.runtime import resource_path as _public_resource
import numpy as np
from scipy.spatial import cKDTree

class ExactTree:
    query_block = 8192

    def __init__(self, legacy):
        self.legacy = legacy
        self.k = legacy.n_neighbors
        self.tree = cKDTree(legacy._fit_X, leafsize=32)

    def kneighbors(self, q, return_distance=False):
        distance, ix = self.tree.query(q, k=min(self.k + 1, len(self.tree.data)), workers=4)
        ties = (np.diff(distance, axis=1) <= 1e-12).any(1)
        result = ix[:, :self.k].copy()
        if ties.any():
            result[ties] = self.legacy.kneighbors(q[ties], return_distance=False)
        return result

def query(raw, cluster, mean, std, pca, nn, pool_x, pool_clusters):
    if not len(raw):
        return (np.empty((0, 5), int), np.empty((0, 5)))
    indices = []
    distances = []
    block = getattr(nn, 'query_block', 512)
    for outer in range(0, len(raw), block):
        normalized = [(raw[start:min(start + 512, outer + block)] - mean) / std for start in range(outer, min(outer + block, len(raw)), 512)]
        projected = np.concatenate([pca.transform(x) for x in normalized])
        candidates = nn.kneighbors(projected, return_distance=False)
        offset = 0
        for x in normalized:
            idx = candidates[offset:offset + len(x)]
            offset += len(x)
            difference = pool_x[idx]
            difference -= x[:, None]
            np.square(difference, out=difference)
            dist = np.sqrt(np.mean(difference, axis=2))
            dist[pool_clusters[idx] == str(cluster)] = np.inf
            order = np.argsort(dist, axis=1)[:, :5]
            indices.append(np.take_along_axis(idx, order, 1))
            distances.append(np.take_along_axis(dist, order, 1))
    return (np.concatenate(indices), np.concatenate(distances))
