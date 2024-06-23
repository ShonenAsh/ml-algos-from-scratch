import numpy as np
from scipy.spatial.distance import cdist


# Hierarchical Clustering with Minimum Linkage
class HierarchicalClustering:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, data: np.ndarray):
        self.data = data
        self.size = data.shape[0]

    def transform(self) -> (np.ndarray, np.ndarray):
        return self._cluster()

    def _cluster(self) -> (np.ndarray, np.ndarray):
        # pairwise-distance normalized
        dist_mat = cdist(self.data, self.data, metric=self.metric)
        np.fill_diagonal(dist_mat, val=np.inf)
        row_ids = np.arange(self.size, dtype=np.uintp)[:, np.newaxis]
        hierarchy = np.tile(row_ids, (1, self.size))
        clusters = np.arange(self.size, dtype=np.uintp)
        for i in range(dist_mat.shape[0] - 1):
            row, col = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
            clusters[clusters == col] = row
            clusters[clusters > col] -= 1

            # Merge by deleting the row & col of 1
            dist_mat = np.delete(dist_mat, col, 0)
            dist_mat = np.delete(dist_mat, col, 1)

            # Update pairwise distances
            for j in range(dist_mat.shape[0]):
                A, B = self.data[clusters == row], self.data[clusters == j]
                dist_mat[row, j] = np.min(cdist(B, A, metric=self.metric))
                np.fill_diagonal(dist_mat, val=np.inf)
                dist_mat[j, row] = dist_mat[row, j]
            hierarchy[i] = clusters

        hierarchy[-1] = 0
        return hierarchy
