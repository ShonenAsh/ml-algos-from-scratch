import numpy as np
from scipy.spatial.distance import cdist


class HierarchicalClustering:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, data: np.ndarray):
        self.data = data
        self.size = data.shape[0]

    def transform(self) -> (np.ndarray, np.ndarray):
        return self._cluster()

    def _cluster(self) -> (np.ndarray, np.ndarray):
        # pairwise-distance
        dist_mat = cdist(self.data, self.data)
        row_ids = np.arange(self.size, dtype=np.uintp)[:, np.newaxis]
        hierarchy = np.tile(row_ids, (1, self.size))
        clusters = np.arange(self.size, dtype=np.uintp)
        for i in range(dist_mat.shape[0] - 1):
            # upper triangle with diagonal offset k=1
            upper_tri = np.triu_indices(dist_mat.shape[0], k=1)
            # print(upper_tri)
            min_val = np.min(dist_mat[upper_tri])
            row, col = np.where(dist_mat == min_val)[0]

            clusters[clusters == col] = row
            clusters[clusters > col] -= 1

            # Merge by deleting the row & col of 1
            dist_mat = np.delete(dist_mat, col, 0)
            dist_mat = np.delete(dist_mat, col, 1)

            # Update pairwise distances
            for j in range(dist_mat.shape[0]):
                A, B = self.data[clusters == row], self.data[clusters == j]
                dist_mat[row, j] = np.min(cdist(B, A, metric=self.metric))
                dist_mat[j, row] = dist_mat[row, j]
            hierarchy[i] = clusters

        hierarchy[-1] = 0
        return clusters, hierarchy
