import numpy as np
from scipy.spatial.distance import cdist


class GaussianKernelPCA:
    def __init__(self, sigma=3, metric='euclidean'):
        self.sigma = sigma
        self.metric = metric

    def fit(self, A):
        self.A = A
        self.dist_mat = cdist(self.A, self.A, metric=self.metric)
        K = np.exp(-self.dist_mat / self.sigma)
        U = np.ones(self.A.shape[0]) / self.A.shape[0]
        Kn = K - U @ K - K @ U + U @ K @ U
        eig_vals, eig_vectors = np.linalg.eigh(Kn)
        eig_val_sorted_ids = np.argsort(eig_vals)[::-1]
        self.eig_vals, self.eig_vectors = eig_vals[eig_val_sorted_ids], eig_vectors[:, eig_val_sorted_ids]
        return self.eig_vals, self.eig_vectors

    def transform(self, B: np.ndarray) -> np.ndarray:
        return self.eig_vectors.dot(B)
