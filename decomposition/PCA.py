import numpy as np


# static function to zero center data
def _zscore_norm(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    std = np.std(data, axis=1).reshape(data.shape[0], 1)
    z_normed = (data - mean) / std
    return np.asarray(z_normed, dtype=np.float32)


class PCA:

    def __init__(self, n_comps=None):
        self.n_comps = n_comps

    def _pca(self) -> (np.ndarray, np.ndarray):
        cov_mat = np.cov(self.X.T)
        eig_vals, eig_vectors = np.linalg.eigh(cov_mat)
        eig_val_sorted_ids = np.argsort(eig_vals)[::-1]
        return eig_vals[eig_val_sorted_ids], eig_vectors[:, eig_val_sorted_ids]

    # only accepts 2-D matrices of shape (n, m)
    def fit(self, X: np.ndarray):
        self.X = _zscore_norm(X)
        self.x_size = self.X.shape[0]
        if self.n_comps == None:
            self.n_comps = self.X.shape[1]
        _, self.eig_vec = self._pca()

    def transform(self, Y: np.ndarray) -> np.ndarray:
        Y = _zscore_norm(Y)
        return Y.dot(self.eig_vec)[:, :self.n_comps]
