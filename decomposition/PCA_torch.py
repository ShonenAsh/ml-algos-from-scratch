import torch


# static function to zero center data
def _zscore_norm(data: torch.Tensor, device) -> torch.Tensor:
    mean = torch.mean(data, 1).reshape(data.shape[0], 1)
    std = torch.std(data, 1).reshape(data.shape[0], 1)
    z_normed = (data - mean) / std
    return torch.as_tensor(z_normed, dtype=torch.float32, device=device)


class PCA:

    def __init__(self, n_comps=None, device=torch.cuda.current_device()):
        self.n_comps = n_comps
        self.device = device

    def _pca(self) -> (torch.Tensor, torch.Tensor):
        cov_mat = torch.cov(self.X.T)
        eig_vals, eig_vectors = torch.linalg.eigh(cov_mat)
        return eig_vals.flip(dims=(-1,)), eig_vectors.flip(dims=(-1,))

    # only accepts 2-D matrices of shape (n, m)
    def fit(self, X: torch.Tensor):
        self.X = _zscore_norm(X, device=self.device)
        self.x_size = self.X.shape[0]
        if self.n_comps == None:
            self.n_comps = self.X.shape[1]
        _, self.eig_vec = self._pca()

    def transform(self, Y: torch.Tensor) -> torch.Tensor:
        Y = _zscore_norm(Y, self.device)
        return torch.matmul(Y, self.eig_vec)[:, :self.n_comps]