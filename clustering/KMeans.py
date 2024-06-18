import numpy as np
from scipy.spatial.distance import cdist


class KMeans:

    def __init__(self, k=10, max_iterations=50, threshold=0.0001, metric='euclidean'):
        self.k = k
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.metric = metric

    def fit(self, data: np.ndarray):
        self.data = data
        self.x_size = self.data.shape[0]

    def transform(self):
        mu_arr = self.data[np.random.choice(self.x_size, self.k, replace=False)]
        pi = None
        obj = np.inf

        for _ in range(self.max_iterations):
            pi = np.zeros((self.x_size, self.k), dtype=np.ushort)
            # E-step
            distances = cdist(self.data, mu_arr, metric=self.metric)
            c_labels = np.argmin(distances, axis=1)
            for i in range(len(c_labels)):
                pi[i, c_labels[i]] = 1  # Set new labels

            # M-step
            for j in range(self.k):
                elements_of_j = np.where(c_labels == j)
                mu_arr[j] = np.mean(self.data[elements_of_j], axis=0)

            new_obj = np.sum(pi * distances)
            if float(abs(obj - new_obj)) < (obj * self.threshold):  # termination criterion
                break
            obj = new_obj

        return obj, pi, mu_arr
