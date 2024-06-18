import numpy as np
from collections import deque
from scipy.spatial.distance import cdist


class DBScan:

    def __init__(self, epsilon: float, min_pts=3, metric='euclidean'):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.metric = metric

    def fit(self, data: np.ndarray):
        self.data = data
        self.x_size = self.data.shape[0]

    def _find_neighbours(self) -> np.ndarray:
        dist = cdist(self.data, self.data, metric=self.metric)
        dist /= np.max(dist)
        dist[dist >= self.epsilon] = 0
        np.fill_diagonal(dist, val=1)
        return np.asarray(dist, dtype=np.float16)

    def transform(self) -> np.ndarray:
        nbs = self._find_neighbours()
        clusters = -1 * np.ones(shape=self.x_size, dtype=np.int32)
        core = np.asarray([i for i in range(self.x_size) if np.count_nonzero(nbs[i]) >= self.min_pts], dtype=np.int32)
        id = 1
        while np.any(clusters[core] < 0):
            q = deque()
            node = np.random.choice(core[clusters[core] < 0])
            q.append(node)

            while len(q) > 0:
                pt = q.popleft()
                clusters[pt] = id
                neighbors = np.argwhere(nbs[pt])
                for i in neighbors:
                    if clusters[i] == -1 and np.sum(nbs[i]) >= self.min_pts:
                        q.append(i)
                    clusters[i] = id
            id += 1
        return clusters
