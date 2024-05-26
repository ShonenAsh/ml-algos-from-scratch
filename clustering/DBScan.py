import numpy as np
from collections import deque
from scipy.spatial.distance import cdist


class DBScan:

    def __init__(self, epsilon: float, min_pts=3, metric='euclidean'):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.metric = metric

    def fit(self, x_train: np.ndarray):
        self.x_train = x_train
        self.x_size = self.x_train.shape[0]

    def _find_neighbours(self) -> np.ndarray:
        dist = cdist(self.x_train, self.x_train, metric=self.metric)
        np.fill_diagonal(dist, 1)
        dist[dist >= self.epsilon] = 0
        dist[dist < self.epsilon] = 1
        return np.asarray(dist, dtype=np.uint8)

    def transform(self) -> np.ndarray:
        nbs = self._find_neighbours()
        clusters = -1 * np.ones(shape=self.x_size, dtype=np.int32)
        core = np.asarray([i for i in range(self.x_size)
                           if np.sum(nbs[i]) >= self.min_pts]
                          , dtype=np.int32)
        id = 1
        print(core[clusters[core] < 0])
        while np.any(clusters[core] < 0):
            q = deque()
            node = np.random.choice(core[clusters[core] < 0])
            q.append(node)

            while len(q) > 0:
                pt = q.popleft()
                clusters[pt] = id
                for i in np.argwhere(nbs[pt]):
                    if clusters[i] == -1 and np.sum(nbs[i]) >= self.min_pts:
                        q.append(i)
                    clusters[i] = id

            id += 1
        return clusters
