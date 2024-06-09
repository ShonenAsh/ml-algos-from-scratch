import scipy as sc
import numpy as np


class KNNClassifier:

    def __init__(self, k=3, metric=None):
        self.k = k
        if metric is None:
            metric = 'euclidean'
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = self._cal_predictions(X_test)
        return y_pred

    def _cal_predictions(self, X):
        idx = None
        dist = sc.spatial.distance.cdist(self.X_train, X, metric=self.metric)
        idx = np.argsort(dist, axis=0)[:self.k]
        # pick the most frequently occuring label from top k neighbors
        k_labels = self.y_train[idx]
        return sc.stats.mode(k_labels, axis=0).mode
