import numpy as np
from scipy.spatial import KDTree


class BaggedNN:
    """Bagged Nearest Neighbors estimator for conditional expectations."""

    def __init__(self, m=7, num_bags=50, bag_fraction=0.5):
        self.m = m
        self.num_bags = num_bags
        self.bag_fraction = bag_fraction
        self.trees = []
        self.y_bags = []
        self.X_mean = None
        self.X_std = None

    def fit(self, X, y):
        """Fit on data X (features), y (target scalar)."""
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std
        n = X.shape[0]
        bag_size = int(self.bag_fraction * n)
        for _ in range(self.num_bags):
            idx = np.random.choice(n, bag_size, replace=False)
            tree = KDTree(X_norm[idx])
            self.trees.append(tree)
            self.y_bags.append(y[idx])
        return self

    def predict(self, X_test, jackknife_bias=True):
        """Predict conditional expectation at X_test."""
        X_test_norm = (X_test - self.X_mean) / self.X_std
        preds = np.zeros((X_test.shape[0], self.num_bags))
        for b in range(self.num_bags):
            dists, idxs = self.trees[b].query(X_test_norm, k=self.m)
            neighbor_ys = self.y_bags[b][idxs]
            bag_pred = np.mean(neighbor_ys, axis=1)
            if jackknife_bias:
                # Jackknife bias reduction: avg leave-one-out on neighbors
                loo_avgs = (np.sum(neighbor_ys, axis=1) - neighbor_ys.T) / (self.m - 1)
                loo_avg = np.mean(loo_avgs, axis=0)
                bag_pred = self.m * bag_pred - (self.m - 1) * loo_avg
            preds[:, b] = bag_pred
        return np.mean(preds, axis=1)
