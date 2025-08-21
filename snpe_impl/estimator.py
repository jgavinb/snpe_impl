import numpy as np
from scipy.spatial import KDTree
from scipy.special import comb  # For large n, but we use recursive weights


class BaggedNN:
    """Bagged Nearest Neighbors with L-representation, jackknife, and bootstrap support."""

    def __init__(self, m=7, jackknife=False, d=None, m_ratio=2.0):
        self.m = m
        self.jackknife = jackknife
        self.d = d  # Dim for jackknife alpha = -2/d
        self.m2 = int(m * m_ratio) if jackknife else None
        self.X_mean = None
        self.X_std = None

    def fit(self, X, y):
        """Fit normalizer; no tree needed for L-rep."""
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-6
        self.X_norm = (X - self.X_mean) / self.X_std
        self.y = y
        self.n = len(y)
        return self

    def _compute_tau(self, X_test_norm, m):
        """Compute bagged NN via L-rep for scale m."""
        preds = np.zeros(X_test_norm.shape[0])
        for i, x0 in enumerate(X_test_norm):
            dists = np.linalg.norm(self.X_norm - x0, axis=1)
            indices = np.argsort(dists)
            sorted_y = self.y[indices]
            # Recursive weights to avoid overflow
            weights = np.zeros(self.n)
            weights[0] = m / self.n
            for k in range(1, self.n):
                weights[k] = weights[k - 1] * (self.n - m - k + 1) / (self.n - k + 1)
                if weights[k] <= 0:
                    break
            weights /= np.sum(weights)  # Normalize if needed, but should sum to 1
            preds[i] = np.dot(weights, sorted_y)
        return preds

    def predict(self, X_test):
        """Predict with optional jackknife bias reduction."""
        X_test_norm = (X_test - self.X_mean) / self.X_std
        tau_m = self._compute_tau(X_test_norm, self.m)
        if not self.jackknife or self.d is None:
            return tau_m
        tau_m2 = self._compute_tau(X_test_norm, self.m2)
        alpha = -2.0 / self.d
        theta1 = self.m**alpha / (self.m**alpha - self.m2**alpha)
        theta2 = -(self.m2**alpha) / (self.m**alpha - self.m2**alpha)
        return theta1 * tau_m + theta2 * tau_m2
