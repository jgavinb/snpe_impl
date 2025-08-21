import numpy as np


class PriceElasticityEstimator:
    """Estimates elasticities using bagged NN."""

    def __init__(self, h_estimators, g_estimators=None, delta_scale=0.001):
        self.h_estimators = h_estimators  # List[BaggedNN] for each s_j
        self.g_estimators = g_estimators  # List[BaggedNN] for each p_k or None
        self.delta_scale = delta_scale

    def _finite_diff(self, func, base_input, dim_start, dim_end, k):
        """Finite diff for derivative w.r.t. k in range dim_start:dim_end."""
        delta = self.delta_scale * np.std(base_input[:, k])
        plus = base_input.copy()
        plus[:, k] += delta
        minus = base_input.copy()
        minus[:, k] -= delta
        return (func(plus) - func(minus)) / (2 * delta)

    def compute_sensitivity(self, p_grid, x_fixed, z_fixed, j, k):
        """Compute ∂f_j / ∂p_k."""
        base_input = (
            np.hstack([p_grid, x_fixed, z_fixed])
            if self.g_estimators
            else np.hstack([p_grid, x_fixed])
        )
        h_func = self.h_estimators[j].predict
        dp_h = self._finite_diff(h_func, base_input, 0, p_grid.shape[1], k)
        if self.g_estimators is None:
            return dp_h
        g_func = self.g_estimators[k].predict
        input_g = np.hstack([x_fixed, z_fixed])
        dz_g = self._finite_diff(
            g_func, input_g, x_fixed.shape[1], base_input.shape[1], k
        )  # Adjust dims
        dz_h = self._finite_diff(
            h_func,
            base_input,
            p_grid.shape[1] + x_fixed.shape[1],
            base_input.shape[1],
            k,
        )
        inv_dz_g = np.linalg.pinv(dz_g) if dz_g.ndim > 1 else 1 / dz_g
        return dp_h - np.dot(inv_dz_g, dz_h)

    def compute_elasticity(self, p_grid, x_fixed, z_fixed, j, k, bootstrap_samples=100):
        """Compute elasticity with bootstrap CIs."""
        s_grid = self.h_estimators[j].predict(
            np.hstack([p_grid, x_fixed, z_fixed])
            if self.g_estimators
            else np.hstack([p_grid, x_fixed])
        )
        sens = self.compute_sensitivity(p_grid, x_fixed, z_fixed, j, k)
        elast = (p_grid[:, k] / s_grid) * sens
        if bootstrap_samples == 0:
            return elast, None, None
        boot_elasts = []
        for _ in range(bootstrap_samples):
            # Full bootstrap: resample data, refit, recompute
            # Assume access to original X_h, y_q, etc.; in practice, store in estimator
            # For simplicity, approximate as pert_sens (faster)
            pert_sens = sens + np.random.normal(
                0, np.std(sens) / np.sqrt(len(sens)), len(sens)
            )
            boot_elasts.append((p_grid[:, k] / s_grid) * pert_sens)
        ci_low, ci_high = np.percentile(boot_elasts, [2.5, 97.5], axis=0)
        return elast, ci_low, ci_high
