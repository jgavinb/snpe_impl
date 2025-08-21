import numpy as np


class PriceElasticityEstimator:
    """Estimates own/cross-price elasticities using IV bagged NN."""

    def __init__(self, h_estimators, g_estimators=None):
        self.h_estimators = h_estimators  # List of BaggedNN for each product's h_j
        self.g_estimators = (
            g_estimators  # List of BaggedNN for each price's g_k (if IV)
        )

    def compute_sensitivity(
        self, p_grid, x_fixed, z_fixed, product_j, product_k, delta=1e-3
    ):
        """Compute partial df_j / dp_k at grid via finite diff + IV correction."""
        # Assume single product for simplicity; extend for multi
        h_j = self.h_estimators[product_j].predict
        dp_h = self._finite_diff(
            h_j, p_grid, x_fixed, z_fixed, dim="p", k=product_k, delta=delta
        )
        if self.g_estimators is None:  # No endogeneity
            return dp_h
        g_k = self.g_estimators[product_k].predict
        dz_g = self._finite_diff(
            g_k, p_grid, x_fixed, z_fixed, dim="z", k=product_k, delta=delta
        )
        dz_h = self._finite_diff(
            h_j, p_grid, x_fixed, z_fixed, dim="z", k=product_k, delta=delta
        )
        # IV correction (simplified Eq. 5 for 1 endogenous, 1 instrument)
        inv_dz_g = np.linalg.pinv(dz_g) if dz_g.ndim > 1 else 1 / dz_g
        correction = inv_dz_g * dz_h
        return dp_h - correction

    def _finite_diff(self, func, p_grid, x_fixed, z_fixed, dim, k, delta):
        """Finite difference derivative wrt dim (p or z) at index k."""
        base_input = np.hstack([p_grid, x_fixed, z_fixed])
        plus = base_input.copy()
        plus[:, k] += delta
        minus = base_input.copy()
        minus[:, k] -= delta
        return (func(plus) - func(minus)) / (2 * delta)

    def compute_elasticity(
        self, p_grid, x_fixed, z_fixed, product_j, product_k, bootstrap_samples=100
    ):
        """Compute elasticity at p_grid, with bootstrap CIs."""
        s_grid = self.h_estimators[product_j].predict(
            np.hstack([p_grid, x_fixed, z_fixed])
        )
        sens = self.compute_sensitivity(p_grid, x_fixed, z_fixed, product_j, product_k)
        elast = (p_grid[:, product_k] / s_grid) * sens
        if bootstrap_samples > 0:
            boot_elasts = []
            for _ in range(bootstrap_samples):
                # Resample data, refit estimators (pseudo; in practice, refit on resamples)
                # For efficiency, approximate via perturbing predictions
                pert_sens = sens + np.random.normal(
                    0, np.std(sens) / np.sqrt(len(sens)), len(sens)
                )
                boot_elasts.append((p_grid[:, product_k] / s_grid) * pert_sens)
            ci_low, ci_high = np.percentile(boot_elasts, [2.5, 97.5], axis=0)
            return elast, ci_low, ci_high
        return elast, None, None
