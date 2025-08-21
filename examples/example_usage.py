import numpy as np
import pandas as pd
from snpe_impl.estimator import BaggedNN
from snpe_impl.elasticity import PriceElasticityEstimator
from snpe_impl.utils import load_data, preprocess_data
from snpe_impl.visuals import plot_elasticity_profile, plot_cross_elasticity_table

# Synthetic daily data: 2 products, optional category
data = {
    "date": np.repeat(np.arange(100), 2),
    "product_id": np.tile([0, 1], 100),
    "category": np.tile(["A", "B"], 100),
    "price": np.random.uniform(10, 20, 200),
    "quantity": np.random.poisson(100, 200),
    "inst_cost": np.random.uniform(0, 1, 200),  # Instrument
}
df = pd.DataFrame(data)
df.to_csv("daily_data.csv", index=False)

df = load_data("daily_data.csv")
X_h, y_s, X_g, y_p, x_dim, z_dim = preprocess_data(df, category=True)

# Fit h_j for each product j, g_k for each price k
h_estimators = [
    BaggedNN(m=7, jackknife=True, d=X_h.shape[1]).fit(X_h, y_s[j])
    for j in range(y_s.shape[0])
]
g_estimators = [
    BaggedNN(m=7, jackknife=True, d=X_g.shape[1]).fit(X_g, y_p[k])
    for k in range(y_p.shape[0])
]

estimator = PriceElasticityEstimator(h_estimators, g_estimators)

# Grid for prices (own/cross)
p_grid = np.linspace(10, 20, 50).reshape(-1, 1)  # For own, extend for multi
x_fixed = np.zeros((50, x_dim))  # Fix covariates
z_fixed = np.zeros((50, z_dim))

# Own elasticity for product 0
elast, ci_low, ci_high = estimator.compute_elasticity(
    p_grid, x_fixed, z_fixed, j=0, k=0
)
plot_elasticity_profile(p_grid, elast, ci_low, ci_high, product="Product 0")

# Cross matrix at median price
median_p = np.median(df["price"].values)
p_median_grid = np.full((1, 2), median_p)  # For 2 products
cross_matrix = np.zeros((2, 2))
for j in range(2):
    for k in range(2):
        cross_matrix[j, k], _, _ = estimator.compute_elasticity(
            p_median_grid, x_fixed[:1], z_fixed[:1], j, k
        )
plot_cross_elasticity_table(cross_matrix, ["Product 0", "Product 1"])
