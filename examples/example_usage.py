import numpy as np
import pandas as pd
from snpe_impl.estimator import BaggedNN
from snpe_impl.elasticity import PriceElasticityEstimator
from snpe_impl.utils import load_data, preprocess_data
from snpe_impl.visuals import plot_elasticity_profile, plot_demand_curve

# Synthetic data for use case: 2 products, 2 channels, limited prices
data = {
    "market_id": np.repeat(np.arange(50), 4),
    "product_id": np.tile([0, 1], 100),
    "channel": np.tile(["Amazon", "DTC"], 100),
    "price": np.random.uniform(10, 20, 200)
    + 5 * (np.random.rand(200) > 0.9),  # Limited changes
    "quantity": np.random.poisson(100),
    "covar_promo": np.random.rand(200),
    "inst_tariff": np.random.uniform(0, 1, 200),  # Instrument e.g. tariff cost
}
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)

# Load and preprocess (one-hot channels)
df = load_data("data.csv")
X_h, y_q, X_g, y_p, x_dim, z_dim = preprocess_data(df)

# Fit estimators (assume single product/channel for sim; loop for multi)
h_est = BaggedNN().fit(X_h, y_q)  # h: quantity on p+x+z
g_est = BaggedNN().fit(X_g, y_p)  # g: price on x+z

estimator = PriceElasticityEstimator(h_estimators=[h_est], g_estimators=[g_est])

# Specify channel (fix in x_fixed, e.g., [1,0] for Amazon if one-hot)
x_fixed_amazon = np.array([[0] * x_dim])  # Placeholder; set channel dummies
x_fixed_amazon[0, 0] = 1  # e.g., chan_Amazon=1
z_fixed = np.array([[0.5]])  # Fixed instrument

# Grid for future prices (post-tariff increase)
p_grid = np.linspace(10, 25, 50).reshape(-1, 1)  # Include higher prices

# Compute elasticity (own for product 0, channel Amazon)
elast, ci_low, ci_high = estimator.compute_elasticity(
    p_grid, x_fixed_amazon, z_fixed, product_j=0, product_k=0
)

# Visuals
plot_elasticity_profile(p_grid, elast, ci_low, ci_high, channel="Amazon")
quant_grid = h_est.predict(np.hstack([p_grid, x_fixed_amazon, z_fixed]))
plot_demand_curve(p_grid, quant_grid)
