import numpy as np
import pandas as pd


def load_data(csv_path):
    """Load CSV: date, product_id, [category], price, quantity, [covar_*], [inst_*]."""
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df, category=True, products=True, instruments=True):
    """Pivot to wide, one-hot category/product, normalize continuous."""
    # Pivot to wide for multi-product
    if products:
        df = df.pivot(
            index="date", columns="product_id", values=["price", "quantity"]
        ).fillna(0)
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    if category:
        df = pd.get_dummies(df, columns=["category"], prefix="cat")
    continuous_cols = [
        c for c in df if c.startswith(("price", "quantity", "covar_", "inst_"))
    ]
    for col in continuous_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    # Extract arrays
    p_cols = [c for c in df if c.startswith("price")]
    s_cols = [c for c in df if c.startswith("quantity")]
    x_cols = [c for c in df if not c.startswith(("price", "quantity", "inst_"))]
    z_cols = [c for c in df if c.startswith("inst_")] if instruments else []
    X_h = df[p_cols + x_cols + z_cols].values  # For h: p + x + z
    y_s = df[s_cols].values.T  # y_s[j] for s_j
    X_g = df[x_cols + z_cols].values  # For g: x + z
    y_p = df[p_cols].values.T  # y_p[k] for p_k
    return X_h, y_s, X_g, y_p, len(x_cols), len(z_cols)
