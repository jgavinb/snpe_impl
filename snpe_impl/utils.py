import numpy as np
import pandas as pd


def load_data(csv_path):
    """Load CSV data: expect columns market_id, product_id, channel, price, quantity, [covariates], [instruments]."""
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df, channels=True, products=True, instruments=True):
    """Preprocess: one-hot channels/products, normalize continuous."""
    if channels:
        df = pd.get_dummies(df, columns=["channel"], prefix="chan")
    if products:
        df = pd.get_dummies(df, columns=["product_id"], prefix="prod")
    continuous_cols = ["price", "quantity"] + [
        c for c in df if c.startswith(("covar_", "inst_"))
    ]
    for col in continuous_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    # Return X (p + x + z), y_quantity, y_price (for g)
    p = df["price"].values.reshape(-1, 1)
    x = df[[c for c in df if c.startswith(("chan_", "prod_", "covar_"))]].values
    z = (
        df[[c for c in df if c.startswith("inst_")]].values
        if instruments
        else np.zeros((len(df), 0))
    )
    y_q = df["quantity"].values
    y_p = p.flatten()  # For g
    X_h = np.hstack([p, x, z])  # For h
    X_g = np.hstack([x, z])  # For g (prices on instruments + x if included)
    return X_h, y_q, X_g, y_p, x.shape[1], z.shape[1]
