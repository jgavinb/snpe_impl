import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_elasticity_profile(
    p_grid,
    elasticities,
    ci_low=None,
    ci_high=None,
    product="Product",
    filename="elasticity_profile.png",
):
    """Plot elasticity vs. price."""
    plt.figure(figsize=(8, 6))
    plt.plot(p_grid, elasticities, label=f"Elasticity ({product})")
    if ci_low is not None:
        plt.fill_between(p_grid.flatten(), ci_low, ci_high, alpha=0.3, label="95% CI")
    plt.xlabel("Price")
    plt.ylabel("Own-Price Elasticity")
    plt.title("Own-Price Elasticity Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_demand_curve(p_grid, quantities, filename="demand_curve.png"):
    """Plot demand vs. price."""
    plt.figure(figsize=(8, 6))
    plt.plot(p_grid, quantities, label="Demand")
    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title("Demand Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_cross_elasticity_table(
    elasticity_matrix, products, filename="cross_elasticity_table.png"
):
    """Table for cross-elasticities at median."""
    df = pd.DataFrame(elasticity_matrix, index=products, columns=products)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("tight")
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    plt.savefig(filename)
    plt.close()
