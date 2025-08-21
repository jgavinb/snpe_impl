import matplotlib.pyplot as plt
import numpy as np


def plot_elasticity_profile(
    p_grid,
    elasticities,
    ci_low=None,
    ci_high=None,
    channel=None,
    filename="elasticity_profile.png",
):
    """Plot elasticity vs. price, like paper Fig. 2/A1."""
    plt.figure(figsize=(8, 6))
    label = f"Elasticity ({channel})" if channel else "Elasticity"
    plt.plot(p_grid, elasticities, label=label)
    if ci_low is not None and ci_high is not None:
        plt.fill_between(p_grid.flatten(), ci_low, ci_high, alpha=0.3, label="95% CI")
    plt.xlabel("Price")
    plt.ylabel("Price Elasticity")
    plt.title("Price Elasticity Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_demand_curve(p_grid, quantities, filename="demand_curve.png"):
    """Plot predicted demand vs. price."""
    plt.figure(figsize=(8, 6))
    plt.plot(p_grid, quantities, label="Demand Curve")
    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title("Demand Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
