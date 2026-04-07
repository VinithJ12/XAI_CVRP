"""
utils/visualize.py
─────────────────────────────────────────────────────────────────────────────
Helper functions for plotting routes and SHAP results.
Kept separate so the main scripts stay clean and readable.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_route(instance, route: list, title: str = "CVRP Route", save_path: str = None):
    """
    Draw a CVRP solution on a 2D map.

    Args:
        instance  : CVRPInstance with coords and demands
        route     : list of customer indices in visit order (e.g. [0,3,1,0,2,0])
                    0 = depot, must start and end at 0
        title     : plot title string
        save_path : if given, save PNG to this path instead of showing
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    coords = instance.coords

    # ── Draw the route as connected arrows ─────────────────────────────────
    for i in range(len(route) - 1):
        start = coords[route[i]]
        end   = coords[route[i + 1]]
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5)
        )

    # ── Plot depot (large red square) ─────────────────────────────────────
    depot = coords[0]
    ax.scatter(*depot, s=200, c="red", marker="s", zorder=5, label="Depot")
    ax.annotate("Depot", depot, textcoords="offset points",
                xytext=(6, 6), fontsize=9, color="red")

    # ── Plot customers (circles, size proportional to demand) ─────────────
    for i in range(1, instance.n_customers + 1):
        c = coords[i]
        demand = instance.demands[i]
        ax.scatter(*c, s=80 + demand * 400, c="steelblue",
                   zorder=4, alpha=0.8)
        ax.annotate(f"{i}\n({demand:.2f})", c,
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="navy")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Legend explaining circle size
    small_patch = mpatches.Patch(color="steelblue", alpha=0.6,
                                  label="Customer (size ∝ demand)")
    ax.legend(handles=[small_patch,
                        mpatches.Patch(color="red", label="Depot")],
               loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved route plot → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_shap_bar(shap_values_mean: np.ndarray, feature_names: list,
                  title: str = "Mean |SHAP| values", save_path: str = None):
    """
    Horizontal bar chart of mean absolute SHAP values.
    Longer bar = more influential feature overall.

    Args:
        shap_values_mean : 1D array of mean |SHAP| per feature
        feature_names    : list of feature name strings (same order)
        title            : chart title
        save_path        : optional save path
    """
    # Sort by importance for readability
    order = np.argsort(shap_values_mean)
    sorted_vals  = shap_values_mean[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_vals)))
    bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP value|  (impact on model decision)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved SHAP bar chart → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_shap_scatter(shap_df, feature: str, save_path: str = None):
    """
    Scatter plot: feature value (x) vs SHAP value (y) for one feature.
    Helps answer: "when distance is HIGH, does it push the model AWAY?"

    Args:
        shap_df  : pandas DataFrame with columns [feature, shap_<feature>]
        feature  : feature name to plot
        save_path: optional save path
    """
    import matplotlib.pyplot as plt

    shap_col = f"shap_{feature}"
    if shap_col not in shap_df.columns or feature not in shap_df.columns:
        print(f"  [skip] columns missing for feature '{feature}'")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(shap_df[feature], shap_df[shap_col],
               alpha=0.4, s=15, c="steelblue")
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)

    ax.set_xlabel(f"Feature value: {feature}")
    ax.set_ylabel("SHAP value (effect on model choice)")
    ax.set_title(f"How '{feature}' affects decisions", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved scatter plot → {save_path}")
    else:
        plt.show()
    plt.close()
