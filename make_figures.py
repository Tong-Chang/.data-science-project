"""Regenerate the summary figures for the King County house price analysis.

The two exploratory figures (fig1, fig2) are produced inside the notebook. This script
reproduces the two summary figures that aggregate the notebook's numeric results:

    images/fig3_model_r2_comparison.png   R^2 of the five fitted models
    images/fig4_feature_correlation.png   Pearson correlation of each feature with price

Values are the results recorded in the notebook run committed alongside this script.

Usage:  python make_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# Model name, R^2, evaluation basis
MODELS = [
    ("Linear: sqft_living only", 0.4929, "in-sample"),
    ("Linear: 11 features", 0.6577, "in-sample"),
    ("Scaler + poly + linear", 0.7513, "in-sample"),
    ("Ridge (a=0.1)", 0.6479, "held-out test"),
    ("Ridge + 2nd-degree poly", 0.7003, "held-out test"),
]

# Feature, Pearson r against price
CORRELATIONS = [
    ("sqft_living", 0.702),
    ("grade", 0.667),
    ("sqft_above", 0.606),
    ("sqft_living15", 0.585),
    ("bathrooms", 0.526),
    ("view", 0.397),
    ("sqft_basement", 0.324),
    ("bedrooms", 0.309),
    ("lat", 0.307),
    ("waterfront", 0.266),
    ("floors", 0.257),
    ("sqft_lot", 0.090),
    ("long", 0.022),
    ("zipcode", -0.053),
]

IN_SAMPLE_COLOR = "#8FAADC"
TEST_COLOR = "#2E75B6"


def plot_model_r2():
    names = [m[0] for m in MODELS]
    scores = [m[1] for m in MODELS]
    colors = [IN_SAMPLE_COLOR if m[2] == "in-sample" else TEST_COLOR for m in MODELS]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("R²")
    ax.set_title("Model comparison: variance in house price explained", fontsize=13)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.015,
            f"{score:.3f}",
            ha="center",
            fontsize=10,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=IN_SAMPLE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=TEST_COLOR),
    ]
    ax.legend(
        handles,
        ["In-sample (all data)", "Held-out test set"],
        loc="upper left",
        frameon=False,
    )

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_model_r2_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlations():
    features = [c[0] for c in CORRELATIONS]
    values = [c[1] for c in CORRELATIONS]
    colors = ["#C00000" if v < 0 else TEST_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(features, values, color=colors, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with price")
    ax.set_title("Feature correlation with sale price", fontsize=13)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for y, v in enumerate(values):
        offset = 0.012 if v >= 0 else -0.012
        ax.text(v + offset, y, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_feature_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for produced in (plot_model_r2(), plot_correlations()):
        print("wrote", produced)
