"""Visualizations for cluster maps, PCA scree plots, and feature importances."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_pca_scree(pca_model, title="PCA Explained Variance"):
    """Plot individual and cumulative explained variance ratios."""
    exp_var = pca_model.explained_variance_ratio_
    cum_var = exp_var.cumsum()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.6, label='Individual')
    ax1.plot(range(1, len(exp_var) + 1), cum_var, 'r-o', label='Cumulative')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(title)
    ax1.legend()
    plt.tight_layout()
    return fig
