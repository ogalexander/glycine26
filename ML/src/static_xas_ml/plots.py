"""Plotting helpers for the static-XAS ML pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_pca_explained(pca, path: str | Path) -> None:
    path = _ensure_parent(path)
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(np.arange(1, ratio.size + 1), ratio)
    ax.plot(np.arange(1, ratio.size + 1), np.cumsum(ratio), marker="o", color="black")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_predicted_examples(
    pixel_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    n_examples: int = 8,
) -> None:
    path = _ensure_parent(path)
    n = min(int(n_examples), y_true.shape[0])
    if n == 0:
        return
    idx = np.linspace(0, y_true.shape[0] - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(2, 1.6 * n)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, idx):
        ax.plot(pixel_axis, y_true[row], label="true reference", lw=1.2)
        ax.plot(pixel_axis, y_pred[row], label="predicted I0", lw=1.0)
        ax.set_ylabel("shape")
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("VLS pixel")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_histograms(errors: dict[str, np.ndarray], path: str | Path) -> None:
    path = _ensure_parent(path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].hist(errors["rmse"], bins=50, color="#4477AA", alpha=0.85)
    axes[0].set_xlabel("RMSE")
    axes[1].hist(errors["centroid_error_px"], bins=50, color="#66AA55", alpha=0.85)
    axes[1].set_xlabel("Centroid error (px)")
    axes[2].hist(errors["width_error_px"], bins=50, color="#AA7744", alpha=0.85)
    axes[2].set_xlabel("RMS width error (px)")
    for ax in axes:
        ax.set_ylabel("Shots")
        ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_binned_absorbance(df: pd.DataFrame, path: str | Path, *, title: str) -> None:
    path = _ensure_parent(path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, group in df.groupby("gmd_bin", sort=False):
        group = group.sort_values("x")
        ax.errorbar(
            group["x"],
            group["absorbance_mean"],
            yerr=group["absorbance_sem"],
            marker="o",
            ms=4,
            lw=1.2,
            capsize=2,
            label=str(label),
        )
    ax.set_xlabel(str(df["x_label"].iloc[0]))
    ax.set_ylabel("ML shape absorbance")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(title="GMD bin", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
