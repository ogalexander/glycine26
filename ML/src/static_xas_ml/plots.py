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
        ax.plot(pixel_axis, y_pred[row], label="predicted virtual I0", lw=1.0)
        ax.set_ylabel("VLS")
        ax.grid(True, alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("VLS pixel")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_histograms(errors: dict[str, np.ndarray], path: str | Path) -> None:
    path = _ensure_parent(path)
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.2), constrained_layout=True)
    axes[0].hist(errors["rmse"], bins=50, color="#4477AA", alpha=0.85)
    axes[0].set_xlabel("RMSE")
    axes[1].hist(errors["integral_relative_error"], bins=50, color="#66AA55", alpha=0.85)
    axes[1].set_xlabel("Integral rel. error")
    axes[2].hist(errors["centroid_error_px"], bins=50, color="#AA7744", alpha=0.85)
    axes[2].set_xlabel("Centroid error (px)")
    axes[3].hist(errors["width_error_px"], bins=50, color="#AA4499", alpha=0.85)
    axes[3].set_xlabel("RMS width error (px)")
    axes[4].hist(errors["similarity"], bins=50, color="#228833", alpha=0.85)
    axes[4].set_xlabel("Similarity")
    for ax in axes:
        ax.set_ylabel("Shots")
        ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_coverage(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_parent(path)
    if df.empty:
        return
    data = df.sort_values("sample_fraction_outside_reference_p01_p99", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.38 * len(data))), constrained_layout=True)
    ax.barh(
        data["feature_name"],
        data["sample_fraction_outside_reference_p01_p99"],
        color="#4477AA",
        alpha=0.85,
    )
    ax.set_xlabel("Sample fraction outside reference p01-p99")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_binned_absorbance(df: pd.DataFrame, path: str | Path, *, title: str) -> None:
    path = _ensure_parent(path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    grouped = list(df.groupby("gmd_bin", sort=False))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(grouped) - 1, 1)) for i in range(len(grouped))]
    for color, (label, group) in zip(colors, grouped):
        group = group.sort_values("x")
        ax.plot(
            group["x"],
            group["absorbance_mean"],
            marker="o",
            lw=1.8,
            color=color,
            label=str(label),
        )
    ax.set_xlabel(str(df["x_label"].iloc[0]))
    ax.set_ylabel("ML absorbance")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="GMD bin", fontsize=8, title_fontsize=9, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)
