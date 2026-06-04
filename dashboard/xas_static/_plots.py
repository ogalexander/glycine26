"""
Pure plot functions — each returns a plotly.graph_objects.Figure.
Inputs are numpy arrays; no Streamlit or I/O dependencies.
"""
from __future__ import annotations

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_TAB_BLUE = "#1f77b4"


def _viridis(n: int) -> list[str]:
    """n colours sampled from Viridis, matching the notebook colour scheme."""
    if n == 1:
        return [pc.sample_colorscale("Viridis", [0.5])[0]]
    return pc.sample_colorscale("Viridis", list(np.linspace(0.05, 0.95, n)))


def plot_correlation(
    vls: np.ndarray,
    gmd: np.ndarray,
    n_shots: np.ndarray,
    n_bins: int = 50,
) -> go.Figure:
    """GMD vs mean-VLS per shot, all energies pooled. 2-panel: hist2d + binned mean."""
    gmd_flat = gmd.reshape(-1)
    vls_flat = vls.reshape(-1, vls.shape[2])
    ok = np.isfinite(gmd_flat) & np.all(np.isfinite(vls_flat), axis=1)
    x = gmd_flat[ok]
    y = np.nanmean(vls_flat[ok], axis=1)

    bin_edges = np.percentile(x, np.linspace(0, 100, 21))
    idx = np.clip(np.digitize(x, bin_edges) - 1, 0, len(bin_edges) - 2)
    x_m = np.array([np.nanmean(x[idx == b]) if (idx == b).any() else np.nan
                    for b in range(len(bin_edges) - 1)])
    y_m = np.array([np.nanmean(y[idx == b]) if (idx == b).any() else np.nan
                    for b in range(len(bin_edges) - 1)])
    y_s = np.array([np.nanstd(y[idx == b])  if (idx == b).any() else np.nan
                    for b in range(len(bin_edges) - 1)])

    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.linspace(float(np.nanmin(x_m)), float(np.nanmax(x_m)), 50)

    x_edges = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    y_edges = np.linspace(np.percentile(y, 1), np.percentile(y, 99), n_bins + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    counts = counts.T  # (ny, nx)
    counts[counts == 0] = np.nan

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f"GMD vs mean VLS  ({ok.sum()} shots)",
                                        "Binned mean ± std"))
    fig.add_trace(go.Heatmap(
        x=0.5 * (x_edges[:-1] + x_edges[1:]),
        y=0.5 * (y_edges[:-1] + y_edges[1:]),
        z=counts,
        colorscale="Viridis",
        colorbar=dict(x=0.45, title="shots"),
        zmin=1, zsmooth=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=xl, y=slope * xl + intercept,
        mode="lines", line=dict(color="white", width=1.5),
        name=f"slope={slope:.3g},  r={r:.3f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_m, y=y_m,
        error_y=dict(type="data", array=y_s, visible=True),
        mode="lines+markers", name="binned mean ± std",
        marker=dict(color=_TAB_BLUE),
    ), row=1, col=2)
    fig.update_xaxes(title_text="GMD (uJ)")
    fig.update_yaxes(title_text="Mean VLS (arb.)", col=1)
    fig.update_yaxes(title_text="Mean VLS (arb.)", col=2)
    fig.update_layout(height=450, showlegend=True)
    return fig


def plot_shot_counts(
    n_per_bin: np.ndarray,
    nominal_energies: np.ndarray,
    gmd_edges: np.ndarray,
) -> go.Figure:
    """Shot-count heatmap (energy × GMD bin) and marginal line."""
    N_GMD = n_per_bin.shape[1]
    bin_labels = [f"[{gmd_edges[i]:.2f}, {gmd_edges[i+1]:.2f})" for i in range(N_GMD)]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Shot count per (energy, GMD) bin",
                                        "Shots per energy (all GMD bins)"))
    fig.add_trace(go.Heatmap(
        x=nominal_energies,
        y=bin_labels,
        z=n_per_bin.T.astype(float),
        colorscale="Viridis",
        colorbar=dict(x=0.45, title="shots"),
        zmin=0,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nominal_energies,
        y=n_per_bin.sum(axis=1),
        mode="lines+markers",
        name="total shots",
        marker=dict(color=_TAB_BLUE),
    ), row=1, col=2)
    fig.update_xaxes(title_text="nominal photon energy (eV)")
    fig.update_yaxes(title_text="GMD bin", col=1)
    fig.update_yaxes(title_text="shots", col=2)
    fig.update_layout(height=450, showlegend=False)
    return fig


def plot_mean_gmd(
    G_mean: np.ndarray,
    n_per_bin: np.ndarray,
    nominal_energies: np.ndarray,
    gmd_edges: np.ndarray,
) -> go.Figure:
    """Mean-GMD heatmap and per-bin vs energy lines."""
    N_GMD = G_mean.shape[1]
    bin_labels = [f"[{gmd_edges[i]:.2f}, {gmd_edges[i+1]:.2f})" for i in range(N_GMD)]
    colors = _viridis(N_GMD)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Mean GMD per (energy, GMD) bin",
                                        "Mean GMD vs energy per bin"))
    fig.add_trace(go.Heatmap(
        x=nominal_energies,
        y=bin_labels,
        z=G_mean.T,
        colorscale="Magma",
        colorbar=dict(x=0.45, title="mean GMD (uJ)"),
    ), row=1, col=1)
    for g in range(N_GMD):
        fig.add_trace(go.Scatter(
            x=nominal_energies,
            y=G_mean[:, g],
            mode="lines+markers",
            name=bin_labels[g],
            line=dict(color=colors[g]),
        ), row=1, col=2)
    fig.update_xaxes(title_text="nominal photon energy (eV)")
    fig.update_yaxes(title_text="GMD bin", col=1)
    fig.update_yaxes(title_text="mean GMD (uJ)", col=2)
    fig.update_layout(height=450)
    return fig


def plot_vls_spectra(
    A_mean: np.ndarray,
    nominal_energies: np.ndarray,
    vls_pixels: np.ndarray,
    gmd_edges: np.ndarray,
    energy_idx: int = 0,
) -> go.Figure:
    """
    Interactive VLS spectra viewer.

    Shows mean VLS intensity as a heatmap (GMD bin × pixel) for one
    selected energy. Use ``energy_idx`` to pick which energy to display;
    the dropdown is handled in the app layer via the slider param.
    """
    N_E, N_GMD, _ = A_mean.shape
    energy_idx = int(np.clip(energy_idx, 0, N_E - 1))
    bin_labels = [f"[{gmd_edges[i]:.2f}, {gmd_edges[i+1]:.2f})" for i in range(N_GMD)]
    z = A_mean[energy_idx]           # (N_GMD, n_pixels)
    finite = np.isfinite(z)
    zmin, zmax = (float(np.nanpercentile(z[finite], 2)),
                  float(np.nanpercentile(z[finite], 98))) if finite.any() else (0.0, 1.0)

    fig = go.Figure(go.Heatmap(
        x=vls_pixels.tolist(),
        y=bin_labels,
        z=z,
        colorscale="Viridis",
        zmin=zmin, zmax=zmax,
        colorbar=dict(title="mean intensity (arb.)"),
    ))
    fig.update_xaxes(title_text="VLS pixel")
    fig.update_yaxes(title_text="GMD bin")
    fig.update_layout(
        title=f"Mean VLS — {nominal_energies[energy_idx]:.2f} eV",
        height=400,
    )
    return fig


def plot_xas(
    A_mean: np.ndarray,
    G_mean: np.ndarray,
    nominal_energies: np.ndarray,
    gmd_edges: np.ndarray,
) -> go.Figure:
    """XAS(E) = <GMD> / sum_pix(<VLS>) per GMD bin."""
    N_GMD = A_mean.shape[1]
    vls_sum = np.nansum(A_mean, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        xas = G_mean / vls_sum
    colors = _viridis(N_GMD)

    fig = go.Figure()
    for g in range(N_GMD):
        fig.add_trace(go.Scatter(
            x=nominal_energies,
            y=xas[:, g],
            mode="lines+markers",
            name=f"[{gmd_edges[g]:.2f}, {gmd_edges[g+1]:.2f})",
            line=dict(color=colors[g], width=1.8),
        ))
    fig.update_xaxes(title_text="nominal photon energy (eV)")
    fig.update_yaxes(title_text="XAS = <GMD> / sum_pix(<VLS>)")
    fig.update_layout(
        title="XAS(E) per GMD bin",
        legend_title="GMD bin (uJ)",
        height=450,
    )
    return fig
