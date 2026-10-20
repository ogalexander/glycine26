"""
Thin thesis-style static-XAS helper for the 5 mmol core notebook.

The analysis follows the thesis normalization:

    OD(E) = log10(sum(GMD) / sum(integrated VLS))

with optional per-nominal-energy intercept correction:

    integrated VLS_corr = integrated VLS - a(E_nominal)

where ``a(E)`` comes from a free-intercept fit of integrated VLS against the
mapped hall GMD.  The error bars are delta-method SEM, matching the existing
thesis diagnostics script.  No values are written back to H5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

import static_xas_analysis as sxa


@dataclass(frozen=True)
class ThesisCoreXASConfig:
    """Configuration for the compact thesis-style XAS notebook."""

    input_h5: str | Path
    vls_intensity_threshold: Optional[float] = 1000.0
    gmd_bin_mode: str = "fixed"  # "fixed" or "quantile"
    gmd_edges: Sequence[float] = field(default_factory=lambda: tuple(np.arange(0.0, 11.0, 1.0)))
    n_gmd_bins: int = 5
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    center_method: str = "com"
    center_half_window: int = 5
    center_chunk_size: int = 50_000
    proxy_pixel_bin_width: int = 1
    correction_mode: str = "per_energy_intercept"  # "none" or "per_energy_intercept"
    unit_scale_mode: str = "none"  # "none", "reference_window", or "global_slope"
    unit_reference_window_ev: Optional[Sequence[float]] = (272.0, 273.0)
    chunk_size: int = 20_000
    pixel_map_config: dict[str, Any] = field(
        default_factory=lambda: {
            "source_to_grating_m": 0.85,
            "grating_to_screen_m": 0.5429,
            "incident_angle_deg": 86.78,
            "groove_density_lines_per_mm": 1200.0,
            "diffraction_order": 1,
            "magnification": 3.0,
            "pixel_pitch_m": 50e-6,
            "axis_sign": 1,
        }
    )
    pixel_map_calibration: dict[str, Any] = field(
        default_factory=lambda: {
            "pixel": 524.0,
            "energy_eV": 284.2,
        }
    )
    inverse_map_range_ev: Sequence[float] = (260.0, 290.0)
    inverse_map_samples: int = 10_000


@dataclass(frozen=True)
class ThesisCoreLinearFit:
    """Per-nominal-energy fit of integrated VLS versus mapped hall GMD."""

    intercept: np.ndarray
    slope: np.ndarray
    r2: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class ThesisCoreBinnedXAS:
    """Summed GMD/VLS state binned by photon-energy coordinate and GMD bin."""

    x: np.ndarray
    gmd_sum: np.ndarray
    trans_sum: np.ndarray
    gmd_sumsq: np.ndarray
    trans_sumsq: np.ndarray
    gmd_trans_sum: np.ndarray
    counts: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    x_label: str
    metadata: dict[str, Any]

    @property
    def od(self) -> np.ndarray:
        return _log10_ratio(self.gmd_sum, self.trans_sum)

    @property
    def ratio(self) -> np.ndarray:
        return _safe_divide(self.gmd_sum, self.trans_sum)

    @property
    def od_sem(self) -> np.ndarray:
        return _delta_log_ratio_sem(
            self.gmd_sum,
            self.trans_sum,
            self.gmd_sumsq,
            self.trans_sumsq,
            self.gmd_trans_sum,
            self.counts,
        )


@dataclass(frozen=True)
class ThesisCoreUnitScale:
    """Single VLS-to-uJ scale applied to binned transmitted intensity."""

    mode: str
    vls_per_uJ: float
    source: str
    reference_window_ev: Optional[tuple[float, float]] = None

    @property
    def global_vls_per_uJ(self) -> float:
        """Backward-compatible alias for older notebook cells."""

        return self.vls_per_uJ


@dataclass(frozen=True)
class ThesisCoreXASResult:
    """Computed state for the compact thesis-style notebook."""

    config: ThesisCoreXASConfig
    path: Path
    attrs: dict[str, Any]
    gmd_source: str
    nominal_energies: np.ndarray
    vls_pixels: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    filter_counts: dict[str, np.ndarray]
    fits: ThesisCoreLinearFit
    nominal_raw: ThesisCoreBinnedXAS
    nominal_corrected: ThesisCoreBinnedXAS
    nominal: ThesisCoreBinnedXAS
    converted: ThesisCoreBinnedXAS
    unit_scale: ThesisCoreUnitScale
    pixel_map_config: Any
    pixel_map_calibration: Any


@dataclass(frozen=True)
class ThesisCoreBootstrapErrors:
    """Shot-level bootstrap OD uncertainty aligned to a thesis-core result."""

    nominal_od_sigma: np.ndarray
    converted_od_sigma: np.ndarray
    nominal_counts: np.ndarray
    converted_counts: np.ndarray
    n_bootstrap: int
    seed: int
    min_shots_per_bin: int
    batch_size: int


def compute_thesis_core_xas(config: ThesisCoreXASConfig) -> ThesisCoreXASResult:
    """Compute nominal and converted/proxy thesis-style static XAS."""

    _validate_config(config)
    path = Path(config.input_h5)
    info = _read_h5_info(path)
    first = _first_pass(path, config, info)
    gmd_edges = _choose_gmd_edges(first["accepted_gmd"], config)
    gmd_labels = _format_gmd_labels(gmd_edges, first["accepted_gmd"])
    fits = _linear_fit_from_accumulators(first["fit"])
    second = _second_pass(path, config, info, gmd_edges, gmd_labels, fits)
    nominal_raw = second["nominal_raw"]
    nominal_corrected = second["nominal_corrected"]
    nominal_active = nominal_corrected if config.correction_mode == "per_energy_intercept" else nominal_raw
    unit_scale = _resolve_unit_scale(config, nominal_active)
    nominal_raw = _apply_unit_scale(nominal_raw, unit_scale)
    nominal_corrected = _apply_unit_scale(nominal_corrected, unit_scale)
    nominal = nominal_corrected if config.correction_mode == "per_energy_intercept" else nominal_raw
    converted = _apply_unit_scale(second["converted"], unit_scale)

    return ThesisCoreXASResult(
        config=config,
        path=path,
        attrs=info["attrs"],
        gmd_source=info["gmd_source"],
        nominal_energies=info["nominal_energies"],
        vls_pixels=info["vls_pixels"],
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        filter_counts=first["filter_counts"],
        fits=fits,
        nominal_raw=nominal_raw,
        nominal_corrected=nominal_corrected,
        nominal=nominal,
        converted=converted,
        unit_scale=unit_scale,
        pixel_map_config=second["pixel_map_config"],
        pixel_map_calibration=second["pixel_map_calibration"],
    )


def compute_thesis_core_bootstrap_errors(
    result: ThesisCoreXASResult,
    *,
    n_bootstrap: int = 500,
    seed: int = 123,
    min_shots_per_bin: int = 100,
    batch_size: int = 50,
) -> ThesisCoreBootstrapErrors:
    """Compute shot-level bootstrap OD sigma for nominal and converted bins.

    The resampling is done independently inside each x-coordinate by GMD-bin
    cell.  Each bootstrap replicate recomputes the same thesis-style quantity:

        OD = log10(sum(GMD) / sum(transmitted VLS))
    """

    if int(n_bootstrap) < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if int(min_shots_per_bin) < 2:
        raise ValueError("min_shots_per_bin must be at least 2")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    config = result.config
    info = _read_h5_info(result.path)
    n_nominal, n_gmd = result.nominal.counts.shape
    n_converted = result.converted.counts.shape[0]
    n_nominal_cells = n_nominal * n_gmd
    n_converted_cells = n_converted * n_gmd
    nominal_g_parts: list[list[np.ndarray]] = [[] for _ in range(n_nominal_cells)]
    nominal_t_parts: list[list[np.ndarray]] = [[] for _ in range(n_nominal_cells)]
    converted_g_parts: list[list[np.ndarray]] = [[] for _ in range(n_converted_cells)]
    converted_t_parts: list[list[np.ndarray]] = [[] for _ in range(n_converted_cells)]

    n_proxy = int(np.ceil(int(info["n_pixels"]) / int(config.proxy_pixel_bin_width)))
    proxy_to_row = _proxy_group_to_converted_row(result, n_proxy)
    scale = float(result.unit_scale.vls_per_uJ)
    if result.unit_scale.mode == "none":
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Bootstrap requires a finite positive unit scale, got {scale}")
    use_corrected = config.correction_mode == "per_energy_intercept"

    for ie, vls, gmd in _iter_energy_chunks(result.path, info, config.chunk_size):
        keep, _ = _quality_mask(vls, gmd, config)
        if not np.any(keep):
            continue
        vv = np.asarray(vls[keep], dtype=np.float64)
        gg = np.asarray(gmd[keep], dtype=np.float64)
        raw_trans = _integrated_vls(vv)
        intercept = result.fits.intercept[ie]
        offset = 0.0 if not np.isfinite(intercept) else float(intercept)
        trans = raw_trans - offset if use_corrected else raw_trans
        trans = trans / scale
        gbin, in_gmd = _gmd_bin_indices(gg, result.gmd_edges)

        nominal_flat = ie * n_gmd + gbin
        _append_bootstrap_parts(nominal_g_parts, nominal_t_parts, nominal_flat, gg, trans, in_gmd)

        spectra_for_center = vv - offset / int(info["n_pixels"]) if use_corrected else vv
        center_rel, _ = sxa.estimate_center_pixels(
            spectra_for_center,
            method=config.center_method,
            half_window=int(config.center_half_window),
            chunk_size=int(config.center_chunk_size),
        )
        valid_center = np.isfinite(center_rel)
        valid_proxy = valid_center & in_gmd & np.isfinite(gg) & np.isfinite(trans)
        if not np.any(valid_proxy):
            continue
        rel = np.clip(np.round(center_rel[valid_proxy]).astype(np.int64), 0, int(info["n_pixels"]) - 1)
        proxy_group = rel // int(config.proxy_pixel_bin_width)
        converted_row = proxy_to_row[proxy_group]
        in_converted = converted_row >= 0
        if not np.any(in_converted):
            continue
        converted_flat = converted_row[in_converted] * n_gmd + gbin[valid_proxy][in_converted]
        _append_bootstrap_parts(
            converted_g_parts,
            converted_t_parts,
            converted_flat,
            gg[valid_proxy][in_converted],
            trans[valid_proxy][in_converted],
            np.ones(int(np.sum(in_converted)), dtype=bool),
        )

    rng = np.random.default_rng(int(seed))
    nominal_sigma, nominal_counts = _bootstrap_sigma_grid(
        nominal_g_parts,
        nominal_t_parts,
        shape=result.nominal.counts.shape,
        n_bootstrap=int(n_bootstrap),
        rng=rng,
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )
    converted_sigma, converted_counts = _bootstrap_sigma_grid(
        converted_g_parts,
        converted_t_parts,
        shape=result.converted.counts.shape,
        n_bootstrap=int(n_bootstrap),
        rng=rng,
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )
    return ThesisCoreBootstrapErrors(
        nominal_od_sigma=nominal_sigma,
        converted_od_sigma=converted_sigma,
        nominal_counts=nominal_counts,
        converted_counts=converted_counts,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )


def plot_thesis_xas_pair_matplotlib(
    result: ThesisCoreXASResult,
    *,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot nominal and converted/proxy OD in two rows with shared x-axis."""

    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, constrained_layout=True)
    _plot_binned_od_on_axis(
        axes[0],
        result.nominal,
        title="Nominal energy",
        xlim=shared_xlim,
        ylim=nominal_ylim,
        show_errorbars=show_errorbars,
        show_legend=True,
    )
    _plot_binned_od_on_axis(
        axes[1],
        result.converted,
        title=f"Converted/proxy energy ({result.config.center_method})",
        xlim=shared_xlim,
        ylim=converted_ylim,
        show_errorbars=show_errorbars,
        show_legend=False,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("photon energy (eV)")
    if title is None:
        title = (
            "5 mmol static XAS "
            f"({result.config.correction_mode}, threshold={result.config.vls_intensity_threshold})"
        )
    fig.suptitle(title)
    return fig, axes


def plot_thesis_xas_pair_plotly(
    result: ThesisCoreXASResult,
    *,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot nominal and converted/proxy OD in two Plotly rows with independent x-axes."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_thesis_xas_pair_plotly") from exc

    title = title or (
        "5 mmol static XAS "
        f"({result.config.correction_mode}, threshold={result.config.vls_intensity_threshold})"
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Nominal energy", f"Converted/proxy energy ({result.config.center_method})"),
        shared_xaxes=False,
        vertical_spacing=0.10,
    )
    colors = [_rgba_to_hex(c) for c in _colors(len(result.gmd_labels))]
    for row, binned in enumerate((result.nominal, result.converted), start=1):
        x = binned.x
        order = _finite_x_order(x)
        x_plot = x[order]
        y = binned.od[order]
        sem = binned.od_sem[order]
        for ig, label in enumerate(result.gmd_labels):
            error_y = {"type": "data", "array": sem[:, ig], "visible": True} if show_errorbars else None
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y[:, ig],
                    error_y=error_y,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=(row == 1),
                    marker={"size": 5, "color": colors[ig]},
                    line={"width": 1.5, "color": colors[ig]},
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="nominal photon energy (eV)", row=1, col=1)
    fig.update_xaxes(title_text="converted photon energy (eV)", row=2, col=1)
    fig.update_yaxes(title_text=_od_y_label(result.nominal), row=1, col=1)
    fig.update_yaxes(title_text=_od_y_label(result.converted), row=2, col=1)
    if nominal_xlim is not None:
        fig.update_xaxes(range=list(nominal_xlim), row=1, col=1)
    if converted_xlim is not None:
        fig.update_xaxes(range=list(converted_xlim), row=2, col=1)
    if nominal_ylim is not None:
        fig.update_yaxes(range=list(nominal_ylim), row=1, col=1)
    if converted_ylim is not None:
        fig.update_yaxes(range=list(converted_ylim), row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=850,
        height=720,
        hovermode="x unified",
        legend_title_text="GMD bin",
        margin={"l": 55, "r": 20, "t": 70, "b": 50},
    )
    return fig


def plot_gmd_difference_matplotlib(
    result: ThesisCoreXASResult,
    *,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    share_xaxis: bool = True,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot Delta OD = OD[bin_b] - OD[bin_a] for nominal and proxy energies."""

    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim) if share_xaxis else None
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=share_xaxis, constrained_layout=True)
    _plot_delta_od_on_axis(
        axes[0],
        result.nominal,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Nominal energy",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=nominal_ylim,
        show_errorbars=show_errorbars,
    )
    _plot_delta_od_on_axis(
        axes[1],
        result.converted,
        bin_a=bin_a,
        bin_b=bin_b,
        title=f"Converted/proxy energy ({result.config.center_method})",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=converted_ylim,
        show_errorbars=show_errorbars,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("photon energy (eV)")
    if title is None:
        ia = _resolve_bin_index(bin_a, len(result.gmd_labels))
        ib = _resolve_bin_index(bin_b, len(result.gmd_labels))
        title = f"5 mmol static XAS GMD-bin difference: bin {ib} - bin {ia}"
    fig.suptitle(title)
    return fig, axes


def plot_gmd_difference_plotly(
    result: ThesisCoreXASResult,
    *,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot Delta OD in two Plotly rows with independent x-axes."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_gmd_difference_plotly") from exc

    ia = _resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = _resolve_bin_index(bin_b, len(result.gmd_labels))
    diff_label = f"{result.gmd_labels[ib]} - {result.gmd_labels[ia]}"
    title = title or f"5 mmol static XAS GMD-bin difference: bin {ib} - bin {ia}"
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Nominal energy", f"Converted/proxy energy ({result.config.center_method})"),
        shared_xaxes=False,
        vertical_spacing=0.11,
    )

    for row, binned in enumerate((result.nominal, result.converted), start=1):
        x, delta, sigma, _, _ = _delta_od_for_bins(binned, bin_a, bin_b)
        error_y = {"type": "data", "array": sigma, "visible": True} if show_errorbars else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                error_y=error_y,
                mode="lines+markers",
                name=diff_label,
                legendgroup="delta",
                showlegend=(row == 1),
                marker={"size": 6, "color": "#1f77b4"},
                line={"width": 1.8, "color": "#1f77b4"},
            ),
            row=row,
            col=1,
        )
        if x.size:
            fig.add_trace(
                go.Scatter(
                    x=[float(np.nanmin(x)), float(np.nanmax(x))],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="zero",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"width": 1.0, "color": "rgba(0,0,0,0.45)", "dash": "dot"},
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="nominal photon energy (eV)", row=1, col=1)
    fig.update_xaxes(title_text="converted photon energy (eV)", row=2, col=1)
    fig.update_yaxes(title_text="Delta OD = OD[B] - OD[A]", row=1, col=1)
    fig.update_yaxes(title_text="Delta OD = OD[B] - OD[A]", row=2, col=1)
    if nominal_xlim is not None:
        fig.update_xaxes(range=list(nominal_xlim), row=1, col=1)
    if converted_xlim is not None:
        fig.update_xaxes(range=list(converted_xlim), row=2, col=1)
    if nominal_ylim is not None:
        fig.update_yaxes(range=list(nominal_ylim), row=1, col=1)
    if converted_ylim is not None:
        fig.update_yaxes(range=list(converted_ylim), row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=850,
        height=680,
        hovermode="x unified",
        legend_title_text="Delta bin",
        margin={"l": 60, "r": 20, "t": 75, "b": 50},
    )
    return fig


def plot_unit_scale_matplotlib(
    result: ThesisCoreXASResult,
    *,
    xlim: Optional[Sequence[float]] = None,
    raw_ylim: Optional[Sequence[float]] = None,
    scaled_ylim: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
):
    """Show the through-origin unit conversion from VLS units to uJ."""

    binned = result.nominal
    factor = float(result.unit_scale.vls_per_uJ)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"vls_per_uJ must be finite and positive, got {factor}")

    mean_gmd = _safe_divide(binned.gmd_sum, binned.counts)
    mean_vls_scaled = _safe_divide(binned.trans_sum, binned.counts)
    mean_vls_raw = mean_vls_scaled * factor if result.unit_scale.mode != "none" else mean_vls_scaled
    mean_vls_in_uJ = mean_vls_raw / factor
    counts = binned.counts.astype(np.float64)
    energy = np.repeat(binned.x[:, None], binned.counts.shape[1], axis=1)
    reference_rows = _unit_reference_mask(binned.x, result.unit_scale.reference_window_ev)
    reference = np.repeat(reference_rows[:, None], binned.counts.shape[1], axis=1)

    valid = np.isfinite(mean_gmd) & np.isfinite(mean_vls_raw) & (counts > 0)
    x = mean_gmd[valid]
    y_raw = mean_vls_raw[valid]
    y_scaled = mean_vls_in_uJ[valid]
    c = energy[valid]
    is_ref = reference[valid]
    sizes = 12.0 + 38.0 * np.sqrt(counts[valid] / np.nanmax(counts[valid]))
    x_line = _line_range(x, xlim)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, constrained_layout=True)
    sc = axes[0].scatter(x, y_raw, c=c, s=sizes, cmap="viridis", alpha=0.72, edgecolors="none")
    if np.any(is_ref):
        axes[0].scatter(
            x[is_ref],
            y_raw[is_ref],
            s=sizes[is_ref] + 18.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="reference window",
        )
    axes[0].plot(x_line, factor * x_line, color="black", lw=1.4, label=f"fit: VLS = {factor:.4g} * GMD")
    axes[0].set_ylabel("mean integrated VLS")
    axes[0].set_title("Before unit conversion")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(x, y_scaled, c=c, s=sizes, cmap="viridis", alpha=0.72, edgecolors="none")
    if np.any(is_ref):
        axes[1].scatter(
            x[is_ref],
            y_scaled[is_ref],
            s=sizes[is_ref] + 18.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="reference window",
        )
    axes[1].plot(x_line, x_line, color="black", lw=1.4, label="after conversion: VLS in uJ = GMD")
    axes[1].set_xlabel("mean GMD / uJ")
    axes[1].set_ylabel("mean VLS in uJ")
    axes[1].set_title("After single-slope unit conversion")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label("nominal photon energy (eV)")
    if title is None:
        title = f"VLS/uJ unit convention ({result.unit_scale.source})"
    fig.suptitle(title)
    if xlim is not None:
        axes[1].set_xlim(*xlim)
    if raw_ylim is not None:
        axes[0].set_ylim(*raw_ylim)
    if scaled_ylim is not None:
        axes[1].set_ylim(*scaled_ylim)
    return fig, axes


def plot_global_unit_scale_matplotlib(*args, **kwargs):
    """Backward-compatible alias for old notebook cells."""

    return plot_unit_scale_matplotlib(*args, **kwargs)


def plot_xas_with_gmd_difference_matplotlib(
    result: ThesisCoreXASResult,
    *,
    bootstrap: Optional[ThesisCoreBootstrapErrors] = None,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    xas_nominal_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    xas_converted_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    share_xaxis: bool = True,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot full GMD-binned OD and the selected GMD-bin difference below it."""

    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim) if share_xaxis else None
    nominal_error = bootstrap.nominal_od_sigma if bootstrap is not None else None
    converted_error = bootstrap.converted_od_sigma if bootstrap is not None else None
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.5), sharex=share_xaxis, constrained_layout=True)
    _plot_binned_od_on_axis(
        axes[0],
        result.nominal,
        title="Nominal energy: GMD-binned XAS",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=xas_nominal_ylim,
        show_errorbars=show_errorbars,
        show_legend=True,
        od_error=nominal_error,
    )
    _plot_delta_od_on_axis(
        axes[1],
        result.nominal,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Nominal energy: GMD-bin difference",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=delta_nominal_ylim,
        show_errorbars=show_errorbars,
        od_error=nominal_error,
    )
    _plot_binned_od_on_axis(
        axes[2],
        result.converted,
        title=f"Converted/proxy energy ({result.config.center_method}): GMD-binned XAS",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=xas_converted_ylim,
        show_errorbars=show_errorbars,
        show_legend=False,
        od_error=converted_error,
    )
    _plot_delta_od_on_axis(
        axes[3],
        result.converted,
        bin_a=bin_a,
        bin_b=bin_b,
        title=f"Converted/proxy energy ({result.config.center_method}): GMD-bin difference",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=delta_converted_ylim,
        show_errorbars=show_errorbars,
        od_error=converted_error,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("photon energy (eV)")
    if title is None:
        error_label = "bootstrap" if bootstrap is not None else "SEM"
        title = (
            "5 mmol static XAS and GMD-bin difference "
            f"({result.config.correction_mode}, {result.unit_scale.mode}, {error_label} errorbars)"
        )
    fig.suptitle(title)
    return fig, axes


def plot_xas_with_gmd_difference_plotly(
    result: ThesisCoreXASResult,
    *,
    bootstrap: Optional[ThesisCoreBootstrapErrors] = None,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    xas_nominal_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    xas_converted_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot full OD and Delta OD in four Plotly rows with independent x-axes."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_xas_with_gmd_difference_plotly") from exc

    ia = _resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = _resolve_bin_index(bin_b, len(result.gmd_labels))
    diff_label = f"{result.gmd_labels[ib]} - {result.gmd_labels[ia]}"
    if title is None:
        error_label = "bootstrap" if bootstrap is not None else "SEM"
        title = (
            "5 mmol static XAS and GMD-bin difference "
            f"({result.config.correction_mode}, {result.unit_scale.mode}, {error_label} errorbars)"
        )
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Nominal energy: GMD-binned XAS",
            "Nominal energy: GMD-bin difference",
            f"Converted/proxy energy ({result.config.center_method}): GMD-binned XAS",
            f"Converted/proxy energy ({result.config.center_method}): GMD-bin difference",
        ),
        shared_xaxes=False,
        vertical_spacing=0.075,
    )
    colors = [_rgba_to_hex(c) for c in _colors(len(result.gmd_labels))]
    error_by_binned = {
        id(result.nominal): bootstrap.nominal_od_sigma if bootstrap is not None else result.nominal.od_sem,
        id(result.converted): bootstrap.converted_od_sigma if bootstrap is not None else result.converted.od_sem,
    }

    for row, binned, showlegend in ((1, result.nominal, True), (3, result.converted, False)):
        order = _finite_x_order(binned.x)
        x = binned.x[order]
        y = binned.od[order]
        sem = error_by_binned[id(binned)][order]
        for ig, label in enumerate(result.gmd_labels):
            error_y = {"type": "data", "array": sem[:, ig], "visible": True} if show_errorbars else None
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[:, ig],
                    error_y=error_y,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=showlegend,
                    marker={"size": 5, "color": colors[ig]},
                    line={"width": 1.4, "color": colors[ig]},
                ),
                row=row,
                col=1,
            )

    for row, binned in ((2, result.nominal), (4, result.converted)):
        x, delta, sigma, _, _ = _delta_od_for_bins(
            binned,
            bin_a,
            bin_b,
            od_error=error_by_binned[id(binned)],
        )
        error_y = {"type": "data", "array": sigma, "visible": True} if show_errorbars else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                error_y=error_y,
                mode="lines+markers",
                name=diff_label,
                legendgroup="delta",
                showlegend=(row == 2),
                marker={"size": 6, "color": "#1f77b4"},
                line={"width": 1.8, "color": "#1f77b4"},
            ),
            row=row,
            col=1,
        )
        if x.size:
            fig.add_trace(
                go.Scatter(
                    x=[float(np.nanmin(x)), float(np.nanmax(x))],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="zero",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"width": 1.0, "color": "rgba(0,0,0,0.45)", "dash": "dot"},
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="nominal photon energy (eV)", row=2, col=1)
    fig.update_xaxes(title_text="converted photon energy (eV)", row=4, col=1)
    fig.update_yaxes(title_text=_od_y_label(result.nominal), row=1, col=1)
    fig.update_yaxes(title_text="Delta OD", row=2, col=1)
    fig.update_yaxes(title_text=_od_y_label(result.converted), row=3, col=1)
    fig.update_yaxes(title_text="Delta OD", row=4, col=1)
    if nominal_xlim is not None:
        fig.update_xaxes(range=list(nominal_xlim), row=1, col=1)
        fig.update_xaxes(range=list(nominal_xlim), row=2, col=1)
    if converted_xlim is not None:
        fig.update_xaxes(range=list(converted_xlim), row=3, col=1)
        fig.update_xaxes(range=list(converted_xlim), row=4, col=1)
    if xas_nominal_ylim is not None:
        fig.update_yaxes(range=list(xas_nominal_ylim), row=1, col=1)
    if delta_nominal_ylim is not None:
        fig.update_yaxes(range=list(delta_nominal_ylim), row=2, col=1)
    if xas_converted_ylim is not None:
        fig.update_yaxes(range=list(xas_converted_ylim), row=3, col=1)
    if delta_converted_ylim is not None:
        fig.update_yaxes(range=list(delta_converted_ylim), row=4, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=1050,
        hovermode="x unified",
        legend_title_text="GMD bin",
        margin={"l": 60, "r": 20, "t": 95, "b": 55},
    )
    return fig


def print_thesis_core_summary(result: ThesisCoreXASResult) -> None:
    """Print a compact sanity summary for the notebook."""

    fc = result.filter_counts
    print(f"input H5            : {result.path}")
    print(f"GMD source          : {result.gmd_source}")
    print(f"run                 : {result.attrs.get('run_no')}")
    print(f"VLS crop ROI        : {result.attrs.get('vls_crop_roi')}")
    print(f"signal bunch range  : {result.attrs.get('signal_bunch_range')}")
    print(f"nominal energies    : {result.nominal_energies.size}")
    print(f"recorded shots      : {int(np.sum(fc['recorded'])):,}")
    print(f"finite shots        : {int(np.sum(fc['finite'])):,}")
    print(f"accepted shots      : {int(np.sum(fc['accepted'])):,}")
    print(f"GMD edges           : {np.array2string(result.gmd_edges, precision=3)}")
    print(f"correction mode     : {result.config.correction_mode}")
    print(f"center method       : {result.config.center_method}")
    _print_bin_count_summary("nominal", result.nominal)
    _print_bin_count_summary("converted", result.converted)


def print_unit_scale_summary(result: ThesisCoreXASResult) -> None:
    """Print the VLS/uJ scale used by the active result."""

    scale = result.unit_scale
    print(f"unit scale mode     : {scale.mode}")
    print(f"slope source        : {scale.source}")
    if scale.reference_window_ev is not None:
        lo, hi = scale.reference_window_ev
        print(f"reference window    : {lo:g}-{hi:g} eV nominal")
    print(f"VLS/uJ slope        : {scale.vls_per_uJ:.8g}")
    if np.isfinite(scale.vls_per_uJ) and scale.vls_per_uJ > 0:
        print(f"log10(slope)        : {np.log10(scale.vls_per_uJ):.8g}")
    else:
        print("log10(slope)        : nan")


def print_gmd_difference_summary(result: ThesisCoreXASResult, *, bin_a: int = 0, bin_b: int = -1) -> None:
    """Print resolved GMD-bin labels for Delta OD."""

    ia = _resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = _resolve_bin_index(bin_b, len(result.gmd_labels))
    print(f"Delta OD direction  : bin {ib} - bin {ia}")
    print(f"bin A               : {result.gmd_labels[ia]}")
    print(f"bin B               : {result.gmd_labels[ib]}")


def print_bootstrap_summary(bootstrap: ThesisCoreBootstrapErrors) -> None:
    """Print a compact bootstrap uncertainty summary."""

    nominal_finite = int(np.sum(np.isfinite(bootstrap.nominal_od_sigma)))
    converted_finite = int(np.sum(np.isfinite(bootstrap.converted_od_sigma)))
    print(f"bootstrap draws     : {bootstrap.n_bootstrap:,}")
    print(f"bootstrap seed      : {bootstrap.seed}")
    print(f"bootstrap min shots : {bootstrap.min_shots_per_bin:,}")
    print(f"nominal boot bins   : {nominal_finite}/{bootstrap.nominal_od_sigma.size} finite")
    print(f"converted boot bins : {converted_finite}/{bootstrap.converted_od_sigma.size} finite")


def _print_bin_count_summary(name: str, binned: ThesisCoreBinnedXAS) -> None:
    counts = binned.counts[np.isfinite(binned.od)]
    if counts.size:
        print(f"{name} shots/bin    : {np.min(counts):,} / {np.median(counts):,.0f} / {np.max(counts):,} min/median/max")
    else:
        print(f"{name} shots/bin    : no finite OD bins")


def _validate_config(config: ThesisCoreXASConfig) -> None:
    if config.correction_mode not in {"none", "per_energy_intercept"}:
        raise ValueError("correction_mode must be 'none' or 'per_energy_intercept'")
    if config.unit_scale_mode not in {"none", "reference_window", "global_slope"}:
        raise ValueError("unit_scale_mode must be 'none', 'reference_window', or 'global_slope'")
    if config.unit_scale_mode == "reference_window":
        _unit_reference_window(config.unit_reference_window_ev)
    if config.gmd_bin_mode not in {"fixed", "quantile"}:
        raise ValueError("gmd_bin_mode must be 'fixed' or 'quantile'")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if int(config.proxy_pixel_bin_width) <= 0:
        raise ValueError("proxy_pixel_bin_width must be positive")


def _read_h5_info(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "ml/gmd_hall_channels" not in h5:
            raise KeyError(f"{path} is missing required mapped hall GMD dataset: ml/gmd_hall_channels")
        return {
            "nominal_energies": h5["nominal_energies"][...].astype(np.float64),
            "n_shots": h5["n_shots"][...].astype(np.int64),
            "vls_pixels": h5["vls_pixels"][...].astype(np.float64),
            "attrs": dict(h5.attrs),
            "gmd_source": "ml/gmd_hall_channels[..., 0]",
            "n_pixels": int(h5["vls"].shape[2]),
        }


def _iter_energy_chunks(path: Path, info: dict[str, Any], chunk_size: int):
    with h5py.File(path, "r") as h5:
        for ie, n in enumerate(info["n_shots"].astype(int)):
            for start in range(0, int(n), int(chunk_size)):
                stop = min(int(n), start + int(chunk_size))
                yield ie, h5["vls"][ie, start:stop, :], sxa.read_hall_gmd_slice(h5, ie, start, stop)


def _quality_mask(vls: np.ndarray, gmd: np.ndarray, config: ThesisCoreXASConfig) -> tuple[np.ndarray, dict[str, int]]:
    finite = np.isfinite(gmd) & np.all(np.isfinite(vls), axis=1)
    keep = finite.copy()
    peak_rejected = 0
    if config.vls_intensity_threshold is not None:
        before = keep.copy()
        keep &= np.nanmax(vls, axis=1) >= float(config.vls_intensity_threshold)
        peak_rejected = int(np.sum(before) - np.sum(keep))
    before_gmd = keep.copy()
    if config.gmd_min_threshold is not None:
        keep &= gmd >= float(config.gmd_min_threshold)
    if config.gmd_max_threshold is not None:
        keep &= gmd <= float(config.gmd_max_threshold)
    return keep, {
        "recorded": int(gmd.size),
        "finite": int(np.sum(finite)),
        "peak_rejected": peak_rejected,
        "gmd_rejected": int(np.sum(before_gmd) - np.sum(keep)),
        "accepted": int(np.sum(keep)),
    }


def _first_pass(path: Path, config: ThesisCoreXASConfig, info: dict[str, Any]) -> dict[str, Any]:
    n_e = info["nominal_energies"].size
    filter_counts = {
        name: np.zeros(n_e, dtype=np.int64)
        for name in ["recorded", "finite", "peak_rejected", "gmd_rejected", "accepted"]
    }
    fit = {name: np.zeros(n_e, dtype=np.float64) for name in ["n", "sx", "sy", "sxx", "syy", "sxy"]}
    accepted_gmd_parts: list[np.ndarray] = []

    for ie, vls, gmd in _iter_energy_chunks(path, info, config.chunk_size):
        keep, summary = _quality_mask(vls, gmd, config)
        for name in filter_counts:
            filter_counts[name][ie] += summary[name]
        if not np.any(keep):
            continue
        g = np.asarray(gmd[keep], dtype=np.float64)
        t = _integrated_vls(vls[keep])
        ok = np.isfinite(g) & np.isfinite(t)
        g = g[ok]
        t = t[ok]
        if g.size == 0:
            continue
        accepted_gmd_parts.append(g)
        fit["n"][ie] += g.size
        fit["sx"][ie] += np.sum(g)
        fit["sy"][ie] += np.sum(t)
        fit["sxx"][ie] += np.sum(g * g)
        fit["syy"][ie] += np.sum(t * t)
        fit["sxy"][ie] += np.sum(g * t)

    accepted_gmd = np.concatenate(accepted_gmd_parts) if accepted_gmd_parts else np.empty(0, dtype=np.float64)
    return {"filter_counts": filter_counts, "fit": fit, "accepted_gmd": accepted_gmd}


def _second_pass(
    path: Path,
    config: ThesisCoreXASConfig,
    info: dict[str, Any],
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    fits: ThesisCoreLinearFit,
) -> dict[str, Any]:
    n_e = info["nominal_energies"].size
    n_g = gmd_edges.size - 1
    n_pix = int(info["n_pixels"])
    nominal_raw = _new_grid(n_e, n_g)
    nominal_corrected = _new_grid(n_e, n_g)
    n_proxy = int(np.ceil(n_pix / int(config.proxy_pixel_bin_width)))
    proxy = _new_grid(n_proxy, n_g)
    proxy_x_sum = np.zeros(n_proxy, dtype=np.float64)
    proxy_x_count = np.zeros(n_proxy, dtype=np.float64)
    use_corrected = config.correction_mode == "per_energy_intercept"

    for ie, vls, gmd in _iter_energy_chunks(path, info, config.chunk_size):
        keep, _ = _quality_mask(vls, gmd, config)
        if not np.any(keep):
            continue
        vv = np.asarray(vls[keep], dtype=np.float64)
        gg = np.asarray(gmd[keep], dtype=np.float64)
        raw_trans = _integrated_vls(vv)
        intercept = fits.intercept[ie]
        offset = 0.0 if not np.isfinite(intercept) else float(intercept)
        corrected_trans = raw_trans - offset
        gbin, in_gmd = _gmd_bin_indices(gg, gmd_edges)
        x_nominal = np.full(gg.shape, ie, dtype=np.int64)
        _add_to_grid(nominal_raw, x_nominal, gbin, in_gmd, gg, raw_trans)
        _add_to_grid(nominal_corrected, x_nominal, gbin, in_gmd, gg, corrected_trans)

        spectra_for_center = vv - offset / n_pix if use_corrected else vv
        trans_for_proxy = corrected_trans if use_corrected else raw_trans
        center_rel, _ = sxa.estimate_center_pixels(
            spectra_for_center,
            method=config.center_method,
            half_window=int(config.center_half_window),
            chunk_size=int(config.center_chunk_size),
        )
        valid_center = np.isfinite(center_rel)
        valid_proxy = valid_center & in_gmd & np.isfinite(gg) & np.isfinite(trans_for_proxy)
        if not np.any(valid_proxy):
            continue
        rel = np.clip(np.round(center_rel[valid_proxy]).astype(np.int64), 0, n_pix - 1)
        proxy_group = rel // int(config.proxy_pixel_bin_width)
        center_pixel = np.interp(
            center_rel[valid_proxy],
            np.arange(n_pix, dtype=np.float64),
            info["vls_pixels"].astype(np.float64),
            left=np.nan,
            right=np.nan,
        )
        _add_to_grid(
            proxy,
            proxy_group,
            gbin[valid_proxy],
            np.ones(proxy_group.shape, dtype=bool),
            gg[valid_proxy],
            trans_for_proxy[valid_proxy],
        )
        ok_x = np.isfinite(center_pixel)
        np.add.at(proxy_x_sum, proxy_group[ok_x], center_pixel[ok_x])
        np.add.at(proxy_x_count, proxy_group[ok_x], 1.0)

    nominal_raw_binned = _make_binned(
        info["nominal_energies"],
        nominal_raw,
        gmd_edges,
        gmd_labels,
        "nominal photon energy (eV)",
        {"mode": "nominal", "correction": "none"},
    )
    nominal_corrected_binned = _make_binned(
        info["nominal_energies"],
        nominal_corrected,
        gmd_edges,
        gmd_labels,
        "nominal photon energy (eV)",
        {"mode": "nominal", "correction": "per_energy_intercept"},
    )

    proxy_x = np.full(n_proxy, np.nan, dtype=np.float64)
    np.divide(proxy_x_sum, proxy_x_count, out=proxy_x, where=proxy_x_count > 0)
    has_proxy = (np.sum(proxy["counts"], axis=1) > 0) & np.isfinite(proxy_x)
    proxy_energy, pixel_map_config, pixel_map_cal = sxa.convert_pixels_to_energy(
        proxy_x[has_proxy],
        config.pixel_map_config,
        config.pixel_map_calibration,
        config.inverse_map_range_ev,
        n_samples=int(config.inverse_map_samples),
    )
    order = _finite_x_order(proxy_energy)
    proxy_group_indices = np.where(has_proxy)[0][order]
    converted = _make_binned(
        proxy_energy[order],
        {name: values[has_proxy][order] for name, values in proxy.items()},
        gmd_edges,
        gmd_labels,
        "converted photon energy (eV)",
        {
            "mode": "converted_proxy",
            "correction": config.correction_mode,
            "center_method": config.center_method,
            "proxy_pixel_bin_width": int(config.proxy_pixel_bin_width),
            "proxy_group_indices": proxy_group_indices,
        },
    )
    return {
        "nominal_raw": nominal_raw_binned,
        "nominal_corrected": nominal_corrected_binned,
        "converted": converted,
        "pixel_map_config": pixel_map_config,
        "pixel_map_calibration": pixel_map_cal,
    }


def _make_binned(
    x: np.ndarray,
    grid: dict[str, np.ndarray],
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    x_label: str,
    metadata: dict[str, Any],
) -> ThesisCoreBinnedXAS:
    return ThesisCoreBinnedXAS(
        x=np.asarray(x, dtype=np.float64),
        gmd_sum=grid["gmd_sum"],
        trans_sum=grid["trans_sum"],
        gmd_sumsq=grid["gmd_sumsq"],
        trans_sumsq=grid["trans_sumsq"],
        gmd_trans_sum=grid["gmd_trans_sum"],
        counts=grid["counts"],
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        x_label=x_label,
        metadata=dict(metadata),
    )


def _proxy_group_to_converted_row(result: ThesisCoreXASResult, n_proxy: int) -> np.ndarray:
    proxy_group_indices = result.converted.metadata.get("proxy_group_indices")
    if proxy_group_indices is None:
        raise ValueError("Converted result is missing proxy_group_indices; recompute result with the current helper.")
    proxy_group_indices = np.asarray(proxy_group_indices, dtype=np.int64)
    if proxy_group_indices.size != result.converted.x.size:
        raise ValueError(
            "Converted proxy_group_indices length does not match converted x length: "
            f"{proxy_group_indices.size} != {result.converted.x.size}"
        )
    mapping = np.full(int(n_proxy), -1, dtype=np.int64)
    valid = (proxy_group_indices >= 0) & (proxy_group_indices < int(n_proxy))
    mapping[proxy_group_indices[valid]] = np.where(valid)[0]
    return mapping


def _append_bootstrap_parts(
    g_parts: list[list[np.ndarray]],
    t_parts: list[list[np.ndarray]],
    flat_idx: np.ndarray,
    gmd: np.ndarray,
    trans: np.ndarray,
    keep: np.ndarray,
) -> None:
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    gmd = np.asarray(gmd, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    valid = keep & (flat_idx >= 0) & (flat_idx < len(g_parts)) & np.isfinite(gmd) & np.isfinite(trans)
    if not np.any(valid):
        return

    idx = flat_idx[valid]
    g = gmd[valid]
    t = trans[valid]
    order = np.argsort(idx)
    idx = idx[order]
    g = g[order]
    t = t[order]
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    stops = np.r_[starts[1:], idx.size]
    for start, stop in zip(starts, stops):
        cell = int(idx[start])
        g_parts[cell].append(g[start:stop])
        t_parts[cell].append(t[start:stop])


def _bootstrap_sigma_grid(
    g_parts: list[list[np.ndarray]],
    t_parts: list[list[np.ndarray]],
    *,
    shape: tuple[int, int],
    n_bootstrap: int,
    rng: np.random.Generator,
    min_shots_per_bin: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(np.prod(shape))
    sigma = np.full(n_cells, np.nan, dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for cell in range(n_cells):
        if not g_parts[cell]:
            continue
        g = np.concatenate(g_parts[cell]).astype(np.float64, copy=False)
        t = np.concatenate(t_parts[cell]).astype(np.float64, copy=False)
        valid = np.isfinite(g) & np.isfinite(t)
        g = g[valid]
        t = t[valid]
        counts[cell] = g.size
        if g.size < int(min_shots_per_bin):
            continue
        sigma[cell] = _bootstrap_od_std(g, t, n_bootstrap=n_bootstrap, rng=rng, batch_size=batch_size)
    return sigma.reshape(shape), counts.reshape(shape)


def _bootstrap_od_std(
    gmd: np.ndarray,
    trans: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    batch_size: int,
) -> float:
    n = int(gmd.size)
    boot = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    write = 0
    for _ in range(0, int(n_bootstrap), int(batch_size)):
        n_batch = min(int(batch_size), int(n_bootstrap) - write)
        sample_idx = rng.integers(0, n, size=(n_batch, n), endpoint=False)
        g_sum = np.sum(gmd[sample_idx], axis=1)
        t_sum = np.sum(trans[sample_idx], axis=1)
        boot[write : write + n_batch] = _log10_ratio(g_sum, t_sum)
        write += n_batch
    finite = boot[np.isfinite(boot)]
    if finite.size < 2:
        return np.nan
    return float(np.std(finite, ddof=1))


def _unit_reference_window(window: Optional[Sequence[float]]) -> tuple[float, float]:
    if window is None:
        raise ValueError("unit_reference_window_ev is required when unit_scale_mode='reference_window'")
    if len(window) != 2:
        raise ValueError("unit_reference_window_ev must contain exactly two values")
    lo, hi = (float(window[0]), float(window[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("unit_reference_window_ev values must be finite")
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        raise ValueError("unit_reference_window_ev must have nonzero width")
    return lo, hi


def _unit_reference_mask(x: np.ndarray, window: Optional[Sequence[float]]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if window is None:
        return np.zeros(values.shape, dtype=bool)
    lo, hi = _unit_reference_window(window)
    return np.isfinite(values) & (values >= lo) & (values <= hi)


def _resolve_unit_scale(config: ThesisCoreXASConfig, active_nominal: ThesisCoreBinnedXAS) -> ThesisCoreUnitScale:
    mode = str(config.unit_scale_mode)
    if mode == "none":
        return ThesisCoreUnitScale(mode="none", vls_per_uJ=1.0, source="not applied")

    if mode == "reference_window":
        reference_window = _unit_reference_window(config.unit_reference_window_ev)
        rows = _unit_reference_mask(active_nominal.x, reference_window)
        if not np.any(rows):
            lo, hi = reference_window
            raise ValueError(
                "Cannot compute reference_window unit scale: "
                f"no nominal energies found in {lo:g}-{hi:g} eV."
            )
        den = float(np.nansum(active_nominal.gmd_sumsq[rows, :]))
        num = float(np.nansum(active_nominal.gmd_trans_sum[rows, :]))
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0 or num <= 0.0:
            lo, hi = reference_window
            raise ValueError(
                "Cannot compute reference_window unit scale: "
                f"window={lo:g}-{hi:g} eV, sum(GMD*VLS)={num:.6g}, sum(GMD^2)={den:.6g}"
            )
        trans_source = "intercept-corrected VLS" if config.correction_mode == "per_energy_intercept" else "raw VLS"
        source = f"nominal {reference_window[0]:g}-{reference_window[1]:g} eV {trans_source}"
        return ThesisCoreUnitScale(
            mode=mode,
            vls_per_uJ=num / den,
            source=source,
            reference_window_ev=reference_window,
        )

    den = float(np.nansum(active_nominal.gmd_sumsq))
    num = float(np.nansum(active_nominal.gmd_trans_sum))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0 or num <= 0.0:
        raise ValueError(
            "Cannot compute global_slope unit scale: "
            f"sum(GMD*VLS)={num:.6g}, sum(GMD^2)={den:.6g}"
        )
    source = "active intercept-corrected VLS" if config.correction_mode == "per_energy_intercept" else "active raw VLS"
    return ThesisCoreUnitScale(mode=mode, vls_per_uJ=num / den, source=source)


def _apply_unit_scale(binned: ThesisCoreBinnedXAS, scale: ThesisCoreUnitScale) -> ThesisCoreBinnedXAS:
    factor = float(scale.vls_per_uJ)
    if scale.mode == "none" or not np.isfinite(factor) or factor == 1.0:
        return binned
    if factor <= 0.0:
        raise ValueError(f"Unit scale must be positive, got {factor}")
    metadata = dict(binned.metadata)
    metadata.update(
        {
            "unit_scale_mode": scale.mode,
            "vls_per_uJ": factor,
            "global_vls_per_uJ": factor,
            "unit_scale_source": scale.source,
            "unit_reference_window_ev": scale.reference_window_ev,
        }
    )
    return ThesisCoreBinnedXAS(
        x=binned.x,
        gmd_sum=binned.gmd_sum,
        trans_sum=binned.trans_sum / factor,
        gmd_sumsq=binned.gmd_sumsq,
        trans_sumsq=binned.trans_sumsq / (factor * factor),
        gmd_trans_sum=binned.gmd_trans_sum / factor,
        counts=binned.counts,
        gmd_edges=binned.gmd_edges,
        gmd_labels=binned.gmd_labels,
        x_label=binned.x_label,
        metadata=metadata,
    )


def _new_grid(n_x: int, n_g: int) -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((n_x, n_g), dtype=np.int64),
        "gmd_sum": np.zeros((n_x, n_g), dtype=np.float64),
        "trans_sum": np.zeros((n_x, n_g), dtype=np.float64),
        "gmd_sumsq": np.zeros((n_x, n_g), dtype=np.float64),
        "trans_sumsq": np.zeros((n_x, n_g), dtype=np.float64),
        "gmd_trans_sum": np.zeros((n_x, n_g), dtype=np.float64),
    }


def _add_to_grid(
    grid: dict[str, np.ndarray],
    xbin: np.ndarray,
    gbin: np.ndarray,
    in_gmd: np.ndarray,
    gmd: np.ndarray,
    trans: np.ndarray,
) -> None:
    valid = in_gmd & np.isfinite(trans) & np.isfinite(gmd) & (xbin >= 0) & (xbin < grid["counts"].shape[0])
    if not np.any(valid):
        return
    idx = (xbin[valid].astype(np.int64), gbin[valid].astype(np.int64))
    gv = gmd[valid]
    tv = trans[valid]
    np.add.at(grid["counts"], idx, 1)
    np.add.at(grid["gmd_sum"], idx, gv)
    np.add.at(grid["trans_sum"], idx, tv)
    np.add.at(grid["gmd_sumsq"], idx, gv * gv)
    np.add.at(grid["trans_sumsq"], idx, tv * tv)
    np.add.at(grid["gmd_trans_sum"], idx, gv * tv)


def _integrated_vls(vls: np.ndarray) -> np.ndarray:
    return np.nansum(np.asarray(vls, dtype=np.float64), axis=1)


def _choose_gmd_edges(accepted_gmd: np.ndarray, config: ThesisCoreXASConfig) -> np.ndarray:
    if config.gmd_bin_mode == "fixed":
        return _validate_edges(np.asarray(config.gmd_edges, dtype=np.float64))
    vals = np.asarray(accepted_gmd, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot build quantile GMD bins because no accepted GMD values are finite.")
    q = np.linspace(0.0, 1.0, int(config.n_gmd_bins) + 1)
    edges = np.quantile(vals, q)
    edges[0] = np.nanmin(vals)
    edges[-1] = np.nanmax(vals)
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Quantile GMD edges collapsed to fewer than two unique values.")
    return _validate_edges(edges)


def _validate_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("GMD edges must be a 1D array with at least two values.")
    if not np.all(np.isfinite(edges)):
        raise ValueError("GMD edges must be finite.")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("GMD edges must be strictly increasing.")
    return edges


def _gmd_bin_indices(gmd: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gmd, dtype=np.float64)
    gbin = np.searchsorted(edges, values, side="right") - 1
    gbin = np.asarray(gbin, dtype=np.int64)
    n_g = edges.size - 1
    gbin[values == edges[-1]] = n_g - 1
    in_range = np.isfinite(values) & (gbin >= 0) & (gbin < n_g)
    return gbin, in_range


def _format_gmd_labels(edges: np.ndarray, accepted_gmd: np.ndarray) -> list[str]:
    labels = []
    vals = np.asarray(accepted_gmd, dtype=np.float64)
    for ig, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        close = "]" if ig == edges.size - 2 else ")"
        sel = (vals >= lo) & (vals < hi)
        if ig == edges.size - 2:
            sel = (vals >= lo) & (vals <= hi)
        if np.any(sel):
            labels.append(f"{lo:.2g}-{hi:.2g}{close} uJ; mean {np.nanmean(vals[sel]):.2g}")
        else:
            labels.append(f"{lo:.2g}-{hi:.2g}{close} uJ")
    return labels


def _linear_fit_from_accumulators(acc: dict[str, np.ndarray]) -> ThesisCoreLinearFit:
    n = acc["n"].astype(np.float64)
    sx = acc["sx"]
    sy = acc["sy"]
    sxx = acc["sxx"]
    syy = acc["syy"]
    sxy = acc["sxy"]
    denom = n * sxx - sx * sx
    intercept = np.full(n.shape, np.nan, dtype=np.float64)
    slope = np.full(n.shape, np.nan, dtype=np.float64)
    r2 = np.full(n.shape, np.nan, dtype=np.float64)
    valid = (n >= 2) & (denom != 0)
    slope[valid] = (n[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / n[valid]
    cov = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    valid_r = (n >= 2) & (var_x > 0) & (var_y > 0)
    r = np.full(n.shape, np.nan, dtype=np.float64)
    r[valid_r] = cov[valid_r] / np.sqrt(var_x[valid_r] * var_y[valid_r])
    r2[valid_r] = r[valid_r] ** 2
    return ThesisCoreLinearFit(intercept=intercept, slope=slope, r2=r2, n=n.astype(np.int64))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den != 0))
    return out


def _log10_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ratio = _safe_divide(num, den)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.log10(ratio[valid])
    return out


def _delta_log_ratio_sem(
    sum_g: np.ndarray,
    sum_t: np.ndarray,
    sum_g2: np.ndarray,
    sum_t2: np.ndarray,
    sum_gt: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    n = np.asarray(counts, dtype=np.float64)
    mean_g = _safe_divide(sum_g, n)
    mean_t = _safe_divide(sum_t, n)
    mean_g2 = _safe_divide(sum_g2, n)
    mean_t2 = _safe_divide(sum_t2, n)
    mean_gt = _safe_divide(sum_gt, n)
    var_g = np.maximum(mean_g2 - mean_g**2, 0.0)
    var_t = np.maximum(mean_t2 - mean_t**2, 0.0)
    cov_gt = mean_gt - mean_g * mean_t
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (1.0 / np.log(10.0) ** 2) * (
            var_g / (n * mean_g**2) + var_t / (n * mean_t**2) - 2.0 * cov_gt / (n * mean_g * mean_t)
        )
    sem = np.sqrt(np.maximum(var, 0.0))
    sem[~np.isfinite(sem)] = np.nan
    return sem


def _plot_binned_od_on_axis(
    ax,
    binned: ThesisCoreBinnedXAS,
    *,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
    show_legend: bool,
    od_error: Optional[np.ndarray] = None,
) -> None:
    order = _finite_x_order(binned.x)
    x = binned.x[order]
    y = binned.od[order]
    sem_source = binned.od_sem if od_error is None else np.asarray(od_error, dtype=np.float64)
    sem = sem_source[order]
    colors = _colors(y.shape[1])
    for ig, color in enumerate(colors):
        if show_errorbars:
            ax.errorbar(x, y[:, ig], yerr=sem[:, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
        else:
            ax.plot(x, y[:, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
    ax.set_title(title)
    ax.set_ylabel(_od_y_label(binned))
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(title="GMD bin", frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_delta_od_on_axis(
    ax,
    binned: ThesisCoreBinnedXAS,
    *,
    bin_a: int,
    bin_b: int,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
    od_error: Optional[np.ndarray] = None,
) -> None:
    x, delta, sigma, ia, ib = _delta_od_for_bins(binned, bin_a, bin_b, od_error=od_error)
    label = f"{binned.gmd_labels[ib]} - {binned.gmd_labels[ia]}"
    if show_errorbars:
        ax.errorbar(x, delta, yerr=sigma, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    else:
        ax.plot(x, delta, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    ax.axhline(0.0, color="0.25", lw=0.9, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Delta OD = OD[B] - OD[A]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _delta_od_for_bins(
    binned: ThesisCoreBinnedXAS,
    bin_a: int,
    bin_b: int,
    *,
    od_error: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n_bins = binned.counts.shape[1]
    ia = _resolve_bin_index(bin_a, n_bins)
    ib = _resolve_bin_index(bin_b, n_bins)
    if ia == ib:
        raise ValueError("DIFF_BIN_A and DIFF_BIN_B must resolve to different GMD bins")
    order = _finite_x_order(binned.x)
    x = binned.x[order]
    od = binned.od[order]
    sem_source = binned.od_sem if od_error is None else np.asarray(od_error, dtype=np.float64)
    sem = sem_source[order]
    delta = od[:, ib] - od[:, ia]
    sigma = np.sqrt(sem[:, ib] ** 2 + sem[:, ia] ** 2)
    valid = np.isfinite(x) & np.isfinite(delta)
    return x[valid], delta[valid], sigma[valid], ia, ib


def _resolve_bin_index(index: int, n_bins: int) -> int:
    idx = int(index)
    if idx < 0:
        idx += int(n_bins)
    if idx < 0 or idx >= int(n_bins):
        raise IndexError(f"GMD bin index {index} is out of range for {n_bins} bins")
    return idx


def _line_range(x: np.ndarray, xlim: Optional[Sequence[float]]) -> np.ndarray:
    if xlim is not None:
        return np.asarray([float(xlim[0]), float(xlim[1])], dtype=np.float64)
    finite = np.asarray(x, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if lo == hi:
        pad = 0.5 if lo == 0.0 else abs(lo) * 0.05
        return np.asarray([lo - pad, hi + pad], dtype=np.float64)
    pad = 0.04 * (hi - lo)
    return np.asarray([lo - pad, hi + pad], dtype=np.float64)


def _od_y_label(binned: ThesisCoreBinnedXAS) -> str:
    if binned.metadata.get("unit_scale_mode") in {"reference_window", "global_slope"}:
        return "OD = log10[sum(GMD) / sum(VLS in uJ)]"
    return "OD = log10[sum(GMD) / sum(VLS)]"


def _colors(n: int):
    return plt.cm.inferno(np.linspace(0.08, 0.92, int(n)))


def _rgba_to_hex(color: Sequence[float]) -> str:
    rgb = np.clip(np.asarray(color[:3], dtype=np.float64), 0.0, 1.0)
    vals = np.round(rgb * 255).astype(np.uint8)
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _resolve_shared_xlim(
    nominal_xlim: Optional[Sequence[float]],
    converted_xlim: Optional[Sequence[float]],
) -> Optional[Sequence[float]]:
    if nominal_xlim is None:
        return converted_xlim
    if converted_xlim is None:
        return nominal_xlim
    nominal = tuple(float(v) for v in nominal_xlim)
    converted = tuple(float(v) for v in converted_xlim)
    if nominal != converted:
        raise ValueError("shared x-axis requires nominal_xlim and converted_xlim to match")
    return nominal


def _finite_x_order(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    return np.where(finite)[0][np.argsort(x[finite])]
