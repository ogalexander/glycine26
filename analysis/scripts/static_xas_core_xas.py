"""
Minimal core static-XAS workflow for thesis-style nominal/proxy plots.

This module keeps the notebook small: it reads a processed static-XAS H5,
requires mapped hall GMD from ``/ml/gmd_hall_channels[..., 0]``,
optionally applies a per-nominal-energy intercept correction, and returns only
the core nominal-energy and converted/proxy-energy XAS curves.
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
class CoreXASConfig:
    """Configuration for the simplified static-XAS notebook."""

    input_h5: str | Path
    vls_intensity_threshold: Optional[float] = 1000.0
    gmd_bin_mode: str = "fixed"
    gmd_edges: Sequence[float] = field(default_factory=lambda: tuple(np.arange(0.0, 11.0, 1.0)))
    n_gmd_bins: int = 5
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    center_method: str = "com"
    center_half_window: int = 5
    center_chunk_size: int = 50_000
    proxy_pixel_bin_width: int = 1
    correction_mode: str = "per_energy_intercept"  # "none" or "per_energy_intercept"
    xas_y_mode: str = "od"  # "od" or "ratio"
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
class CoreLinearFit:
    """Per-nominal-energy fit of integrated VLS versus mapped hall GMD."""

    intercept: np.ndarray
    slope: np.ndarray
    r2: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class CoreBinnedXAS:
    """Summed GMD and transmitted VLS binned by x and GMD."""

    x: np.ndarray
    gmd_sum: np.ndarray
    trans_sum: np.ndarray
    counts: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    x_label: str
    metadata: dict[str, Any]

    @property
    def ratio(self) -> np.ndarray:
        return _safe_divide(self.gmd_sum, self.trans_sum)

    @property
    def od(self) -> np.ndarray:
        return _log10_ratio(self.gmd_sum, self.trans_sum)

    def values(self, y_mode: str) -> np.ndarray:
        y_mode = str(y_mode).lower()
        if y_mode == "od":
            return self.od
        if y_mode == "ratio":
            return self.ratio
        raise ValueError("y_mode must be 'od' or 'ratio'")


@dataclass(frozen=True)
class CoreXASResult:
    """Computed state for the core notebook."""

    config: CoreXASConfig
    path: Path
    attrs: dict[str, Any]
    gmd_source: str
    nominal_energies: np.ndarray
    vls_pixels: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    filter_counts: dict[str, np.ndarray]
    fits: CoreLinearFit
    nominal_raw: CoreBinnedXAS
    nominal_corrected: CoreBinnedXAS
    nominal: CoreBinnedXAS
    converted: CoreBinnedXAS
    pixel_map_config: Any
    pixel_map_calibration: Any


def compute_core_xas(config: CoreXASConfig) -> CoreXASResult:
    """Compute nominal-energy and converted/proxy-energy thesis-style XAS."""

    _validate_config(config)
    path = Path(config.input_h5)
    info = _read_h5_info(path)
    first = _first_pass(path, config, info)
    gmd_edges = _choose_gmd_edges(first["accepted_gmd"], config)
    gmd_labels = _format_gmd_labels(gmd_edges, first["accepted_gmd"])
    fits = _linear_fit_from_accumulators(first["fit"])
    second = _second_pass(path, config, info, gmd_edges, gmd_labels, fits)
    nominal = second["nominal_corrected"] if config.correction_mode == "per_energy_intercept" else second["nominal_raw"]

    return CoreXASResult(
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
        nominal_raw=second["nominal_raw"],
        nominal_corrected=second["nominal_corrected"],
        nominal=nominal,
        converted=second["converted"],
        pixel_map_config=second["pixel_map_config"],
        pixel_map_calibration=second["pixel_map_calibration"],
    )


def plot_correction_colormap(
    result: CoreXASResult,
    *,
    xlim: Optional[Sequence[float]] = None,
    clim: Optional[Sequence[float]] = None,
    cmap: str = "coolwarm",
):
    """Plot active correction effect as OD(active) - OD(raw)."""

    x = result.nominal_raw.x
    active = result.nominal.od
    raw = result.nominal_raw.od
    delta = active - raw
    order = _finite_x_order(x)
    x_plot = x[order]
    delta_plot = delta[order].T

    fig, ax = plt.subplots(figsize=(9.0, 3.6), constrained_layout=True)
    extent = _imshow_extent_from_centers(x_plot, delta_plot.shape[0])
    vmax = np.nanmax(np.abs(delta_plot)) if np.any(np.isfinite(delta_plot)) else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin, vmax = (-vmax, vmax) if clim is None else (float(clim[0]), float(clim[1]))
    im = ax.imshow(delta_plot, aspect="auto", origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("OD(active correction) - OD(raw)")
    ax.set_yticks(np.arange(len(result.gmd_labels)) + 0.5)
    ax.set_yticklabels(result.gmd_labels)
    ax.set_xlabel("nominal photon energy (eV)")
    ax.set_ylabel("GMD bin")
    ax.set_title(
        "Per-energy intercept correction effect "
        f"({result.config.correction_mode}, VLS threshold={result.config.vls_intensity_threshold})"
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig, ax


def plot_core_xas_with_counts(
    binned: CoreBinnedXAS,
    *,
    title: str,
    y_mode: str = "od",
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    counts_ylim: Optional[Sequence[float]] = None,
    marker: str = "o",
):
    """Plot core XAS curves with a shot-count panel."""

    x = binned.x
    order = _finite_x_order(x)
    x_plot = x[order]
    y = binned.values(y_mode)[order]
    counts = binned.counts[order]
    colors = _colors(y.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, constrained_layout=True)
    for ig, color in enumerate(colors):
        axes[0].plot(x_plot, y[:, ig], marker=marker, ms=3, lw=1.2, color=color, label=binned.gmd_labels[ig])
        axes[1].plot(x_plot, counts[:, ig], marker=".", lw=1.0, color=color)

    axes[0].set_ylabel(_y_label(y_mode))
    axes[0].set_title(title)
    axes[0].legend(title="GMD bin", frameon=False, fontsize=8, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("shot count")
    axes[1].set_xlabel(binned.x_label)
    axes[1].grid(alpha=0.25)
    if xlim is not None:
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    if counts_ylim is not None:
        axes[1].set_ylim(*counts_ylim)
    return fig, axes


def plot_core_xas_pair_matplotlib(
    result: CoreXASResult,
    *,
    y_mode: Optional[str] = None,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
):
    """Plot nominal and converted/proxy XAS in two rows with Matplotlib."""

    y_mode = result.config.xas_y_mode if y_mode is None else y_mode
    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, constrained_layout=True)
    _plot_core_xas_on_axis(
        axes[0],
        result.nominal,
        y_mode=y_mode,
        title="Nominal energy",
        xlim=shared_xlim,
        ylim=nominal_ylim,
        show_legend=True,
    )
    _plot_core_xas_on_axis(
        axes[1],
        result.converted,
        y_mode=y_mode,
        title=f"Converted/proxy energy ({result.config.center_method})",
        xlim=shared_xlim,
        ylim=converted_ylim,
        show_legend=False,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("photon energy (eV)")
    if title is None:
        title = f"5 mmol static XAS ({result.config.correction_mode}, threshold={result.config.vls_intensity_threshold})"
    fig.suptitle(title)
    return fig, axes


def plot_core_xas_pair_plotly(
    result: CoreXASResult,
    *,
    y_mode: Optional[str] = None,
    nominal_xlim: Optional[Sequence[float]] = None,
    nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    converted_ylim: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
):
    """Plot nominal and converted/proxy XAS in two rows with Plotly."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_core_xas_pair_plotly") from exc

    y_mode = result.config.xas_y_mode if y_mode is None else y_mode
    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim)
    title = title or f"5 mmol static XAS ({result.config.correction_mode}, threshold={result.config.vls_intensity_threshold})"
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Nominal energy", f"Converted/proxy energy ({result.config.center_method})"),
        shared_xaxes=True,
        vertical_spacing=0.10,
    )
    colors = [_rgba_to_hex(c) for c in _colors(len(result.gmd_labels))]
    for row, binned in enumerate((result.nominal, result.converted), start=1):
        x = binned.x
        order = _finite_x_order(x)
        x_plot = x[order]
        y = binned.values(y_mode)[order]
        for ig, label in enumerate(result.gmd_labels):
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y[:, ig],
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

    fig.update_xaxes(title_text="photon energy (eV)", row=2, col=1)
    fig.update_yaxes(title_text=_y_label(y_mode), row=1, col=1)
    fig.update_yaxes(title_text=_y_label(y_mode), row=2, col=1)
    if shared_xlim is not None:
        fig.update_xaxes(range=list(shared_xlim), row=1, col=1)
        fig.update_xaxes(range=list(shared_xlim), row=2, col=1)
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


def print_core_summary(result: CoreXASResult) -> None:
    """Print the short run/config summary for the notebook."""

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


def print_intercept_fit_table(result: CoreXASResult, *, max_rows: Optional[int] = None) -> None:
    """Print a compact per-energy intercept fit table."""

    n = result.nominal_energies.size if max_rows is None else min(int(max_rows), result.nominal_energies.size)
    print("energy_eV       n      intercept         slope        R2")
    for i in range(n):
        print(
            f"{result.nominal_energies[i]:9.2f} "
            f"{int(result.fits.n[i]):7d} "
            f"{result.fits.intercept[i]:14.4g} "
            f"{result.fits.slope[i]:13.4g} "
            f"{result.fits.r2[i]:9.4f}"
        )


def _validate_config(config: CoreXASConfig) -> None:
    if config.correction_mode not in {"none", "per_energy_intercept"}:
        raise ValueError("correction_mode must be 'none' or 'per_energy_intercept'")
    if config.xas_y_mode not in {"od", "ratio"}:
        raise ValueError("xas_y_mode must be 'od' or 'ratio'")
    if config.gmd_bin_mode not in {"fixed", "quantile"}:
        raise ValueError("gmd_bin_mode must be 'fixed' or 'quantile'")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")


def _read_h5_info(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "ml/gmd_hall_channels" not in h5:
            raise KeyError(f"{path} is missing required mapped hall GMD dataset: ml/gmd_hall_channels")
        gmd_source = "ml/gmd_hall_channels[..., 0]"
        return {
            "nominal_energies": h5["nominal_energies"][...].astype(np.float64),
            "n_shots": h5["n_shots"][...].astype(np.int64),
            "vls_pixels": h5["vls_pixels"][...].astype(np.float64),
            "attrs": dict(h5.attrs),
            "gmd_source": gmd_source,
            "n_pixels": int(h5["vls"].shape[2]),
        }


def _iter_energy_chunks(path: Path, info: dict[str, Any], chunk_size: int):
    with h5py.File(path, "r") as h5:
        for ie, n in enumerate(info["n_shots"].astype(int)):
            for start in range(0, int(n), int(chunk_size)):
                stop = min(int(n), start + int(chunk_size))
                yield ie, h5["vls"][ie, start:stop, :], sxa.read_hall_gmd_slice(h5, ie, start, stop)


def _quality_mask(vls: np.ndarray, gmd: np.ndarray, config: CoreXASConfig) -> tuple[np.ndarray, dict[str, int]]:
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


def _first_pass(path: Path, config: CoreXASConfig, info: dict[str, Any]) -> dict[str, Any]:
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
    config: CoreXASConfig,
    info: dict[str, Any],
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    fits: CoreLinearFit,
) -> dict[str, Any]:
    n_e = info["nominal_energies"].size
    n_g = gmd_edges.size - 1
    n_pix = int(info["n_pixels"])
    nominal_raw = _new_grid(n_e, n_g)
    nominal_corrected = _new_grid(n_e, n_g)
    n_proxy = int(np.ceil(n_pix / max(1, int(config.proxy_pixel_bin_width))))
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
        group = rel // max(1, int(config.proxy_pixel_bin_width))
        center_pixel = np.interp(
            center_rel[valid_proxy],
            np.arange(n_pix, dtype=np.float64),
            info["vls_pixels"].astype(np.float64),
            left=np.nan,
            right=np.nan,
        )
        _add_to_grid(
            proxy,
            group,
            gbin[valid_proxy],
            np.ones(group.shape, dtype=bool),
            gg[valid_proxy],
            trans_for_proxy[valid_proxy],
        )
        ok_x = np.isfinite(center_pixel)
        np.add.at(proxy_x_sum, group[ok_x], center_pixel[ok_x])
        np.add.at(proxy_x_count, group[ok_x], 1.0)

    nominal_raw_binned = CoreBinnedXAS(
        x=info["nominal_energies"],
        gmd_sum=nominal_raw["gmd_sum"],
        trans_sum=nominal_raw["trans_sum"],
        counts=nominal_raw["counts"],
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        x_label="nominal photon energy (eV)",
        metadata={"mode": "nominal", "correction": "none"},
    )
    nominal_corrected_binned = CoreBinnedXAS(
        x=info["nominal_energies"],
        gmd_sum=nominal_corrected["gmd_sum"],
        trans_sum=nominal_corrected["trans_sum"],
        counts=nominal_corrected["counts"],
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        x_label="nominal photon energy (eV)",
        metadata={"mode": "nominal", "correction": "per_energy_intercept"},
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
    converted = CoreBinnedXAS(
        x=proxy_energy[order],
        gmd_sum=proxy["gmd_sum"][has_proxy][order],
        trans_sum=proxy["trans_sum"][has_proxy][order],
        counts=proxy["counts"][has_proxy][order],
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        x_label="converted photon energy (eV)",
        metadata={
            "mode": "converted_proxy",
            "correction": config.correction_mode,
            "center_method": config.center_method,
            "proxy_pixel_bin_width": int(config.proxy_pixel_bin_width),
        },
    )
    return {
        "nominal_raw": nominal_raw_binned,
        "nominal_corrected": nominal_corrected_binned,
        "converted": converted,
        "pixel_map_config": pixel_map_config,
        "pixel_map_calibration": pixel_map_cal,
    }


def _new_grid(n_x: int, n_g: int) -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((n_x, n_g), dtype=np.int64),
        "gmd_sum": np.zeros((n_x, n_g), dtype=np.float64),
        "trans_sum": np.zeros((n_x, n_g), dtype=np.float64),
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
    np.add.at(grid["counts"], idx, 1)
    np.add.at(grid["gmd_sum"], idx, gmd[valid])
    np.add.at(grid["trans_sum"], idx, trans[valid])


def _integrated_vls(vls: np.ndarray) -> np.ndarray:
    return np.nansum(np.asarray(vls, dtype=np.float64), axis=1)


def _choose_gmd_edges(accepted_gmd: np.ndarray, config: CoreXASConfig) -> np.ndarray:
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


def _linear_fit_from_accumulators(acc: dict[str, np.ndarray]) -> CoreLinearFit:
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
    return CoreLinearFit(intercept=intercept, slope=slope, r2=r2, n=n.astype(np.int64))


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


def _colors(n: int):
    return plt.cm.inferno(np.linspace(0.08, 0.92, int(n)))


def _plot_core_xas_on_axis(
    ax,
    binned: CoreBinnedXAS,
    *,
    y_mode: str,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_legend: bool,
) -> None:
    x = binned.x
    order = _finite_x_order(x)
    x_plot = x[order]
    y = binned.values(y_mode)[order]
    colors = _colors(y.shape[1])
    for ig, color in enumerate(colors):
        ax.plot(x_plot, y[:, ig], marker="o", ms=3, lw=1.2, color=color, label=binned.gmd_labels[ig])
    ax.set_title(title)
    ax.set_xlabel(binned.x_label)
    ax.set_ylabel(_y_label(y_mode))
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(title="GMD bin", frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


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


def _imshow_extent_from_centers(x: np.ndarray, n_y: int) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        dx = 0.5
    else:
        dx = float(np.nanmedian(np.diff(np.sort(x)))) / 2.0
        if not np.isfinite(dx) or dx == 0:
            dx = 0.5
    return [float(np.nanmin(x) - dx), float(np.nanmax(x) + dx), 0.0, float(n_y)]


def _y_label(y_mode: str) -> str:
    if str(y_mode).lower() == "ratio":
        return "sum(GMD) / sum(VLS)"
    return "OD = log10[sum(GMD) / sum(VLS)]"
