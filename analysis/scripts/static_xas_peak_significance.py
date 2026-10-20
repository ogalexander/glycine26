"""
Local-peak significance diagnostics for static XAS.

This module is intentionally small at the notebook boundary: it computes
pooled thesis-style XAS curves, subtracts a smooth continuum, and estimates
local residual uncertainty by shot-level bootstrap resampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - only hit in incomplete environments.
    savgol_filter = None

import static_xas_analysis as sxa
import static_xas_core_xas as core


@dataclass(frozen=True)
class PeakSignificanceConfig:
    """Configuration for local-peak significance analysis."""

    input_h5: str | Path
    vls_intensity_threshold: Optional[float] = 1000.0
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    center_method: str = "com"
    center_half_window: int = 5
    center_chunk_size: int = 50_000
    proxy_pixel_bin_width: int = 1
    correction_mode: str = "per_energy_intercept"
    xas_y_mode: str = "od"
    smooth_method: str = "savgol"
    smooth_window_ev: float = 1.5
    smooth_polyorder: int = 2
    n_bootstrap: int = 500
    bootstrap_seed: int = 123
    min_shots_per_point: int = 100
    bootstrap_batch: int = 50
    chunk_size: int = 20_000
    feature_windows_ev: Sequence[Sequence[float]] = field(default_factory=tuple)
    z_markers: Sequence[float] = (2.0, 3.0)
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
class PeakCurve:
    """One pooled XAS curve with bootstrap local-residual diagnostics."""

    x: np.ndarray
    xas: np.ndarray
    smooth: np.ndarray
    residual: np.ndarray
    xas_sigma: np.ndarray
    residual_sigma: np.ndarray
    z: np.ndarray
    counts: np.ndarray
    gmd_sum: np.ndarray
    trans_sum: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PeakSignificanceResult:
    """Nominal and converted/proxy peak-significance result."""

    config: PeakSignificanceConfig
    path: Path
    attrs: dict[str, Any]
    gmd_source: str
    filter_counts: dict[str, np.ndarray]
    fits: core.CoreLinearFit
    nominal: PeakCurve
    converted: PeakCurve
    pixel_map_config: Any
    pixel_map_calibration: Any


@dataclass(frozen=True)
class _GroupedShots:
    x: np.ndarray
    gmd: list[np.ndarray]
    trans: list[np.ndarray]
    x_label: str
    metadata: dict[str, Any]


def compute_peak_significance(config: PeakSignificanceConfig) -> PeakSignificanceResult:
    """Compute pooled nominal/proxy XAS local residual significance."""

    _validate_config(config)
    core_config = _to_core_config(config)
    path = Path(config.input_h5)
    info = core._read_h5_info(path)
    first = core._first_pass(path, core_config, info)
    fits = core._linear_fit_from_accumulators(first["fit"])
    grouped, pixel_map_config, pixel_map_calibration = _collect_grouped_shots(path, info, config, core_config, fits)
    rng = np.random.default_rng(int(config.bootstrap_seed))
    nominal = _compute_curve(grouped["nominal"], config, rng)
    converted = _compute_curve(grouped["converted"], config, rng)
    return PeakSignificanceResult(
        config=config,
        path=path,
        attrs=info["attrs"],
        gmd_source=info["gmd_source"],
        filter_counts=first["filter_counts"],
        fits=fits,
        nominal=nominal,
        converted=converted,
        pixel_map_config=pixel_map_config,
        pixel_map_calibration=pixel_map_calibration,
    )


def plot_peak_significance_summary(
    result: PeakSignificanceResult,
    *,
    xlim: Optional[Sequence[float]] = None,
    od_ylim: Optional[Sequence[float]] = None,
    residual_ylim: Optional[Sequence[float]] = None,
):
    """Plot pooled XAS and local residual significance for nominal/proxy energy."""

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9.2, 9.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.0, 1.45, 1.0]},
    )
    _plot_xas_axis(axes[0], result.nominal, "Nominal-energy pooled XAS", result.config)
    _plot_residual_axis(axes[1], result.nominal, "Nominal-energy residual", result.config)
    _plot_xas_axis(axes[2], result.converted, f"Converted/proxy-energy pooled XAS ({result.config.center_method})", result.config)
    _plot_residual_axis(axes[3], result.converted, "Converted/proxy-energy residual", result.config)

    for ax in axes:
        _shade_feature_windows(ax, result.config.feature_windows_ev)
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(*xlim)
    if od_ylim is not None:
        axes[0].set_ylim(*od_ylim)
        axes[2].set_ylim(*od_ylim)
    if residual_ylim is not None:
        axes[1].set_ylim(*residual_ylim)
        axes[3].set_ylim(*residual_ylim)

    axes[-1].set_xlabel("photon energy (eV)")
    fig.suptitle(
        "Static XAS local-peak significance "
        f"({result.config.correction_mode}, bootstrap n={result.config.n_bootstrap})"
    )
    return fig, axes


def _validate_config(config: PeakSignificanceConfig) -> None:
    if config.correction_mode not in {"none", "per_energy_intercept"}:
        raise ValueError("correction_mode must be 'none' or 'per_energy_intercept'")
    if config.xas_y_mode not in {"od", "ratio"}:
        raise ValueError("xas_y_mode must be 'od' or 'ratio'")
    if config.smooth_method != "savgol":
        raise ValueError("smooth_method must currently be 'savgol'")
    if config.n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if config.bootstrap_batch <= 0:
        raise ValueError("bootstrap_batch must be positive")
    if config.min_shots_per_point < 1:
        raise ValueError("min_shots_per_point must be positive")
    if config.smooth_window_ev <= 0:
        raise ValueError("smooth_window_ev must be positive")
    if config.smooth_polyorder < 0:
        raise ValueError("smooth_polyorder must be non-negative")


def _to_core_config(config: PeakSignificanceConfig) -> core.CoreXASConfig:
    return core.CoreXASConfig(
        input_h5=config.input_h5,
        vls_intensity_threshold=config.vls_intensity_threshold,
        gmd_min_threshold=config.gmd_min_threshold,
        gmd_max_threshold=config.gmd_max_threshold,
        center_method=config.center_method,
        center_half_window=config.center_half_window,
        center_chunk_size=config.center_chunk_size,
        proxy_pixel_bin_width=config.proxy_pixel_bin_width,
        correction_mode=config.correction_mode,
        xas_y_mode=config.xas_y_mode,
        chunk_size=config.chunk_size,
        pixel_map_config=config.pixel_map_config,
        pixel_map_calibration=config.pixel_map_calibration,
        inverse_map_range_ev=config.inverse_map_range_ev,
        inverse_map_samples=config.inverse_map_samples,
    )


def _collect_grouped_shots(
    path: Path,
    info: dict[str, Any],
    config: PeakSignificanceConfig,
    core_config: core.CoreXASConfig,
    fits: core.CoreLinearFit,
) -> tuple[dict[str, _GroupedShots], Any, Any]:
    n_e = info["nominal_energies"].size
    n_pix = int(info["n_pixels"])
    proxy_width = max(1, int(config.proxy_pixel_bin_width))
    n_proxy = int(np.ceil(n_pix / proxy_width))
    nominal_g = [[] for _ in range(n_e)]
    nominal_t = [[] for _ in range(n_e)]
    proxy_g = [[] for _ in range(n_proxy)]
    proxy_t = [[] for _ in range(n_proxy)]
    proxy_x_sum = np.zeros(n_proxy, dtype=np.float64)
    proxy_x_count = np.zeros(n_proxy, dtype=np.float64)
    use_corrected = config.correction_mode == "per_energy_intercept"

    with h5py.File(path, "r") as h5:
        for ie, n in enumerate(info["n_shots"].astype(int)):
            for start in range(0, int(n), int(config.chunk_size)):
                stop = min(int(n), start + int(config.chunk_size))
                vls = h5["vls"][ie, start:stop, :]
                gmd = sxa.read_hall_gmd_slice(h5, ie, start, stop)
                keep, _ = core._quality_mask(vls, gmd, core_config)
                if not np.any(keep):
                    continue
                vv = np.asarray(vls[keep], dtype=np.float64)
                gg = np.asarray(gmd[keep], dtype=np.float64)
                raw_trans = core._integrated_vls(vv)
                intercept = fits.intercept[ie]
                offset = 0.0 if not np.isfinite(intercept) else float(intercept)
                trans = raw_trans - offset if use_corrected else raw_trans
                valid_nominal = np.isfinite(gg) & np.isfinite(trans)
                if np.any(valid_nominal):
                    nominal_g[ie].append(gg[valid_nominal])
                    nominal_t[ie].append(trans[valid_nominal])

                spectra_for_center = vv - offset / n_pix if use_corrected else vv
                center_rel, _ = sxa.estimate_center_pixels(
                    spectra_for_center,
                    method=config.center_method,
                    half_window=int(config.center_half_window),
                    chunk_size=int(config.center_chunk_size),
                )
                valid_center = np.isfinite(center_rel) & np.isfinite(gg) & np.isfinite(trans)
                if not np.any(valid_center):
                    continue
                rel = np.clip(np.round(center_rel[valid_center]).astype(np.int64), 0, n_pix - 1)
                proxy_bin = rel // proxy_width
                center_pixel = np.interp(
                    center_rel[valid_center],
                    np.arange(n_pix, dtype=np.float64),
                    info["vls_pixels"].astype(np.float64),
                    left=np.nan,
                    right=np.nan,
                )
                gg_proxy = gg[valid_center]
                trans_proxy = trans[valid_center]
                for ibin in np.unique(proxy_bin):
                    sel = proxy_bin == ibin
                    proxy_g[int(ibin)].append(gg_proxy[sel])
                    proxy_t[int(ibin)].append(trans_proxy[sel])
                ok_x = np.isfinite(center_pixel)
                np.add.at(proxy_x_sum, proxy_bin[ok_x], center_pixel[ok_x])
                np.add.at(proxy_x_count, proxy_bin[ok_x], 1.0)

    nominal = _GroupedShots(
        x=info["nominal_energies"],
        gmd=_concat_groups(nominal_g),
        trans=_concat_groups(nominal_t),
        x_label="nominal photon energy (eV)",
        metadata={"mode": "nominal", "correction": config.correction_mode},
    )

    proxy_x = np.full(n_proxy, np.nan, dtype=np.float64)
    np.divide(proxy_x_sum, proxy_x_count, out=proxy_x, where=proxy_x_count > 0)
    has_proxy = np.array([len(parts) > 0 for parts in proxy_g]) & np.isfinite(proxy_x)
    proxy_energy, pixel_map_config, pixel_map_calibration = sxa.convert_pixels_to_energy(
        proxy_x[has_proxy],
        config.pixel_map_config,
        config.pixel_map_calibration,
        config.inverse_map_range_ev,
        n_samples=int(config.inverse_map_samples),
    )
    original_bins = np.where(has_proxy)[0]
    order = np.where(np.isfinite(proxy_energy))[0][np.argsort(proxy_energy[np.isfinite(proxy_energy)])]
    ordered_bins = original_bins[order]
    converted = _GroupedShots(
        x=proxy_energy[order],
        gmd=[_concat_parts(proxy_g[int(i)]) for i in ordered_bins],
        trans=[_concat_parts(proxy_t[int(i)]) for i in ordered_bins],
        x_label="converted photon energy (eV)",
        metadata={
            "mode": "converted_proxy",
            "correction": config.correction_mode,
            "center_method": config.center_method,
            "proxy_pixel_bin_width": proxy_width,
        },
    )
    return {"nominal": nominal, "converted": converted}, pixel_map_config, pixel_map_calibration


def _compute_curve(grouped: _GroupedShots, config: PeakSignificanceConfig, rng: np.random.Generator) -> PeakCurve:
    grouped = _sort_grouped_by_x(grouped)
    counts = np.array([g.size for g in grouped.gmd], dtype=np.int64)
    gmd_sum = np.array([np.sum(g, dtype=np.float64) for g in grouped.gmd], dtype=np.float64)
    trans_sum = np.array([np.sum(t, dtype=np.float64) for t in grouped.trans], dtype=np.float64)
    xas = _xas_from_sums(gmd_sum, trans_sum, config.xas_y_mode)
    xas[counts < int(config.min_shots_per_point)] = np.nan
    smooth = _smooth_curve(grouped.x, xas, config)
    residual = xas - smooth
    boot_xas = _bootstrap_xas(grouped, config, rng)
    boot_smooth = np.full_like(boot_xas, np.nan)
    for iboot in range(boot_xas.shape[0]):
        boot_smooth[iboot] = _smooth_curve(grouped.x, boot_xas[iboot], config)
    boot_residual = boot_xas - boot_smooth
    xas_sigma = _nanstd_or_nan(boot_xas)
    residual_sigma = _nanstd_or_nan(boot_residual)
    z = np.full(residual.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(residual) & np.isfinite(residual_sigma) & (residual_sigma > 0)
    z[ok] = residual[ok] / residual_sigma[ok]
    return PeakCurve(
        x=grouped.x,
        xas=xas,
        smooth=smooth,
        residual=residual,
        xas_sigma=xas_sigma,
        residual_sigma=residual_sigma,
        z=z,
        counts=counts,
        gmd_sum=gmd_sum,
        trans_sum=trans_sum,
        x_label=grouped.x_label,
        y_label=_y_label(config.xas_y_mode),
        metadata=grouped.metadata,
    )


def _bootstrap_xas(grouped: _GroupedShots, config: PeakSignificanceConfig, rng: np.random.Generator) -> np.ndarray:
    n_boot = int(config.n_bootstrap)
    n_bins = len(grouped.gmd)
    boot = np.full((n_boot, n_bins), np.nan, dtype=np.float64)
    if n_boot == 0:
        return boot
    batch_size = int(config.bootstrap_batch)
    min_shots = int(config.min_shots_per_point)
    for ibin, (gmd, trans) in enumerate(zip(grouped.gmd, grouped.trans)):
        n = int(gmd.size)
        if n < min_shots:
            continue
        for start in range(0, n_boot, batch_size):
            stop = min(n_boot, start + batch_size)
            sample = rng.integers(0, n, size=(stop - start, n), endpoint=False)
            g_sum = np.sum(gmd[sample], axis=1, dtype=np.float64)
            t_sum = np.sum(trans[sample], axis=1, dtype=np.float64)
            boot[start:stop, ibin] = _xas_from_sums(g_sum, t_sum, config.xas_y_mode)
    return boot


def _smooth_curve(x: np.ndarray, y: np.ndarray, config: PeakSignificanceConfig) -> np.ndarray:
    if savgol_filter is None:
        raise ImportError("scipy.signal.savgol_filter is required for SMOOTH_METHOD='savgol'")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    n_finite = int(np.sum(finite))
    min_window = int(config.smooth_polyorder) + 2
    if min_window % 2 == 0:
        min_window += 1
    if n_finite < min_window:
        return out
    finite_idx = np.where(finite)[0]
    finite_order = finite_idx[np.argsort(x[finite])]
    x_finite = x[finite_order]
    y_finite = y[finite_order]
    dx = np.nanmedian(np.diff(x_finite))
    if not np.isfinite(dx) or dx <= 0:
        window = min_window
    else:
        window = int(np.ceil(float(config.smooth_window_ev) / dx))
        if window % 2 == 0:
            window += 1
        window = max(window, min_window)
    max_window = n_finite if n_finite % 2 == 1 else n_finite - 1
    window = min(window, max_window)
    if window <= int(config.smooth_polyorder):
        return out
    interp = np.interp(x, x_finite, y_finite)
    smooth = savgol_filter(interp, window_length=window, polyorder=int(config.smooth_polyorder), mode="interp")
    out[finite] = smooth[finite]
    return out


def _plot_xas_axis(ax, curve: PeakCurve, title: str, config: PeakSignificanceConfig) -> None:
    ax.plot(curve.x, curve.xas, color="#1f77b4", lw=1.2, marker="o", ms=3, label="pooled XAS")
    ax.plot(curve.x, curve.smooth, color="#222222", lw=1.4, label=f"smooth baseline ({config.smooth_window_ev:g} eV)")
    ok = np.isfinite(curve.xas) & np.isfinite(curve.xas_sigma)
    if np.any(ok):
        ax.fill_between(
            curve.x[ok],
            curve.xas[ok] - curve.xas_sigma[ok],
            curve.xas[ok] + curve.xas_sigma[ok],
            color="#1f77b4",
            alpha=0.18,
            linewidth=0,
            label="bootstrap +/-1 sigma",
        )
    ax.set_title(title)
    ax.set_ylabel(curve.y_label)
    ax.legend(frameon=False, fontsize=8, loc="best")


def _plot_residual_axis(ax, curve: PeakCurve, title: str, config: PeakSignificanceConfig) -> None:
    ax.axhline(0.0, color="#222222", lw=0.9, alpha=0.8)
    ax.plot(curve.x, curve.residual, color="#4c78a8", lw=1.1, marker="o", ms=3, label="residual")
    ok = np.isfinite(curve.residual) & np.isfinite(curve.residual_sigma)
    if np.any(ok):
        ax.fill_between(
            curve.x[ok],
            curve.residual[ok] - curve.residual_sigma[ok],
            curve.residual[ok] + curve.residual_sigma[ok],
            color="#4c78a8",
            alpha=0.18,
            linewidth=0,
            label="bootstrap +/-1 sigma",
        )
    marker_styles = [("o", "#d98c00"), ("*", "#c0392b"), ("D", "#6f3fa0")]
    for threshold, (marker, color) in zip(sorted(float(z) for z in config.z_markers), marker_styles):
        sel = ok & (np.abs(curve.z) >= threshold)
        if np.any(sel):
            ax.scatter(
                curve.x[sel],
                curve.residual[sel],
                s=35 if marker != "*" else 55,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
                label=f"|z| >= {threshold:g}",
            )
    ax.set_title(title)
    ax.set_ylabel(f"residual {curve.y_label}")
    ax.legend(frameon=False, fontsize=8, loc="best")


def _shade_feature_windows(ax, windows: Sequence[Sequence[float]]) -> None:
    for window in windows:
        if len(window) != 2:
            continue
        lo, hi = float(window[0]), float(window[1])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(min(lo, hi), max(lo, hi), color="#888888", alpha=0.12, linewidth=0)


def _xas_from_sums(gmd_sum: np.ndarray, trans_sum: np.ndarray, y_mode: str) -> np.ndarray:
    ratio = np.full(np.broadcast_shapes(np.shape(gmd_sum), np.shape(trans_sum)), np.nan, dtype=np.float64)
    valid = np.isfinite(gmd_sum) & np.isfinite(trans_sum) & (trans_sum != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(gmd_sum, trans_sum, out=ratio, where=valid)
    if str(y_mode).lower() == "ratio":
        return ratio
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(ratio) & (ratio > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.log10(ratio[ok])
    return out


def _nanstd_or_nan(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    finite_counts = np.sum(np.isfinite(values), axis=0)
    ok = finite_counts >= 2
    if np.any(ok):
        out[ok] = np.nanstd(values[:, ok], axis=0, ddof=1)
    return out


def _concat_groups(groups: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [_concat_parts(parts) for parts in groups]


def _sort_grouped_by_x(grouped: _GroupedShots) -> _GroupedShots:
    order = np.where(np.isfinite(grouped.x))[0][np.argsort(grouped.x[np.isfinite(grouped.x)])]
    return _GroupedShots(
        x=np.asarray(grouped.x, dtype=np.float64)[order],
        gmd=[grouped.gmd[int(i)] for i in order],
        trans=[grouped.trans[int(i)] for i in order],
        x_label=grouped.x_label,
        metadata=grouped.metadata,
    )


def _concat_parts(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64, copy=False)


def _y_label(y_mode: str) -> str:
    if str(y_mode).lower() == "ratio":
        return "sum(GMD) / sum(VLS)"
    return "OD = log10[sum(GMD) / sum(VLS)]"
