"""
Reusable downstream helpers for static XAS analysis.

The raw/static processing entry point is still ``compute_static_xas.py``.
This module starts from the processed static-XAS H5 layout written by
that script and keeps notebook logic focused on configuration, plots,
and scientific choices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:  # Imported lazily by the functions that need it, but cached here.
    from scipy.signal import find_peaks, savgol_filter
except Exception:  # pragma: no cover - exercised only on incomplete envs.
    find_peaks = None
    savgol_filter = None

from gotthard_energy_pixel_map import (
    CalibrationPoint,
    SpectrometerConfig,
    pixel_to_energy,
)


@dataclass(frozen=True)
class StaticXASRun:
    """Processed static-XAS H5 contents."""

    path: Optional[Path]
    vls: np.ndarray
    gmd: np.ndarray
    n_shots: np.ndarray
    nominal_energies: np.ndarray
    vls_pixels: np.ndarray
    attrs: dict[str, Any]
    gmd_tunnel: Optional[np.ndarray] = None

    @property
    def n_energy(self) -> int:
        return int(self.nominal_energies.size)

    @property
    def n_pixels(self) -> int:
        return int(self.vls_pixels.size)

    @property
    def stored_shots(self) -> int:
        return int(np.nansum(self.n_shots))


@dataclass(frozen=True)
class PreparedShots:
    """Finite, quality-filtered shot table for proxy-energy analysis."""

    gmd: np.ndarray
    vls: np.ndarray
    nominal_energy: np.ndarray
    nominal_index: np.ndarray
    peak_vls: np.ndarray
    center_rel: np.ndarray
    proxy_rel: np.ndarray
    center_pixel: np.ndarray
    center_method: str
    summary: dict[str, Any]
    double_peak_diagnostics: Optional[dict[str, np.ndarray]] = None
    rejected_double_peaks: Optional[dict[str, np.ndarray]] = None

    @property
    def n_shots(self) -> int:
        return int(self.gmd.size)


@dataclass(frozen=True)
class BinnedXAS:
    """Binned XAS result with enough state for diagnostics and replots."""

    x: np.ndarray
    xas: np.ndarray
    n_per_bin: np.ndarray
    gmd_edges: np.ndarray
    g_mean: np.ndarray
    a_mean: np.ndarray
    x_label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AbsorbanceCurve:
    """Traditional absorbance curve from transmitted and incident intensities."""

    x: np.ndarray
    absorbance: np.ndarray
    trans_intensity: np.ndarray
    incident_intensity: np.ndarray
    n_trans: np.ndarray
    n_incident: np.ndarray
    x_label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GmdBinnedAbsorbanceCurve:
    """Traditional absorbance curves split by GMD bin."""

    x: np.ndarray
    absorbance: np.ndarray
    trans_intensity: np.ndarray
    incident_intensity: np.ndarray
    n_trans: np.ndarray
    n_incident: np.ndarray
    gmd_edges: np.ndarray
    x_label: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Loading and finite-shot flattening
# ---------------------------------------------------------------------------


def load_static_xas_run(path: str | Path, *, include_gmd_tunnel: bool = True) -> StaticXASRun:
    """Load a processed static-XAS H5 file into memory."""

    p = Path(path)
    with h5py.File(p, "r") as f:
        required = ("vls", "gmd", "n_shots", "nominal_energies", "vls_pixels")
        missing = [key for key in required if key not in f]
        if missing:
            raise KeyError(f"{p} is missing required dataset(s): {missing}")
        gmd_tunnel = f["gmd_tunnel"][...] if include_gmd_tunnel and "gmd_tunnel" in f else None
        return StaticXASRun(
            path=p,
            vls=f["vls"][...],
            gmd=f["gmd"][...],
            n_shots=f["n_shots"][...],
            nominal_energies=f["nominal_energies"][...],
            vls_pixels=f["vls_pixels"][...],
            attrs=dict(f.attrs),
            gmd_tunnel=gmd_tunnel,
        )


def finite_shot_blocks(run: StaticXASRun):
    """Yield finite shots per nominal-energy section, excluding NaN padding."""

    for ie in range(run.n_energy):
        n = int(run.n_shots[ie])
        if n <= 0:
            continue
        a = run.vls[ie, :n, :]
        g = run.gmd[ie, :n]
        finite = np.isfinite(g) & np.all(np.isfinite(a), axis=1)
        if not np.any(finite):
            continue
        count = int(np.sum(finite))
        yield {
            "energy_index": np.full(count, ie, dtype=np.int32),
            "nominal_energy": np.full(count, run.nominal_energies[ie], dtype=np.float64),
            "gmd": g[finite],
            "vls": a[finite],
        }


def flatten_finite_shots(run: StaticXASRun) -> dict[str, np.ndarray]:
    """Flatten processed H5 data to finite shots without using padded rows."""

    blocks = list(finite_shot_blocks(run))
    if not blocks:
        return {
            "gmd": np.empty(0, dtype=np.float64),
            "vls": np.empty((0, run.n_pixels), dtype=np.float64),
            "nominal_energy": np.empty(0, dtype=np.float64),
            "nominal_index": np.empty(0, dtype=np.int32),
        }
    return {
        "gmd": np.concatenate([b["gmd"] for b in blocks]),
        "vls": np.concatenate([b["vls"] for b in blocks], axis=0),
        "nominal_energy": np.concatenate([b["nominal_energy"] for b in blocks]),
        "nominal_index": np.concatenate([b["energy_index"] for b in blocks]),
    }


# ---------------------------------------------------------------------------
# Shot metrics and center-pixel methods
# ---------------------------------------------------------------------------


def vls_peak_metrics(spectra: np.ndarray) -> dict[str, np.ndarray]:
    """Return simple per-shot VLS intensity metrics."""

    a = np.asarray(spectra, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("spectra must be 2D: (n_shots, n_pixels)")
    if a.shape[0] == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"peak_vls": empty, "sum_vls": empty, "sum_positive_vls": empty, "mean_vls": empty}
    return {
        "peak_vls": np.nanmax(a, axis=1),
        "sum_vls": np.nansum(a, axis=1),
        "sum_positive_vls": np.nansum(np.maximum(a, 0.0), axis=1),
        "mean_vls": np.nanmean(a, axis=1),
    }


def _positive_spectra(spectra: np.ndarray) -> np.ndarray:
    """Return a finite, positive 2D VLS array for center calculations."""

    a = np.asarray(spectra, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("spectra must be 2D")
    return np.maximum(np.nan_to_num(a, nan=0.0), 0.0)


def _peak_indices(a_positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strongest positive peak index and value for each shot."""

    if a_positive.ndim != 2:
        raise ValueError("spectra must be 2D")
    if a_positive.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    peak_idx = np.argmax(a_positive, axis=1).astype(np.int64)
    peak_val = a_positive[np.arange(a_positive.shape[0]), peak_idx]
    return peak_idx, peak_val


def _windowed_com_from_peaks(
    a_positive: np.ndarray,
    peak_idx: np.ndarray,
    peak_val: np.ndarray,
    *,
    half_window: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Positive COM inside peak +/- half_window, clipped to the stored ROI."""

    n_shots, n_pixels = a_positive.shape
    centers = np.full(n_shots, np.nan, dtype=np.float64)
    if n_shots == 0 or n_pixels == 0:
        return centers

    half_window = max(0, int(half_window))
    chunk_size = max(1, int(chunk_size))
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)

    for start in range(0, n_shots, chunk_size):
        stop = min(n_shots, start + chunk_size)
        local_peak = peak_idx[start:stop]
        local_peak_val = peak_val[start:stop]
        idx = local_peak[:, None] + offsets[None, :]
        valid = (idx >= 0) & (idx < n_pixels) & (local_peak_val[:, None] > 0)
        idx_clip = np.clip(idx, 0, n_pixels - 1)
        y = np.take_along_axis(a_positive[start:stop], idx_clip, axis=1)
        y = np.where(valid, y, 0.0)
        weight = y.sum(axis=1)
        pix_weight = (y * np.where(valid, idx, 0.0)).sum(axis=1)
        ok = weight > 0
        chunk_centers = centers[start:stop]
        chunk_centers[ok] = pix_weight[ok] / weight[ok]
    return centers


def center_com(
    spectra: np.ndarray,
    *,
    half_window: Optional[int] = None,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Positive-weight centre of mass on the relative pixel axis.

    If ``half_window`` is provided, the COM is evaluated only inside each
    shot's strongest-peak +/- half_window local ROI. Calling
    ``center_com(spectra)`` keeps the historical full-axis behavior.
    """

    a = _positive_spectra(spectra)
    if a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if half_window is not None:
        peak_idx, peak_val = _peak_indices(a)
        return _windowed_com_from_peaks(
            a,
            peak_idx,
            peak_val,
            half_window=int(half_window),
            chunk_size=int(chunk_size),
        )

    pix = np.arange(a.shape[1], dtype=np.float64)
    weight = a.sum(axis=1)
    center = np.full(a.shape[0], np.nan, dtype=np.float64)
    ok = weight > 0
    center[ok] = (a[ok] @ pix) / weight[ok]
    return center


def center_peak(spectra: np.ndarray) -> np.ndarray:
    """Argmax center on the relative pixel axis."""

    a = _positive_spectra(spectra)
    if a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    peak_idx, peak_val = _peak_indices(a)
    center = peak_idx.astype(np.float64)
    center[peak_val <= 0] = np.nan
    return center


def center_smooth_peak(
    spectra: np.ndarray,
    *,
    half_window: int = 12,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Local quadratic refinement around the strongest positive peak."""

    a = np.asarray(spectra, dtype=np.float64)
    n_shots, n_pixels = a.shape
    half_window = int(half_window)
    chunk_size = int(chunk_size)
    centers = center_com(a, half_window=half_window, chunk_size=chunk_size)
    if half_window < 1 or n_shots == 0:
        return centers

    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    x = offsets.astype(np.float64)
    pinv = np.linalg.pinv(np.column_stack([x**2, x, np.ones_like(x)]))

    for start in range(0, n_shots, chunk_size):
        stop = min(n_shots, start + chunk_size)
        y = np.maximum(np.nan_to_num(a[start:stop], nan=0.0), 0.0)
        peak = np.argmax(y, axis=1)
        peak_val = y[np.arange(y.shape[0]), peak]
        edge_ok = (peak_val > 0) & (peak >= half_window) & (peak + half_window < n_pixels)
        rows = np.where(edge_ok)[0]
        if rows.size == 0:
            continue
        idx = peak[rows, None] + offsets[None, :]
        yw = y[rows[:, None], idx]
        coeff = yw @ pinv.T
        quad = coeff[:, 0]
        lin = coeff[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = -lin / (2.0 * quad)
        ok = np.isfinite(delta) & (quad < 0) & (np.abs(delta) <= half_window)
        centers[start + rows[ok]] = peak[rows[ok]] + delta[ok]
    return centers


def center_gaussian_logfit(
    spectra: np.ndarray,
    *,
    half_window: int = 12,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Fast Gaussian-like peak center from a local log-quadratic fit."""

    a = np.asarray(spectra, dtype=np.float64)
    n_shots, n_pixels = a.shape
    centers = np.full(n_shots, np.nan, dtype=np.float64)
    fallback = center_com(a, half_window=half_window, chunk_size=chunk_size)
    half_window = int(half_window)
    chunk_size = int(chunk_size)
    if half_window < 1 or n_shots == 0:
        return fallback

    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    x = offsets.astype(np.float64)
    pinv = np.linalg.pinv(np.column_stack([x**2, x, np.ones_like(x)]))
    eps = np.finfo(np.float64).tiny

    for start in range(0, n_shots, chunk_size):
        stop = min(n_shots, start + chunk_size)
        y = np.maximum(np.nan_to_num(a[start:stop], nan=0.0), 0.0)
        peak = np.argmax(y, axis=1)
        peak_val = y[np.arange(y.shape[0]), peak]
        edge_ok = (peak_val > 0) & (peak >= half_window) & (peak + half_window < n_pixels)
        rows = np.where(edge_ok)[0]
        if rows.size == 0:
            continue
        idx = peak[rows, None] + offsets[None, :]
        yw = y[rows[:, None], idx]
        baseline = np.percentile(yw, 10, axis=1)
        signal = np.maximum(yw - baseline[:, None], eps)
        coeff = np.log(signal) @ pinv.T
        quad = coeff[:, 0]
        lin = coeff[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = -lin / (2.0 * quad)
        ok = np.isfinite(delta) & (quad < 0) & (np.abs(delta) <= half_window)
        centers[start + rows[ok]] = peak[rows[ok]] + delta[ok]

    missing = ~np.isfinite(centers)
    centers[missing] = fallback[missing]
    return centers


def estimate_center_pixels(
    spectra: np.ndarray,
    *,
    method: str = "com",
    half_window: int = 12,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, str]:
    """Estimate relative center pixel for each shot."""

    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    if key == "com":
        return center_com(spectra, half_window=half_window, chunk_size=chunk_size), "com"
    if key in {"peak", "max", "argmax"}:
        return center_peak(spectra), "peak"
    if key in {"smooth_peak", "peak_smooth", "parabolic", "parabola"}:
        return center_smooth_peak(spectra, half_window=half_window, chunk_size=chunk_size), "smooth_peak"
    if key in {"gaussian", "gauss", "fit", "gaussian_logfit", "logfit"}:
        return center_gaussian_logfit(spectra, half_window=half_window, chunk_size=chunk_size), "gaussian_logfit"
    raise ValueError("method must be one of: com, peak, smooth_peak, gaussian_logfit")


# ---------------------------------------------------------------------------
# Single/double peak quality filtering
# ---------------------------------------------------------------------------


def _require_scipy_signal() -> None:
    if find_peaks is None or savgol_filter is None:
        raise ImportError("scipy.signal is required for double-peak rejection")


def _valid_savgol_window(n_pixels: int, window: int, poly: int) -> int:
    window = int(window)
    if window % 2 == 0:
        window -= 1
    window = min(window, n_pixels if n_pixels % 2 == 1 else n_pixels - 1)
    if window <= poly:
        window = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    if window > n_pixels:
        window = n_pixels if n_pixels % 2 == 1 else n_pixels - 1
    if window <= poly or window < 3:
        raise ValueError("not enough pixels for Savitzky-Golay smoothing")
    return int(window)


def preprocess_shots_for_peakfinding(
    spectra: np.ndarray,
    *,
    baseline_q: float = 0.15,
    sg_window: int = 9,
    sg_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline-correct, smooth, and estimate robust shot noise."""

    _require_scipy_signal()
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("spectra must be 2D")
    if x.shape[0] == 0:
        return x.copy(), x.copy(), np.empty(0, dtype=np.float64)

    baseline = np.quantile(x, baseline_q, axis=1, keepdims=True)
    xcorr = x - baseline
    xcorr[xcorr < 0] = 0.0
    window = _valid_savgol_window(xcorr.shape[1], sg_window, sg_poly)
    xsm = savgol_filter(xcorr, window_length=window, polyorder=int(sg_poly), axis=1, mode="interp")

    d = np.diff(xcorr, axis=1)
    med = np.median(d, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(d - med), axis=1) / np.sqrt(2.0)
    floor = np.median(noise[noise > 0]) if np.any(noise > 0) else 1.0
    noise = np.where(noise > 0, noise, floor)
    return xcorr, xsm, noise


def detect_double_peaks_pairscan(
    xcorr: np.ndarray,
    xsm: np.ndarray,
    noise: np.ndarray,
    *,
    proxy_pixel: Optional[np.ndarray] = None,
    min_distance_px: int = 3,
    max_distance_px: int = 17,
    min_prom_sigma: float = 2.5,
    min_height_sigma: float = 2.0,
    min_rel_peak_height: float = 0.35,
    min_width_px: float = 1.0,
    max_width_px: float = 40.0,
    min_second_ratio: float = 0.05,
    min_bridge_drop: float = 0.015,
    min_split_sigma: float = 0.75,
    max_peaks_to_score: int = 20,
    proxy_weight: float = 0.10,
) -> dict[str, np.ndarray]:
    """Detect shots with two resolved VLS peaks."""

    _require_scipy_signal()
    yall = np.asarray(xsm, dtype=np.float64)
    nshots, _ = yall.shape

    is_candidate = np.zeros(nshots, dtype=bool)
    peak1_px = np.full(nshots, -1, dtype=np.int16)
    peak2_px = np.full(nshots, -1, dtype=np.int16)
    second_ratio = np.full(nshots, np.nan, dtype=np.float64)
    valley_ratio = np.full(nshots, np.nan, dtype=np.float64)
    bridge_drop = np.full(nshots, np.nan, dtype=np.float64)
    split_sigma = np.full(nshots, np.nan, dtype=np.float64)
    sep = np.full(nshots, np.nan, dtype=np.float64)
    score = np.full(nshots, -np.inf, dtype=np.float64)
    n_peaks = np.zeros(nshots, dtype=np.int16)

    for i in range(nshots):
        y = yall[i]
        sig = max(float(noise[i]), 1e-9)
        shot_max = np.nanmax(y)
        if not np.isfinite(shot_max) or shot_max <= 0:
            continue

        height_floor = max(min_height_sigma * sig, min_rel_peak_height * shot_max)
        peaks, props = find_peaks(
            y,
            distance=int(min_distance_px),
            prominence=float(min_prom_sigma) * sig,
            height=height_floor,
            width=(float(min_width_px), float(max_width_px)),
        )
        n_peaks[i] = len(peaks)
        if len(peaks) < 2:
            continue

        order = np.argsort(props["prominences"])[::-1][: int(max_peaks_to_score)]
        peaks = peaks[order]
        heights = props["peak_heights"][order]
        prominences = props["prominences"][order]
        best = None

        for ia in range(len(peaks)):
            for ib in range(ia + 1, len(peaks)):
                p1, p2 = int(peaks[ia]), int(peaks[ib])
                h1, h2 = float(heights[ia]), float(heights[ib])
                pr1, pr2 = float(prominences[ia]), float(prominences[ib])
                lo, hi = sorted((p1, p2))
                s = hi - lo
                if s < min_distance_px or s > max_distance_px:
                    continue
                weaker = min(h1, h2)
                stronger = max(h1, h2)
                if weaker <= 0 or stronger <= 0:
                    continue
                valley = float(np.min(y[lo : hi + 1]))
                sr = weaker / stronger
                vr = valley / weaker
                bd = max((weaker - valley) / weaker, 0.0)
                ss = max((weaker - valley) / sig, 0.0)
                proxy_bonus = 1.0
                if proxy_pixel is not None:
                    proxy_bonus += proxy_weight * float(lo <= proxy_pixel[i] <= hi)
                sc = (
                    np.sqrt(max(min(pr1, pr2) / sig, 1e-6))
                    * np.sqrt(max(s, 1))
                    * sr
                    * (0.05 + bd)
                    * (1.0 + 0.20 * ss)
                    * proxy_bonus
                )
                if best is None or sc > best[0]:
                    best = (sc, lo, hi, sr, vr, bd, ss, s)

        if best is None:
            continue
        sc, lo, hi, sr, vr, bd, ss, s = best
        peak1_px[i] = lo
        peak2_px[i] = hi
        second_ratio[i] = sr
        valley_ratio[i] = vr
        bridge_drop[i] = bd
        split_sigma[i] = ss
        sep[i] = s
        score[i] = sc
        is_candidate[i] = (sr >= min_second_ratio) and ((bd >= min_bridge_drop) or (ss >= min_split_sigma))

    return {
        "is_candidate": is_candidate,
        "peak1_px": peak1_px,
        "peak2_px": peak2_px,
        "second_ratio": second_ratio,
        "valley_ratio": valley_ratio,
        "bridge_drop": bridge_drop,
        "split_sigma": split_sigma,
        "sep": sep,
        "score": score,
        "n_peaks": n_peaks,
    }


def prepare_proxy_shots(
    run: StaticXASRun,
    *,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    center_method: str = "com",
    center_half_window: int = 12,
    center_chunk_size: int = 50_000,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
) -> PreparedShots:
    """Flatten finite shots, apply quality filters, and assign proxy pixels."""

    flat = flatten_finite_shots(run)
    g = flat["gmd"]
    a = flat["vls"]
    nominal = flat["nominal_energy"]
    nominal_index = flat["nominal_index"]
    summary: dict[str, Any] = {
        "stored_shots": run.stored_shots,
        "finite_shots": int(g.size),
        "vls_intensity_threshold": vls_intensity_threshold,
        "gmd_min_threshold": gmd_min_threshold,
        "gmd_max_threshold": gmd_max_threshold,
        "after_peak_threshold": int(g.size),
        "after_gmd_filter": int(g.size),
        "low_gmd_rejected": 0,
        "high_gmd_rejected": 0,
        "double_peak_rejected": 0,
        "after_double_peak": int(g.size),
        "after_center": 0,
        "final_shots": 0,
        "center_half_window": center_half_window,
    }

    peak_vls = vls_peak_metrics(a)["peak_vls"]
    if vls_intensity_threshold is not None:
        keep = peak_vls >= float(vls_intensity_threshold)
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )
    summary["after_peak_threshold"] = int(g.size)

    if gmd_min_threshold is not None or gmd_max_threshold is not None:
        keep = np.ones(g.shape, dtype=bool)
        if gmd_min_threshold is not None:
            low = g < float(gmd_min_threshold)
            summary["low_gmd_rejected"] = int(np.sum(low))
            keep &= ~low
        if gmd_max_threshold is not None:
            high = g > float(gmd_max_threshold)
            summary["high_gmd_rejected"] = int(np.sum(high))
            keep &= ~high
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )
    summary["after_gmd_filter"] = int(g.size)

    diagnostics = None
    rejected_double_peaks = None
    if reject_double_peaks and g.size:
        xcorr, xsm, noise = preprocess_shots_for_peakfinding(
            a,
            baseline_q=peakfind_baseline_q,
            sg_window=peakfind_sg_window,
            sg_poly=peakfind_sg_poly,
        )
        kwargs = dict(double_peak_kwargs or {})
        diagnostics = detect_double_peaks_pairscan(xcorr, xsm, noise, **kwargs)
        double = diagnostics["is_candidate"]
        keep = ~double
        summary["double_peak_rejected"] = int(np.sum(double))
        if np.any(double):
            rejected_double_peaks = {
                "vls": a[double],
                "gmd": g[double],
                "nominal_energy": nominal[double],
                "nominal_index": nominal_index[double],
                "peak_vls": peak_vls[double],
            }
            for name, values in diagnostics.items():
                if values.shape[0] == double.shape[0]:
                    rejected_double_peaks[name] = values[double]
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )
    summary["after_double_peak"] = int(g.size)

    center_rel, canonical_method = estimate_center_pixels(
        a,
        method=center_method,
        half_window=center_half_window,
        chunk_size=center_chunk_size,
    )
    valid_center = np.isfinite(center_rel)
    g, a, nominal, nominal_index, peak_vls, center_rel = (
        g[valid_center],
        a[valid_center],
        nominal[valid_center],
        nominal_index[valid_center],
        peak_vls[valid_center],
        center_rel[valid_center],
    )
    summary["after_center"] = int(g.size)

    proxy_rel = np.clip(np.round(center_rel).astype(np.int64), 0, run.n_pixels - 1)
    center_pixel = np.interp(
        center_rel,
        np.arange(run.n_pixels, dtype=np.float64),
        run.vls_pixels.astype(np.float64),
        left=np.nan,
        right=np.nan,
    )

    summary["final_shots"] = int(g.size)
    summary["center_method"] = canonical_method
    summary["reject_double_peaks"] = bool(reject_double_peaks)
    return PreparedShots(
        gmd=g,
        vls=a,
        nominal_energy=nominal,
        nominal_index=nominal_index,
        peak_vls=peak_vls,
        center_rel=center_rel,
        proxy_rel=proxy_rel,
        center_pixel=center_pixel,
        center_method=canonical_method,
        summary=summary,
        double_peak_diagnostics=diagnostics,
        rejected_double_peaks=rejected_double_peaks,
    )


# ---------------------------------------------------------------------------
# Binning and pixel/energy conversion
# ---------------------------------------------------------------------------


def _empty_binned(n_x: int, n_gmd: int, n_pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_sum = np.zeros((n_x, n_gmd, n_pixels), dtype=np.float64)
    g_sum = np.zeros((n_x, n_gmd), dtype=np.float64)
    n_per_bin = np.zeros((n_x, n_gmd), dtype=np.int64)
    return a_sum, g_sum, n_per_bin, np.zeros(n_x, dtype=np.float64)


def _validated_gmd_edges(gmd_edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(gmd_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("gmd_edges must contain at least two edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("gmd_edges must be finite")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("gmd_edges must be strictly increasing")
    return edges


def _gmd_bin_indices(gmd: np.ndarray, gmd_edges: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return GMD bin indices with the final edge included in the last bin."""

    edges = _validated_gmd_edges(gmd_edges)
    values = np.asarray(gmd, dtype=np.float64)
    gbin = np.searchsorted(edges, values, side="right") - 1
    gbin = np.asarray(gbin, dtype=np.int64)
    n_gmd = edges.size - 1
    gbin[values == edges[-1]] = n_gmd - 1
    in_range = np.isfinite(values) & (gbin >= 0) & (gbin < n_gmd)
    return gbin, in_range


def _gmd_bin_label(gmd_edges: Sequence[float], index: int) -> str:
    edges = np.asarray(gmd_edges, dtype=np.float64)
    close = "]" if index == edges.size - 2 else ")"
    return f"[{edges[index]:.2f}, {edges[index + 1]:.2f}{close} uJ"


def _finish_binned(
    x: np.ndarray,
    a_sum: np.ndarray,
    g_sum: np.ndarray,
    n_per_bin: np.ndarray,
    gmd_edges: np.ndarray,
    *,
    x_label: str,
    metadata: Optional[dict[str, Any]] = None,
) -> BinnedXAS:
    safe_n = np.where(n_per_bin > 0, n_per_bin, 1).astype(np.float64)
    a_mean = np.where(n_per_bin[:, :, None] > 0, a_sum / safe_n[:, :, None], np.nan)
    g_mean = np.where(n_per_bin > 0, g_sum / safe_n, np.nan)
    vls_sum = np.nansum(a_sum, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        xas = np.where(n_per_bin > 0, g_sum / vls_sum, np.nan)
    return BinnedXAS(
        x=np.asarray(x, dtype=np.float64),
        xas=xas,
        n_per_bin=n_per_bin,
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        g_mean=g_mean,
        a_mean=a_mean,
        x_label=x_label,
        metadata=dict(metadata or {}),
    )


def bin_nominal_energy_run(
    run: StaticXASRun,
    gmd_edges: Sequence[float],
    *,
    vls_intensity_threshold: Optional[float] = None,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
) -> BinnedXAS:
    """Bin XAS by nominal photon energy and GMD."""

    edges = _validated_gmd_edges(gmd_edges)
    n_gmd = edges.size - 1
    a_sum, g_sum, n_per_bin, _ = _empty_binned(run.n_energy, n_gmd, run.n_pixels)

    for ie in range(run.n_energy):
        n = int(run.n_shots[ie])
        if n <= 0:
            continue
        a = run.vls[ie, :n, :]
        g = run.gmd[ie, :n]
        finite = np.isfinite(g) & np.all(np.isfinite(a), axis=1)
        if vls_intensity_threshold is not None:
            peak = vls_peak_metrics(a)["peak_vls"]
            finite &= peak >= float(vls_intensity_threshold)
        if gmd_min_threshold is not None:
            finite &= g >= float(gmd_min_threshold)
        if gmd_max_threshold is not None:
            finite &= g <= float(gmd_max_threshold)
        if not np.any(finite):
            continue
        a = a[finite]
        g = g[finite]
        gb, in_range = _gmd_bin_indices(g, edges)
        if not np.any(in_range):
            continue
        np.add.at(n_per_bin, (np.full(np.sum(in_range), ie), gb[in_range]), 1)
        np.add.at(g_sum, (np.full(np.sum(in_range), ie), gb[in_range]), g[in_range])
        np.add.at(a_sum, (np.full(np.sum(in_range), ie), gb[in_range]), a[in_range])

    return _finish_binned(
        run.nominal_energies,
        a_sum,
        g_sum,
        n_per_bin,
        edges,
        x_label="nominal photon energy (eV)",
        metadata={
            "mode": "nominal",
            "vls_intensity_threshold": vls_intensity_threshold,
            "gmd_min_threshold": gmd_min_threshold,
            "gmd_max_threshold": gmd_max_threshold,
        },
    )


def bin_proxy_energy(
    prepared: PreparedShots,
    vls_pixels: np.ndarray,
    gmd_edges: Sequence[float],
    *,
    pixel_bin_width: int = 1,
) -> BinnedXAS:
    """Bin XAS by shot-resolved center-pixel proxy and GMD."""

    edges = _validated_gmd_edges(gmd_edges)
    n_gmd = edges.size - 1
    n_pixels = int(vls_pixels.size)
    width = max(1, int(pixel_bin_width))
    if prepared.n_shots == 0:
        return _finish_binned(
            np.empty(0),
            np.zeros((0, n_gmd, n_pixels)),
            np.zeros((0, n_gmd)),
            np.zeros((0, n_gmd), dtype=np.int64),
            edges,
            x_label="shot-resolved photon-energy proxy (Gotthard pixel)",
            metadata={"mode": "proxy_pixel", "pixel_bin_width": width},
        )

    rel_min = int(np.nanmin(prepared.proxy_rel))
    rel_max = int(np.nanmax(prepared.proxy_rel))
    x_group = (prepared.proxy_rel - rel_min) // width
    n_x = int(x_group.max()) + 1
    a_sum, g_sum, n_per_bin, x_weight_sum = _empty_binned(n_x, n_gmd, n_pixels)
    x_count = np.zeros(n_x, dtype=np.float64)

    gb, in_range = _gmd_bin_indices(prepared.gmd, edges)
    if np.any(in_range):
        xb = x_group[in_range]
        gb = gb[in_range]
        g = prepared.gmd[in_range]
        a = prepared.vls[in_range]
        np.add.at(n_per_bin, (xb, gb), 1)
        np.add.at(g_sum, (xb, gb), g)
        np.add.at(a_sum, (xb, gb), a)

    np.add.at(x_weight_sum, x_group, prepared.center_pixel)
    np.add.at(x_count, x_group, 1.0)
    x = np.full(n_x, np.nan, dtype=np.float64)
    np.divide(x_weight_sum, x_count, out=x, where=x_count > 0)
    labels = []
    for i in range(n_x):
        lo_rel = rel_min + i * width
        hi_rel = min(rel_max, lo_rel + width - 1)
        lo_abs = vls_pixels[lo_rel]
        hi_abs = vls_pixels[hi_rel]
        labels.append(f"{int(lo_abs)}-{int(hi_abs)}" if lo_abs != hi_abs else f"{int(lo_abs)}")

    return _finish_binned(
        x,
        a_sum,
        g_sum,
        n_per_bin,
        edges,
        x_label="shot-resolved photon-energy proxy (Gotthard pixel)",
        metadata={
            "mode": "proxy_pixel",
            "pixel_bin_width": width,
            "x_bin_labels": labels,
            "center_method": prepared.center_method,
        },
    )


def build_pixel_map_config(
    config_dict: Mapping[str, Any],
    calibration_dict: Mapping[str, Any],
) -> tuple[SpectrometerConfig, CalibrationPoint]:
    """Build pixel-map dataclasses from notebook-style dictionaries."""

    return SpectrometerConfig(**dict(config_dict)), CalibrationPoint(**dict(calibration_dict))


def convert_pixels_to_energy(
    pixels: np.ndarray,
    config_dict: Mapping[str, Any],
    calibration_dict: Mapping[str, Any],
    e_range: Sequence[float],
    n_samples: int = 10_000,
) -> tuple[np.ndarray, SpectrometerConfig, CalibrationPoint]:
    """Convert absolute Gotthard pixels to photon energy in eV."""

    cfg, cal = build_pixel_map_config(config_dict, calibration_dict)
    energies = pixel_to_energy(
        np.asarray(pixels, dtype=np.float64),
        cfg,
        cal,
        e_min=float(e_range[0]),
        e_max=float(e_range[1]),
        n_samples=int(n_samples),
    )
    return energies, cfg, cal


def convert_binned_pixels_to_energy(
    binned: BinnedXAS,
    config_dict: Mapping[str, Any],
    calibration_dict: Mapping[str, Any],
    e_range: Sequence[float],
    n_samples: int = 10_000,
    *,
    sort: bool = True,
) -> tuple[BinnedXAS, SpectrometerConfig, CalibrationPoint]:
    """Return a copy of a proxy-pixel binned result with x in eV."""

    energy, cfg, cal = convert_pixels_to_energy(
        binned.x,
        config_dict,
        calibration_dict,
        e_range,
        n_samples=n_samples,
    )
    out = replace(
        binned,
        x=energy,
        x_label="converted photon energy (eV)",
        metadata={**binned.metadata, "converted_from": binned.x_label},
    )
    if sort and energy.size:
        order = np.argsort(out.x)
        out = replace(
            out,
            x=out.x[order],
            xas=out.xas[order],
            n_per_bin=out.n_per_bin[order],
            g_mean=out.g_mean[order],
            a_mean=out.a_mean[order],
        )
    return out, cfg, cal


# ---------------------------------------------------------------------------
# Traditional absorbance helpers
# ---------------------------------------------------------------------------


def _pixel_roi_mask(
    pixel_axis: Optional[np.ndarray],
    pixel_roi: Optional[Sequence[float]],
    n_pixels: int,
) -> np.ndarray:
    if pixel_roi is None:
        return np.ones(n_pixels, dtype=bool)
    if pixel_axis is None:
        raise ValueError("pixel_axis is required when pixel_roi is provided")
    pixels = np.asarray(pixel_axis, dtype=np.float64)
    if pixels.shape[0] != n_pixels:
        raise ValueError("pixel_axis length must match the VLS pixel dimension")
    lo, hi = float(pixel_roi[0]), float(pixel_roi[1])
    mask = (pixels >= lo) & (pixels <= hi)
    if not np.any(mask):
        raise ValueError("pixel_roi does not overlap the VLS pixel axis")
    return mask


def integrated_vls_intensity(
    spectra: np.ndarray,
    *,
    pixel_axis: Optional[np.ndarray] = None,
    pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
) -> float:
    """Integrated scalar intensity from the mean VLS spectrum.

    The intended Beer-Lambert scalar is I(E) = sum_pixels(<VLS(E)>).
    Negative mean-spectrum pixels are clipped to zero by default because the
    processed spectra are background-subtracted.
    """

    a = np.asarray(spectra, dtype=np.float64)
    if a.ndim == 1:
        mean_spectrum = a.copy()
    elif a.ndim == 2:
        if a.shape[0] == 0:
            return np.nan
        finite = np.all(np.isfinite(a), axis=1)
        if not np.any(finite):
            return np.nan
        mean_spectrum = np.nanmean(a[finite], axis=0)
    else:
        raise ValueError("spectra must be 1D or 2D")

    mask = _pixel_roi_mask(pixel_axis, pixel_roi, mean_spectrum.shape[0])
    y = mean_spectrum[mask]
    finite_y = np.isfinite(y)
    if not np.any(finite_y):
        return np.nan
    y = y[finite_y]
    if clip_negative:
        y = np.maximum(y, 0.0)
    intensity = float(np.sum(y))
    return intensity if np.isfinite(intensity) else np.nan


def _filter_flat_shots_for_absorbance(
    run: StaticXASRun,
    *,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Flatten and quality-filter shots without assigning center pixels."""

    flat = flatten_finite_shots(run)
    g = flat["gmd"]
    a = flat["vls"]
    nominal = flat["nominal_energy"]
    nominal_index = flat["nominal_index"]
    summary: dict[str, Any] = {
        "stored_shots": run.stored_shots,
        "finite_shots": int(g.size),
        "vls_intensity_threshold": vls_intensity_threshold,
        "gmd_min_threshold": gmd_min_threshold,
        "gmd_max_threshold": gmd_max_threshold,
        "after_peak_threshold": int(g.size),
        "after_gmd_filter": int(g.size),
        "low_gmd_rejected": 0,
        "high_gmd_rejected": 0,
        "double_peak_rejected": 0,
        "final_shots": 0,
    }

    peak_vls = vls_peak_metrics(a)["peak_vls"]
    if vls_intensity_threshold is not None:
        keep = peak_vls >= float(vls_intensity_threshold)
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )
    summary["after_peak_threshold"] = int(g.size)

    if gmd_min_threshold is not None or gmd_max_threshold is not None:
        keep = np.ones(g.shape, dtype=bool)
        if gmd_min_threshold is not None:
            low = g < float(gmd_min_threshold)
            summary["low_gmd_rejected"] = int(np.sum(low))
            keep &= ~low
        if gmd_max_threshold is not None:
            high = g > float(gmd_max_threshold)
            summary["high_gmd_rejected"] = int(np.sum(high))
            keep &= ~high
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )
    summary["after_gmd_filter"] = int(g.size)

    diagnostics = None
    if reject_double_peaks and g.size:
        xcorr, xsm, noise = preprocess_shots_for_peakfinding(
            a,
            baseline_q=peakfind_baseline_q,
            sg_window=peakfind_sg_window,
            sg_poly=peakfind_sg_poly,
        )
        kwargs = dict(double_peak_kwargs or {})
        diagnostics = detect_double_peaks_pairscan(xcorr, xsm, noise, **kwargs)
        double = diagnostics["is_candidate"]
        keep = ~double
        summary["double_peak_rejected"] = int(np.sum(double))
        g, a, nominal, nominal_index, peak_vls = (
            g[keep],
            a[keep],
            nominal[keep],
            nominal_index[keep],
            peak_vls[keep],
        )

    summary["final_shots"] = int(g.size)
    return {
        "gmd": g,
        "vls": a,
        "nominal_energy": nominal,
        "nominal_index": nominal_index,
        "peak_vls": peak_vls,
        "summary": summary,
        "double_peak_diagnostics": diagnostics,
    }


def _group_intensity_by_x(
    x: np.ndarray,
    spectra: np.ndarray,
    pixel_axis: np.ndarray,
    *,
    pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    if x.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    keys = np.round(x, int(decimals))
    finite = np.isfinite(keys)
    unique = np.unique(keys[finite])
    intensity = np.full(unique.shape, np.nan, dtype=np.float64)
    counts = np.zeros(unique.shape, dtype=np.int64)
    for i, key in enumerate(unique):
        keep = finite & (keys == key)
        counts[i] = int(np.sum(keep))
        intensity[i] = integrated_vls_intensity(
            spectra[keep],
            pixel_axis=pixel_axis,
            pixel_roi=pixel_roi,
            clip_negative=clip_negative,
        )
    return unique.astype(np.float64), intensity, counts


def _absorbance_from_intensities(
    x: np.ndarray,
    trans_intensity: np.ndarray,
    incident_intensity: np.ndarray,
    n_trans: np.ndarray,
    n_incident: np.ndarray,
    *,
    x_label: str,
    metadata: Optional[dict[str, Any]] = None,
) -> AbsorbanceCurve:
    x = np.asarray(x, dtype=np.float64)
    trans = np.asarray(trans_intensity, dtype=np.float64)
    incident = np.asarray(incident_intensity, dtype=np.float64)
    absorbance = np.full(x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(trans) & np.isfinite(incident) & (trans > 0) & (incident > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        absorbance[valid] = -np.log10(trans[valid] / incident[valid])
    return AbsorbanceCurve(
        x=x,
        absorbance=absorbance,
        trans_intensity=trans,
        incident_intensity=incident,
        n_trans=np.asarray(n_trans, dtype=np.int64),
        n_incident=np.asarray(n_incident, dtype=np.int64),
        x_label=x_label,
        metadata=dict(metadata or {}),
    )



def _absorbance_from_binned_intensities(
    x: np.ndarray,
    trans_intensity: np.ndarray,
    incident_intensity: np.ndarray,
    n_trans: np.ndarray,
    n_incident: np.ndarray,
    gmd_edges: Sequence[float],
    *,
    x_label: str,
    metadata: Optional[dict[str, Any]] = None,
) -> GmdBinnedAbsorbanceCurve:
    x = np.asarray(x, dtype=np.float64)
    trans = np.asarray(trans_intensity, dtype=np.float64)
    incident = np.asarray(incident_intensity, dtype=np.float64)
    absorbance = np.full(trans.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(trans) & np.isfinite(incident) & (trans > 0) & (incident > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        absorbance[valid] = -np.log10(trans[valid] / incident[valid])
    return GmdBinnedAbsorbanceCurve(
        x=x,
        absorbance=absorbance,
        trans_intensity=trans,
        incident_intensity=incident,
        n_trans=np.asarray(n_trans, dtype=np.int64),
        n_incident=np.asarray(n_incident, dtype=np.int64),
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        x_label=x_label,
        metadata=dict(metadata or {}),
    )


def _group_intensity_by_x_gmd(
    x: np.ndarray,
    gmd: np.ndarray,
    spectra: np.ndarray,
    pixel_axis: np.ndarray,
    gmd_edges: Sequence[float],
    *,
    pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    gmd = np.asarray(gmd, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    edges = _validated_gmd_edges(gmd_edges)
    n_gmd = edges.size - 1
    if x.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.zeros((0, n_gmd), dtype=np.float64),
            np.zeros((0, n_gmd), dtype=np.int64),
        )

    keys = np.round(x, int(decimals))
    finite_x = np.isfinite(keys)
    x_unique = np.unique(keys[finite_x])
    intensity = np.full((x_unique.size, n_gmd), np.nan, dtype=np.float64)
    counts = np.zeros((x_unique.size, n_gmd), dtype=np.int64)
    gbin, in_gmd = _gmd_bin_indices(gmd, edges)

    for ix, key in enumerate(x_unique):
        same_x = finite_x & (keys == key)
        for ig in range(n_gmd):
            keep = same_x & in_gmd & (gbin == ig)
            counts[ix, ig] = int(np.sum(keep))
            if counts[ix, ig]:
                intensity[ix, ig] = integrated_vls_intensity(
                    spectra[keep],
                    pixel_axis=pixel_axis,
                    pixel_roi=pixel_roi,
                    clip_negative=clip_negative,
                )
    return x_unique.astype(np.float64), intensity, counts


def build_nominal_absorbance_curve(
    trans_run: StaticXASRun,
    ref_run: StaticXASRun,
    *,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    nominal_decimals: int = 6,
) -> AbsorbanceCurve:
    """Build -log10(I_trans/I_incident) against common nominal energies."""

    trans = _filter_flat_shots_for_absorbance(
        trans_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )
    ref = _filter_flat_shots_for_absorbance(
        ref_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )

    tx, ti, tn = _group_intensity_by_x(
        trans["nominal_energy"],
        trans["vls"],
        trans_run.vls_pixels,
        pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
        decimals=nominal_decimals,
    )
    rx, ri, rn = _group_intensity_by_x(
        ref["nominal_energy"],
        ref["vls"],
        ref_run.vls_pixels,
        pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
        decimals=nominal_decimals,
    )

    common = np.intersect1d(tx, rx)
    tpos = np.searchsorted(tx, common)
    rpos = np.searchsorted(rx, common)
    return _absorbance_from_intensities(
        common,
        ti[tpos],
        ri[rpos],
        tn[tpos],
        rn[rpos],
        x_label="nominal photon energy (eV)",
        metadata={
            "mode": "nominal_absorbance",
            "trans_summary": trans["summary"],
            "incident_summary": ref["summary"],
            "intensity_pixel_roi": None if intensity_pixel_roi is None else tuple(intensity_pixel_roi),
            "clip_negative": bool(clip_negative),
            "common_bins": int(common.size),
        },
    )


def build_nominal_gmd_binned_absorbance_curve(
    trans_run: StaticXASRun,
    ref_run: StaticXASRun,
    gmd_edges: Sequence[float],
    *,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    nominal_decimals: int = 6,
) -> GmdBinnedAbsorbanceCurve:
    """Build GMD-binned -log10(I_trans/I_incident) vs common nominal energies."""

    trans = _filter_flat_shots_for_absorbance(
        trans_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )
    ref = _filter_flat_shots_for_absorbance(
        ref_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )

    tx, ti, tn = _group_intensity_by_x_gmd(
        trans["nominal_energy"],
        trans["gmd"],
        trans["vls"],
        trans_run.vls_pixels,
        gmd_edges,
        pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
        decimals=nominal_decimals,
    )
    rx, ri, rn = _group_intensity_by_x_gmd(
        ref["nominal_energy"],
        ref["gmd"],
        ref["vls"],
        ref_run.vls_pixels,
        gmd_edges,
        pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
        decimals=nominal_decimals,
    )

    common = np.intersect1d(tx, rx)
    tpos = np.searchsorted(tx, common)
    rpos = np.searchsorted(rx, common)
    return _absorbance_from_binned_intensities(
        common,
        ti[tpos],
        ri[rpos],
        tn[tpos],
        rn[rpos],
        gmd_edges,
        x_label="nominal photon energy (eV)",
        metadata={
            "mode": "nominal_gmd_binned_absorbance",
            "trans_summary": trans["summary"],
            "incident_summary": ref["summary"],
            "intensity_pixel_roi": None if intensity_pixel_roi is None else tuple(intensity_pixel_roi),
            "clip_negative": bool(clip_negative),
            "common_bins": int(common.size),
        },
    )


def _fixed_abs_pixel_bin_intensities(
    prepared: PreparedShots,
    vls_pixels: np.ndarray,
    *,
    pixel_min: int,
    pixel_max: int,
    pixel_bin_width: int,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(1, int(pixel_bin_width))
    starts = np.arange(int(pixel_min), int(pixel_max) + 1, width, dtype=np.int64)
    stops = np.minimum(starts + width - 1, int(pixel_max))
    centers = starts.astype(np.float64) + 0.5 * (stops - starts).astype(np.float64)
    intensity = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    if prepared.n_shots == 0 or centers.size == 0:
        return centers, intensity, counts

    pixels = np.asarray(vls_pixels, dtype=np.int64)
    abs_proxy = pixels[prepared.proxy_rel]
    in_range = (abs_proxy >= int(pixel_min)) & (abs_proxy <= int(pixel_max))
    if not np.any(in_range):
        return centers, intensity, counts
    groups = ((abs_proxy[in_range] - int(pixel_min)) // width).astype(np.int64)
    spectra = prepared.vls[in_range]
    for group in np.unique(groups):
        keep = groups == group
        counts[group] = int(np.sum(keep))
        intensity[group] = integrated_vls_intensity(
            spectra[keep],
            pixel_axis=vls_pixels,
            pixel_roi=intensity_pixel_roi,
            clip_negative=clip_negative,
        )
    return centers, intensity, counts


def _fixed_abs_pixel_gmd_bin_intensities(
    prepared: PreparedShots,
    vls_pixels: np.ndarray,
    gmd_edges: Sequence[float],
    *,
    pixel_min: int,
    pixel_max: int,
    pixel_bin_width: int,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(1, int(pixel_bin_width))
    starts = np.arange(int(pixel_min), int(pixel_max) + 1, width, dtype=np.int64)
    stops = np.minimum(starts + width - 1, int(pixel_max))
    centers = starts.astype(np.float64) + 0.5 * (stops - starts).astype(np.float64)
    edges = _validated_gmd_edges(gmd_edges)
    n_gmd = edges.size - 1
    intensity = np.full((centers.size, n_gmd), np.nan, dtype=np.float64)
    counts = np.zeros((centers.size, n_gmd), dtype=np.int64)
    if prepared.n_shots == 0 or centers.size == 0:
        return centers, intensity, counts

    pixels = np.asarray(vls_pixels, dtype=np.int64)
    abs_proxy = pixels[prepared.proxy_rel]
    in_pixel = (abs_proxy >= int(pixel_min)) & (abs_proxy <= int(pixel_max))
    gbin, in_gmd = _gmd_bin_indices(prepared.gmd, edges)
    valid = in_pixel & in_gmd
    if not np.any(valid):
        return centers, intensity, counts

    groups = ((abs_proxy[valid] - int(pixel_min)) // width).astype(np.int64)
    gvalid = gbin[valid]
    spectra = prepared.vls[valid]
    for group in np.unique(groups):
        same_group = groups == group
        for ig in np.unique(gvalid[same_group]):
            keep = same_group & (gvalid == ig)
            counts[group, ig] = int(np.sum(keep))
            intensity[group, ig] = integrated_vls_intensity(
                spectra[keep],
                pixel_axis=vls_pixels,
                pixel_roi=intensity_pixel_roi,
                clip_negative=clip_negative,
            )
    return centers, intensity, counts


def build_actual_absorbance_curve(
    trans_run: StaticXASRun,
    ref_run: StaticXASRun,
    config_dict: Mapping[str, Any],
    calibration_dict: Mapping[str, Any],
    e_range: Sequence[float],
    *,
    n_samples: int = 10_000,
    pixel_bin_width: int = 1,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    center_method: str = "com",
    center_half_window: int = 12,
    center_chunk_size: int = 50_000,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    sort: bool = True,
) -> AbsorbanceCurve:
    """Build -log10(I_trans/I_incident) against common actual-energy bins."""

    trans_shots = prepare_proxy_shots(
        trans_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        center_method=center_method,
        center_half_window=center_half_window,
        center_chunk_size=center_chunk_size,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )
    ref_shots = prepare_proxy_shots(
        ref_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        center_method=center_method,
        center_half_window=center_half_window,
        center_chunk_size=center_chunk_size,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )

    trans_pixels = np.asarray(trans_run.vls_pixels, dtype=np.int64)
    ref_pixels = np.asarray(ref_run.vls_pixels, dtype=np.int64)
    pixel_min = int(max(np.nanmin(trans_pixels), np.nanmin(ref_pixels)))
    pixel_max = int(min(np.nanmax(trans_pixels), np.nanmax(ref_pixels)))
    if pixel_min > pixel_max:
        raise ValueError("transmitted and incident VLS pixel axes do not overlap")

    px, ti, tn = _fixed_abs_pixel_bin_intensities(
        trans_shots,
        trans_run.vls_pixels,
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        pixel_bin_width=pixel_bin_width,
        intensity_pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
    )
    rx, ri, rn = _fixed_abs_pixel_bin_intensities(
        ref_shots,
        ref_run.vls_pixels,
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        pixel_bin_width=pixel_bin_width,
        intensity_pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
    )
    if px.shape != rx.shape or not np.allclose(px, rx, equal_nan=True):
        raise RuntimeError("internal pixel-bin mismatch")

    common = (tn > 0) & (rn > 0)
    common_px = px[common]
    if common_px.size:
        energy, _, _ = convert_pixels_to_energy(
            common_px,
            config_dict,
            calibration_dict,
            e_range,
            n_samples=n_samples,
        )
    else:
        energy = np.empty(0, dtype=np.float64)

    curve = _absorbance_from_intensities(
        energy,
        ti[common],
        ri[common],
        tn[common],
        rn[common],
        x_label="converted photon energy (eV)",
        metadata={
            "mode": "actual_absorbance",
            "pixel_bin_width": int(max(1, int(pixel_bin_width))),
            "pixel_bin_centers": common_px,
            "pixel_range": (pixel_min, pixel_max),
            "trans_summary": trans_shots.summary,
            "incident_summary": ref_shots.summary,
            "intensity_pixel_roi": None if intensity_pixel_roi is None else tuple(intensity_pixel_roi),
            "clip_negative": bool(clip_negative),
            "center_method": trans_shots.center_method,
            "common_bins": int(common_px.size),
        },
    )
    if sort and curve.x.size:
        order = np.argsort(curve.x)
        metadata = dict(curve.metadata)
        metadata["pixel_bin_centers"] = np.asarray(metadata["pixel_bin_centers"])[order]
        curve = replace(
            curve,
            x=curve.x[order],
            absorbance=curve.absorbance[order],
            trans_intensity=curve.trans_intensity[order],
            incident_intensity=curve.incident_intensity[order],
            n_trans=curve.n_trans[order],
            n_incident=curve.n_incident[order],
            metadata=metadata,
        )
    return curve


def build_actual_gmd_binned_absorbance_curve(
    trans_run: StaticXASRun,
    ref_run: StaticXASRun,
    gmd_edges: Sequence[float],
    config_dict: Mapping[str, Any],
    calibration_dict: Mapping[str, Any],
    e_range: Sequence[float],
    *,
    n_samples: int = 10_000,
    pixel_bin_width: int = 1,
    vls_intensity_threshold: Optional[float] = 1000.0,
    gmd_min_threshold: Optional[float] = None,
    gmd_max_threshold: Optional[float] = None,
    center_method: str = "com",
    center_half_window: int = 12,
    center_chunk_size: int = 50_000,
    reject_double_peaks: bool = False,
    peakfind_baseline_q: float = 0.15,
    peakfind_sg_window: int = 9,
    peakfind_sg_poly: int = 2,
    double_peak_kwargs: Optional[Mapping[str, Any]] = None,
    intensity_pixel_roi: Optional[Sequence[float]] = None,
    clip_negative: bool = True,
    sort: bool = True,
) -> GmdBinnedAbsorbanceCurve:
    """Build GMD-binned -log10(I_trans/I_incident) vs common actual-energy bins."""

    trans_shots = prepare_proxy_shots(
        trans_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        center_method=center_method,
        center_half_window=center_half_window,
        center_chunk_size=center_chunk_size,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )
    ref_shots = prepare_proxy_shots(
        ref_run,
        vls_intensity_threshold=vls_intensity_threshold,
        gmd_min_threshold=gmd_min_threshold,
        gmd_max_threshold=gmd_max_threshold,
        center_method=center_method,
        center_half_window=center_half_window,
        center_chunk_size=center_chunk_size,
        reject_double_peaks=reject_double_peaks,
        peakfind_baseline_q=peakfind_baseline_q,
        peakfind_sg_window=peakfind_sg_window,
        peakfind_sg_poly=peakfind_sg_poly,
        double_peak_kwargs=double_peak_kwargs,
    )

    trans_pixels = np.asarray(trans_run.vls_pixels, dtype=np.int64)
    ref_pixels = np.asarray(ref_run.vls_pixels, dtype=np.int64)
    pixel_min = int(max(np.nanmin(trans_pixels), np.nanmin(ref_pixels)))
    pixel_max = int(min(np.nanmax(trans_pixels), np.nanmax(ref_pixels)))
    if pixel_min > pixel_max:
        raise ValueError("transmitted and incident VLS pixel axes do not overlap")

    px, ti, tn = _fixed_abs_pixel_gmd_bin_intensities(
        trans_shots,
        trans_run.vls_pixels,
        gmd_edges,
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        pixel_bin_width=pixel_bin_width,
        intensity_pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
    )
    rx, ri, rn = _fixed_abs_pixel_gmd_bin_intensities(
        ref_shots,
        ref_run.vls_pixels,
        gmd_edges,
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        pixel_bin_width=pixel_bin_width,
        intensity_pixel_roi=intensity_pixel_roi,
        clip_negative=clip_negative,
    )
    if px.shape != rx.shape or not np.allclose(px, rx, equal_nan=True):
        raise RuntimeError("internal pixel-bin mismatch")

    common_x = np.any((tn > 0) & (rn > 0), axis=1)
    common_px = px[common_x]
    if common_px.size:
        energy, _, _ = convert_pixels_to_energy(
            common_px,
            config_dict,
            calibration_dict,
            e_range,
            n_samples=n_samples,
        )
    else:
        energy = np.empty(0, dtype=np.float64)

    curve = _absorbance_from_binned_intensities(
        energy,
        ti[common_x],
        ri[common_x],
        tn[common_x],
        rn[common_x],
        gmd_edges,
        x_label="converted photon energy (eV)",
        metadata={
            "mode": "actual_gmd_binned_absorbance",
            "pixel_bin_width": int(max(1, int(pixel_bin_width))),
            "pixel_bin_centers": common_px,
            "pixel_range": (pixel_min, pixel_max),
            "trans_summary": trans_shots.summary,
            "incident_summary": ref_shots.summary,
            "intensity_pixel_roi": None if intensity_pixel_roi is None else tuple(intensity_pixel_roi),
            "clip_negative": bool(clip_negative),
            "center_method": trans_shots.center_method,
            "common_bins": int(common_px.size),
        },
    )
    if sort and curve.x.size:
        order = np.argsort(curve.x)
        metadata = dict(curve.metadata)
        metadata["pixel_bin_centers"] = np.asarray(metadata["pixel_bin_centers"])[order]
        curve = replace(
            curve,
            x=curve.x[order],
            absorbance=curve.absorbance[order],
            trans_intensity=curve.trans_intensity[order],
            incident_intensity=curve.incident_intensity[order],
            n_trans=curve.n_trans[order],
            n_incident=curve.n_incident[order],
            metadata=metadata,
        )
    return curve


def plot_absorbance_curve(
    curve: AbsorbanceCurve,
    *,
    title: str,
    ylim: Optional[Sequence[float]] = None,
    ax=None,
    linewidth: float = 1.8,
    marker: Optional[str] = "o",
):
    """Plot a traditional absorbance curve."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
    else:
        fig = ax.figure
    ax.plot(curve.x, curve.absorbance, lw=linewidth, marker=marker, color="tab:blue")
    ax.set_xlabel(curve.x_label)
    ax.set_ylabel("Absorbance = -log10(I_trans / I_inc)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Calibration quicklook
# ---------------------------------------------------------------------------


def _quadratic_refine_peak(x: np.ndarray, y: np.ndarray, i: int) -> tuple[float, float]:
    if i <= 0 or i >= y.size - 1:
        return float(x[i]), float(y[i])
    xw = x[i - 1 : i + 2].astype(np.float64)
    yw = y[i - 1 : i + 2].astype(np.float64)
    if not np.all(np.isfinite(yw)):
        return float(x[i]), float(y[i])
    a, b, c = np.polyfit(xw, yw, 2)
    if not np.isfinite(a) or a >= 0:
        return float(x[i]), float(y[i])
    xp = -b / (2.0 * a)
    if xp < xw[0] or xp > xw[-1]:
        return float(x[i]), float(y[i])
    return float(xp), float(a * xp**2 + b * xp + c)


def calibration_peak_summary(
    run: StaticXASRun,
    nominal_energy_eV: float,
    *,
    pixel_roi: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Mean-spectrum peak quicklook for a nominal energy section."""

    ie = int(np.argmin(np.abs(run.nominal_energies - float(nominal_energy_eV))))
    n = int(run.n_shots[ie])
    if n <= 0:
        raise RuntimeError(f"no shots stored for energy index {ie}")
    shots = run.vls[ie, :n, :]
    finite = np.all(np.isfinite(shots), axis=1)
    if not np.any(finite):
        raise RuntimeError(f"no finite VLS shots for energy index {ie}")
    mean_vls = np.nanmean(shots[finite], axis=0)
    pixels = run.vls_pixels.astype(np.float64)
    if pixel_roi is None:
        roi = np.ones(pixels.shape, dtype=bool)
    else:
        lo, hi = float(pixel_roi[0]), float(pixel_roi[1])
        roi = (pixels >= lo) & (pixels <= hi)
        if not np.any(roi):
            raise RuntimeError("pixel_roi does not overlap vls_pixels")
    roi_pixels = pixels[roi]
    roi_mean = mean_vls[roi]
    peak_i = int(np.nanargmax(roi_mean))
    peak_pixel, peak_value = _quadratic_refine_peak(roi_pixels, roi_mean, peak_i)
    return {
        "requested_energy_eV": float(nominal_energy_eV),
        "selected_energy_eV": float(run.nominal_energies[ie]),
        "energy_index": ie,
        "shots_used": int(np.sum(finite)),
        "pixel_roi": None if pixel_roi is None else tuple(pixel_roi),
        "pixel_axis": pixels,
        "mean_vls": mean_vls,
        "roi_pixels": roi_pixels,
        "roi_mean_vls": roi_mean,
        "peak_pixel": peak_pixel,
        "peak_value": peak_value,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _colors(n: int):
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def format_shot_summary(prepared: PreparedShots, label: str = "shots") -> str:
    """Compact text summary for notebook logs."""

    s = prepared.summary
    parts = [
        f"{label}: finite={int(s.get('finite_shots', 0))}",
        f"after peak={int(s.get('after_peak_threshold', 0))}",
    ]
    if s.get("gmd_min_threshold") is not None or s.get("gmd_max_threshold") is not None:
        parts.append(f"after GMD range={int(s.get('after_gmd_filter', 0))}")
        if s.get("gmd_min_threshold") is not None:
            parts.append(f"low-GMD removed={int(s.get('low_gmd_rejected', 0))}")
        if s.get("gmd_max_threshold") is not None:
            parts.append(f"high-GMD removed={int(s.get('high_gmd_rejected', 0))}")
    if s.get("reject_double_peaks"):
        parts.append(f"double rejected={int(s.get('double_peak_rejected', 0))}")
        parts.append(f"after double={int(s.get('after_double_peak', 0))}")
    parts.extend([
        f"after center={int(s.get('after_center', 0))}",
        f"final={int(s.get('final_shots', prepared.n_shots))}",
        f"center={s.get('center_method')}",
    ])
    return ", ".join(parts)


def plot_xas_curves(
    binned: BinnedXAS,
    *,
    title: str,
    ylim: Optional[Sequence[float]] = None,
    ax=None,
    linewidth: float = 1.8,
    marker: Optional[str] = None,
):
    """Plot XAS curves by GMD bin."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
    else:
        fig = ax.figure
    n_gmd = binned.xas.shape[1]
    for ig, color in enumerate(_colors(n_gmd)):
        label = _gmd_bin_label(binned.gmd_edges, ig)
        ax.plot(binned.x, binned.xas[:, ig], lw=linewidth, marker=marker, color=color, label=label)
    ax.set_xlabel(binned.x_label)
    ax.set_ylabel("XAS = sum(GMD) / sum(VLS)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(title="GMD bin", fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    return fig, ax




def plot_gmd_binned_absorbance_curves(
    curve: GmdBinnedAbsorbanceCurve,
    *,
    title: str,
    ylim: Optional[Sequence[float]] = None,
    ax=None,
    linewidth: float = 1.8,
    marker: Optional[str] = "o",
):
    """Plot traditional absorbance curves split by GMD bin."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
    else:
        fig = ax.figure
    n_gmd = curve.absorbance.shape[1]
    for ig, color in enumerate(_colors(n_gmd)):
        label = _gmd_bin_label(curve.gmd_edges, ig)
        ax.plot(curve.x, curve.absorbance[:, ig], lw=linewidth, marker=marker, color=color, label=label)
    ax.set_xlabel(curve.x_label)
    ax.set_ylabel("Absorbance = -log10(I_trans / I_inc)")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(title="GMD bin", fontsize=8, title_fontsize=9, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_shot_count_heatmap(
    binned: BinnedXAS,
    *,
    title: str,
    ax=None,
):
    """Plot retained shots per x/GMD bin."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
    else:
        fig = ax.figure
    n_gmd = binned.n_per_bin.shape[1]
    if binned.x.size == 0:
        ax.set_title(title + " (empty)")
        return fig, ax
    im = ax.imshow(
        binned.n_per_bin.T,
        aspect="auto",
        origin="lower",
        extent=[float(np.nanmin(binned.x)), float(np.nanmax(binned.x)), 0, n_gmd],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="shots / bin")
    ax.set_xlabel(binned.x_label)
    ax.set_ylabel("GMD bin")
    ax.set_yticks(np.arange(n_gmd) + 0.5)
    ax.set_yticklabels([_gmd_bin_label(binned.gmd_edges, i).replace(" uJ", "") for i in range(n_gmd)])
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_calibration_peak(summary: Mapping[str, Any], *, ax=None):
    """Plot mean VLS spectrum and selected peak used for calibration checks."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
    else:
        fig = ax.figure
    ax.plot(summary["pixel_axis"], summary["mean_vls"], lw=1.5, color="0.7", label="Mean VLS (full axis)")
    ax.plot(summary["roi_pixels"], summary["roi_mean_vls"], lw=2.0, color="tab:blue", label="Mean VLS (search ROI)")
    ax.axvline(summary["peak_pixel"], color="tab:red", ls="--", lw=1.5, label=f"Peak pixel = {summary['peak_pixel']:.2f}")
    ax.plot([summary["peak_pixel"]], [summary["peak_value"]], "o", color="tab:red", ms=6)
    ax.set_xlabel("Gotthard pixel")
    ax.set_ylabel("Mean VLS intensity (arb.)")
    ax.set_title(f"Calibration quicklook near {summary['selected_energy_eV']:.2f} eV")
    if summary.get("pixel_roi") is not None:
        lo, hi = summary["pixel_roi"]
        ax.set_xlim(float(lo), float(hi))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def _select_gmd_tail_shots(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    side: str,
    vls_intensity_threshold: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Return finite shots in one GMD tail for inspection."""

    flat = flatten_finite_shots(run)
    threshold = float(gmd_threshold)
    if side == "low":
        keep = flat["gmd"] < threshold
    elif side == "high":
        keep = flat["gmd"] > threshold
    else:
        raise ValueError("side must be 'low' or 'high'")
    if vls_intensity_threshold is not None and flat["vls"].size:
        keep &= vls_peak_metrics(flat["vls"])["peak_vls"] >= float(vls_intensity_threshold)
    return {
        "gmd": flat["gmd"][keep],
        "vls": flat["vls"][keep],
        "nominal_energy": flat["nominal_energy"][keep],
        "nominal_index": flat["nominal_index"][keep],
    }


def select_low_gmd_shots(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    vls_intensity_threshold: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Return finite shots below a GMD threshold for inspection."""

    return _select_gmd_tail_shots(
        run,
        gmd_threshold,
        side="low",
        vls_intensity_threshold=vls_intensity_threshold,
    )


def select_high_gmd_shots(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    vls_intensity_threshold: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Return finite shots above a GMD threshold for inspection."""

    return _select_gmd_tail_shots(
        run,
        gmd_threshold,
        side="high",
        vls_intensity_threshold=vls_intensity_threshold,
    )


def _plot_gmd_tail_spectra(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    side: str,
    vls_intensity_threshold: Optional[float] = None,
    n_examples: int = 12,
    random_seed: int = 7,
    ax=None,
):
    selected = _select_gmd_tail_shots(
        run,
        gmd_threshold,
        side=side,
        vls_intensity_threshold=vls_intensity_threshold,
    )
    n_total = selected["gmd"].size
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
    else:
        fig = ax.figure
    label = "low" if side == "low" else "high"
    op = "<" if side == "low" else ">"
    if n_total == 0:
        ax.text(0.5, 0.5, f"No {label}-GMD shots found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax

    rng = np.random.default_rng(random_seed)
    pick = rng.choice(n_total, size=min(int(n_examples), n_total), replace=False)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, pick.size))
    for color, idx in zip(colors, pick):
        ax.plot(
            run.vls_pixels,
            selected["vls"][idx],
            lw=1.0,
            alpha=0.75,
            color=color,
            label=f"GMD={selected['gmd'][idx]:.2g} uJ, E={selected['nominal_energy'][idx]:.1f}",
        )
    ax.set_xlabel("Gotthard pixel")
    ax.set_ylabel("Processed VLS intensity (arb.)")
    ax.set_title(f"Example VLS spectra with GMD {op} {gmd_threshold:g} uJ ({n_total} shots)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_low_gmd_spectra(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    vls_intensity_threshold: Optional[float] = None,
    n_examples: int = 12,
    random_seed: int = 7,
    ax=None,
):
    """Plot example VLS spectra below a low-GMD threshold."""

    return _plot_gmd_tail_spectra(
        run,
        gmd_threshold,
        side="low",
        vls_intensity_threshold=vls_intensity_threshold,
        n_examples=n_examples,
        random_seed=random_seed,
        ax=ax,
    )


def plot_high_gmd_spectra(
    run: StaticXASRun,
    gmd_threshold: float,
    *,
    vls_intensity_threshold: Optional[float] = None,
    n_examples: int = 12,
    random_seed: int = 7,
    ax=None,
):
    """Plot example VLS spectra above a high-GMD threshold."""

    return _plot_gmd_tail_spectra(
        run,
        gmd_threshold,
        side="high",
        vls_intensity_threshold=vls_intensity_threshold,
        n_examples=n_examples,
        random_seed=random_seed,
        ax=ax,
    )


def plot_double_peak_examples(
    prepared: PreparedShots,
    vls_pixels: np.ndarray,
    *,
    n_examples: int = 12,
    random_seed: int = 11,
    sort_by_score: bool = True,
):
    """Plot spectra rejected by the double-peak detector."""

    rejected = prepared.rejected_double_peaks
    if not rejected or rejected["vls"].shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, "No double-peak spectra rejected", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig, ax

    n_total = rejected["vls"].shape[0]
    if sort_by_score and "score" in rejected:
        order = np.argsort(rejected["score"])[::-1]
        pick = order[: min(int(n_examples), n_total)]
    else:
        rng = np.random.default_rng(random_seed)
        pick = rng.choice(n_total, size=min(int(n_examples), n_total), replace=False)

    ncols = min(4, max(1, pick.size))
    nrows = int(np.ceil(pick.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 2.8 * nrows),
        sharex=True,
        squeeze=False,
    )

    for ax, idx in zip(axes.ravel(), pick):
        ax.plot(vls_pixels, rejected["vls"][idx], color="0.25", lw=1.0)
        for key, color in (("peak1_px", "tab:red"), ("peak2_px", "tab:blue")):
            if key in rejected and rejected[key][idx] >= 0:
                rel = int(rejected[key][idx])
                if 0 <= rel < len(vls_pixels):
                    ax.axvline(vls_pixels[rel], color=color, lw=1.0, alpha=0.9)
        score = rejected.get("score", np.full(n_total, np.nan))[idx]
        sep = rejected.get("sep", np.full(n_total, np.nan))[idx]
        ax.set_title(
            f"GMD={rejected['gmd'][idx]:.2g} uJ, E={rejected['nominal_energy'][idx]:.1f}\n"
            f"sep={sep:.0f}px, score={score:.2g}",
            fontsize=8,
        )
        ax.grid(alpha=0.15)
    for ax in axes.ravel()[pick.size:]:
        ax.axis("off")
    fig.suptitle(f"Rejected double-peak VLS spectra ({n_total} total)", fontsize=11)
    fig.tight_layout()
    return fig, axes


def plot_random_shots_with_centers(
    prepared: PreparedShots,
    vls_pixels: np.ndarray,
    *,
    n_random: int = 3,
    random_seed: int = 6,
    ax=None,
):
    """Plot random retained shots with their assigned center pixels."""

    if prepared.n_shots == 0:
        raise RuntimeError("no prepared shots to plot")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 5.2))
    else:
        fig = ax.figure
    rng = np.random.default_rng(random_seed)
    picks = rng.choice(prepared.n_shots, size=min(n_random, prepared.n_shots), replace=False)
    cmap = plt.get_cmap("tab10")
    for j, idx in enumerate(picks):
        color = cmap(j % 10)
        ax.plot(vls_pixels, prepared.vls[idx], lw=1.1, color=color, alpha=0.9, label=f"shot {idx}, GMD={prepared.gmd[idx]:.2g} uJ")
        ax.axvline(prepared.center_pixel[idx], color=color, ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel("VLS pixel number")
    ax.set_ylabel("Processed VLS intensity (arb.)")
    ax.set_title(f"Random retained shots ({prepared.center_method} center)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax
