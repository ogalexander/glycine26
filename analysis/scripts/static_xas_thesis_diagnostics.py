"""
Thesis-style downstream diagnostics for static XAS GMD-bin baselines.

This module keeps notebooks small.  It starts from the processed
``run*_static_xas.h5`` layout and follows the thesis-style transmission
XAS idea:

    OD(E) = log10(I0 / IT) ~= log10(sum(GMD) / sum(integrated VLS))

No values are written back to H5.  Background/intercept corrections are
applied only to temporary arrays during aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

import static_xas_analysis as sxa


DEFAULT_DOUBLE_PEAK_KWARGS = {
    "min_distance_px": 3,
    "max_distance_px": 17,
    "min_prom_sigma": 2.5,
    "min_height_sigma": 2.0,
    "min_rel_peak_height": 0.35,
    "min_second_ratio": 0.05,
    "min_bridge_drop": 0.015,
    "min_split_sigma": 0.75,
}


@dataclass(frozen=True)
class ThesisXASConfig:
    """Configuration for the thesis-style static XAS diagnostics."""

    gmd_bin_mode: str = "quantile"  # "quantile" or "fixed"
    n_gmd_bins: int = 5
    fixed_gmd_edges: Sequence[float] = field(default_factory=lambda: tuple(np.arange(0.0, 11.0, 1.0)))
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    vls_intensity_threshold: Optional[float] = 1000.0
    reject_double_peaks: bool = False
    double_peak_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_DOUBLE_PEAK_KWARGS))
    center_method: str = "com"
    center_half_window: int = 5
    center_chunk_size: int = 50_000
    edge_pixel_count: int = 20
    selected_energies_ev: Sequence[float] = (273.5, 274.0, 275.0, 275.6, 277.1, 283.0, 284.0)
    selected_max_shots_per_energy: int = 20_000
    chunk_size: int = 20_000
    random_seed: int = 7
    gmd_hist_bins: int = 120
    min_shots_for_interpretation: int = 200


@dataclass(frozen=True)
class LinearFitSummary:
    """Per-energy fit of integrated VLS versus GMD."""

    intercept: np.ndarray
    slope: np.ndarray
    origin_slope: np.ndarray
    r: np.ndarray
    r2: np.ndarray
    n: np.ndarray


@dataclass
class ThesisDiagnostics:
    """Aggregated state used by notebook plots."""

    path: Path
    config: ThesisXASConfig
    energies: np.ndarray
    pixels: np.ndarray
    gmd_edges: np.ndarray
    gmd_centers: np.ndarray
    gmd_labels: list[str]
    filter_counts: dict[str, np.ndarray]
    accepted_gmd: np.ndarray
    fits: LinearFitSummary
    counts: np.ndarray
    gmd_sum: np.ndarray
    trans_raw_sum: np.ndarray
    trans_positive_sum: np.ndarray
    trans_corrected_sum: np.ndarray
    nonpositive_corrected_counts: np.ndarray
    spec_raw_sum: np.ndarray
    spec_corrected_sum: np.ndarray
    stats_sum: dict[str, np.ndarray]
    stats_sumsq: dict[str, np.ndarray]
    covariance_sums: dict[str, np.ndarray]
    selected_samples: dict[int, dict[str, np.ndarray]]
    double_peak_summary: Optional[dict[str, np.ndarray]] = None

    @property
    def mean_gmd(self) -> np.ndarray:
        return _safe_divide(self.gmd_sum, self.counts)

    @property
    def mean_trans_raw(self) -> np.ndarray:
        return _safe_divide(self.trans_raw_sum, self.counts)

    @property
    def mean_trans_positive(self) -> np.ndarray:
        return _safe_divide(self.trans_positive_sum, self.counts)

    @property
    def mean_trans_corrected(self) -> np.ndarray:
        return _safe_divide(self.trans_corrected_sum, self.counts)

    @property
    def od_raw(self) -> np.ndarray:
        return _log10_ratio(self.gmd_sum, self.trans_raw_sum)

    @property
    def od_corrected(self) -> np.ndarray:
        return _log10_ratio(self.gmd_sum, self.trans_corrected_sum)

    @property
    def simple_ratio_raw(self) -> np.ndarray:
        return _safe_divide(self.gmd_sum, self.trans_raw_sum)

    @property
    def total_od_raw(self) -> np.ndarray:
        return _log10_ratio(np.nansum(self.gmd_sum, axis=1), np.nansum(self.trans_raw_sum, axis=1))

    @property
    def total_od_corrected(self) -> np.ndarray:
        return _log10_ratio(np.nansum(self.gmd_sum, axis=1), np.nansum(self.trans_corrected_sum, axis=1))

    @property
    def total_simple_ratio_raw(self) -> np.ndarray:
        return _safe_divide(np.nansum(self.gmd_sum, axis=1), np.nansum(self.trans_raw_sum, axis=1))

    @property
    def mean_spec_raw(self) -> np.ndarray:
        return _safe_divide_3d(self.spec_raw_sum, self.counts)

    @property
    def mean_spec_corrected(self) -> np.ndarray:
        return _safe_divide_3d(self.spec_corrected_sum, self.counts)

    def stat_mean(self, name: str) -> np.ndarray:
        return _safe_divide(self.stats_sum[name], self.counts)

    def stat_sem(self, name: str) -> np.ndarray:
        mean = self.stat_mean(name)
        mean_sq = _safe_divide(self.stats_sumsq[name], self.counts)
        var = np.maximum(mean_sq - mean**2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(var / np.where(self.counts > 0, self.counts, np.nan))

    def od_sem(self, corrected: bool) -> np.ndarray:
        prefix = "corrected" if corrected else "raw"
        return _delta_log_ratio_sem(
            self.covariance_sums["g"],
            self.covariance_sums[prefix],
            self.covariance_sums["g2"],
            self.covariance_sums[f"{prefix}2"],
            self.covariance_sums[f"g_{prefix}"],
            self.counts,
        )


def compute_thesis_diagnostics(path: str | Path, config: ThesisXASConfig) -> ThesisDiagnostics:
    """Compute all notebook-facing thesis-style static XAS diagnostics."""

    path = Path(path)
    info = read_h5_info(path)
    energies = info["nominal_energies"]
    pixels = info["vls_pixels"]
    selected_indices = nearest_indices(energies, config.selected_energies_ev)

    first = _first_pass(path, config, energies, pixels)
    gmd_edges = choose_gmd_edges(first["accepted_gmd"], config)
    gmd_centers = 0.5 * (gmd_edges[:-1] + gmd_edges[1:])
    gmd_labels = format_gmd_labels(gmd_edges, first["accepted_gmd"])
    fits = _linear_fit_from_accumulators(first["fit"])

    second = _second_pass(path, config, energies, pixels, gmd_edges, selected_indices, fits)
    return ThesisDiagnostics(
        path=path,
        config=config,
        energies=energies,
        pixels=pixels,
        gmd_edges=gmd_edges,
        gmd_centers=gmd_centers,
        gmd_labels=gmd_labels,
        filter_counts=first["filter_counts"],
        accepted_gmd=first["accepted_gmd"],
        fits=fits,
        counts=second["counts"],
        gmd_sum=second["gmd_sum"],
        trans_raw_sum=second["trans_raw_sum"],
        trans_positive_sum=second["trans_positive_sum"],
        trans_corrected_sum=second["trans_corrected_sum"],
        nonpositive_corrected_counts=second["nonpositive_corrected_counts"],
        spec_raw_sum=second["spec_raw_sum"],
        spec_corrected_sum=second["spec_corrected_sum"],
        stats_sum=second["stats_sum"],
        stats_sumsq=second["stats_sumsq"],
        covariance_sums=second["covariance_sums"],
        selected_samples=second["selected_samples"],
    )


def read_h5_info(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {
            "nominal_energies": h5["nominal_energies"][...].astype(np.float64),
            "n_shots": h5["n_shots"][...].astype(np.int64),
            "vls_pixels": h5["vls_pixels"][...].astype(np.float64),
        }


def nearest_indices(values: np.ndarray, targets: Sequence[float]) -> list[int]:
    values = np.asarray(values, dtype=np.float64)
    return sorted({int(np.nanargmin(np.abs(values - float(target)))) for target in targets})


def nearest_energy_index(energies: np.ndarray, target_ev: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(energies, dtype=np.float64) - float(target_ev))))


def choose_gmd_edges(accepted_gmd: np.ndarray, config: ThesisXASConfig) -> np.ndarray:
    """Choose thesis-style quantile bins or fixed bins."""

    if config.gmd_bin_mode == "fixed":
        edges = np.asarray(config.fixed_gmd_edges, dtype=np.float64)
    elif config.gmd_bin_mode == "quantile":
        vals = np.asarray(accepted_gmd, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            edges = np.asarray(config.fixed_gmd_edges, dtype=np.float64)
        else:
            q = np.linspace(0.0, 1.0, int(config.n_gmd_bins) + 1)
            edges = np.quantile(vals, q)
            edges[0] = np.nanmin(vals)
            edges[-1] = np.nanmax(vals)
            edges = np.unique(edges)
            if edges.size < 3:
                edges = np.linspace(np.nanmin(vals), np.nanmax(vals), int(config.n_gmd_bins) + 1)
    else:
        raise ValueError("gmd_bin_mode must be 'quantile' or 'fixed'")

    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError(f"Invalid GMD edges: {edges}")
    return edges.astype(np.float64)


def format_gmd_labels(edges: np.ndarray, accepted_gmd: Optional[np.ndarray] = None) -> list[str]:
    labels = []
    for ig, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        close = "]" if ig == len(edges) - 2 else ")"
        label = f"{lo:.2g}-{hi:.2g}{close} uJ"
        if accepted_gmd is not None and accepted_gmd.size:
            sel = (accepted_gmd >= lo) & (accepted_gmd < hi)
            if ig == len(edges) - 2:
                sel = (accepted_gmd >= lo) & (accepted_gmd <= hi)
            if np.any(sel):
                label = f"mean {np.nanmean(accepted_gmd[sel]):.2g} uJ"
        labels.append(label)
    return labels


def edge_mask(n_pixels: int, edge_count: int) -> np.ndarray:
    n_edge = max(1, min(int(edge_count), int(n_pixels) // 2))
    mask = np.zeros(int(n_pixels), dtype=bool)
    mask[:n_edge] = True
    mask[-n_edge:] = True
    return mask


def integrated_vls(vls: np.ndarray, *, clip_negative: bool = False) -> np.ndarray:
    y = np.asarray(vls, dtype=np.float64)
    if clip_negative:
        y = np.maximum(y, 0.0)
    return np.nansum(y, axis=1)


def spectral_moments(vls: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.maximum(np.nan_to_num(vls, nan=0.0), 0.0)
    weight = np.sum(y, axis=1)
    centroid = np.full(y.shape[0], np.nan, dtype=np.float64)
    width = np.full(y.shape[0], np.nan, dtype=np.float64)
    ok = weight > 0
    if np.any(ok):
        pix = np.asarray(pixels, dtype=np.float64)
        centroid[ok] = y[ok] @ pix / weight[ok]
        second = y[ok] @ (pix**2) / weight[ok]
        width[ok] = np.sqrt(np.maximum(second - centroid[ok] ** 2, 0.0))
    return centroid, width, 2.3548 * width


def quality_mask(vls: np.ndarray, gmd: np.ndarray, config: ThesisXASConfig) -> tuple[np.ndarray, dict[str, int]]:
    keep = np.isfinite(gmd) & np.all(np.isfinite(vls), axis=1)
    summary = {
        "input": int(gmd.size),
        "finite": int(np.sum(keep)),
        "peak": 0,
        "gmd": 0,
        "double_peak": 0,
        "kept": 0,
    }

    if config.vls_intensity_threshold is not None:
        before = keep.copy()
        keep &= np.nanmax(vls, axis=1) >= float(config.vls_intensity_threshold)
        summary["peak"] = int(np.sum(before) - np.sum(keep))

    before_gmd = keep.copy()
    if config.gmd_min_threshold is not None:
        keep &= gmd >= float(config.gmd_min_threshold)
    if config.gmd_max_threshold is not None:
        keep &= gmd <= float(config.gmd_max_threshold)
    summary["gmd"] = int(np.sum(before_gmd) - np.sum(keep))

    if config.reject_double_peaks and np.any(keep):
        idx = np.where(keep)[0]
        corrected, smoothed, noise = sxa.preprocess_shots_for_peakfinding(vls[idx])
        diag = sxa.detect_double_peaks_pairscan(corrected, smoothed, noise, **config.double_peak_kwargs)
        bad = np.asarray(diag["is_candidate"], dtype=bool)
        summary["double_peak"] = int(np.sum(bad))
        keep[idx[bad]] = False

    summary["kept"] = int(np.sum(keep))
    return keep, summary


def iter_energy_chunks(path: Path, config: ThesisXASConfig, energy_indices: Optional[Iterable[int]] = None):
    with h5py.File(path, "r") as h5:
        energies = h5["nominal_energies"][...]
        n_shots = h5["n_shots"][...].astype(np.int64)
        if energy_indices is None:
            indices = range(energies.size)
        else:
            indices = list(dict.fromkeys(int(i) for i in energy_indices))
        for ie in indices:
            n = int(n_shots[ie])
            for start in range(0, n, int(config.chunk_size)):
                stop = min(n, start + int(config.chunk_size))
                yield ie, h5["vls"][ie, start:stop, :], sxa.read_hall_gmd_slice(h5, ie, start, stop)


def _first_pass(path: Path, config: ThesisXASConfig, energies: np.ndarray, pixels: np.ndarray) -> dict:
    n_e = energies.size
    filter_counts = {
        "recorded": np.zeros(n_e, dtype=np.int64),
        "finite": np.zeros(n_e, dtype=np.int64),
        "peak_rejected": np.zeros(n_e, dtype=np.int64),
        "gmd_rejected": np.zeros(n_e, dtype=np.int64),
        "double_peak_rejected": np.zeros(n_e, dtype=np.int64),
        "kept": np.zeros(n_e, dtype=np.int64),
    }
    fit = {name: np.zeros(n_e, dtype=np.float64) for name in ["n", "sx", "sy", "sxx", "syy", "sxy"]}
    accepted_gmd_parts: list[np.ndarray] = []

    for ie, vls, gmd in iter_energy_chunks(path, config):
        keep, summary = quality_mask(vls, gmd, config)
        filter_counts["recorded"][ie] += summary["input"]
        filter_counts["finite"][ie] += summary["finite"]
        filter_counts["peak_rejected"][ie] += summary["peak"]
        filter_counts["gmd_rejected"][ie] += summary["gmd"]
        filter_counts["double_peak_rejected"][ie] += summary["double_peak"]
        filter_counts["kept"][ie] += summary["kept"]
        if not np.any(keep):
            continue

        g = np.asarray(gmd[keep], dtype=np.float64)
        t = integrated_vls(vls[keep], clip_negative=False)
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


def _linear_fit_from_accumulators(acc: dict[str, np.ndarray]) -> LinearFitSummary:
    n = acc["n"].astype(np.float64)
    sx = acc["sx"]
    sy = acc["sy"]
    sxx = acc["sxx"]
    syy = acc["syy"]
    sxy = acc["sxy"]
    denom = n * sxx - sx * sx

    intercept = np.full(n.shape, np.nan, dtype=np.float64)
    slope = np.full(n.shape, np.nan, dtype=np.float64)
    origin_slope = np.full(n.shape, np.nan, dtype=np.float64)
    r = np.full(n.shape, np.nan, dtype=np.float64)
    r2 = np.full(n.shape, np.nan, dtype=np.float64)

    valid = (n >= 2) & (denom != 0)
    slope[valid] = (n[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / n[valid]

    valid_origin = (n >= 1) & (sxx != 0)
    origin_slope[valid_origin] = sxy[valid_origin] / sxx[valid_origin]

    cov = n * sxy - sx * sy
    var_x = n * sxx - sx * sx
    var_y = n * syy - sy * sy
    valid_r = (n >= 2) & (var_x > 0) & (var_y > 0)
    r[valid_r] = cov[valid_r] / np.sqrt(var_x[valid_r] * var_y[valid_r])
    r2[valid_r] = r[valid_r] ** 2
    return LinearFitSummary(intercept=intercept, slope=slope, origin_slope=origin_slope, r=r, r2=r2, n=n.astype(np.int64))


def _second_pass(
    path: Path,
    config: ThesisXASConfig,
    energies: np.ndarray,
    pixels: np.ndarray,
    gmd_edges: np.ndarray,
    selected_indices: Sequence[int],
    fits: LinearFitSummary,
) -> dict:
    n_e = energies.size
    n_g = gmd_edges.size - 1
    n_pix = pixels.size
    counts = np.zeros((n_e, n_g), dtype=np.int64)
    gmd_sum = np.zeros((n_e, n_g), dtype=np.float64)
    trans_raw_sum = np.zeros((n_e, n_g), dtype=np.float64)
    trans_positive_sum = np.zeros((n_e, n_g), dtype=np.float64)
    trans_corrected_sum = np.zeros((n_e, n_g), dtype=np.float64)
    nonpositive_corrected_counts = np.zeros((n_e, n_g), dtype=np.int64)
    spec_raw_sum = np.zeros((n_e, n_g, n_pix), dtype=np.float64)
    spec_corrected_sum = np.zeros((n_e, n_g, n_pix), dtype=np.float64)

    stat_names = ["edge_baseline", "negative_fraction", "peak_height", "centroid", "width", "fwhm", "raw_sum", "positive_sum", "corrected_sum"]
    stats_sum = {name: np.zeros((n_e, n_g), dtype=np.float64) for name in stat_names}
    stats_sumsq = {name: np.zeros((n_e, n_g), dtype=np.float64) for name in stat_names}
    covariance_sums = {
        name: np.zeros((n_e, n_g), dtype=np.float64)
        for name in ["g", "raw", "corrected", "g2", "raw2", "corrected2", "g_raw", "g_corrected"]
    }
    selected_set = set(int(i) for i in selected_indices)
    selected_samples: dict[int, dict[str, list[np.ndarray]]] = {
        int(ie): {key: [] for key in ["gmd", "vls_raw", "vls_corrected", "trans_raw", "trans_corrected", "centroid", "width", "fwhm", "peak_height", "edge_baseline", "negative_fraction"]}
        for ie in selected_set
    }
    selected_counts = {int(ie): 0 for ie in selected_set}

    edge = edge_mask(n_pix, config.edge_pixel_count)

    for ie, vls, gmd in iter_energy_chunks(path, config):
        keep, _ = quality_mask(vls, gmd, config)
        if not np.any(keep):
            continue
        vls = np.asarray(vls[keep], dtype=np.float64)
        gmd = np.asarray(gmd[keep], dtype=np.float64)
        gbin, in_range = sxa._gmd_bin_indices(gmd, gmd_edges)
        if not np.any(in_range):
            continue

        intercept = fits.intercept[ie]
        scalar_offset = 0.0 if not np.isfinite(intercept) else float(intercept) / n_pix
        corrected_vls = vls - scalar_offset
        trans_raw = integrated_vls(vls, clip_negative=False)
        trans_positive = integrated_vls(vls, clip_negative=True)
        trans_corrected = trans_raw - (0.0 if not np.isfinite(intercept) else float(intercept))
        centroid, width, fwhm = spectral_moments(vls, pixels)
        edge_baseline = np.nanmedian(vls[:, edge], axis=1)
        negative_fraction = np.nanmean(vls < 0.0, axis=1)
        peak_height = np.nanmax(vls, axis=1)
        metrics = {
            "edge_baseline": edge_baseline,
            "negative_fraction": negative_fraction,
            "peak_height": peak_height,
            "centroid": centroid,
            "width": width,
            "fwhm": fwhm,
            "raw_sum": trans_raw,
            "positive_sum": trans_positive,
            "corrected_sum": trans_corrected,
        }

        for ig in np.unique(gbin[in_range]):
            sel = in_range & (gbin == ig)
            if not np.any(sel):
                continue
            n_sel = int(np.sum(sel))
            counts[ie, ig] += n_sel
            gmd_sum[ie, ig] += np.nansum(gmd[sel])
            trans_raw_sum[ie, ig] += np.nansum(trans_raw[sel])
            trans_positive_sum[ie, ig] += np.nansum(trans_positive[sel])
            trans_corrected_sum[ie, ig] += np.nansum(trans_corrected[sel])
            nonpositive_corrected_counts[ie, ig] += int(np.sum(trans_corrected[sel] <= 0.0))
            spec_raw_sum[ie, ig] += np.nansum(vls[sel], axis=0)
            spec_corrected_sum[ie, ig] += np.nansum(corrected_vls[sel], axis=0)

            g = gmd[sel]
            raw = trans_raw[sel]
            corr = trans_corrected[sel]
            covariance_sums["g"][ie, ig] += np.nansum(g)
            covariance_sums["raw"][ie, ig] += np.nansum(raw)
            covariance_sums["corrected"][ie, ig] += np.nansum(corr)
            covariance_sums["g2"][ie, ig] += np.nansum(g * g)
            covariance_sums["raw2"][ie, ig] += np.nansum(raw * raw)
            covariance_sums["corrected2"][ie, ig] += np.nansum(corr * corr)
            covariance_sums["g_raw"][ie, ig] += np.nansum(g * raw)
            covariance_sums["g_corrected"][ie, ig] += np.nansum(g * corr)

            for name, values in metrics.items():
                vals = np.asarray(values[sel], dtype=np.float64)
                stats_sum[name][ie, ig] += np.nansum(vals)
                stats_sumsq[name][ie, ig] += np.nansum(vals * vals)

        if ie in selected_set and selected_counts[ie] < config.selected_max_shots_per_energy:
            remaining = int(config.selected_max_shots_per_energy - selected_counts[ie])
            take_idx = np.where(in_range)[0][:remaining]
            if take_idx.size:
                selected_counts[ie] += int(take_idx.size)
                store = selected_samples[ie]
                store["gmd"].append(gmd[take_idx])
                store["vls_raw"].append(vls[take_idx])
                store["vls_corrected"].append(corrected_vls[take_idx])
                store["trans_raw"].append(trans_raw[take_idx])
                store["trans_corrected"].append(trans_corrected[take_idx])
                for name in ["centroid", "width", "fwhm", "peak_height", "edge_baseline", "negative_fraction"]:
                    store[name].append(metrics[name][take_idx])

    merged_samples: dict[int, dict[str, np.ndarray]] = {}
    for ie, fields in selected_samples.items():
        merged_samples[ie] = {}
        for key, pieces in fields.items():
            if pieces:
                merged_samples[ie][key] = np.concatenate(pieces, axis=0)
            elif key.startswith("vls"):
                merged_samples[ie][key] = np.empty((0, n_pix), dtype=np.float64)
            else:
                merged_samples[ie][key] = np.empty(0, dtype=np.float64)

    return {
        "counts": counts,
        "gmd_sum": gmd_sum,
        "trans_raw_sum": trans_raw_sum,
        "trans_positive_sum": trans_positive_sum,
        "trans_corrected_sum": trans_corrected_sum,
        "nonpositive_corrected_counts": nonpositive_corrected_counts,
        "spec_raw_sum": spec_raw_sum,
        "spec_corrected_sum": spec_corrected_sum,
        "stats_sum": stats_sum,
        "stats_sumsq": stats_sumsq,
        "covariance_sums": covariance_sums,
        "selected_samples": merged_samples,
    }


def compute_double_peak_summary(
    path: str | Path,
    config: ThesisXASConfig,
    gmd_edges: Optional[np.ndarray] = None,
    *,
    max_shots_per_energy: Optional[int] = 20_000,
) -> dict[str, np.ndarray]:
    """Summarize VLS multi-peak proxy diagnostics by nominal energy and GMD bin."""

    path = Path(path)
    info = read_h5_info(path)
    energies = info["nominal_energies"]
    if gmd_edges is None:
        gmd_edges = np.asarray(config.fixed_gmd_edges, dtype=np.float64)
    n_e = energies.size
    n_g = gmd_edges.size - 1
    total = np.zeros((n_e, n_g), dtype=np.int64)
    rejected = np.zeros((n_e, n_g), dtype=np.int64)
    sep_sum = np.zeros((n_e, n_g), dtype=np.float64)
    ratio_sum = np.zeros((n_e, n_g), dtype=np.float64)

    scan_config = ThesisXASConfig(
        **{
            **config.__dict__,
            "reject_double_peaks": False,
            "chunk_size": int(config.chunk_size),
        }
    )

    for ie, vls, gmd in iter_energy_chunks(path, scan_config):
        if max_shots_per_energy is not None:
            already = int(np.sum(total[ie]))
            if already >= int(max_shots_per_energy):
                continue
            keep_n = max(0, int(max_shots_per_energy) - already)
            vls = vls[:keep_n]
            gmd = gmd[:keep_n]
        keep, _ = quality_mask(vls, gmd, scan_config)
        if not np.any(keep):
            continue
        vv = np.asarray(vls[keep], dtype=np.float64)
        gg = np.asarray(gmd[keep], dtype=np.float64)
        corrected, smoothed, noise = sxa.preprocess_shots_for_peakfinding(vv)
        diag = sxa.detect_double_peaks_pairscan(corrected, smoothed, noise, **config.double_peak_kwargs)
        bad = np.asarray(diag["is_candidate"], dtype=bool)
        sep = np.asarray(diag["sep"], dtype=np.float64)
        ratio = np.asarray(diag["second_ratio"], dtype=np.float64)
        gbin, in_range = sxa._gmd_bin_indices(gg, gmd_edges)
        for ig in np.unique(gbin[in_range]):
            sel = in_range & (gbin == ig)
            bad_sel = sel & bad
            total[ie, ig] += int(np.sum(sel))
            rejected[ie, ig] += int(np.sum(bad_sel))
            sep_sum[ie, ig] += np.nansum(sep[bad_sel])
            ratio_sum[ie, ig] += np.nansum(ratio[bad_sel])

    return {
        "energies": energies,
        "total": total,
        "rejected": rejected,
        "fraction": _safe_divide(rejected, total),
        "mean_separation_px": _safe_divide(sep_sum, rejected),
        "mean_second_main_ratio": _safe_divide(ratio_sum, rejected),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def colors_for_bins(n_bins: int):
    return plt.cm.inferno(np.linspace(0.08, 0.92, int(n_bins)))


def apply_limits(ax, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_filter_audit(diag: ThesisDiagnostics, *, xlim=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False, constrained_layout=True)
    e = diag.energies
    fc = diag.filter_counts
    axes[0].plot(e, fc["recorded"], label="recorded")
    axes[0].plot(e, fc["finite"], label="finite")
    axes[0].plot(e, fc["kept"], label="accepted")
    axes[0].set_ylabel("shots")
    axes[0].set_title("Shot filtering audit")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    apply_limits(axes[0], xlim=xlim)

    axes[1].plot(e, fc["peak_rejected"], label="peak threshold")
    axes[1].plot(e, fc["gmd_rejected"], label="GMD threshold")
    axes[1].plot(e, fc["double_peak_rejected"], label="double peak")
    axes[1].set_ylabel("rejected shots")
    axes[1].set_xlabel("nominal photon energy / eV")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    apply_limits(axes[1], xlim=xlim)

    axes[2].hist(diag.accepted_gmd[np.isfinite(diag.accepted_gmd)], bins=diag.config.gmd_hist_bins, color="0.25")
    for edge in diag.gmd_edges:
        axes[2].axvline(edge, color="tab:red", lw=0.8, alpha=0.7)
    axes[2].set_xlabel("accepted GMD / uJ")
    axes[2].set_ylabel("shots")
    axes[2].set_title("Accepted GMD distribution and active GMD bin edges")
    axes[2].grid(alpha=0.2)
    return fig, axes


def plot_gmd_vls_correlation(diag: ThesisDiagnostics, energy_ev: float, *, corrected: bool = False, xlim=None, ylim=None):
    ie = nearest_energy_index(diag.energies, energy_ev)
    sample = diag.selected_samples.get(ie)
    if sample is None or sample["gmd"].size == 0:
        raise ValueError(f"No selected-shot sample stored for nominal energy near {energy_ev} eV")
    x = sample["gmd"]
    y = sample["trans_corrected"] if corrected else sample["trans_raw"]
    fit = diag.fits

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    hb = ax.hexbin(x, y, gridsize=80, mincnt=1, bins="log", cmap="magma")
    fig.colorbar(hb, ax=ax, label="log10 shot count")
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    if corrected:
        yy = fit.slope[ie] * xx
        title_suffix = "intercept-corrected"
        line_label = "intercept-corrected fit: b * GMD"
        ylabel = "integrated VLS - intercept"
    else:
        yy = fit.intercept[ie] + fit.slope[ie] * xx
        title_suffix = "raw"
        line_label = "raw free-intercept fit: a + b * GMD"
        ylabel = "integrated VLS"
    ax.plot(xx, yy, color="cyan", lw=1.5, label=line_label)
    if not corrected:
        ax.plot(
            xx,
            fit.origin_slope[ie] * xx,
            color="white",
            lw=1.0,
            ls="--",
            label="raw through-origin fit",
        )
    ax.scatter([0], [0], s=18, color="lime", label="origin")
    ax.set_xlabel("GMD / uJ")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"GMD vs transmitted VLS, nominal {diag.energies[ie]:.2f} eV ({title_suffix})\n"
        f"a={fit.intercept[ie]:.3g}, b={fit.slope[ie]:.3g}, R2={fit.r2[ie]:.3f}"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    apply_limits(ax, xlim=xlim, ylim=ylim)
    return fig, ax


def plot_fit_summary(diag: ThesisDiagnostics, *, xlim=None, intercept_ylim=None, slope_ylim=None, r2_ylim=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    e = diag.energies
    axes[0].plot(e, diag.fits.intercept, "o-", ms=3)
    axes[0].axhline(0.0, color="0.25", lw=0.8)
    axes[0].set_ylabel("fit intercept")
    axes[0].set_title("Per-energy fit of integrated VLS = a(E) + b(E) * GMD")
    axes[0].grid(alpha=0.25)

    axes[1].plot(e, diag.fits.slope, "o-", ms=3, label="raw free-intercept slope")
    axes[1].plot(e, diag.fits.origin_slope, "s--", ms=3, label="raw through-origin slope")
    axes[1].set_ylabel("slope")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    axes[2].plot(e, diag.fits.r2, "o-", ms=3)
    axes[2].set_ylabel("R2")
    axes[2].set_xlabel("nominal photon energy / eV")
    axes[2].grid(alpha=0.25)

    apply_limits(axes[0], xlim=xlim, ylim=intercept_ylim)
    apply_limits(axes[1], xlim=xlim, ylim=slope_ylim)
    apply_limits(axes[2], xlim=xlim, ylim=r2_ylim)
    return fig, axes


def plot_total_od(diag: ThesisDiagnostics, *, xlim=None, od_ylim=None, ratio_ylim=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(diag.energies, diag.total_od_raw, "o-", ms=3, label="raw thesis-style OD")
    axes[0].plot(diag.energies, diag.total_od_corrected, "s--", ms=3, label="intercept-corrected OD")
    axes[0].set_ylabel("log10(sum GMD / sum VLS)")
    axes[0].set_title("Non-GMD-resolved static XAS")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(diag.energies, diag.total_simple_ratio_raw, "o-", ms=3, color="tab:gray")
    axes[1].set_ylabel("simple ratio")
    axes[1].set_xlabel("nominal photon energy / eV")
    axes[1].grid(alpha=0.25)
    apply_limits(axes[0], xlim=xlim, ylim=od_ylim)
    apply_limits(axes[1], xlim=xlim, ylim=ratio_ylim)
    return fig, axes


def plot_gmd_binned_od(
    diag: ThesisDiagnostics,
    *,
    corrected: bool = False,
    xlim=None,
    ylim=None,
    show_errorbars: bool = True,
):
    y = diag.od_corrected if corrected else diag.od_raw
    sem = diag.od_sem(corrected=corrected)
    title_suffix = "intercept-corrected" if corrected else "raw"
    colors = colors_for_bins(y.shape[1])

    fig, axes = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True, constrained_layout=True)
    for ig, color in enumerate(colors):
        label = diag.gmd_labels[ig]
        if show_errorbars:
            axes[0].errorbar(diag.energies, y[:, ig], yerr=sem[:, ig], marker="o", ms=3, lw=1.0, color=color, label=label)
        else:
            axes[0].plot(diag.energies, y[:, ig], marker="o", ms=3, lw=1.0, color=color, label=label)
        axes[1].plot(diag.energies, diag.counts[:, ig], marker=".", lw=1.0, color=color)
        axes[2].plot(diag.energies, diag.mean_gmd[:, ig], marker=".", lw=1.0, color=color)
        trans = diag.mean_trans_corrected if corrected else diag.mean_trans_raw
        axes[3].plot(diag.energies, trans[:, ig], marker=".", lw=1.0, color=color)

    axes[0].set_ylabel("OD")
    axes[0].set_title(f"GMD-binned thesis-style OD ({title_suffix})")
    axes[0].legend(title="GMD bin", frameon=False, fontsize=8, ncol=2)
    axes[1].set_ylabel("shots")
    axes[1].set_title("Shot counts")
    axes[2].set_ylabel("mean GMD / uJ")
    axes[2].set_title("Mean GMD")
    axes[3].set_ylabel("mean VLS sum")
    axes[3].set_title("Mean transmitted VLS")
    axes[3].set_xlabel("nominal photon energy / eV")
    for ax in axes:
        ax.grid(alpha=0.25)
        apply_limits(ax, xlim=xlim)
    apply_limits(axes[0], xlim=xlim, ylim=ylim)
    return fig, axes


def plot_correction_effect(diag: ThesisDiagnostics, *, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    colors = colors_for_bins(diag.counts.shape[1])
    delta = diag.od_corrected - diag.od_raw
    for ig, color in enumerate(colors):
        ax.plot(diag.energies, delta[:, ig], marker="o", ms=3, lw=1.0, color=color, label=diag.gmd_labels[ig])
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("nominal photon energy / eV")
    ax.set_ylabel("corrected OD - raw OD")
    ax.set_title("Effect of thesis-style intercept correction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    apply_limits(ax, xlim=xlim, ylim=ylim)
    return fig, ax


def plot_vls_stats_at_energy(diag: ThesisDiagnostics, energy_ev: float, *, corrected: bool = False, pixel_xlim=None, spectrum_ylim=None):
    ie = nearest_energy_index(diag.energies, energy_ev)
    spectra = diag.mean_spec_corrected[ie] if corrected else diag.mean_spec_raw[ie]
    sample = diag.selected_samples.get(ie, {})
    sample_spectra = sample.get("vls_corrected" if corrected else "vls_raw", np.empty((0, diag.pixels.size)))
    sample_gmd = sample.get("gmd", np.empty(0))
    gbin, in_range = sxa._gmd_bin_indices(sample_gmd, diag.gmd_edges) if sample_gmd.size else (np.empty(0, dtype=int), np.empty(0, dtype=bool))
    colors = colors_for_bins(diag.counts.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ig, color in enumerate(colors):
        axes[0, 0].plot(diag.pixels, spectra[ig], color=color, lw=1.0, label=diag.gmd_labels[ig])
        if sample_spectra.size:
            sel = in_range & (gbin == ig)
            if np.any(sel):
                axes[0, 1].plot(diag.pixels, np.nanmedian(sample_spectra[sel], axis=0), color=color, lw=1.0)

    axes[0, 0].set_title("Mean VLS spectrum")
    axes[0, 1].set_title("Median VLS spectrum from selected shots")
    for ax in axes[0, :2]:
        ax.set_xlabel("VLS pixel")
        ax.set_ylabel("VLS")
        ax.grid(alpha=0.2)
        apply_limits(ax, xlim=pixel_xlim, ylim=spectrum_ylim)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    axes[0, 2].plot(diag.gmd_centers, diag.gmd_sum[ie], "o-", label="sum GMD")
    ax2 = axes[0, 2].twinx()
    ax2.plot(diag.gmd_centers, diag.trans_raw_sum[ie], "s-", color="tab:red", label="sum VLS raw")
    ax2.plot(diag.gmd_centers, diag.trans_corrected_sum[ie], "^-", color="tab:orange", label="sum VLS corrected")
    axes[0, 2].set_title("Numerator/denominator decomposition")
    axes[0, 2].set_xlabel("GMD bin center / uJ")
    axes[0, 2].set_ylabel("sum GMD")
    ax2.set_ylabel("sum VLS")
    lines, labels = axes[0, 2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[0, 2].legend(lines + lines2, labels + labels2, frameon=False, fontsize=7)
    axes[0, 2].grid(alpha=0.25)

    axes[1, 0].plot(diag.gmd_centers, diag.stat_mean("edge_baseline")[ie], "o-", label="edge baseline")
    axes[1, 0].plot(diag.gmd_centers, diag.stat_mean("negative_fraction")[ie], "s-", label="negative fraction")
    axes[1, 0].set_title("Baseline residual metrics")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(diag.gmd_centers, diag.stat_mean("raw_sum")[ie], "o-", label="raw")
    axes[1, 1].plot(diag.gmd_centers, diag.stat_mean("positive_sum")[ie], "s-", label="positive")
    axes[1, 1].plot(diag.gmd_centers, diag.stat_mean("corrected_sum")[ie], "^-", label="corrected")
    axes[1, 1].set_title("Mean integrated VLS per shot")
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].plot(diag.gmd_centers, diag.stat_mean("peak_height")[ie], "o-", label="peak height")
    ax3 = axes[1, 2].twinx()
    ax3.plot(diag.gmd_centers, diag.stat_mean("centroid")[ie], "s-", color="tab:green", label="centroid")
    ax3.plot(diag.gmd_centers, diag.stat_mean("fwhm")[ie], "^-", color="tab:purple", label="FWHM proxy")
    axes[1, 2].set_title("Peak shape metrics")
    lines, labels = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    axes[1, 2].legend(lines + lines2, labels + labels2, frameon=False, fontsize=7)

    for ax in axes[1, :]:
        ax.set_xlabel("GMD bin center / uJ")
        ax.grid(alpha=0.25)
    fig.suptitle(f"VLS statistics at nominal {diag.energies[ie]:.2f} eV")
    return fig, axes


def plot_shape_binning(diag: ThesisDiagnostics, energy_ev: float, *, by: str = "centroid", n_bins: int = 7, corrected: bool = False, pixel_xlim=None, ylim=None):
    if by not in {"centroid", "fwhm"}:
        raise ValueError("by must be 'centroid' or 'fwhm'")
    ie = nearest_energy_index(diag.energies, energy_ev)
    sample = diag.selected_samples.get(ie)
    if sample is None or sample["gmd"].size == 0:
        raise ValueError(f"No selected-shot sample stored for nominal energy near {energy_ev} eV")
    spectra = sample["vls_corrected"] if corrected else sample["vls_raw"]
    metric = sample[by]
    valid = np.isfinite(metric) & np.all(np.isfinite(spectra), axis=1)
    metric = metric[valid]
    spectra = spectra[valid]
    if metric.size == 0:
        raise ValueError(f"No finite {by} values available")
    edges = np.unique(np.quantile(metric, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 3:
        edges = np.linspace(np.nanmin(metric), np.nanmax(metric), int(n_bins) + 1)
    bin_idx = np.searchsorted(edges, metric, side="right") - 1
    bin_idx[metric == edges[-1]] = edges.size - 2
    in_range = (bin_idx >= 0) & (bin_idx < edges.size - 1)

    colors = colors_for_bins(edges.size - 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    counts = np.zeros(edges.size - 1, dtype=int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ib, color in enumerate(colors):
        sel = in_range & (bin_idx == ib)
        counts[ib] = int(np.sum(sel))
        if not counts[ib]:
            continue
        mean = np.nanmean(spectra[sel], axis=0)
        denom = np.nanmax(mean)
        if np.isfinite(denom) and denom != 0:
            mean = mean / denom
        axes[0].plot(diag.pixels, mean, color=color, lw=1.2, label=f"{centers[ib]:.2f}")
    axes[1].bar(centers, counts, width=np.diff(edges) * 0.85, color=colors, edgecolor="0.2")

    axes[0].set_xlabel("VLS pixel")
    axes[0].set_ylabel("normalized mean VLS")
    axes[0].set_title(f"Mean spectra binned by {by}")
    axes[0].legend(title=by, frameon=False, fontsize=7)
    axes[0].grid(alpha=0.2)
    apply_limits(axes[0], xlim=pixel_xlim, ylim=ylim)

    axes[1].set_xlabel(by)
    axes[1].set_ylabel("shot count")
    axes[1].set_title(f"{by} bin statistics")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Spectral-shape binning at nominal {diag.energies[ie]:.2f} eV")
    return fig, axes


def plot_double_peak_summary(summary: dict[str, np.ndarray], gmd_labels: Sequence[str], *, xlim=None):
    colors = colors_for_bins(len(gmd_labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, constrained_layout=True)
    e = summary["energies"]
    for ig, color in enumerate(colors):
        label = gmd_labels[ig]
        axes[0].plot(e, summary["fraction"][:, ig], "o-", ms=3, lw=1.0, color=color, label=label)
        axes[1].plot(e, summary["mean_separation_px"][:, ig], "o-", ms=3, lw=1.0, color=color)
        axes[2].plot(e, summary["mean_second_main_ratio"][:, ig], "o-", ms=3, lw=1.0, color=color)
    axes[0].set_ylabel("rejected fraction")
    axes[1].set_ylabel("peak separation / px")
    axes[2].set_ylabel("second/main ratio")
    axes[2].set_xlabel("nominal photon energy / eV")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25)
        apply_limits(ax, xlim=xlim)
    fig.suptitle("VLS double-peak proxy by GMD bin")
    return fig, axes


def print_interpretation_table(diag: ThesisDiagnostics, energy_ev: float) -> None:
    ie = nearest_energy_index(diag.energies, energy_ev)
    print(f"Interpretation metrics near nominal {diag.energies[ie]:.2f} eV")
    print("metric                         range/span")
    rows = [
        ("raw OD span", diag.od_raw[ie]),
        ("corrected OD span", diag.od_corrected[ie]),
        ("edge baseline span", diag.stat_mean("edge_baseline")[ie]),
        ("centroid span", diag.stat_mean("centroid")[ie]),
        ("FWHM proxy span", diag.stat_mean("fwhm")[ie]),
        ("mean GMD span", diag.mean_gmd[ie]),
        ("mean VLS raw span", diag.mean_trans_raw[ie]),
    ]
    for name, values in rows:
        finite = np.asarray(values, dtype=np.float64)
        if np.any(np.isfinite(finite)):
            span = np.nanmax(finite) - np.nanmin(finite)
        else:
            span = np.nan
        print(f"{name:<30} {span:.6g}")
    low_count = diag.counts[ie] < diag.config.min_shots_for_interpretation
    if np.any(low_count):
        labels = [diag.gmd_labels[i] for i in np.where(low_count)[0]]
        print(f"Low-stat bins (<{diag.config.min_shots_for_interpretation} shots): {labels}")


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den != 0))
    return out


def _safe_divide_3d(num: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, counts[:, :, None], out=out, where=counts[:, :, None] > 0)
    return out


def _log10_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ratio = _safe_divide(num, den)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0)
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
