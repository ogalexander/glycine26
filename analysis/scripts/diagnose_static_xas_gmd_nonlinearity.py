"""
Diagnose GMD-dependent trends in traditional static XAS absorbance.

This script is deliberately downstream-only. It starts from the processed
static-XAS H5 products and writes diagnostic figures plus a short markdown
report to ./diagnostics by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import static_xas_analysis as sxa


REPO_ROOT = Path(__file__).resolve().parents[2]


PARAMS = {
    "SAMPLE_H5": REPO_ROOT / "11022188" / "processed" / "xas_static" / "run58794_static_xas.h5",
    "REFERENCE_H5": REPO_ROOT / "11022188" / "processed" / "xas_static" / "reference_frame.h5",
    "OUTPUT_DIR": REPO_ROOT / "diagnostics",
    "GMD_EDGES": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    "GMD_MIN_THRESHOLD": None,
    "GMD_MAX_THRESHOLD": 10.0,
    "VLS_INTENSITY_THRESHOLD": 1000.0,
    "REJECT_DOUBLE_PEAKS": False,
    "DOUBLE_PEAK_KWARGS": {
        "min_distance_px": 3,
        "max_distance_px": 17,
        "min_prom_sigma": 2.5,
        "min_height_sigma": 2.0,
        "min_rel_peak_height": 0.35,
        "min_second_ratio": 0.05,
        "min_bridge_drop": 0.015,
        "min_split_sigma": 0.75,
    },
    "CENTER_METHOD": "com",
    "CENTER_HALF_WINDOW": 5,
    "CENTER_CHUNK_SIZE": 50_000,
    "PROXY_PIXEL_BIN_WIDTH": 1,
    "INTENSITY_PIXEL_ROI": None,
    "SHOT_INTENSITY_CLIP_NEGATIVE": False,
    "CURVATURE_GMD_FINE_EDGES": np.linspace(0.0, 10.0, 41),
    # Edit these for the final science run. The processed sample nominal range
    # in run58794 is 271-286 eV, so these defaults choose high-nominal-energy
    # points in that scan plus one low-energy off-resonant point.
    "OFF_NOMINAL_ENERGY_EV": 272.0,
    "PEAK_NOMINAL_ENERGIES_EV": np.array([273.5, 275.0, 278.0]),
    "PEAK_ACTUAL_ENERGIES_EV": np.array([274.5, 276.0, 279.0]),
    "PROXY_DIAGNOSTIC_NOMINAL_ENERGY_EV": 284.0,
    "ROBUSTNESS_ROIS": [None, (470.0, 630.0), (500.0, 600.0)],
    "ROBUSTNESS_DOUBLE_PEAK_MODES": [False, True],
    "RUN_DOUBLE_PEAK_ROBUSTNESS": False,
    "PIXEL_MAP_CONFIG": {
        "source_to_grating_m": 0.85,
        "grating_to_screen_m": 0.5429,
        "incident_angle_deg": 86.78,
        "groove_density_lines_per_mm": 1200.0,
        "diffraction_order": 1,
        "magnification": 3.0,
        "pixel_pitch_m": 50e-6,
        "axis_sign": 1,
    },
    "PIXEL_MAP_CALIBRATION": {
        "pixel": 524.0,
        "energy_eV": 284.2,
    },
    "INVERSE_MAP_RANGE_EV": (260.0, 290.0),
    "INVERSE_MAP_SAMPLES": 10_000,
    "SATURATION_PHOTON_ENERGY_EV": 288.0,
    "RESONANT_CROSS_SECTION_CM2": 1.0e-18,
    "FOCUS_DIAMETER_UM": None,
    "CHUNK_SIZE": 20_000,
    "QUICK_MAX_SHOTS_PER_ENERGY": 4000,
}


@dataclass
class FilterConfig:
    vls_intensity_threshold: Optional[float]
    gmd_min_threshold: Optional[float]
    gmd_max_threshold: Optional[float]
    reject_double_peaks: bool
    double_peak_kwargs: dict


@dataclass
class SpectrumGrid:
    x: np.ndarray
    spec_sum: np.ndarray
    counts: np.ndarray
    gmd_edges: np.ndarray
    pixels: np.ndarray
    x_label: str
    metadata: dict


@dataclass
class CurveGrid:
    x: np.ndarray
    absorbance: np.ndarray
    trans_intensity: np.ndarray
    incident_intensity: np.ndarray
    n_trans: np.ndarray
    n_incident: np.ndarray
    gmd_edges: np.ndarray
    x_label: str


def finite_name(value: object) -> str:
    if value is None:
        return "full"
    if isinstance(value, tuple):
        return f"{value[0]:.0f}_{value[1]:.0f}"
    return str(value).replace(" ", "_").replace("/", "_")


def get_h5_info(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {
            "nominal_energies": h5["nominal_energies"][...].astype(np.float64),
            "n_shots": h5["n_shots"][...].astype(np.int64),
            "vls_pixels": h5["vls_pixels"][...].astype(np.float64),
            "vls_shape": tuple(h5["vls"].shape),
        }


def gmd_labels(edges: Sequence[float]) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        close = "]" if i == len(edges) - 2 else ")"
        labels.append(f"[{edges[i]:.1f}, {edges[i + 1]:.1f}{close}")
    return labels


def gmd_centers(edges: Sequence[float]) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def nearest_indices(values: np.ndarray, targets: Sequence[float]) -> list[int]:
    values = np.asarray(values, dtype=np.float64)
    return [int(np.nanargmin(np.abs(values - float(target)))) for target in targets]


def nearest_common_rows(x: np.ndarray, targets: Sequence[float]) -> list[int]:
    return nearest_indices(np.asarray(x, dtype=np.float64), targets)


def roi_mask(pixels: np.ndarray, roi: Optional[Sequence[float]]) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    if roi is None:
        return np.ones(pixels.shape, dtype=bool)
    lo, hi = float(roi[0]), float(roi[1])
    mask = (pixels >= lo) & (pixels <= hi)
    if not np.any(mask):
        raise ValueError(f"ROI {roi} does not overlap pixel axis {pixels[0]}..{pixels[-1]}")
    return mask


def shot_intensity(vls: np.ndarray, pixels: np.ndarray, roi: Optional[Sequence[float]], *, clip_negative: bool) -> np.ndarray:
    y = np.asarray(vls, dtype=np.float64)[:, roi_mask(pixels, roi)]
    if clip_negative:
        y = np.maximum(y, 0.0)
    return np.sum(y, axis=1)


def spectral_moments(vls: np.ndarray, pixels: np.ndarray, roi: Optional[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    mask = roi_mask(pixels, roi)
    pix = np.asarray(pixels, dtype=np.float64)[mask]
    y = np.maximum(np.nan_to_num(vls[:, mask], nan=0.0), 0.0)
    weight = np.sum(y, axis=1)
    centroid = np.full(y.shape[0], np.nan, dtype=np.float64)
    width = np.full(y.shape[0], np.nan, dtype=np.float64)
    ok = weight > 0
    centroid[ok] = np.sum(y[ok] * pix[None, :], axis=1) / weight[ok]
    var = np.sum(y[ok] * (pix[None, :] - centroid[ok, None]) ** 2, axis=1) / weight[ok]
    width[ok] = np.sqrt(np.maximum(var, 0.0))
    return centroid, width


def gmd_bins(gmd: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gbin, in_range = sxa._gmd_bin_indices(gmd, edges)
    return gbin, in_range


def apply_quality_filters(vls: np.ndarray, gmd: np.ndarray, cfg: FilterConfig) -> tuple[np.ndarray, dict]:
    keep = np.isfinite(gmd) & np.all(np.isfinite(vls), axis=1)
    summary = {
        "input": int(gmd.size),
        "finite": int(np.sum(keep)),
        "peak_rejected": 0,
        "low_gmd_rejected": 0,
        "high_gmd_rejected": 0,
        "double_peak_rejected": 0,
    }
    if cfg.vls_intensity_threshold is not None:
        peak = np.nanmax(vls, axis=1)
        reject = keep & (peak < float(cfg.vls_intensity_threshold))
        summary["peak_rejected"] = int(np.sum(reject))
        keep &= peak >= float(cfg.vls_intensity_threshold)
    if cfg.gmd_min_threshold is not None:
        reject = keep & (gmd < float(cfg.gmd_min_threshold))
        summary["low_gmd_rejected"] = int(np.sum(reject))
        keep &= gmd >= float(cfg.gmd_min_threshold)
    if cfg.gmd_max_threshold is not None:
        reject = keep & (gmd > float(cfg.gmd_max_threshold))
        summary["high_gmd_rejected"] = int(np.sum(reject))
        keep &= gmd <= float(cfg.gmd_max_threshold)
    if cfg.reject_double_peaks and np.any(keep):
        local = np.where(keep)[0]
        xcorr, xsm, noise = sxa.preprocess_shots_for_peakfinding(vls[local])
        diag = sxa.detect_double_peaks_pairscan(xcorr, xsm, noise, **cfg.double_peak_kwargs)
        reject_local = diag["is_candidate"]
        summary["double_peak_rejected"] = int(np.sum(reject_local))
        keep[local[reject_local]] = False
    summary["kept"] = int(np.sum(keep))
    return keep, summary


def iter_energy_chunks(
    path: Path,
    *,
    energy_indices: Optional[Iterable[int]],
    chunk_size: int,
    max_shots_per_energy: Optional[int],
):
    with h5py.File(path, "r") as h5:
        energies = h5["nominal_energies"][...].astype(np.float64)
        n_shots = h5["n_shots"][...].astype(np.int64)
        pixels = h5["vls_pixels"][...].astype(np.float64)
        if energy_indices is None:
            indices = range(energies.size)
        else:
            indices = list(dict.fromkeys(int(i) for i in energy_indices))
        for ie in indices:
            n = int(n_shots[ie])
            if max_shots_per_energy is not None:
                n = min(n, int(max_shots_per_energy))
            for start in range(0, n, int(chunk_size)):
                stop = min(n, start + int(chunk_size))
                yield {
                    "energy_index": ie,
                    "nominal_energy": float(energies[ie]),
                    "vls": h5["vls"][ie, start:stop, :],
                    "gmd": h5["gmd"][ie, start:stop],
                    "pixels": pixels,
                }


def accumulate_nominal_grid(
    path: Path,
    gmd_edges: np.ndarray,
    cfg: FilterConfig,
    *,
    chunk_size: int,
    max_shots_per_energy: Optional[int],
) -> SpectrumGrid:
    info = get_h5_info(path)
    keys = np.round(info["nominal_energies"], 6)
    x = np.unique(keys[np.isfinite(keys)])
    x.sort()
    x_pos = {float(key): i for i, key in enumerate(x)}
    n_gmd = gmd_edges.size - 1
    n_pix = info["vls_pixels"].size
    spec_sum = np.zeros((x.size, n_gmd, n_pix), dtype=np.float64)
    counts = np.zeros((x.size, n_gmd), dtype=np.int64)

    for chunk in iter_energy_chunks(path, energy_indices=None, chunk_size=chunk_size, max_shots_per_energy=max_shots_per_energy):
        keep, _ = apply_quality_filters(chunk["vls"], chunk["gmd"], cfg)
        if not np.any(keep):
            continue
        vls = chunk["vls"][keep]
        gmd = chunk["gmd"][keep]
        gb, in_gmd = gmd_bins(gmd, gmd_edges)
        if not np.any(in_gmd):
            continue
        ix = x_pos[float(np.round(chunk["nominal_energy"], 6))]
        for ig in np.unique(gb[in_gmd]):
            sel = in_gmd & (gb == ig)
            counts[ix, ig] += int(np.sum(sel))
            spec_sum[ix, ig] += np.sum(vls[sel], axis=0)

    return SpectrumGrid(
        x=x.astype(np.float64),
        spec_sum=spec_sum,
        counts=counts,
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        pixels=info["vls_pixels"],
        x_label="nominal photon energy (eV)",
        metadata={"mode": "nominal", "path": str(path)},
    )


def accumulate_actual_grid(
    path: Path,
    gmd_edges: np.ndarray,
    cfg: FilterConfig,
    params: dict,
    *,
    chunk_size: int,
    max_shots_per_energy: Optional[int],
) -> SpectrumGrid:
    info = get_h5_info(path)
    pixels = info["vls_pixels"].astype(np.int64)
    pixel_min = int(np.nanmin(pixels))
    pixel_max = int(np.nanmax(pixels))
    width = max(1, int(params["PROXY_PIXEL_BIN_WIDTH"]))
    starts = np.arange(pixel_min, pixel_max + 1, width, dtype=np.int64)
    stops = np.minimum(starts + width - 1, pixel_max)
    centers = starts.astype(np.float64) + 0.5 * (stops - starts).astype(np.float64)
    n_gmd = gmd_edges.size - 1
    spec_sum = np.zeros((centers.size, n_gmd, pixels.size), dtype=np.float64)
    counts = np.zeros((centers.size, n_gmd), dtype=np.int64)

    for chunk in iter_energy_chunks(path, energy_indices=None, chunk_size=chunk_size, max_shots_per_energy=max_shots_per_energy):
        keep, _ = apply_quality_filters(chunk["vls"], chunk["gmd"], cfg)
        if not np.any(keep):
            continue
        vls = chunk["vls"][keep]
        gmd = chunk["gmd"][keep]
        center_rel, _ = sxa.estimate_center_pixels(
            vls,
            method=params["CENTER_METHOD"],
            half_window=int(params["CENTER_HALF_WINDOW"]),
            chunk_size=int(params["CENTER_CHUNK_SIZE"]),
        )
        valid_center = np.isfinite(center_rel)
        if not np.any(valid_center):
            continue
        vls = vls[valid_center]
        gmd = gmd[valid_center]
        proxy_rel = np.clip(np.round(center_rel[valid_center]).astype(np.int64), 0, pixels.size - 1)
        abs_proxy = pixels[proxy_rel]
        xb = ((abs_proxy - pixel_min) // width).astype(np.int64)
        gb, in_gmd = gmd_bins(gmd, gmd_edges)
        valid = in_gmd & (xb >= 0) & (xb < centers.size)
        if not np.any(valid):
            continue
        for ix in np.unique(xb[valid]):
            same_x = valid & (xb == ix)
            for ig in np.unique(gb[same_x]):
                sel = same_x & (gb == ig)
                counts[ix, ig] += int(np.sum(sel))
                spec_sum[ix, ig] += np.sum(vls[sel], axis=0)

    occupied = np.any(counts > 0, axis=1)
    centers = centers[occupied]
    spec_sum = spec_sum[occupied]
    counts = counts[occupied]
    if centers.size:
        energy, _, _ = sxa.convert_pixels_to_energy(
            centers,
            params["PIXEL_MAP_CONFIG"],
            params["PIXEL_MAP_CALIBRATION"],
            params["INVERSE_MAP_RANGE_EV"],
            n_samples=params["INVERSE_MAP_SAMPLES"],
        )
    else:
        energy = np.empty(0, dtype=np.float64)
    order = np.argsort(energy)
    return SpectrumGrid(
        x=energy[order],
        spec_sum=spec_sum[order],
        counts=counts[order],
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        pixels=info["vls_pixels"],
        x_label="converted photon energy (eV)",
        metadata={"mode": "actual", "path": str(path), "pixel_centers": centers[order]},
    )


def intensity_from_grid(grid: SpectrumGrid, roi: Optional[Sequence[float]], *, clip_mode: str) -> np.ndarray:
    mask = roi_mask(grid.pixels, roi)
    counts = np.where(grid.counts > 0, grid.counts, 1).astype(np.float64)
    mean_spec = grid.spec_sum / counts[:, :, None]
    mean_spec = np.where(grid.counts[:, :, None] > 0, mean_spec, np.nan)
    y = mean_spec[:, :, mask]
    if clip_mode == "positive":
        y = np.maximum(y, 0.0)
    elif clip_mode == "raw":
        pass
    else:
        raise ValueError("clip_mode must be 'positive' or 'raw'")
    return np.nansum(y, axis=2)


def absorbance_from_grids(trans: SpectrumGrid, ref: SpectrumGrid, roi: Optional[Sequence[float]], *, clip_mode: str) -> CurveGrid:
    common = np.intersect1d(np.round(trans.x, 6), np.round(ref.x, 6))
    tx = np.round(trans.x, 6)
    rx = np.round(ref.x, 6)
    tpos = np.array([int(np.where(tx == c)[0][0]) for c in common], dtype=np.int64)
    rpos = np.array([int(np.where(rx == c)[0][0]) for c in common], dtype=np.int64)
    ti = intensity_from_grid(trans, roi, clip_mode=clip_mode)[tpos]
    ri = intensity_from_grid(ref, roi, clip_mode=clip_mode)[rpos]
    nt = trans.counts[tpos]
    nr = ref.counts[rpos]
    absorbance = np.full(ti.shape, np.nan, dtype=np.float64)
    valid = (nt > 0) & (nr > 0) & np.isfinite(ti) & np.isfinite(ri) & (ti > 0) & (ri > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        absorbance[valid] = -np.log10(ti[valid] / ri[valid])
    return CurveGrid(
        x=common.astype(np.float64),
        absorbance=absorbance,
        trans_intensity=ti,
        incident_intensity=ri,
        n_trans=nt,
        n_incident=nr,
        gmd_edges=trans.gmd_edges,
        x_label=trans.x_label,
    )


def fit_origin_models(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[valid].astype(np.float64)
    y = y[valid].astype(np.float64)
    if x.size < 3:
        return {"n": int(x.size), "a_lin": np.nan, "a_quad": np.nan, "b_quad": np.nan, "b_se": np.nan, "b_t": np.nan}
    a_lin = float(np.sum(x * y) / np.sum(x * x))
    X = np.column_stack([x, x**2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, x.size - 2)
    sigma2 = float(np.sum(resid**2) / dof)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        b_se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        b_se = np.nan
    b_t = float(beta[1] / b_se) if np.isfinite(b_se) and b_se > 0 else np.nan
    return {
        "n": int(x.size),
        "a_lin": a_lin,
        "a_quad": float(beta[0]),
        "b_quad": float(beta[1]),
        "b_se": b_se,
        "b_t": b_t,
    }


def binned_xy(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gb, in_range = gmd_bins(x, edges)
    centers = gmd_centers(edges)
    mean = np.full(centers.shape, np.nan)
    sem = np.full(centers.shape, np.nan)
    counts = np.zeros(centers.shape, dtype=np.int64)
    for ig in range(centers.size):
        keep = in_range & (gb == ig) & np.isfinite(y)
        counts[ig] = int(np.sum(keep))
        if counts[ig]:
            vals = y[keep]
            mean[ig] = float(np.mean(vals))
            sem[ig] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return centers, mean, sem, counts


def collect_sample_energy_shots(path: Path, energy_indices: Sequence[int], cfg: FilterConfig, params: dict, *, quick: bool) -> dict[int, dict]:
    max_shots = params["QUICK_MAX_SHOTS_PER_ENERGY"] if quick else None
    out: dict[int, dict[str, list[np.ndarray]]] = {
        int(ie): {"gmd": [], "itrans": [], "centroid": [], "width": [], "peak": []} for ie in energy_indices
    }
    for chunk in iter_energy_chunks(path, energy_indices=energy_indices, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots):
        keep, _ = apply_quality_filters(chunk["vls"], chunk["gmd"], cfg)
        if not np.any(keep):
            continue
        vls = chunk["vls"][keep]
        gmd = chunk["gmd"][keep]
        ie = int(chunk["energy_index"])
        out[ie]["gmd"].append(gmd)
        out[ie]["itrans"].append(
            shot_intensity(
                vls,
                chunk["pixels"],
                params["INTENSITY_PIXEL_ROI"],
                clip_negative=bool(params["SHOT_INTENSITY_CLIP_NEGATIVE"]),
            )
        )
        centroid, width = spectral_moments(vls, chunk["pixels"], params["INTENSITY_PIXEL_ROI"])
        out[ie]["centroid"].append(centroid)
        out[ie]["width"].append(width)
        out[ie]["peak"].append(np.nanmax(vls, axis=1))
    merged = {}
    for ie, fields in out.items():
        merged[ie] = {}
        for key, pieces in fields.items():
            merged[ie][key] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)
    return merged


def collect_actual_center_hists(
    path: Path,
    energy_indices: Sequence[int],
    cfg: FilterConfig,
    params: dict,
    *,
    quick: bool,
    bins: np.ndarray,
) -> dict[int, np.ndarray]:
    max_shots = params["QUICK_MAX_SHOTS_PER_ENERGY"] if quick else None
    hists = {int(ie): np.zeros(bins.size - 1, dtype=np.int64) for ie in energy_indices}
    for chunk in iter_energy_chunks(path, energy_indices=energy_indices, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots):
        keep, _ = apply_quality_filters(chunk["vls"], chunk["gmd"], cfg)
        if not np.any(keep):
            continue
        vls = chunk["vls"][keep]
        center_rel, _ = sxa.estimate_center_pixels(
            vls,
            method=params["CENTER_METHOD"],
            half_window=int(params["CENTER_HALF_WINDOW"]),
            chunk_size=int(params["CENTER_CHUNK_SIZE"]),
        )
        valid = np.isfinite(center_rel)
        if not np.any(valid):
            continue
        center_pixel = np.interp(
            center_rel[valid],
            np.arange(chunk["pixels"].size, dtype=np.float64),
            chunk["pixels"].astype(np.float64),
            left=np.nan,
            right=np.nan,
        )
        hist, _ = np.histogram(center_pixel[np.isfinite(center_pixel)], bins=bins)
        hists[int(chunk["energy_index"])] += hist
    return hists


def plot_check1(sample_shots: dict[int, dict], energy_names: list[str], nominal_energies: np.ndarray, params: dict, outdir: Path) -> tuple[str, list[str], dict]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()
    report_lines = []
    fits = {}
    fine_edges = np.asarray(params["CURVATURE_GMD_FINE_EDGES"], dtype=np.float64)
    for ax, (ie, name) in zip(axes, zip(sample_shots.keys(), energy_names)):
        gmd = sample_shots[ie]["gmd"]
        y = sample_shots[ie]["itrans"]
        fit = fit_origin_models(gmd, y)
        fits[name] = fit
        xc, ym, yerr, counts = binned_xy(gmd, y, fine_edges)
        keep = counts > 0
        ax.errorbar(xc[keep], ym[keep], yerr=yerr[keep], fmt="o", ms=3, lw=0.8, label="fine GMD-bin mean")
        xfit = np.linspace(np.nanmin(xc[keep]) if np.any(keep) else 0.0, np.nanmax(xc[keep]) if np.any(keep) else 1.0, 200)
        ax.plot(xfit, fit["a_lin"] * xfit, label="linear through 0")
        ax.plot(xfit, fit["a_quad"] * xfit + fit["b_quad"] * xfit**2, label="quadratic through 0")
        ax.set_title(f"{name}: nominal {nominal_energies[ie]:.2f} eV\nb={fit['b_quad']:.3g}, t={fit['b_t']:.2f}, n={fit['n']}")
        ax.set_xlabel("GMD / incident pulse energy (uJ)")
        ax.set_ylabel("per-shot integrated I_trans (a.u.)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        verdict = "INCONCLUSIVE"
        if np.isfinite(fit["b_t"]) and abs(fit["b_t"]) >= 3.0:
            verdict = "PHYSICS-CANDIDATE" if name != "off" else "ARTIFACT/BEAM"
        report_lines.append(f"- {name}: b={fit['b_quad']:.6g}, se={fit['b_se']:.6g}, t={fit['b_t']:.2f}, n={fit['n']} -> {verdict}")
    fig.suptitle("Check 1: reference-free I_trans vs GMD curvature")
    fig.tight_layout()
    path = outdir / "check1_reference_free_curvature.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path), report_lines, fits


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def plot_check2(diag: dict, ie: int, nominal_energy: float, params: dict, outdir: Path) -> tuple[str, list[str], dict]:
    gmd = diag["gmd"]
    metrics = [
        ("centroid", diag["centroid"], "centroid (Gotthard pixel)"),
        ("width", diag["width"], "2nd-moment width (pixels)"),
        ("peak", diag["peak"], "VLS peak intensity (a.u.)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    lines = []
    stats = {}
    for ax, (name, y, ylabel) in zip(axes, metrics):
        r = corrcoef(gmd, y)
        stats[name] = r
        xc, ym, yerr, counts = binned_xy(gmd, y, np.asarray(params["CURVATURE_GMD_FINE_EDGES"], dtype=np.float64))
        keep = counts > 0
        ax.errorbar(xc[keep], ym[keep], yerr=yerr[keep], fmt="o", ms=3, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name}: Pearson r={r:.3f}")
        ax.grid(alpha=0.25)
        verdict = "ARTIFACT/BEAM" if np.isfinite(r) and abs(r) > 0.2 else "INCONCLUSIVE/OK"
        lines.append(f"- {name}: Pearson r={r:.4f} at nominal {nominal_energy:.2f} eV -> {verdict}")
    axes[-1].set_xlabel("GMD / incident pulse energy (uJ)")
    fig.suptitle(f"Check 2: is GMD also changing spectrum shape? nominal {nominal_energy:.2f} eV")
    fig.tight_layout()
    path = outdir / "check2_gmd_proxy_shape_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path), lines, stats


def plot_grid_curves(curve: CurveGrid, outdir: Path, *, filename: str, title: str) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    labels = gmd_labels(curve.gmd_edges)
    for ig, label in enumerate(labels):
        axes[0].plot(curve.x, curve.trans_intensity[:, ig], label=label)
        axes[1].plot(curve.x, curve.incident_intensity[:, ig], label=label)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = curve.trans_intensity[:, ig] / curve.incident_intensity[:, ig]
        axes[2].plot(curve.x, ratio, label=label)
    axes[0].set_ylabel("I_trans (a.u.)")
    axes[1].set_ylabel("I_inc (a.u.)")
    axes[2].set_ylabel("I_trans / I_inc")
    axes[2].set_xlabel(curve.x_label)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(title="GMD bin", fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_absorbance_at_peaks(
    nominal: CurveGrid,
    actual: Optional[CurveGrid],
    peak_nominal: Sequence[float],
    peak_actual: Optional[Sequence[float]],
    outdir: Path,
    *,
    filename: str,
    title: str,
) -> str:
    centers = gmd_centers(nominal.gmd_edges)
    n_peaks = len(peak_nominal)
    fig, axes = plt.subplots(1, n_peaks, figsize=(4.2 * n_peaks, 3.8), sharey=True)
    if n_peaks == 1:
        axes = [axes]
    if peak_actual is None:
        peak_actual = peak_nominal
    for i, ax in enumerate(axes):
        ni = nearest_common_rows(nominal.x, [peak_nominal[i]])[0]
        ax.plot(centers, nominal.absorbance[ni], "o-", label=f"nominal {nominal.x[ni]:.2f} eV")
        if actual is not None and actual.x.size:
            ai = nearest_common_rows(actual.x, [peak_actual[i]])[0]
            ax.plot(centers, actual.absorbance[ai], "s--", label=f"actual {actual.x[ai]:.2f} eV")
        ax.set_xlabel("GMD bin center (uJ)")
        ax.set_title(f"peak {i + 1}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Absorbance")
    fig.suptitle(title)
    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_check5_clip_sensitivity(clip_on: CurveGrid, clip_off: CurveGrid, peak_nominal: Sequence[float], outdir: Path) -> str:
    centers = gmd_centers(clip_on.gmd_edges)
    fig, axes = plt.subplots(1, len(peak_nominal), figsize=(4.2 * len(peak_nominal), 3.8), sharey=True)
    if len(peak_nominal) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        row_on = nearest_common_rows(clip_on.x, [peak_nominal[i]])[0]
        row_off = nearest_common_rows(clip_off.x, [peak_nominal[i]])[0]
        ax.plot(centers, clip_on.absorbance[row_on], "o-", label="negative clip ON")
        ax.plot(centers, clip_off.absorbance[row_off], "s--", label="negative clip OFF")
        ax.set_title(f"peak {i + 1}: nearest {clip_on.x[row_on]:.2f} eV")
        ax.set_xlabel("GMD bin center (uJ)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Absorbance")
    fig.suptitle("Check 5: negative-clipping sensitivity")
    fig.tight_layout()
    path = outdir / "check5_negative_clipping_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_check6_counts_and_hists(
    sample_grid: SpectrumGrid,
    ref_grid: SpectrumGrid,
    sample_hists: dict[int, np.ndarray],
    ref_hists: dict[int, np.ndarray],
    hist_bins: np.ndarray,
    sample_energies: np.ndarray,
    outdir: Path,
) -> str:
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    im0 = ax0.imshow(sample_grid.counts.T, aspect="auto", origin="lower")
    im1 = ax1.imshow(ref_grid.counts.T, aspect="auto", origin="lower")
    ax0.set_title("sample surviving counts")
    ax1.set_title("reference surviving counts")
    ax0.set_ylabel("GMD bin")
    ax1.set_ylabel("GMD bin")
    ax0.set_xlabel("nominal-energy row")
    ax1.set_xlabel("nominal-energy row")
    fig.colorbar(im0, ax=ax0, fraction=0.046)
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    hist_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])]
    centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    for ax, ie in zip(hist_axes, sample_hists.keys()):
        ax.plot(centers, sample_hists[ie], label="sample")
        ax.plot(centers, ref_hists.get(ie, np.zeros_like(centers)), label="reference")
        ax.set_title(f"actual-center pixel distribution, nominal {sample_energies[ie]:.2f} eV")
        ax.set_xlabel("center pixel")
        ax.set_ylabel("shots")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Check 6: shot counts and actual-energy distribution mismatch")
    fig.tight_layout()
    path = outdir / "check6_counts_and_actual_energy_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_check7_robustness(results: list[tuple[str, CurveGrid]], peak_nominal: Sequence[float], outdir: Path) -> str:
    centers = gmd_centers(results[0][1].gmd_edges)
    fig, axes = plt.subplots(1, len(peak_nominal), figsize=(4.2 * len(peak_nominal), 3.8), sharey=True)
    if len(peak_nominal) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for label, curve in results:
            row = nearest_common_rows(curve.x, [peak_nominal[i]])[0]
            ax.plot(centers, curve.absorbance[row], marker="o", label=label)
        ax.set_title(f"peak {i + 1}")
        ax.set_xlabel("GMD bin center (uJ)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Absorbance")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Check 7: ROI and filter robustness")
    fig.tight_layout()
    path = outdir / "check7_roi_filter_robustness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_check8_fluence(params: dict, outdir: Path) -> tuple[str, list[str]]:
    edges = np.asarray(params["GMD_EDGES"], dtype=np.float64)
    centers_uj = gmd_centers(edges)
    photon_energy_j = float(params["SATURATION_PHOTON_ENERGY_EV"]) * 1.602176634e-19
    photons = centers_uj * 1.0e-6 / photon_energy_j
    sigma = float(params["RESONANT_CROSS_SECTION_CM2"])
    f_sat = photon_energy_j / (2.0 * sigma)
    lines = [
        f"- photon energy: {params['SATURATION_PHOTON_ENERGY_EV']:.3f} eV",
        f"- cross section: {sigma:.3g} cm^2",
        f"- F_sat = {f_sat:.3g} J/cm^2",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if params["FOCUS_DIAMETER_UM"] is None:
        ax.plot(centers_uj, photons, "o-")
        ax.set_ylabel("photons / pulse")
        ax.set_title("Check 8: fluence sanity needs FOCUS_DIAMETER_UM")
        verdict = "INCONCLUSIVE: no focus size configured"
        lines.append(f"- verdict: {verdict}")
    else:
        diameter_cm = float(params["FOCUS_DIAMETER_UM"]) * 1.0e-4
        area_cm2 = np.pi * (0.5 * diameter_cm) ** 2
        fluence = centers_uj * 1.0e-6 / area_cm2
        ax.plot(centers_uj, fluence, "o-", label="GMD-bin fluence")
        ax.axhline(f_sat, color="k", ls="--", label=f"F_sat {f_sat:.2g} J/cm2")
        ax.set_ylabel("fluence (J/cm2)")
        ax.legend()
        verdict = "PHYSICS-POSSIBLE" if np.nanmax(fluence) > 0.1 * f_sat else "SATURATION UNLIKELY"
        lines.append(f"- focus diameter: {params['FOCUS_DIAMETER_UM']} um")
        lines.append(f"- fluence range: {np.nanmin(fluence):.3g}..{np.nanmax(fluence):.3g} J/cm2")
        lines.append(f"- verdict: {verdict}")
    ax.set_xlabel("GMD bin center (uJ)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "check8_fluence_saturation_sanity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path), lines


def survival_table(grid: SpectrumGrid, run_label: str) -> list[str]:
    labels = gmd_labels(grid.gmd_edges)
    counts = np.sum(grid.counts, axis=0)
    total = int(np.sum(counts))
    lines = [f"| {run_label} | {label} | {int(count)} | {count / total if total else np.nan:.4f} |" for label, count in zip(labels, counts)]
    return lines


def report_verdict_from_checks(fits: dict, check5_delta: float) -> str:
    if not np.isfinite(check5_delta):
        return "INCONCLUSIVE/ARTIFACT-RISK: clipping-off did not produce finite peak absorbance with the current ROI; rerun with a tighter INTENSITY_PIXEL_ROI before claiming nonlinear physics."
    off = fits.get("off", {})
    off_curved = np.isfinite(off.get("b_t", np.nan)) and abs(off.get("b_t", np.nan)) >= 3.0
    peak_curved = [
        np.isfinite(v.get("b_t", np.nan)) and abs(v.get("b_t", np.nan)) >= 3.0
        for k, v in fits.items()
        if k != "off"
    ]
    if check5_delta > 0.05:
        return "ARTIFACT: absorbance trend is sensitive to negative clipping."
    if off_curved:
        return "ARTIFACT/BEAM: off-resonant I_trans also curves with GMD."
    if any(peak_curved):
        return "INCONCLUSIVE/PHYSICS-CANDIDATE: at least one resonance shows energy-selective curvature; inspect checks 3-7."
    return "ARTIFACT/INCONCLUSIVE: no clear energy-selective reference-free curvature was found."


def run(params: dict, *, quick: bool) -> None:
    outdir = Path(params["OUTPUT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    gmd_edges = np.asarray(params["GMD_EDGES"], dtype=np.float64)
    sample_h5 = Path(params["SAMPLE_H5"])
    ref_h5 = Path(params["REFERENCE_H5"])
    cfg = FilterConfig(
        vls_intensity_threshold=params["VLS_INTENSITY_THRESHOLD"],
        gmd_min_threshold=params["GMD_MIN_THRESHOLD"],
        gmd_max_threshold=params["GMD_MAX_THRESHOLD"],
        reject_double_peaks=bool(params["REJECT_DOUBLE_PEAKS"]),
        double_peak_kwargs=dict(params["DOUBLE_PEAK_KWARGS"]),
    )

    sample_info = get_h5_info(sample_h5)
    ref_info = get_h5_info(ref_h5)
    peak_nominal = np.asarray(params["PEAK_NOMINAL_ENERGIES_EV"], dtype=np.float64)
    peak_actual = params["PEAK_ACTUAL_ENERGIES_EV"]
    if peak_actual is not None:
        peak_actual = np.asarray(peak_actual, dtype=np.float64)
    energy_targets = [float(params["OFF_NOMINAL_ENERGY_EV"])] + [float(x) for x in peak_nominal]
    energy_names = ["off", "peak1", "peak2", "peak3"][: len(energy_targets)]
    sample_energy_indices = nearest_indices(sample_info["nominal_energies"], energy_targets)
    proxy_diag_ie = nearest_indices(sample_info["nominal_energies"], [params["PROXY_DIAGNOSTIC_NOMINAL_ENERGY_EV"]])[0]

    max_shots = params["QUICK_MAX_SHOTS_PER_ENERGY"] if quick else None
    sample_shots = collect_sample_energy_shots(sample_h5, sample_energy_indices + [proxy_diag_ie], cfg, params, quick=quick)
    check1_path, check1_lines, fits = plot_check1(
        {ie: sample_shots[ie] for ie in sample_energy_indices},
        energy_names,
        sample_info["nominal_energies"],
        params,
        outdir,
    )
    check2_path, check2_lines, _ = plot_check2(
        sample_shots[proxy_diag_ie],
        proxy_diag_ie,
        sample_info["nominal_energies"][proxy_diag_ie],
        params,
        outdir,
    )

    trans_nominal = accumulate_nominal_grid(sample_h5, gmd_edges, cfg, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
    ref_nominal = accumulate_nominal_grid(ref_h5, gmd_edges, cfg, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
    nominal_clip = absorbance_from_grids(trans_nominal, ref_nominal, params["INTENSITY_PIXEL_ROI"], clip_mode="positive")
    nominal_raw = absorbance_from_grids(trans_nominal, ref_nominal, params["INTENSITY_PIXEL_ROI"], clip_mode="raw")
    check3_path = plot_grid_curves(
        nominal_clip,
        outdir,
        filename="check3_itrans_iinc_ratio_by_gmd.png",
        title="Check 3: I_trans, I_inc, and raw ratio by GMD bin (nominal energy)",
    )

    trans_actual = accumulate_actual_grid(sample_h5, gmd_edges, cfg, params, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
    ref_actual = accumulate_actual_grid(ref_h5, gmd_edges, cfg, params, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
    actual_clip = absorbance_from_grids(trans_actual, ref_actual, params["INTENSITY_PIXEL_ROI"], clip_mode="positive")
    check4_path = plot_absorbance_at_peaks(
        nominal_clip,
        actual_clip,
        peak_nominal,
        peak_actual,
        outdir,
        filename="check4_nominal_vs_actual_binning.png",
        title="Check 4: nominal-energy vs transmitted-center actual-energy binning",
    )
    check5_path = plot_check5_clip_sensitivity(nominal_clip, nominal_raw, peak_nominal, outdir)

    hist_bins = np.arange(sample_info["vls_pixels"][0] - 0.5, sample_info["vls_pixels"][-1] + 1.5, 1.0)
    peak_indices = nearest_indices(sample_info["nominal_energies"], peak_nominal)
    sample_hists = collect_actual_center_hists(sample_h5, peak_indices, cfg, params, quick=quick, bins=hist_bins)
    ref_hist_indices = nearest_indices(ref_info["nominal_energies"], sample_info["nominal_energies"][peak_indices])
    ref_hists_raw = collect_actual_center_hists(ref_h5, ref_hist_indices, cfg, params, quick=quick, bins=hist_bins)
    ref_hists = {sample_ie: ref_hists_raw[ref_ie] for sample_ie, ref_ie in zip(peak_indices, ref_hist_indices)}
    check6_path = plot_check6_counts_and_hists(
        trans_nominal,
        ref_nominal,
        sample_hists,
        ref_hists,
        hist_bins,
        sample_info["nominal_energies"],
        outdir,
    )

    robustness_results: list[tuple[str, CurveGrid]] = []
    for roi in params["ROBUSTNESS_ROIS"]:
        robustness_results.append((f"ROI {finite_name(roi)}", absorbance_from_grids(trans_nominal, ref_nominal, roi, clip_mode="positive")))
    ran_double_peak_robustness = bool(params["RUN_DOUBLE_PEAK_ROBUSTNESS"] and not cfg.reject_double_peaks)
    if ran_double_peak_robustness:
        dp_cfg = FilterConfig(
            vls_intensity_threshold=params["VLS_INTENSITY_THRESHOLD"],
            gmd_min_threshold=params["GMD_MIN_THRESHOLD"],
            gmd_max_threshold=params["GMD_MAX_THRESHOLD"],
            reject_double_peaks=True,
            double_peak_kwargs=dict(params["DOUBLE_PEAK_KWARGS"]),
        )
        dp_trans = accumulate_nominal_grid(sample_h5, gmd_edges, dp_cfg, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
        dp_ref = accumulate_nominal_grid(ref_h5, gmd_edges, dp_cfg, chunk_size=params["CHUNK_SIZE"], max_shots_per_energy=max_shots)
        robustness_results.append(("double-peak rejected", absorbance_from_grids(dp_trans, dp_ref, params["INTENSITY_PIXEL_ROI"], clip_mode="positive")))
    check7_path = plot_check7_robustness(robustness_results, peak_nominal, outdir)
    check8_path, check8_lines = plot_check8_fluence(params, outdir)

    peak_rows = nearest_common_rows(nominal_clip.x, peak_nominal)
    clip_diff = np.abs(nominal_clip.absorbance[peak_rows] - nominal_raw.absorbance[peak_rows])
    clip_delta = float(np.nanmax(clip_diff)) if np.any(np.isfinite(clip_diff)) else np.nan

    report = [
        "# Static XAS GMD-Dependence Diagnostics",
        "",
        f"Mode: {'quick smoke run' if quick else 'full run'}",
        f"Sample: `{sample_h5}`",
        f"Reference: `{ref_h5}`",
        f"Max shots per energy: `{max_shots}`",
        f"GMD edges: `{gmd_edges.tolist()}`",
        f"VLS peak threshold: `{params['VLS_INTENSITY_THRESHOLD']}`",
        f"Reject double peaks: `{params['REJECT_DOUBLE_PEAKS']}`",
        f"Intensity ROI: `{params['INTENSITY_PIXEL_ROI']}`",
        "",
        "## Selected Energies",
        "",
    ]
    for name, target, ie in zip(energy_names, energy_targets, sample_energy_indices):
        report.append(f"- {name}: requested {target:.3f} eV, nearest sample nominal {sample_info['nominal_energies'][ie]:.3f} eV")
    report += [
        "",
        "## Check 1 - Reference-Free Curvature",
        "",
        f"Figure: `{Path(check1_path).name}`",
        *check1_lines,
        "",
        "## Check 2 - Is GMD A Valid Fluence Proxy?",
        "",
        f"Figure: `{Path(check2_path).name}`",
        *check2_lines,
        "",
        "## Check 3 - I_trans, I_inc, Ratio By GMD",
        "",
        f"Figure: `{Path(check3_path).name}`",
        "- Verdict: ARTIFACT if the off-resonant ratio baseline or I_inc shape moves strongly with GMD; otherwise inspect resonance-specific ratio changes.",
        "",
        "## Check 4 - Nominal vs Actual Binning",
        "",
        f"Figure: `{Path(check4_path).name}`",
        "- Verdict: ARTIFACT if the GMD ordering changes or disappears only under transmitted-center actual-energy binning.",
        "",
        "## Check 5 - Negative-Clipping Sensitivity",
        "",
        f"Figure: `{Path(check5_path).name}`",
        (
            f"- Max peak absorbance difference between clip ON and OFF: {clip_delta:.6g}"
            if np.isfinite(clip_delta)
            else "- Max peak absorbance difference between clip ON and OFF: not finite; clip-off produced no valid positive intensity ratio in the selected peak cells."
        ),
        "- Verdict: ARTIFACT if clipping changes the GMD ordering or amplitude materially.",
        "",
        "## Check 6 - Counts And Actual-Energy Distributions",
        "",
        f"Figure: `{Path(check6_path).name}`",
        "- Verdict: ARTIFACT if reference counts collapse or sample/reference actual-energy distributions diverge in the affected cells.",
        "",
        "## Check 7 - ROI And Filter Robustness",
        "",
        f"Figure: `{Path(check7_path).name}`",
        "- Verdict: PHYSICS only if the GMD ordering is stable across ROIs and filtering choices.",
        f"- Double-peak robustness pass run: `{ran_double_peak_robustness}`",
        "",
        "### Survival Counts By GMD Bin",
        "",
        "| run | GMD bin | surviving shots | fraction |",
        "|---|---:|---:|---:|",
        *survival_table(trans_nominal, "sample"),
        *survival_table(ref_nominal, "reference"),
        "",
        "## Check 8 - Fluence vs Saturation Sanity",
        "",
        f"Figure: `{Path(check8_path).name}`",
        *check8_lines,
        "",
        "## Decision",
        "",
        report_verdict_from_checks(fits, clip_delta),
        "",
        "A nonlinear-physics claim is credible only if the effect survives nominal-energy binning, survives clipping-off, lives in I_trans rather than I_inc, shows energy-selective reference-free curvature, and is compatible with the fluence scale.",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote diagnostics to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Limit shots per energy for a fast smoke run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override diagnostics output directory.")
    parser.add_argument("--focus-diameter-um", type=float, default=None, help="Set focus diameter for Check 8 fluence.")
    parser.add_argument(
        "--double-peak-robustness",
        action="store_true",
        help="Also run the expensive Check 7 with double-peak rejection enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = dict(PARAMS)
    if args.output_dir is not None:
        params["OUTPUT_DIR"] = args.output_dir
    if args.focus_diameter_um is not None:
        params["FOCUS_DIAMETER_UM"] = args.focus_diameter_um
    if args.double_peak_robustness:
        params["RUN_DOUBLE_PEAK_ROBUSTNESS"] = True
    run(params, quick=bool(args.quick))


if __name__ == "__main__":
    main()
