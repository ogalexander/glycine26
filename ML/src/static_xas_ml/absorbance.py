"""Absorbance helpers for predicted virtual-I0 spectra."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shape_absorbance(
    transmitted_shape: np.ndarray,
    incident_shape: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Per-shot spectral absorbance: -log10(transmitted shape / incident shape)."""
    t = np.asarray(transmitted_shape, dtype=np.float32)
    i0 = np.asarray(incident_shape, dtype=np.float32)
    ratio = np.maximum(t, float(eps)) / np.maximum(i0, float(eps))
    return (-np.log10(ratio)).astype(np.float32)


def pixel_roi_mask(pixel_axis: np.ndarray, roi: tuple[float, float] | None) -> np.ndarray:
    pixels = np.asarray(pixel_axis, dtype=np.float64)
    if roi is None:
        return np.ones(pixels.shape, dtype=bool)
    lo, hi = float(roi[0]), float(roi[1])
    return (pixels >= lo) & (pixels < hi)


def scalar_absorbance(
    absorbance_pixel: np.ndarray,
    pixel_axis: np.ndarray,
    *,
    pixel_roi: tuple[float, float] | None = None,
) -> np.ndarray:
    mask = pixel_roi_mask(pixel_axis, pixel_roi)
    if not np.any(mask):
        raise ValueError(f"pixel_roi {pixel_roi} selects no pixels.")
    return np.nanmean(absorbance_pixel[:, mask], axis=1)


def _gmd_bin_labels(gmd_edges: np.ndarray) -> list[str]:
    edges = np.asarray(gmd_edges, dtype=float)
    return [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(edges.size - 1)]


def _gmd_bin_indices(gmd: np.ndarray, gmd_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(gmd_edges, dtype=float)
    values = np.asarray(gmd, dtype=float)
    gbin = np.searchsorted(edges, values, side="right") - 1
    n_gmd = edges.size - 1
    gbin[values == edges[-1]] = n_gmd - 1
    in_range = np.isfinite(values) & (gbin >= 0) & (gbin < n_gmd)
    return gbin.astype(int), in_range


def _integrated_mean_spectrum_intensity(
    spectra: np.ndarray,
    pixel_axis: np.ndarray,
    *,
    pixel_roi: tuple[float, float] | None = None,
    clip_negative: bool = True,
) -> float:
    mask = pixel_roi_mask(pixel_axis, pixel_roi)
    if not np.any(mask):
        raise ValueError(f"pixel_roi {pixel_roi} selects no pixels.")
    y = np.asarray(spectra, dtype=np.float64)
    if y.ndim == 1:
        mean_spectrum = y
    elif y.ndim == 2:
        finite = np.all(np.isfinite(y), axis=1)
        if not np.any(finite):
            return np.nan
        mean_spectrum = np.nanmean(y[finite], axis=0)
    else:
        raise ValueError("spectra must be 1D or 2D.")
    selected = mean_spectrum[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return np.nan
    if clip_negative:
        selected = np.maximum(selected, 0.0)
    return float(np.nansum(selected))


def _row_integrated_intensities(
    spectra: np.ndarray,
    pixel_axis: np.ndarray,
    *,
    pixel_roi: tuple[float, float] | None = None,
    clip_negative: bool = True,
) -> np.ndarray:
    mask = pixel_roi_mask(pixel_axis, pixel_roi)
    if not np.any(mask):
        raise ValueError(f"pixel_roi {pixel_roi} selects no pixels.")
    y = np.asarray(spectra, dtype=np.float64)
    selected = y[:, mask]
    if clip_negative:
        selected = np.maximum(selected, 0.0)
    return np.nansum(selected, axis=1)


def _absorbed_intensity_row(
    *,
    transmitted_spectra: np.ndarray,
    incident_spectra: np.ndarray,
    pixel_axis: np.ndarray,
    mask: np.ndarray,
    pixel_roi: tuple[float, float] | None,
    clip_negative: bool,
    eps: float,
) -> dict[str, float]:
    n = int(np.sum(mask))
    trans = transmitted_spectra[mask]
    inc = incident_spectra[mask]
    trans_intensity = _integrated_mean_spectrum_intensity(
        trans,
        pixel_axis,
        pixel_roi=pixel_roi,
        clip_negative=clip_negative,
    )
    incident_intensity = _integrated_mean_spectrum_intensity(
        inc,
        pixel_axis,
        pixel_roi=pixel_roi,
        clip_negative=clip_negative,
    )
    if (
        trans_intensity > 0
        and incident_intensity > 0
        and np.isfinite(trans_intensity)
        and np.isfinite(incident_intensity)
    ):
        absorbance = float(-np.log10(max(trans_intensity, eps) / max(incident_intensity, eps)))
    else:
        absorbance = np.nan

    sem = np.nan
    if n > 1 and np.isfinite(absorbance):
        trans_rows = _row_integrated_intensities(
            trans,
            pixel_axis,
            pixel_roi=pixel_roi,
            clip_negative=clip_negative,
        )
        inc_rows = _row_integrated_intensities(
            inc,
            pixel_axis,
            pixel_roi=pixel_roi,
            clip_negative=clip_negative,
        )
        trans_rows = trans_rows[np.isfinite(trans_rows)]
        inc_rows = inc_rows[np.isfinite(inc_rows)]
        if trans_rows.size > 1 and inc_rows.size > 1:
            trans_sem = np.nanstd(trans_rows, ddof=1) / np.sqrt(trans_rows.size)
            inc_sem = np.nanstd(inc_rows, ddof=1) / np.sqrt(inc_rows.size)
            if trans_intensity > 0 and incident_intensity > 0:
                sem = float(
                    np.sqrt((trans_sem / trans_intensity) ** 2 + (inc_sem / incident_intensity) ** 2)
                    / np.log(10.0)
                )

    return {
        "n_shots": n,
        "absorbance_mean": absorbance,
        "absorbance_sem": sem,
        "trans_intensity": float(trans_intensity),
        "incident_intensity": float(incident_intensity),
    }


def bin_integrated_absorbance_by_nominal(
    transmitted_spectra: np.ndarray,
    incident_spectra: np.ndarray,
    nominal_energy: np.ndarray,
    gmd: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    gmd_edges: np.ndarray,
    pixel_roi: tuple[float, float] | None = None,
    clip_negative: bool = True,
    eps: float = 1e-9,
    energy_decimals: int = 6,
) -> pd.DataFrame:
    """Bin spectra first, integrate mean spectra, then compute ML absorbance."""
    transmitted_spectra = np.asarray(transmitted_spectra, dtype=np.float64)
    incident_spectra = np.asarray(incident_spectra, dtype=np.float64)
    if transmitted_spectra.shape != incident_spectra.shape:
        raise ValueError("transmitted_spectra and incident_spectra must have the same shape.")

    nominal_key = np.round(np.asarray(nominal_energy, dtype=float), int(energy_decimals))
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin, in_gmd = _gmd_bin_indices(gmd, gmd_edges)
    labels = _gmd_bin_labels(gmd_edges)

    rows = []
    finite_rows = (
        np.isfinite(nominal_key)
        & in_gmd
        & np.all(np.isfinite(transmitted_spectra), axis=1)
        & np.all(np.isfinite(incident_spectra), axis=1)
    )
    for energy in np.unique(nominal_key[finite_rows]):
        for ibin, label in enumerate(labels):
            mask = finite_rows & (nominal_key == energy) & (gmd_bin == ibin)
            if not np.any(mask):
                continue
            row = _absorbed_intensity_row(
                transmitted_spectra=transmitted_spectra,
                incident_spectra=incident_spectra,
                pixel_axis=pixel_axis,
                mask=mask,
                pixel_roi=pixel_roi,
                clip_negative=clip_negative,
                eps=eps,
            )
            rows.append(
                {
                    "binning": "nominal_energy",
                    "nominal_energy_eV": float(energy),
                    "x": float(energy),
                    "x_label": "Nominal photon energy (eV)",
                    "gmd_bin": label,
                    "gmd_left": float(gmd_edges[ibin]),
                    "gmd_right": float(gmd_edges[ibin + 1]),
                    **row,
                }
            )
    return pd.DataFrame(rows).sort_values(["gmd_left", "x"]).reset_index(drop=True)


def bin_integrated_absorbance_by_energy(
    transmitted_spectra: np.ndarray,
    incident_spectra: np.ndarray,
    energy: np.ndarray,
    gmd: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    gmd_edges: np.ndarray,
    binning: str,
    x_label: str,
    energy_column: str,
    pixel_roi: tuple[float, float] | None = None,
    clip_negative: bool = True,
    eps: float = 1e-9,
    energy_decimals: int = 6,
) -> pd.DataFrame:
    """Bin spectra by an existing shot-energy coordinate, integrate, then compute absorbance."""
    transmitted_spectra = np.asarray(transmitted_spectra, dtype=np.float64)
    incident_spectra = np.asarray(incident_spectra, dtype=np.float64)
    if transmitted_spectra.shape != incident_spectra.shape:
        raise ValueError("transmitted_spectra and incident_spectra must have the same shape.")

    energy_key = np.round(np.asarray(energy, dtype=float), int(energy_decimals))
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin, in_gmd = _gmd_bin_indices(gmd, gmd_edges)
    labels = _gmd_bin_labels(gmd_edges)
    finite_rows = (
        np.isfinite(energy_key)
        & in_gmd
        & np.all(np.isfinite(transmitted_spectra), axis=1)
        & np.all(np.isfinite(incident_spectra), axis=1)
    )

    rows = []
    for energy_value in np.unique(energy_key[finite_rows]):
        for ibin, label in enumerate(labels):
            mask = finite_rows & (energy_key == energy_value) & (gmd_bin == ibin)
            if not np.any(mask):
                continue
            row = _absorbed_intensity_row(
                transmitted_spectra=transmitted_spectra,
                incident_spectra=incident_spectra,
                pixel_axis=pixel_axis,
                mask=mask,
                pixel_roi=pixel_roi,
                clip_negative=clip_negative,
                eps=eps,
            )
            rows.append(
                {
                    "binning": binning,
                    energy_column: float(energy_value),
                    "x": float(energy_value),
                    "x_label": x_label,
                    "gmd_bin": label,
                    "gmd_left": float(gmd_edges[ibin]),
                    "gmd_right": float(gmd_edges[ibin + 1]),
                    **row,
                }
            )
    return pd.DataFrame(rows).sort_values(["gmd_left", "x"]).reset_index(drop=True)


def baseline_slope_rows(
    df: pd.DataFrame,
    *,
    energy_range: tuple[float, float] = (271.0, 280.0),
    exclude_windows: tuple[tuple[float, float], ...] = ((272.7, 273.4), (274.6, 275.4)),
) -> list[dict[str, object]]:
    """Fit a simple pre-edge slope after excluding feature windows."""
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    lo, hi = energy_range
    base_mask = (df["x"] >= float(lo)) & (df["x"] <= float(hi))
    for wlo, whi in exclude_windows:
        base_mask &= ~((df["x"] >= float(wlo)) & (df["x"] <= float(whi)))
    for gmd_bin, group in df[base_mask].groupby("gmd_bin", sort=False):
        x = np.asarray(group["x"], dtype=float)
        y = np.asarray(group["absorbance_mean"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if np.sum(ok) < 2:
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept = np.polyfit(x[ok], y[ok], deg=1)
        rows.append(
            {
                "binning": str(group["binning"].iloc[0]),
                "gmd_bin": gmd_bin,
                "energy_min_eV": float(lo),
                "energy_max_eV": float(hi),
                "n_points": int(np.sum(ok)),
                "slope_absorbance_per_eV": float(slope),
                "intercept": float(intercept),
            }
        )
    return rows


def bin_absorbance_by_nominal(
    absorbance_pixel: np.ndarray,
    nominal_energy: np.ndarray,
    gmd: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    gmd_edges: np.ndarray,
    pixel_roi: tuple[float, float] | None = None,
    energy_decimals: int = 6,
) -> pd.DataFrame:
    """Mean scalar ML absorbance by nominal energy and GMD bin."""
    scalar = scalar_absorbance(absorbance_pixel, pixel_axis, pixel_roi=pixel_roi)
    nominal_key = np.round(np.asarray(nominal_energy, dtype=float), int(energy_decimals))
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin, in_gmd = _gmd_bin_indices(gmd, gmd_edges)
    labels = _gmd_bin_labels(gmd_edges)

    rows = []
    for energy in np.unique(nominal_key[np.isfinite(nominal_key)]):
        for ibin, label in enumerate(labels):
            mask = in_gmd & (nominal_key == energy) & (gmd_bin == ibin) & np.isfinite(scalar)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "binning": "nominal_energy",
                    "nominal_energy_eV": float(energy),
                    "x": float(energy),
                    "x_label": "Nominal photon energy (eV)",
                    "gmd_bin": label,
                    "gmd_left": float(gmd_edges[ibin]),
                    "gmd_right": float(gmd_edges[ibin + 1]),
                    "n_shots": int(mask.sum()),
                    "absorbance_mean": float(np.nanmean(scalar[mask])),
                    "absorbance_sem": float(np.nanstd(scalar[mask]) / np.sqrt(mask.sum())),
                }
            )
    return pd.DataFrame(rows).sort_values(["gmd_left", "x"]).reset_index(drop=True)


def centroid_pixel(pixel_axis: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixel_axis, dtype=np.float64)
    y = np.asarray(shapes, dtype=np.float64)
    area = np.nansum(y, axis=1)
    center = np.nansum(y * pixels[None, :], axis=1) / area
    center[area <= 0] = np.nan
    return center


def bin_absorbance_by_actual_pixel(
    absorbance_pixel: np.ndarray,
    transmitted_shape: np.ndarray,
    gmd: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    gmd_edges: np.ndarray,
    pixel_bin_width: float,
    pixel_to_energy,
    center_pixel: np.ndarray | None = None,
    pixel_roi: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Mean scalar ML absorbance by actual/proxy center pixel and GMD bin."""
    scalar = scalar_absorbance(absorbance_pixel, pixel_axis, pixel_roi=pixel_roi)
    if center_pixel is None:
        center = centroid_pixel(pixel_axis, transmitted_shape)
    else:
        center = np.asarray(center_pixel, dtype=np.float64)
        if center.shape[0] != scalar.shape[0]:
            raise ValueError("center_pixel must have one value per absorbance row.")
    finite_center = center[np.isfinite(center)]
    if finite_center.size == 0:
        return pd.DataFrame()
    width = float(pixel_bin_width)
    lo = np.floor(np.nanmin(finite_center) / width) * width
    hi = np.ceil(np.nanmax(finite_center) / width) * width + width
    pixel_edges = np.arange(lo, hi + 0.5 * width, width)
    pixel_bin = np.digitize(center, pixel_edges) - 1
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin, in_gmd = _gmd_bin_indices(gmd, gmd_edges)
    labels = _gmd_bin_labels(gmd_edges)

    rows = []
    for pbin in range(pixel_edges.size - 1):
        center_pixel = 0.5 * (pixel_edges[pbin] + pixel_edges[pbin + 1])
        try:
            energy = float(pixel_to_energy(np.asarray([center_pixel], dtype=float))[0])
        except Exception:
            energy = np.nan
        for ibin, label in enumerate(labels):
            mask = in_gmd & (pixel_bin == pbin) & (gmd_bin == ibin) & np.isfinite(scalar)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "binning": "actual_pixel",
                    "center_pixel": float(center_pixel),
                    "actual_energy_eV": energy,
                    "x": energy,
                    "x_label": "Converted actual photon energy (eV)",
                    "gmd_bin": label,
                    "gmd_left": float(gmd_edges[ibin]),
                    "gmd_right": float(gmd_edges[ibin + 1]),
                    "n_shots": int(mask.sum()),
                    "absorbance_mean": float(np.nanmean(scalar[mask])),
                    "absorbance_sem": float(np.nanstd(scalar[mask]) / np.sqrt(mask.sum())),
                }
            )
    return pd.DataFrame(rows).sort_values(["gmd_left", "x"]).reset_index(drop=True)


def bin_integrated_absorbance_by_actual_pixel(
    transmitted_spectra: np.ndarray,
    incident_spectra: np.ndarray,
    gmd: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    gmd_edges: np.ndarray,
    pixel_bin_width: float,
    pixel_to_energy,
    center_pixel: np.ndarray | None = None,
    pixel_roi: tuple[float, float] | None = None,
    clip_negative: bool = True,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Bin spectra by center pixel/GMD, integrate mean spectra, then compute absorbance."""
    transmitted_spectra = np.asarray(transmitted_spectra, dtype=np.float64)
    incident_spectra = np.asarray(incident_spectra, dtype=np.float64)
    if transmitted_spectra.shape != incident_spectra.shape:
        raise ValueError("transmitted_spectra and incident_spectra must have the same shape.")

    if center_pixel is None:
        center = centroid_pixel(pixel_axis, transmitted_spectra)
    else:
        center = np.asarray(center_pixel, dtype=np.float64)
        if center.shape[0] != transmitted_spectra.shape[0]:
            raise ValueError("center_pixel must have one value per spectrum row.")

    finite_center = center[np.isfinite(center)]
    if finite_center.size == 0:
        return pd.DataFrame()
    width = float(pixel_bin_width)
    lo = np.floor(np.nanmin(finite_center) / width) * width
    hi = np.ceil(np.nanmax(finite_center) / width) * width + width
    pixel_edges = np.arange(lo, hi + 0.5 * width, width)
    pixel_bin = np.digitize(center, pixel_edges) - 1
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin, in_gmd = _gmd_bin_indices(gmd, gmd_edges)
    labels = _gmd_bin_labels(gmd_edges)
    finite_rows = (
        np.isfinite(center)
        & in_gmd
        & np.all(np.isfinite(transmitted_spectra), axis=1)
        & np.all(np.isfinite(incident_spectra), axis=1)
    )

    rows = []
    for pbin in range(pixel_edges.size - 1):
        center_pixel_value = 0.5 * (pixel_edges[pbin] + pixel_edges[pbin + 1])
        try:
            energy = float(pixel_to_energy(np.asarray([center_pixel_value], dtype=float))[0])
        except Exception:
            energy = np.nan
        for ibin, label in enumerate(labels):
            mask = finite_rows & (pixel_bin == pbin) & (gmd_bin == ibin)
            if not np.any(mask):
                continue
            row = _absorbed_intensity_row(
                transmitted_spectra=transmitted_spectra,
                incident_spectra=incident_spectra,
                pixel_axis=pixel_axis,
                mask=mask,
                pixel_roi=pixel_roi,
                clip_negative=clip_negative,
                eps=eps,
            )
            rows.append(
                {
                    "binning": "actual_pixel",
                    "center_pixel": float(center_pixel_value),
                    "actual_energy_eV": energy,
                    "x": energy,
                    "x_label": "Converted actual photon energy (eV)",
                    "gmd_bin": label,
                    "gmd_left": float(gmd_edges[ibin]),
                    "gmd_right": float(gmd_edges[ibin + 1]),
                    **row,
                }
            )
    return pd.DataFrame(rows).sort_values(["gmd_left", "x"]).reset_index(drop=True)
