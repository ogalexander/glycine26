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
    gmd_bin = np.digitize(np.asarray(gmd, dtype=float), gmd_edges) - 1
    labels = _gmd_bin_labels(gmd_edges)

    rows = []
    for energy in np.unique(nominal_key[np.isfinite(nominal_key)]):
        for ibin, label in enumerate(labels):
            mask = (nominal_key == energy) & (gmd_bin == ibin) & np.isfinite(scalar)
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
    pixel_roi: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Mean scalar ML absorbance by transmitted-shape centroid pixel and GMD bin."""
    scalar = scalar_absorbance(absorbance_pixel, pixel_axis, pixel_roi=pixel_roi)
    center = centroid_pixel(pixel_axis, transmitted_shape)
    finite_center = center[np.isfinite(center)]
    if finite_center.size == 0:
        return pd.DataFrame()
    width = float(pixel_bin_width)
    lo = np.floor(np.nanmin(finite_center) / width) * width
    hi = np.ceil(np.nanmax(finite_center) / width) * width + width
    pixel_edges = np.arange(lo, hi + 0.5 * width, width)
    pixel_bin = np.digitize(center, pixel_edges) - 1
    gmd_edges = np.asarray(gmd_edges, dtype=float)
    gmd_bin = np.digitize(np.asarray(gmd, dtype=float), gmd_edges) - 1
    labels = _gmd_bin_labels(gmd_edges)

    rows = []
    for pbin in range(pixel_edges.size - 1):
        center_pixel = 0.5 * (pixel_edges[pbin] + pixel_edges[pbin + 1])
        try:
            energy = float(pixel_to_energy(np.asarray([center_pixel], dtype=float))[0])
        except Exception:
            energy = np.nan
        for ibin, label in enumerate(labels):
            mask = (pixel_bin == pbin) & (gmd_bin == ibin) & np.isfinite(scalar)
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
