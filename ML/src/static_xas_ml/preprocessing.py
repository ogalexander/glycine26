"""Spectrum preprocessing for virtual-I0 modeling."""

from __future__ import annotations

import numpy as np


def preprocess_spectra(
    spectra: np.ndarray,
    *,
    baseline_quantile: float | None = 0.05,
    clip_negative: bool = True,
    normalize_area: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optionally baseline-correct, clip, and area-normalize spectra row by row.

    Returns
    -------
    spectra_out, areas, baselines
        If ``normalize_area`` is True, ``spectra_out`` integrates to 1 over
        pixels for rows with positive area. Otherwise it is the corrected
        spectrum in the same intensity units as the input.
    """
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array (shots, pixels).")
    if baseline_quantile is None:
        baselines = np.zeros(spectra.shape[0], dtype=np.float32)
    else:
        baselines = np.nanquantile(spectra, float(baseline_quantile), axis=1)
    corrected = spectra - baselines[:, None]
    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)
    corrected = np.where(np.isfinite(corrected), corrected, 0.0)
    areas = np.sum(corrected, axis=1)
    if normalize_area:
        spectra_out = corrected / np.maximum(areas[:, None], float(eps))
        bad = areas <= float(eps)
        if np.any(bad):
            spectra_out[bad] = np.nan
    else:
        spectra_out = corrected
    return spectra_out.astype(np.float32), areas.astype(np.float32), baselines.astype(np.float32)


def renormalize_shapes(shapes: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    shapes = np.asarray(shapes, dtype=np.float32)
    shapes = np.clip(shapes, 0.0, None)
    area = np.sum(shapes, axis=1)
    out = shapes / np.maximum(area[:, None], float(eps))
    out[area <= float(eps)] = np.nan
    return out.astype(np.float32)


def spectrum_moments(pixel_axis: np.ndarray, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid and RMS width for area-normalized spectra."""
    x = np.asarray(pixel_axis, dtype=np.float64)
    y = np.asarray(shapes, dtype=np.float64)
    area = np.nansum(y, axis=1)
    centroid = np.nansum(y * x[None, :], axis=1) / area
    variance = np.nansum(y * (x[None, :] - centroid[:, None]) ** 2, axis=1) / area
    width = np.sqrt(np.maximum(variance, 0.0))
    centroid[area <= 0] = np.nan
    width[area <= 0] = np.nan
    return centroid, width
