"""Metrics for predicted incident spectral shapes."""

from __future__ import annotations

import numpy as np

from .preprocessing import spectrum_moments


def shape_error_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pixel_axis: np.ndarray,
    label: str,
) -> dict[str, float | str | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse_per_shot = np.sqrt(np.nanmean(err ** 2, axis=1))
    mae_per_shot = np.nanmean(np.abs(err), axis=1)

    c_true, w_true = spectrum_moments(pixel_axis, y_true)
    c_pred, w_pred = spectrum_moments(pixel_axis, y_pred)
    centroid_error = c_pred - c_true
    width_error = w_pred - w_true

    return {
        "label": label,
        "n_shots": int(y_true.shape[0]),
        "mean_rmse": float(np.nanmean(rmse_per_shot)),
        "median_rmse": float(np.nanmedian(rmse_per_shot)),
        "mean_mae": float(np.nanmean(mae_per_shot)),
        "mean_centroid_error_px": float(np.nanmean(centroid_error)),
        "std_centroid_error_px": float(np.nanstd(centroid_error)),
        "mean_width_error_px": float(np.nanmean(width_error)),
        "std_width_error_px": float(np.nanstd(width_error)),
    }


def per_shot_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pixel_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    c_true, w_true = spectrum_moments(pixel_axis, y_true)
    c_pred, w_pred = spectrum_moments(pixel_axis, y_pred)
    return {
        "rmse": np.sqrt(np.nanmean(err ** 2, axis=1)),
        "mae": np.nanmean(np.abs(err), axis=1),
        "centroid_error_px": c_pred - c_true,
        "width_error_px": w_pred - w_true,
    }
