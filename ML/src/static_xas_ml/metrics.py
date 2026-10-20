"""Metrics for predicted incident spectra."""

from __future__ import annotations

import numpy as np

from .preprocessing import spectrum_moments


def spectral_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ML_XAS cosine similarity between true and predicted spectra."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    numerator = np.nansum(true * pred, axis=1)
    true_norm = np.sqrt(np.nansum(true * true, axis=1))
    pred_norm = np.sqrt(np.nansum(pred * pred, axis=1))
    denom = true_norm * pred_norm
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denom) & (denom > 0)
    out[valid] = numerator[valid] / denom[valid]
    return out


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
    true_integral = np.nansum(y_true, axis=1)
    pred_integral = np.nansum(y_pred, axis=1)
    integral_error = pred_integral - true_integral
    integral_rel_error = integral_error / true_integral
    integral_rel_error[~np.isfinite(integral_rel_error)] = np.nan

    c_true, w_true = spectrum_moments(pixel_axis, y_true)
    c_pred, w_pred = spectrum_moments(pixel_axis, y_pred)
    centroid_error = c_pred - c_true
    width_error = w_pred - w_true
    similarity = spectral_similarity(y_true, y_pred)

    return {
        "label": label,
        "n_shots": int(y_true.shape[0]),
        "mean_rmse": float(np.nanmean(rmse_per_shot)),
        "median_rmse": float(np.nanmedian(rmse_per_shot)),
        "mean_mae": float(np.nanmean(mae_per_shot)),
        "mean_integral_error": float(np.nanmean(integral_error)),
        "median_integral_error": float(np.nanmedian(integral_error)),
        "mean_integral_relative_error": float(np.nanmean(integral_rel_error)),
        "median_integral_relative_error": float(np.nanmedian(integral_rel_error)),
        "mean_centroid_error_px": float(np.nanmean(centroid_error)),
        "std_centroid_error_px": float(np.nanstd(centroid_error)),
        "mean_width_error_px": float(np.nanmean(width_error)),
        "std_width_error_px": float(np.nanstd(width_error)),
        "mean_similarity": float(np.nanmean(similarity)),
        "median_similarity": float(np.nanmedian(similarity)),
        "std_similarity": float(np.nanstd(similarity)),
        "min_similarity": float(np.nanmin(similarity)),
        "max_similarity": float(np.nanmax(similarity)),
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
    true_integral = np.nansum(y_true, axis=1)
    pred_integral = np.nansum(y_pred, axis=1)
    integral_error = pred_integral - true_integral
    integral_rel_error = integral_error / true_integral
    integral_rel_error[~np.isfinite(integral_rel_error)] = np.nan
    return {
        "rmse": np.sqrt(np.nanmean(err ** 2, axis=1)),
        "mae": np.nanmean(np.abs(err), axis=1),
        "integral_error": integral_error,
        "integral_relative_error": integral_rel_error,
        "centroid_error_px": c_pred - c_true,
        "width_error_px": w_pred - w_true,
        "similarity": spectral_similarity(y_true, y_pred),
    }
