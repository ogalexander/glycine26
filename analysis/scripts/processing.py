import numpy as np
from typing import Optional, Tuple
from experiment_data import ExperimentData

def count_hits(
    tofs: np.ndarray,
    tof_min: Optional[float] = None,
    tof_max: Optional[float] = None,
) -> np.ndarray:
    """
    Count non-zero TOF hits per shot, optionally within a TOF range.

    Parameters
    ----------
    tofs : np.ndarray
        TOF hits array, shape (n, m, max_hits). Zero-padded.
        Units: ns.
    tof_min : float, optional
        Minimum TOF value to count (ns). If None, no lower bound.
    tof_max : float, optional
        Maximum TOF value to count (ns). If None, no upper bound.

    Returns
    -------
    counts : np.ndarray
        Hit counts per shot, shape (n, m).
    """
    nonzero = tofs != 0

    if tof_min is not None:
        nonzero = nonzero & (tofs >= tof_min)
    if tof_max is not None:
        nonzero = nonzero & (tofs <= tof_max)

    return nonzero.sum(axis=-1)  # (n, m)


def hits_to_spectrum(
    tofs: np.ndarray,
    tof_edges: np.ndarray,
) -> np.ndarray:
    """
    Convert a zero-padded TOF hits array into histogrammed spectra per shot.

    Uses np.apply_along_axis to histogram each shot's hits independently.

    Parameters
    ----------
    tofs : np.ndarray
        Zero-padded TOF hits, shape (n, m, max_hits). Units: ns.
    tof_edges : np.ndarray
        Bin edges for the TOF histogram, length n_bins+1. Units: ns.

    Returns
    -------
    spectra : np.ndarray
        Histogrammed spectra, shape (n, m, n_bins).
    """
    n_bins = len(tof_edges) - 1
    n, m, _ = tofs.shape

    def _hist_shot(shot_hits: np.ndarray) -> np.ndarray:
        # Exclude zero-padding before histogramming
        valid = shot_hits[shot_hits != 0]
        counts, _ = np.histogram(valid, bins=tof_edges)
        return counts.astype(float)

    # Reshape to (n*m, max_hits), apply, reshape back
    flat = tofs.reshape(-1, tofs.shape[-1])
    spectra_flat = np.apply_along_axis(_hist_shot, axis=1, arr=flat)  # (n*m, n_bins)
    return spectra_flat.reshape(n, m, n_bins)


def average_spectrum(
    tofs: np.ndarray,
    tof_edges: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the average spectrum over all (or masked) shots.

    Parameters
    ----------
    tofs : np.ndarray
        Zero-padded TOF hits, shape (n, m, max_hits). Units: ns.
    tof_edges : np.ndarray
        Bin edges for the TOF histogram, length n_bins+1. Units: ns.
    mask : np.ndarray, optional
        Boolean mask of shape (n, m). If None, all shots are used.

    Returns
    -------
    tof_cents : np.ndarray
        Bin centres, shape (n_bins,). Units: ns.
    mean_spectrum : np.ndarray
        Mean spectrum, shape (n_bins,).
    std_spectrum : np.ndarray
        Std of spectrum, shape (n_bins,).
    """
    tof_cents = (tof_edges[:-1] + tof_edges[1:]) / 2
    spectra = hits_to_spectrum(tofs, tof_edges)  # (n, m, n_bins)

    if mask is not None:
        flat_spectra = spectra[mask]  # (n_good, n_bins)
    else:
        flat_spectra = spectra.reshape(-1, spectra.shape[-1])

    return tof_cents, np.nanmean(flat_spectra, axis=0), np.nanstd(flat_spectra, axis=0)


def select_bunch_range(
    data: ExperimentData,
    bunch_start: int,
    bunch_end: int,
) -> dict:
    """
    Select a contiguous range of bunch indices and flatten trains × bunches
    into a single shot axis.

    Parameters
    ----------
    data : ExperimentData
        Loaded experiment data.
    bunch_start : int
        First bunch index to include (inclusive).
    bunch_end : int
        Last bunch index to include (inclusive).

    Returns
    -------
    flat : dict
        Dictionary of flattened arrays with keys matching ExperimentData fields.
        All arrays have shape (n * n_bunches_selected,...).
    """
    sl = slice(bunch_start, bunch_end + 1)

    flat = {
        "gmd":  data.gmd[:, sl].reshape(-1),
        "mpe":  data.mpe_broadcast[:, sl].reshape(-1),
        "z":    data.z[:, sl].reshape(-1),
        "tID":  np.repeat(data.tID, bunch_end - bunch_start + 1),
        "bunch_idx": np.tile(
            np.arange(bunch_start, bunch_end + 1), data.n_trains
        ),
    }

    if data.tofs_e is not None:
        flat["tofs_e"] = data.tofs_e[:, sl, :].reshape(-1, data.tofs_e.shape[-1])
    if data.tofs_i is not None:
        flat["tofs_i"] = data.tofs_i[:, sl, :].reshape(-1, data.tofs_i.shape[-1])
    if data.liq_tofs_e is not None:
        flat["liq_tofs_e"] = data.liq_tofs_e[:, sl, :].reshape(-1, data.liq_tofs_e.shape[-1])
    if data.vls is not None:
        flat["vls"] = data.vls[:, sl, :].reshape(-1, data.vls.shape[-1])

    return flat


def crop_to_roi(
    spectra: np.ndarray,
    roi_min: int,
    roi_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a spectrum array along its last (pixel) axis to a region of interest.

    Intended for VLS spectra of shape ``(n_trains, n_bunches, n_pixels)``,
    but works on any array whose last axis is the pixel axis — 1D
    ``(n_pixels,)``, 2D ``(n_shots, n_pixels)``, 3D, etc.

    Parameters
    ----------
    spectra : np.ndarray
        Spectrum array; pixel axis must be the last axis.
    roi_min : int
        First pixel index to keep (inclusive).
    roi_max : int
        Last pixel index to keep (exclusive — half-open like Python slicing).

    Returns
    -------
    cropped : np.ndarray
        ``spectra[..., roi_min:roi_max]``.
    pixel_ax : np.ndarray
        1D array of the original pixel indices kept, shape
        ``(roi_max - roi_min,)``. Use this as the x-axis when plotting so
        ticks reflect the source pixel numbers, not the cropped offsets.
    """
    n_pixels = spectra.shape[-1]
    if not 0 <= roi_min < roi_max <= n_pixels:
        raise ValueError(
            f"Invalid ROI ({roi_min}, {roi_max}) for pixel axis of length "
            f"{n_pixels}: require 0 <= roi_min < roi_max <= n_pixels."
        )
    cropped = spectra[..., roi_min:roi_max]
    pixel_ax = np.arange(roi_min, roi_max)
    return cropped, pixel_ax


def spectral_moments(
    spectra: np.ndarray,
    pixel_ax: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the sum, centre-of-mass, and width of each spectrum.

    Acts along the last axis. Works on any shape whose last axis is the
    pixel axis (e.g. ``(n_pixels,)``, ``(n_shots, n_pixels)``, or
    ``(n_trains, n_bunches, n_pixels)``).

    Parameters
    ----------
    spectra : np.ndarray
        Spectrum array; pixel axis must be the last axis.
    pixel_ax : np.ndarray
        1D pixel coordinate array, shape ``(n_pixels,)``. The COM and
        width are returned in these coordinates, so pass the
        source-pixel indices (e.g. ``ExperimentData.vls_pixels``) if you
        want results in the original-pixel frame.

    Returns
    -------
    sums : np.ndarray
        Integrated intensity per spectrum, shape ``spectra.shape[:-1]``.
    coms : np.ndarray
        Intensity-weighted centre of mass per spectrum, in ``pixel_ax``
        units. ``NaN`` where ``sum == 0``.
    widths : np.ndarray
        ``sqrt`` of the second central moment in ``pixel_ax`` units.
        Negative variances (possible with background-subtracted spectra
        whose net weight is small) are absolute-valued before the
        square root, so ``widths`` is always real and non-negative;
        treat large widths next to a near-zero sum with suspicion.

    Notes
    -----
    Background-subtracted spectra can have negative pixel values and
    near-zero sums in dark regions, which makes the COM/width
    interpretation noisy there. Filter by ``sums > threshold`` before
    histogramming if this matters.
    """
    sums = np.sum(spectra, axis=-1)
    safe = np.where(sums != 0, sums, np.nan)
    coms = np.sum(spectra * pixel_ax, axis=-1) / safe
    diff = pixel_ax - coms[..., None]
    var = np.sum(spectra * diff**2, axis=-1) / safe
    widths = np.sqrt(np.abs(var))
    return sums, coms, widths