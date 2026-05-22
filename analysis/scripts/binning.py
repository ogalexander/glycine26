import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Your original binning functions (unchanged, with added type hints/docstrings)
# ---------------------------------------------------------------------------

def arb_bool_ar(
    bin_edges: np.ndarray,
    bin_on: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a boolean mask array from arbitrary bin edges.

    Parameters
    ----------
    bin_edges : np.ndarray
        Array of bin edges, length N+1 for N bins.
    bin_on : np.ndarray
        1D array of values to bin, length M.

    Returns
    -------
    bin_cents : np.ndarray
        Bin centres, shape (N,).
    bool_ar : np.ndarray
        Boolean mask array, shape (N, M). bool_ar[i, j] is True if
        bin_on[j] falls in bin i.
    """
    lowers = bin_edges[:-1]
    uppers = bin_edges[1:]
    bin_cents = (lowers + uppers) / 2

    bool_ar = np.zeros((len(lowers), len(bin_on)), dtype=bool)
    for idx, (lower, upper) in enumerate(zip(lowers, uppers)):
        bool_ar[idx] = (bin_on >= lower) & (bin_on <= upper)

    return bin_cents, bool_ar


def bin_bool_ar(
    start: float,
    stop: float,
    n_bins: int,
    bin_on: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate boolean mask array using uniformly spaced bin edges.

    Parameters
    ----------
    start : float
        Left edge of the first bin.
    stop : float
        Right edge of the last bin.
    n_bins : int
        Number of bins.
    bin_on : np.ndarray
        1D array of values to bin.

    Returns
    -------
    bin_cents : np.ndarray
        Bin centres, shape (n_bins,).
    bool_ar : np.ndarray
        Boolean mask array, shape (n_bins, M).
    """
    bin_edges = np.linspace(start, stop, n_bins + 1)
    return arb_bool_ar(bin_edges, bin_on)


def bin_bool_ar_pc(
    start: float,
    stop: float,
    n_bins: int,
    bin_on: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate boolean mask array using percentile-based bin edges.

    Parameters
    ----------
    start : float
        Starting percentile (0–100).
    stop : float
        Ending percentile (0–100).
    n_bins : int
        Number of bins.
    bin_on : np.ndarray
        1D array of values to bin.

    Returns
    -------
    bin_cents : np.ndarray
        Bin centres (in data units), shape (n_bins,).
    bool_ar : np.ndarray
        Boolean mask array, shape (n_bins, M).
    """
    bin_edges_pc = np.linspace(start, stop, n_bins + 1)
    bin_edges = np.array(
        [np.nanpercentile(bin_on, p) for p in bin_edges_pc]
    )
    return arb_bool_ar(bin_edges, bin_on)


def bin_and_average(
    bool_ar: np.ndarray,
    to_average: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average data within bins defined by a boolean mask array.

    Parameters
    ----------
    bool_ar : np.ndarray
        Boolean mask array of shape (N, M) from arb_bool_ar / bin_bool_ar.
    to_average : np.ndarray
        Data to average, shape (M,) or (M,...).

    Returns
    -------
    binned : np.ndarray
        Mean of to_average in each bin, shape (N,) or (N,...).
    binned_std : np.ndarray
        Std of to_average in each bin, shape (N,) or (N,...).
    binned_n : np.ndarray
        Number of shots in each bin, shape (N,), dtype int.
    """
    n_bin = bool_ar.shape[0]

    if to_average.ndim == 1:
        binned     = np.empty(n_bin)
        binned_std = np.empty(n_bin)
    else:
        binned     = np.empty((n_bin, *to_average.shape[1:]))
        binned_std = np.empty((n_bin, *to_average.shape[1:]))

    binned_n = np.empty(n_bin, dtype=int)

    for idx, mask in enumerate(bool_ar):
        masked = to_average[mask]
        binned[idx]     = np.nanmean(masked, axis=0)
        binned_std[idx] = np.nanstd(masked, axis=0)
        binned_n[idx]   = mask.sum()

    return binned, binned_std, binned_n


def bin_and_sum_ratio(
    bool_ar: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sum(numerator) / sum(denominator) within each bin.

    This is the energy-normalised averaging: rather than mean(x/y),
    compute sum(x) / sum(y) per bin.

    Parameters
    ----------
    bool_ar : np.ndarray
        Boolean mask array, shape (N, M).
    numerator : np.ndarray
        Numerator data, shape (M,) or (M, K).
    denominator : np.ndarray
        Denominator data, shape (M,). Must be 1D.

    Returns
    -------
    ratio : np.ndarray
        sum(numerator) / sum(denominator) per bin, shape (N,) or (N, K).
    binned_n : np.ndarray
        Number of shots per bin, shape (N,).
    """
    n_bin = bool_ar.shape[0]

    if numerator.ndim == 1:
        ratio = np.empty(n_bin)
    else:
        ratio = np.empty((n_bin, *numerator.shape[1:]))

    binned_n = np.empty(n_bin, dtype=int)

    for idx, mask in enumerate(bool_ar):
        num = numerator[mask]
        den = denominator[mask]
        den_sum = np.nansum(den)
        if den_sum == 0:
            ratio[idx] = np.nan
        else:
            ratio[idx] = np.nansum(num, axis=0) / den_sum
        binned_n[idx] = mask.sum()

    return ratio, binned_n