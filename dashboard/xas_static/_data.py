"""Data loading and GMD binning — both cached by (path, gmd_edges)."""
from __future__ import annotations

import h5py
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def load_h5(path: str):
    """Load all datasets from a static-XAS H5 file. Cached by path."""
    with h5py.File(path, "r") as f:
        return (
            f["vls"][...],
            f["gmd"][...],
            f["n_shots"][...],
            f["nominal_energies"][...],
            f["vls_pixels"][...],
            dict(f.attrs),
        )


@st.cache_data(show_spinner=False)
def bin_by_gmd(path: str, gmd_edges_tuple: tuple):
    """
    Bin per-shot data by GMD. Cached by (path, gmd_edges).

    Returns
    -------
    A_mean    : (N_E, N_GMD, n_pixels)
    G_mean    : (N_E, N_GMD)
    n_per_bin : (N_E, N_GMD)
    """
    vls, gmd, n_shots, _, _, _ = load_h5(path)
    gmd_edges = np.array(gmd_edges_tuple)
    N_E, _, n_pixels = vls.shape
    N_GMD = len(gmd_edges) - 1

    A_sum     = np.zeros((N_E, N_GMD, n_pixels), dtype=np.float64)
    G_sum     = np.zeros((N_E, N_GMD),           dtype=np.float64)
    n_per_bin = np.zeros((N_E, N_GMD),           dtype=np.int64)

    for ie in range(N_E):
        n = int(n_shots[ie])
        G = gmd[ie, :n]
        A = vls[ie, :n, :]
        finite  = np.isfinite(G) & np.all(np.isfinite(A), axis=1)
        bins    = np.digitize(G[finite], gmd_edges) - 1
        in_rng  = (bins >= 0) & (bins < N_GMD)
        A_f, G_f, bins_f = A[finite][in_rng], G[finite][in_rng], bins[in_rng]
        for b in np.unique(bins_f):
            m = bins_f == b
            n_per_bin[ie, b] += int(m.sum())
            G_sum[ie, b]     += G_f[m].sum()
            A_sum[ie, b]     += A_f[m].sum(axis=0)

    safe_n = np.where(n_per_bin > 0, n_per_bin, 1).astype(np.float64)
    G_mean = np.where(n_per_bin > 0, G_sum / safe_n, np.nan)
    A_mean = np.where(n_per_bin[:, :, None] > 0,
                      A_sum / safe_n[:, :, None], np.nan)
    return A_mean, G_mean, n_per_bin
