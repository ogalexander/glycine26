"""
TOF calibrations of the gas-phase spectrometers (beamtime 11024642). Times in ns.

Electrons:  KE(t) = (a / (t - t0))**2 + E0        [eV]
    from ``analysis/notebooks/etof_calibration_argon.ipynb``: argon run 60538
    (hv 275.52 eV nominal), Ar 2p / 3s / 3p photolines and L2,3MM Auger lines
    with the energies of D. Schwickert's thesis (Hamburg 2022, p. 48), all
    eight lines, weighted least squares: t0 = 14.7 +- 2.3 ns,
    a = 1225 +- 37 ns sqrt(eV) (L = 0.726 m), E0 = -0.6 +- 1.1 eV.
Ions:       m/q(t) = ((t - t0) / k)**2             [u/e]
    from ``analysis/notebooks/inspect_itof_argon.ipynb``: argon run 60538,
    Ar+ - Ar4+, N2+, O2+; t0 = -18.4 +- 7.9 ns, k = 481.4 +- 1.6 ns.

Spectra change variable with the Jacobian: a histogram in TOF bins becomes a
density per eV (per unit m/q) by dividing each bin by its width in eV (m/q),
``np.abs(np.diff(ke_of_tof(tof_edges)))``; second moments / covariances by the
product of the widths of both axes (:func:`per_unit`). The electron energy axis
runs backwards relative to TOF (fast = high energy).
"""

import numpy as np

__all__ = ["ETOF", "ITOF", "ke_of_tof", "tof_of_ke", "mq_of_tof", "tof_of_mq", "per_unit"]

ETOF = dict(t0=14.747516097742794, a=1224.7287482836796, E0=-0.6438657505042036)
ITOF = dict(t0=-18.41892520939264, k=481.36689120299553)


def ke_of_tof(t, t0=ETOF["t0"], a=ETOF["a"], E0=ETOF["E0"]):
    """Electron kinetic energy (eV) for a TOF (ns); NaN for t <= t0."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > t0, (a / (t - t0)) ** 2 + E0, np.nan)


def tof_of_ke(ke, t0=ETOF["t0"], a=ETOF["a"], E0=ETOF["E0"]):
    """Electron TOF (ns) for a kinetic energy (eV); NaN for ke <= E0."""
    ke = np.asarray(ke, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ke > E0, t0 + a / np.sqrt(ke - E0), np.nan)


def mq_of_tof(t, t0=ITOF["t0"], k=ITOF["k"]):
    """Ion mass-to-charge ratio (u/e) for a TOF (ns); NaN for t <= t0."""
    t = np.asarray(t, dtype=float)
    return np.where(t > t0, ((t - t0) / k) ** 2, np.nan)


def tof_of_mq(mq, t0=ITOF["t0"], k=ITOF["k"]):
    """Ion TOF (ns) for a mass-to-charge ratio (u/e)."""
    return t0 + k * np.sqrt(np.asarray(mq, dtype=float))


def per_unit(arr, *widths):
    """
    Divide the trailing axes of ``arr`` by bin widths (one 1D array per axis):
    counts per TOF bin -> per eV (per u/e); for a second moment pass both axes'
    widths, e.g. ``per_unit(DtC, dke, dmq)``.
    """
    out = np.asarray(arr, dtype=float)
    n = len(widths)
    for i, w in enumerate(widths):
        shape = [1] * out.ndim
        shape[out.ndim - n + i] = len(w)
        out = out / np.asarray(w, dtype=float).reshape(shape)
    return out
