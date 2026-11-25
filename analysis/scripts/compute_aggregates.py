"""
Compute binned matrix aggregates for covariance / ADMM ghost imaging.

For each GMD bin, accumulates over all kept shots:

    <A>   = (1/N) Σ Aᵢ                 mean VLS spectrum
    <AtA> = (1/N) Σ Aᵢ Aᵢᵀ
    <D>   = (1/N) Σ Dᵢ                 mean binned eTOF spectrum
    <DtD> = (1/N) Σ Dᵢ Dᵢᵀ
    <AtD> = (1/N) Σ Aᵢ Dᵢᵀ

The H5 file is iterated in chunks of trains so the full
``(n_shots, n_pixels)`` and ``(n_shots, n_tof)`` matrices never live in
memory; only the chunk and the small per-bin aggregates do.

Config file
-----------
A Python module exposing the module-level names below. Example file:
``analysis/configs/aggregates_example.py``.

    CROP_ROI    : (roi_min, roi_max) half-open VLS pixel ROI
    BACKGROUND  : dict
        {"type": "auto",  "roi": (bunch_start, bunch_end)}   or
        {"type": "array", "path": "background.npy"}          or
        {"type": "none"}
    TOF_EDGES   : 1D array-like of TOF bin edges (100 ps)
    GMD_EDGES   : 1D array-like of GMD bin edges (µJ); shots outside
                  the first/last edges are dropped. May be None when
                  GMD_PERCENTILE_BINS is given.
    GMD_PERCENTILE_BINS : int, optional - GMD edges at equally spaced
                  percentiles (0-100) of the shots that are binned
                  (equal shot counts per GMD bin overall)
    GMD_MIN     : float, optional - with GMD_PERCENTILE_BINS, shots below
                  this GMD (noise floor, negative readings) are left out
    VLS_CROP_ROI : (lo, hi), config 1 - also aggregate the VLS (A, AtA,
                  AtD, AtC, AtG) over this half-open pixel range. Per shot:
                  minus the background (VLS_BACKGROUND, default the mean of
                  the dark bunch slot 100 = {"type": "auto", "roi": (100, 101)},
                  over every VLS_BG_STRIDE-th kept train), minus the median
                  over VLS_CM_PIXELS (common mode, optional), VLS_BAD_PIXELS
                  set to 0, then cropped. Shots without a frame are skipped.
                  VLS_RAW_RUN = N reads the frames from run N's raw files
                  (combined files written without VLS; VLS_OFFSET as in
                  write_h5.py, default config.VLS_OFFSET).
    MODE        : "spectral" (default) | "time_resolved" (Z_EDGES) |
                  "energy" (config 1: bins on a per-shot photon energy,
                  ENERGY_EDGES, read from ENERGY_KEY in the input H5,
                  default "photon_energy/measured" from photon_energy.py)
    CONFIG      : 1 or 2  (H5 layout — picks tofs_e vs liq_tofs_e)
    CHUNK_SIZE  : trains per chunk         (optional, default 200)
    BUNCH_RANGE : (bunch_start, bunch_end) half-open bunch slice that is
                  aggregated (optional, default all bunches)
    MAX_HITS_E  : drop shots with more electron hits than this (counted
                  over the whole TOF range; optional, default no cut)
    MAX_HITS_I  : same for ion hits (config 1; optional)
    TRIM_START  : trains trimmed from the start (optional, 0)
    TRIM_END    : trains trimmed from the end   (optional, 0)
    WORKERS     : config 1, worker processes (optional, 1; --workers on the CLI overrides)
    OUTPUT_SUFFIX : str appended to the default output filename
                    (optional, default ""). Inserted before ".h5", so
                    a value of "_v2" turns the default
                    "<input>_aggregates.h5" into
                    "<input>_aggregates_v2.h5". Ignored when ``-o`` is
                    passed on the CLI.

CLI
---
    python compute_aggregates.py CONFIG.py INPUT.h5 [-o OUT.h5]

Output H5 layout (means per bin):
    /A          (N_GMD, n_pixels)
    /AtA        (N_GMD, n_pixels, n_pixels)
    /D          (N_GMD, n_tof)
    /DtD        (N_GMD, n_tof, n_tof)
    /AtD        (N_GMD, n_pixels, n_tof)
    /n_per_bin  (N_GMD,)              int64
    /gmd_edges  (N_GMD+1,)
    /tof_edges  (n_tof+1,)
    /vls_pixels (n_pixels,)           source-pixel indices
    /background (n_pixels,)           background that was subtracted
    attrs:      provenance + summary counts
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np


__all__ = [
    "AggregatesData",
    "compute_aggregates",
    "load_aggregates",
    "load_python_config",
    "main",
]


# ----------------------------------------------------------------------
# Public data container + loader
# ----------------------------------------------------------------------

@dataclass
class AggregatesData:
    """
    Container for the binned aggregates produced by
    :func:`compute_aggregates`. All matrix fields are *means* over the
    shots that fell into each bin (empty bins give NaN).

    The shapes and which fields are populated depend on the
    experiment configuration (``config`` attribute in :attr:`metadata`):

    Config 2 — VLS + liquid-jet eTOF  (A, D, plus G aggregates):
        A    (N_GMD[, N_Z], n_pixels)
        AtA  (N_GMD[, N_Z], n_pixels, n_pixels)
        D    (N_GMD[, N_Z], n_tof)
        DtD  (N_GMD[, N_Z], n_tof, n_tof)
        AtD  (N_GMD[, N_Z], n_pixels, n_tof)
        AtG  (N_GMD[, N_Z], n_pixels)
        DtG  (N_GMD[, N_Z], n_tof)

    Config 1 — electron + ion TOF (D = eTOF, C = ion TOF):
        D    (N_GMD[, N_Z], n_tof)
        DtD  (N_GMD[, N_Z], n_tof, n_tof)
        C    (N_GMD[, N_Z], n_tof_i)
        CtC  (N_GMD[, N_Z], n_tof_i, n_tof_i)
        DtC  (N_GMD[, N_Z], n_tof, n_tof_i)
        DtG  (N_GMD[, N_Z], n_tof)
        CtG  (N_GMD[, N_Z], n_tof_i)
      plus, with VLS_CROP_ROI, the VLS (A = cropped, background-subtracted
      Gotthard spectrum): A, AtA, AtD, AtC (n_pixels, n_tof_i), AtG,
      vls_pixels, background.

    Both configs always populate the GMD aggregates and bookkeeping:
        G          (per-bin mean GMD, scalar per bin)
        GtG        (per-bin mean GMD², for Var(G))
        n_per_bin  (shot counts per bin)
        gmd_edges, tof_edges, metadata

    Time-resolved mode: an extra ``N_Z`` axis is inserted before the
    per-shot dimensions and ``z_edges`` is populated.

    Energy mode (config 1): bins are ``(N_E, N_GMD)`` on a per-shot photon
    energy (e.g. the VLS-measured one, ``photon_energy/measured``) and
    GMD; every aggregate has these two leading axes and ``energy_edges``
    is populated.

    * Config 2: ``tof_edges`` collapses to ``[tof_roi_min, tof_roi_max]``
      (``n_tof = 1``) — the eTOF is reduced per shot to a scalar count.
    * Config 1: ``tof_edges`` and ``ion_tof_edges`` keep their full
      spectral resolution (just with an extra ``N_Z`` axis on every
      aggregate). No ``tof_roi`` is used.

    XAS-scan mode (config 2, written by ``compute_xas_aggregates.py``):
    bins are ``(N_E, N_GMD)`` indexed by per-section nominal photon
    energy and GMD. Only the VLS aggregates (``A``, ``AtA``, ``AtG``)
    and GMD aggregates (``G``, ``GtG``, ``n_per_bin``) are populated;
    eTOF fields (``D``, ``DtD``, ``DtG``, ``tof_edges``) are ``None``.
    ``nominal_energies`` carries the (N_E,) energy axis in eV.
    """
    # Bookkeeping (always present).
    G: np.ndarray
    GtG: np.ndarray
    n_per_bin: np.ndarray
    gmd_edges: np.ndarray
    # eTOF (spectral / time-resolved modes; None in xas_scan).
    D: Optional[np.ndarray] = None
    DtD: Optional[np.ndarray] = None
    DtG: Optional[np.ndarray] = None
    tof_edges: Optional[np.ndarray] = None
    # Config 2 only.
    A: Optional[np.ndarray] = None
    AtA: Optional[np.ndarray] = None
    AtD: Optional[np.ndarray] = None
    AtG: Optional[np.ndarray] = None
    AtC: Optional[np.ndarray] = None          # config 1 with VLS
    vls_pixels: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    # Config 1 only.
    C: Optional[np.ndarray] = None
    CtC: Optional[np.ndarray] = None
    DtC: Optional[np.ndarray] = None
    CtG: Optional[np.ndarray] = None
    ion_tof_edges: Optional[np.ndarray] = None
    # TR mode (config 2).
    z_edges: Optional[np.ndarray] = None
    # XAS-scan mode (config 2).
    nominal_energies: Optional[np.ndarray] = None
    # Energy mode (config 1): per-shot photon-energy bin edges (eV).
    energy_edges: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    @property
    def config(self) -> int:
        return int(self.metadata.get("config", 2))

    @property
    def mode(self) -> str:
        if self.energy_edges is not None:
            return "energy"
        if self.nominal_energies is not None:
            return "xas_scan"
        if self.z_edges is not None:
            return "time_resolved"
        return "spectral"

    @property
    def n_gmd_bins(self) -> int:
        return int(self.gmd_edges.shape[0]) - 1

    @property
    def n_z_bins(self) -> Optional[int]:
        return None if self.z_edges is None else int(self.z_edges.shape[0]) - 1

    @property
    def n_energy_bins(self) -> Optional[int]:
        if self.energy_edges is not None:
            return int(self.energy_edges.shape[0]) - 1
        return None if self.nominal_energies is None else int(self.nominal_energies.shape[0])

    @property
    def energy_centres(self) -> Optional[np.ndarray]:
        if self.energy_edges is None:
            return None
        return 0.5 * (self.energy_edges[:-1] + self.energy_edges[1:])

    @property
    def n_pixels(self) -> Optional[int]:
        if self.vls_pixels is None:
            return None
        return int(self.vls_pixels.shape[0])

    @property
    def n_tof(self) -> Optional[int]:
        if self.tof_edges is None:
            return None
        return int(self.tof_edges.shape[0]) - 1

    @property
    def n_tof_i(self) -> Optional[int]:
        if self.ion_tof_edges is None:
            return None
        return int(self.ion_tof_edges.shape[0]) - 1

    @property
    def gmd_centres(self) -> np.ndarray:
        return 0.5 * (self.gmd_edges[:-1] + self.gmd_edges[1:])

    @property
    def z_centres(self) -> Optional[np.ndarray]:
        if self.z_edges is None:
            return None
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def tof_centres(self) -> Optional[np.ndarray]:
        if self.tof_edges is None:
            return None
        return 0.5 * (self.tof_edges[:-1] + self.tof_edges[1:])

    @property
    def ion_tof_centres(self) -> Optional[np.ndarray]:
        if self.ion_tof_edges is None:
            return None
        return 0.5 * (self.ion_tof_edges[:-1] + self.ion_tof_edges[1:])

    @property
    def var_G(self) -> np.ndarray:
        """Per-bin variance of G: ``<G²> − <G>²``. Useful for partial
        covariances against GMD."""
        return self.GtG - self.G ** 2


def load_aggregates(path: Union[str, Path]) -> AggregatesData:
    """
    Load aggregates produced by :func:`compute_aggregates`. The mode is
    inferred from whether ``z_edges`` is present in the file.

    Parameters
    ----------
    path : str or Path
        Path to the aggregates H5 file.

    Returns
    -------
    AggregatesData
        All aggregate arrays + metadata.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        meta = {}
        for k in f.attrs:
            v = f.attrs[k]
            meta[k] = v.tolist() if isinstance(v, np.ndarray) else v

        def _opt(k):
            return f[k][:] if k in f else None

        return AggregatesData(
            # always present
            G=f["G"][:],
            GtG=f["GtG"][:],
            n_per_bin=f["n_per_bin"][:],
            gmd_edges=f["gmd_edges"][:],
            # eTOF (spectral / TR modes)
            D=_opt("D"),
            DtD=_opt("DtD"),
            DtG=_opt("DtG"),
            tof_edges=_opt("tof_edges"),
            # config 2
            A=_opt("A"),
            AtA=_opt("AtA"),
            AtD=_opt("AtD"),
            AtG=_opt("AtG"),
            AtC=_opt("AtC"),
            vls_pixels=_opt("vls_pixels"),
            background=_opt("background"),
            # config 1
            C=_opt("C"),
            CtC=_opt("CtC"),
            DtC=_opt("DtC"),
            CtG=_opt("CtG"),
            ion_tof_edges=_opt("ion_tof_edges"),
            # TR mode
            z_edges=_opt("z_edges"),
            # XAS-scan mode
            nominal_energies=_opt("nominal_energies"),
            # Energy mode
            energy_edges=_opt("energy_edges"),
            metadata=meta,
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_python_config(path: Path):
    spec = importlib.util.spec_from_file_location("aggregates_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Public alias for use by sibling scripts (compute_xas_aggregates.py).
load_python_config = _load_python_config


def _iter_chunks(n_total: int, chunk_size: int):
    for s in range(0, n_total, chunk_size):
        yield s, min(s + chunk_size, n_total)


def _build_keep_mask(
    f: h5py.File, trim_start: int, trim_end: int
) -> np.ndarray:
    """
    Boolean mask of length ``n_total``: True for trains that pass the
    bad-train filter (``between_tdc_files == False``) and are not in
    the first ``trim_start`` or last ``trim_end`` of the surviving set.
    """
    btf = f["between_tdc_files"][:].astype(bool)
    n_total = btf.shape[0]
    good_pos = np.where(~btf)[0]
    if trim_end > 0:
        good_pos = good_pos[trim_start:-trim_end]
    else:
        good_pos = good_pos[trim_start:]
    keep_mask = np.zeros(n_total, dtype=bool)
    keep_mask[good_pos] = True
    return keep_mask


def _histogram_tof_chunk(
    tofs: np.ndarray, tof_edges: np.ndarray
) -> np.ndarray:
    """
    Convert ``(n, m, max_hits)`` zero-padded TOF hits to ``(n, m, n_bins)``
    histograms via a single ``np.histogram2d`` call. Zero entries are
    treated as padding and excluded.
    """
    n, m, _ = tofs.shape
    n_bins = len(tof_edges) - 1
    valid = tofs != 0
    if not valid.any():
        return np.zeros((n, m, n_bins), dtype=np.float64)
    tof_h = tofs[valid]
    train_of_hit = np.broadcast_to(
        np.arange(n)[:, None, None], tofs.shape
    )[valid]
    bunch_of_hit = np.broadcast_to(
        np.arange(m)[None, :, None], tofs.shape
    )[valid]
    shot_id = train_of_hit * m + bunch_of_hit
    H, _, _ = np.histogram2d(
        shot_id, tof_h,
        bins=[np.arange(n * m + 1) - 0.5, tof_edges],
    )
    return H.reshape(n, m, n_bins).astype(np.float64)


def _compute_auto_background(
    f: h5py.File,
    keep_mask: np.ndarray,
    bg_start: int,
    bg_end: int,
    roi_min: int,
    roi_max: int,
    chunk_size: int,
) -> np.ndarray:
    """
    Stream-average VLS over (kept trains, bunches in
    ``[bg_start, bg_end)``) within the cropped pixel ROI. Returns a
    ``(roi_max - roi_min,)`` float64 spectrum.
    """
    accum = np.zeros(roi_max - roi_min, dtype=np.float64)
    count = 0
    for s, e in _iter_chunks(keep_mask.shape[0], chunk_size):
        sub_mask = keep_mask[s:e]
        if not sub_mask.any():
            continue
        block = f["vls"][s:e, bg_start:bg_end, roi_min:roi_max][sub_mask]
        accum += block.sum(axis=(0, 1), dtype=np.float64)
        count += block.shape[0] * block.shape[1]
    if count == 0:
        raise RuntimeError("no shots available for auto background")
    return accum / count


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def _verify_h5_for_config(f: h5py.File, config: int) -> None:
    """
    Sanity-check that the H5 file actually contains the per-shot
    datasets that match the ``config`` declared in the config module.
    """
    keys = set(f.keys())
    if config == 1:
        # vls is a legitimate (optional) part of config 1 files; the
        # config 1 aggregates don't use it yet.
        required = {"gmd", "tofs_e", "tofs_i", "between_tdc_files", "tID"}
        wrong   = {"liq_tofs_e"} & keys
    elif config == 2:
        required = {"gmd", "vls", "liq_tofs_e", "between_tdc_files", "tID"}
        wrong   = {"tofs_e", "tofs_i"} & keys
    else:
        raise ValueError(f"config must be 1 or 2, got {config}")
    missing = required - keys
    if missing:
        raise RuntimeError(
            f"CONFIG = {config} but the H5 file is missing the expected "
            f"datasets {sorted(missing)}. Check the config file."
        )
    if wrong:
        print(f"warning: CONFIG = {config} but the H5 file also contains "
              f"{sorted(wrong)}; ignoring those datasets.")


# ----------------------------------------------------------------------
# Config 1: sparse accumulation (hit lists), optionally in worker processes
# ----------------------------------------------------------------------

def _hit_list(tofs, edges):
    """
    Shot index (row-major over the first two axes) and bin of every hit of a
    zero-padded ``(n, m, max_hits)`` TOF block that falls inside ``edges``,
    with the bin rules of ``np.histogram2d`` (``[e_i, e_i+1)``, last bin
    closed). Shot indices come out ascending.
    """
    m = tofs.shape[1]
    nz = np.nonzero(tofs)
    t = tofs[nz].astype(np.float64)
    j = np.searchsorted(edges, t, side="right") - 1
    j[t == edges[-1]] = len(edges) - 2
    inside = (j >= 0) & (j < len(edges) - 1)
    return (nz[0] * m + nz[1])[inside], j[inside]


def _pairs_within_shot(qa, qb):
    """
    Indices (ia, ib) of every pair of an a-hit and a b-hit in the same shot
    (``qa``, ``qb`` ascending shot indices; self-pairs included when a and b
    are the same list).
    """
    if len(qa) == 0 or len(qb) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    uq, b_start, b_cnt = np.unique(qb, return_index=True, return_counts=True)
    pos = np.searchsorted(uq, qa)
    pos_c = np.minimum(pos, len(uq) - 1)
    a_idx = np.flatnonzero((pos < len(uq)) & (uq[pos_c] == qa))
    cnt = b_cnt[pos_c[a_idx]]
    ia = np.repeat(a_idx, cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ib = np.repeat(b_start[pos_c[a_idx]], cnt) + offs
    return ia, ib


def _group_sum(keys, rows):
    """Sum the rows of ``rows`` per distinct key: (unique keys, (n_keys, ...) sums)."""
    o = np.argsort(keys, kind="stable")
    uk, st = np.unique(keys[o], return_index=True)
    return uk, np.add.reduceat(rows[o], st, axis=0)


def _vls_preprocess_block(x, b0, b1, roi_min, roi_max, ped, cm_px, bad_in_roi):
    """(n, m, n_px) raw frames -> (n, b1-b0, n_roi) float64 spectra and frame flags (as in compute_aggregates)."""
    x = x[:, b0:b1]
    ok = np.isfinite(x[:, :, roi_min])
    a = x[:, :, roi_min:roi_max].astype(np.float64) - ped[roi_min:roi_max]
    if cm_px is not None:
        a -= np.median(x[:, :, cm_px] - ped[cm_px], axis=-1, keepdims=True)
    a[..., bad_in_roi] = 0.0
    a[~ok] = 0.0
    return a, ok


def _accumulate_cfg1(task):
    """
    Worker: config-1 sums over the chunks ``task["chunks"]`` of
    ``task["input_h5"]``. D / C enter as hit lists, so every hit product is a
    scatter-add of the hits (pairs) actually present; only the VLS products
    ``AtA`` stay dense. Returns the sums (not means) and the skip counters.
    """
    import h5py as _h5py
    p = task
    n_cells, n_tof, n_tof_i = p["n_cells"], p["n_tof"], p["n_tof_i"]
    vls_on, n_px = p["vls_on"], p["n_pixels"]
    out = {"n": np.zeros(n_cells, np.int64), "G": np.zeros(n_cells), "GtG": np.zeros(n_cells),
           "D": np.zeros(n_cells * n_tof), "DtG": np.zeros(n_cells * n_tof),
           "DtD": np.zeros(n_cells * n_tof * n_tof),
           "C": np.zeros(n_cells * n_tof_i), "CtG": np.zeros(n_cells * n_tof_i),
           "CtC": np.zeros(n_cells * n_tof_i * n_tof_i), "DtC": np.zeros(n_cells * n_tof * n_tof_i)}
    if vls_on:
        out.update(A=np.zeros((n_cells, n_px)), AtG=np.zeros((n_cells, n_px)),
                   AtA=np.zeros((n_cells, n_px, n_px)), AtD=np.zeros((n_cells, n_px, n_tof)),
                   AtC=np.zeros((n_cells, n_px, n_tof_i)))
    cnt = dict.fromkeys(("gmd_nan", "gmd_oor", "max_hits", "z_nan", "z_oor", "e_nan", "e_oor", "vls"), 0)
    b0, b1 = p["b0"], p["b1"]
    mode, n_gmd = p["mode"], p["n_gmd"]
    keep_mask = p["keep_mask"]
    reader = None
    with _h5py.File(p["input_h5"], "r") as f:
        if vls_on and p["vls_raw_run"] is not None:
            import data_loading
            reader = data_loading.RawVLSReader(p["vls_raw_run"], f["tID"][...], f["gmd"].shape[1],
                                               vls_offset=p["vls_offset"])
        for s, e in p["chunks"]:
            sub = keep_mask[s:e]
            if not sub.any():
                continue
            gmd_block = f["gmd"][s:e, b0:b1][sub]
            te = f["tofs_e"][s:e, b0:b1][sub]
            ti = f["tofs_i"][s:e, b0:b1][sub]
            if mode == "time_resolved":
                z_block = f["z"][s:e, b0:b1][sub]
            if mode == "energy":
                e_block = f[p["energy_key"]][s:e, b0:b1][sub]
            if vls_on:
                rows = np.arange(s, e)[sub]
                if reader is not None:
                    frames = reader.read(rows)
                elif rows[-1] - rows[0] + 1 == len(rows):
                    frames = f["vls"][rows[0]:rows[-1] + 1]
                else:
                    frames = f["vls"][rows]
                A_block, vls_ok = _vls_preprocess_block(frames, b0, b1, p["roi_min"], p["roi_max"],
                                                        p["vls_ped"], p["cm_px"], p["bad_in_roi"])
                del frames

            # per-shot cell and validity (same rules as the dense loop)
            g = gmd_block.ravel()
            gbin = np.digitize(g, p["gmd_edges"]) - 1
            gnan = ~np.isfinite(g)
            goor = ~gnan & ((gbin < 0) | (gbin >= n_gmd))
            valid = ~gnan & ~goor
            cnt["gmd_nan"] += int(gnan.sum())
            cnt["gmd_oor"] += int(goor.sum())
            too_many = np.zeros_like(valid)
            if p["max_hits_e"] is not None:
                too_many |= (te != 0).sum(axis=-1).ravel() > int(p["max_hits_e"])
            if p["max_hits_i"] is not None:
                too_many |= (ti != 0).sum(axis=-1).ravel() > int(p["max_hits_i"])
            cnt["max_hits"] += int((too_many & valid).sum())
            valid &= ~too_many
            if vls_on:
                cnt["vls"] += int((valid & ~vls_ok.ravel()).sum())
                valid &= vls_ok.ravel()
            if mode == "spectral":
                cell = gbin
            elif mode == "energy":
                ev = e_block.ravel()
                ebin = np.digitize(ev, p["e_edges"]) - 1
                enan = ~np.isfinite(ev)
                eoor = ~enan & ((ebin < 0) | (ebin >= p["n_e"]))
                cnt["e_nan"] += int((enan & valid).sum())
                cnt["e_oor"] += int((eoor & valid).sum())
                valid &= ~enan & ~eoor
                cell = ebin * n_gmd + gbin
            else:
                zv = z_block.ravel()
                zbin = np.digitize(zv, p["z_edges"]) - 1
                znan = ~np.isfinite(zv)
                zoor = ~znan & ((zbin < 0) | (zbin >= p["n_z"]))
                cnt["z_nan"] += int(znan.sum())
                cnt["z_oor"] += int(zoor.sum())
                valid &= ~znan & ~zoor
                cell = gbin * p["n_z"] + zbin

            # shots
            vq = np.flatnonzero(valid)
            cv, gv = cell[vq], g[vq]
            np.add.at(out["n"], cv, 1)
            np.add.at(out["G"], cv, gv)
            np.add.at(out["GtG"], cv, gv * gv)

            # hits of valid shots
            qe, je = _hit_list(te, p["tof_edges"])
            qi, ji = _hit_list(ti, p["ion_tof_edges"])
            ke, ki = valid[qe], valid[qi]
            qe, je, qi, ji = qe[ke], je[ke], qi[ki], ji[ki]
            ce, ci = cell[qe], cell[qi]
            np.add.at(out["D"], ce * n_tof + je, 1.0)
            np.add.at(out["DtG"], ce * n_tof + je, g[qe])
            np.add.at(out["C"], ci * n_tof_i + ji, 1.0)
            np.add.at(out["CtG"], ci * n_tof_i + ji, g[qi])
            a, b = _pairs_within_shot(qe, qe)
            np.add.at(out["DtD"], (ce[a] * n_tof + je[a]) * n_tof + je[b], 1.0)
            a, b = _pairs_within_shot(qi, qi)
            np.add.at(out["CtC"], (ci[a] * n_tof_i + ji[a]) * n_tof_i + ji[b], 1.0)
            a, b = _pairs_within_shot(qe, qi)
            np.add.at(out["DtC"], (ce[a] * n_tof + je[a]) * n_tof_i + ji[b], 1.0)

            if vls_on:
                A_flat = A_block.reshape(-1, n_px)
                o = vq[np.argsort(cv, kind="stable")]
                cells_p, st = np.unique(cell[o], return_index=True)
                en = np.r_[st[1:], len(o)]
                A_s = A_flat[o]
                out["A"][cells_p] += np.add.reduceat(A_s, st, axis=0)
                out["AtG"][cells_p] += np.add.reduceat(A_s * g[o][:, None], st, axis=0)
                for c_, a0, a1 in zip(cells_p, st, en):
                    blk = A_s[a0:a1]
                    out["AtA"][c_] += blk.T @ blk
                if len(qe):
                    uk, R = _group_sum(ce * n_tof + je, A_flat[qe])
                    out["AtD"][uk // n_tof, :, uk % n_tof] += R
                if len(qi):
                    uk, R = _group_sum(ci * n_tof_i + ji, A_flat[qi])
                    out["AtC"][uk // n_tof_i, :, uk % n_tof_i] += R
    if reader is not None:
        reader.close()
    out["counters"] = cnt
    return out


def _pedestal_part(task):
    """Worker: sum and finite count over bunch slots [bg0, bg1) of the VLS frames of ``task["rows"]``."""
    import h5py as _h5py
    rows, bg0, bg1, chunk = task["rows"], task["bg0"], task["bg1"], task["chunk"]
    with _h5py.File(task["input_h5"], "r") as f:
        reader = None
        if task["vls_raw_run"] is not None:
            import data_loading
            reader = data_loading.RawVLSReader(task["vls_raw_run"], f["tID"][...], f["gmd"].shape[1],
                                               vls_offset=task["vls_offset"])
        tot = cnt = None
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            x = reader.read(r) if reader is not None else f["vls"][r]
            x = x[:, bg0:bg1].reshape(-1, x.shape[-1])
            if tot is None:
                tot, cnt = np.zeros(x.shape[-1]), np.zeros(x.shape[-1])
            tot += np.nansum(x, axis=0)
            cnt += np.isfinite(x).sum(axis=0)
        if reader is not None:
            reader.close()
    return tot, cnt


def _pool_map(fn, tasks, log):
    """
    Run fn over tasks in spawned processes, BLAS threads split between them;
    yields results as they finish. A worker that dies (e.g. one that cannot
    import the caller's main module: run scripts from a file with an
    ``if __name__ == "__main__":`` guard, not from stdin) raises
    ``BrokenProcessPool`` instead of being respawned forever.
    """
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    threads = str(max(1, n_cpu // len(tasks)))
    saved = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    for k in saved:
        os.environ[k] = threads
    log(f"workers            : {len(tasks)} processes x {threads} BLAS threads ({n_cpu} CPUs) - {fn.__name__}")
    try:
        with ProcessPoolExecutor(len(tasks), mp_context=mp.get_context("spawn")) as ex:
            for fut in as_completed([ex.submit(fn, t) for t in tasks]):
                yield fut.result()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_cfg1(task, chunks, workers, log):
    """Split ``chunks`` over ``workers`` processes (spawn), run _accumulate_cfg1 and add the results."""
    import os
    workers = max(1, int(workers))
    if workers == 1:
        return _accumulate_cfg1(dict(task, chunks=chunks))
    groups = [g for g in np.array_split(np.arange(len(chunks)), workers) if len(g)]
    tasks = [dict(task, chunks=[chunks[i] for i in g]) for g in groups]
    total = None
    for k_done, res in enumerate(_pool_map(_accumulate_cfg1, tasks, log), 1):
        if total is None:
            total = res
        else:
            for k, v in res.items():
                if k == "counters":
                    for kk, vv in v.items():
                        total["counters"][kk] += vv
                else:
                    total[k] += v
        log(f"  worker {k_done}/{len(tasks)} done")
    return total


def compute_aggregates(
    input_h5: Union[str, Path],
    output_h5: Union[str, Path],
    *,
    gmd_edges: np.ndarray,
    config: int,
    mode: str = "spectral",
    tof_edges: Optional[np.ndarray] = None,
    ion_tof_edges: Optional[np.ndarray] = None,
    crop_roi: Optional[Tuple[int, int]] = None,
    background_spec: Optional[dict] = None,
    z_edges: Optional[np.ndarray] = None,
    tof_roi: Optional[Tuple[float, float]] = None,
    energy_edges: Optional[np.ndarray] = None,
    energy_key: str = "photon_energy/measured",
    gmd_percentile_bins: Optional[int] = None,
    gmd_min: Optional[float] = None,
    vls_crop_roi: Optional[Tuple[int, int]] = None,
    vls_background: Optional[dict] = None,
    vls_cm_pixels: Optional[Tuple[int, int]] = None,
    vls_bad_pixels=None,
    vls_raw_run: Optional[int] = None,
    vls_offset: Optional[int] = None,
    vls_bg_stride: int = 10,
    workers: int = 1,
    chunk_size: int = 200,
    trim_start: int = 0,
    trim_end: int = 0,
    bunch_range: Optional[Tuple[int, int]] = None,
    max_hits_e: Optional[int] = None,
    max_hits_i: Optional[int] = None,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """
    Stream the input H5 in chunks and write binned aggregates to disk.

    Supports two H5 layouts:

    * ``config = 1`` — electron + ion TOF (``tofs_e``, ``tofs_i``); no
      VLS. Computes per-bin ``D`` (electron spectrum), ``C`` (ion
      spectrum) and the cross-product ``DtC``. ``crop_roi`` and
      ``background_spec`` are unused.

    * ``config = 2`` — VLS + liquid-jet eTOF (``vls``, ``liq_tofs_e``).
      Computes ``A`` (VLS), ``D`` (eTOF), and ``AtD``. ``crop_roi`` and
      ``background_spec`` are required.

    Both configs always compute the GMD aggregates ``G = <G_i>``,
    ``GtG = <G_i²>``, ``DtG = <D_i G_i>`` (and ``AtG`` / ``CtG`` where
    those modalities are present). These let downstream code form
    partial covariances against GMD.

    Modes
    -----
    ``mode = "spectral"`` (default)
        Bin by GMD only. Aggregate shapes have a leading ``N_GMD`` axis.
    ``mode = "time_resolved"``
        Bin by GMD × stage z. Aggregate shapes have leading
        ``(N_GMD, N_Z)`` axes.

        * ``config = 1``: the full electron / ion TOF spectra are kept,
          so the eTOF axis is ``n_tof`` and the ion-TOF axis is
          ``n_tof_i`` (same as spectral mode, just with an extra ``N_Z``
          axis prepended). ``tof_roi`` is unused.
        * ``config = 2``: the eTOF is reduced per shot to a scalar count
          inside ``tof_roi`` (``n_tof`` collapses to 1). VLS keeps its
          full ``n_pixels`` axis.
    ``mode = "energy"`` (config 1)
        Bin by photon energy x GMD, leading axes ``(N_E, N_GMD)``. The
        per-shot energy is read from ``energy_key`` (an ``(n, m)``
        dataset of the input H5; NaN shots are skipped).

    Parameters
    ----------
    input_h5, output_h5 : str or Path
    gmd_edges : array-like or None
        GMD bin edges. Shots outside the first / last edge are dropped.
        None together with ``gmd_percentile_bins``.
    gmd_percentile_bins : int, optional
        Put the GMD edges at equally spaced percentiles of the GMD of the
        shots that will be binned (kept trains, ``bunch_range``, finite
        GMD and, in energy mode, an energy inside ``energy_edges``).
    gmd_min : float, optional
        With ``gmd_percentile_bins``: leave out shots below this GMD (the
        lowest edge is then at or above it, so they fall outside the bins).
    energy_edges : array-like
        Required for ``mode = "energy"``: photon-energy bin edges (eV).
    energy_key : str
        Dataset holding the per-shot photon energy (energy mode).
    vls_crop_roi : (int, int), optional
        Config 1: also aggregate the VLS over this half-open pixel range
        (``A``, ``AtA``, ``AtD``, ``AtC``, ``AtG``). Shots without a VLS
        frame are then skipped for every aggregate.
    vls_background : dict, optional
        ``{"type": "auto", "roi": (s, e)}`` (mean over bunch slots
        ``[s, e)`` of every ``vls_bg_stride``-th kept train, full pixel
        width; default ``(100, 101)``, the dark slot) or
        ``{"type": "none"}``.
    vls_cm_pixels : (int, int), optional
        Pixels whose per-shot median (after the background) is subtracted.
    vls_bad_pixels : sequence of int, optional
        Set to 0 after the subtraction.
    vls_raw_run, vls_offset : int, optional
        Read the frames from the raw files of this run instead of the
        file's ``vls`` dataset (``data_loading.RawVLSReader``).
    workers : int
        Config 1: split the trains over this many processes (each reads its
        own share of the frames and accumulates partial sums, which are
        then added). Config 1 always accumulates from hit lists: every
        product involving D or C is a scatter-add over the hits (pairs)
        present, so only the VLS ``AtA`` needs dense products.
    config : int  (1 or 2)
    mode : {"spectral", "time_resolved"}
    tof_edges : array-like
        eTOF histogram edges (100 ps) — required for spectral mode.
    ion_tof_edges : array-like
        Ion-TOF histogram edges (100 ps) — required for ``config = 1``.
    crop_roi : (int, int)
        Half-open VLS pixel ROI — required for ``config = 2``.
    background_spec : dict
        ``{"type": "auto",  "roi": (s, e)}`` |
        ``{"type": "array", "path": "..."}`` |
        ``{"type": "none"}`` — required for ``config = 2``.
    z_edges : array-like
        Required for ``mode = "time_resolved"`` (both configs).
    tof_roi : array-like
        Required for ``mode = "time_resolved"`` with ``config = 2``;
        unused for ``config = 1`` time-resolved (which keeps the full
        eTOF / ion-TOF spectra).
    chunk_size, trim_start, trim_end : int
    bunch_range : (int, int), optional
        Half-open ``[bunch_start, bunch_end)`` slice of the bunch axis to
        aggregate. ``None`` (default) uses every bunch slot.
    max_hits_e, max_hits_i : int, optional
        Drop shots with more electron (ion, config 1) hits than this,
        counted over the full TOF range before histogramming. Removes
        burst / hit-cap-clipped shots, which otherwise dominate the
        second moments. ``None`` (default) applies no cut.
    config_path : Path, optional
        Recorded as a provenance attribute.
    verbose : bool
    """
    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    config = int(config)
    if gmd_edges is None and not gmd_percentile_bins:
        raise ValueError("give gmd_edges or gmd_percentile_bins")

    # ----- per-config validation + key resolution ------------------------
    if config == 1:
        if mode not in ("spectral", "time_resolved", "energy"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "energy" and energy_edges is None:
            raise ValueError("mode='energy' requires energy_edges")
        if tof_edges is None or ion_tof_edges is None:
            raise ValueError("config=1 requires tof_edges and ion_tof_edges")
        tof_edges     = np.asarray(tof_edges, dtype=np.float64)
        ion_tof_edges = np.asarray(ion_tof_edges, dtype=np.float64)
        n_tof   = len(tof_edges) - 1
        n_tof_i = len(ion_tof_edges) - 1
        n_pixels = None
        e_tof_key = "tofs_e"
        if vls_crop_roi is not None:
            roi_min, roi_max = int(vls_crop_roi[0]), int(vls_crop_roi[1])
            n_pixels = roi_max - roi_min
            bad = sorted({int(q) for q in (vls_bad_pixels or ())})
            bad_in_roi = np.array([q - roi_min for q in bad if roi_min <= q < roi_max], dtype=int)
            cm_px = (None if vls_cm_pixels is None else
                     np.array([q for q in range(int(vls_cm_pixels[0]), int(vls_cm_pixels[1])) if q not in bad]))
            if vls_background is None:
                vls_background = {"type": "auto", "roi": (100, 101)}
            elif isinstance(vls_background, (tuple, list)):
                vls_background = {"type": "auto", "roi": tuple(vls_background)}
    elif config == 2:
        if tof_edges is None and mode == "spectral":
            raise ValueError("mode='spectral' (config=2) requires tof_edges")
        if crop_roi is None or background_spec is None:
            raise ValueError("config=2 requires crop_roi and background_spec")
        e_tof_key = "liq_tofs_e"
        n_tof_i = None
        ion_tof_edges = None
        roi_min, roi_max = int(crop_roi[0]), int(crop_roi[1])
        n_pixels = roi_max - roi_min
        if mode == "spectral":
            tof_edges = np.asarray(tof_edges, dtype=np.float64)
            n_tof = len(tof_edges) - 1
        elif mode == "time_resolved":
            if z_edges is None or tof_roi is None:
                raise ValueError(
                    "mode='time_resolved' requires z_edges and tof_roi"
                )
            tof_roi_arr = np.asarray(tof_roi, dtype=np.float64)
            if tof_roi_arr.shape != (2,):
                raise ValueError(f"tof_roi must be (2,), got {tof_roi_arr.shape}")
            tof_edges = tof_roi_arr.copy()
            n_tof = 1
        else:
            raise ValueError(f"unknown mode {mode!r}")
    else:
        raise ValueError(f"config must be 1 or 2, got {config}")

    if mode == "time_resolved":
        z_edges_arr = np.asarray(z_edges, dtype=np.float64)
        n_z = len(z_edges_arr) - 1
    else:
        z_edges_arr = None
        n_z = None
    if mode == "energy":
        e_edges_arr = np.asarray(energy_edges, dtype=np.float64)
        n_e = len(e_edges_arr) - 1
    else:
        e_edges_arr = None
        n_e = None

    log = print if verbose else (lambda *a, **k: None)
    cfg1_vls = config == 1 and vls_crop_roi is not None

    with h5py.File(input_h5, "r") as f:
        _verify_h5_for_config(f, config)
        keep_mask = _build_keep_mask(f, trim_start, trim_end)
        n_total = keep_mask.shape[0]
        m = f["gmd"].shape[1]
        b0, b1 = (0, m) if bunch_range is None else (int(bunch_range[0]), int(bunch_range[1]))
        if not 0 <= b0 < b1 <= m:
            raise ValueError(f"bunch_range {bunch_range} outside [0, {m}]")

        if gmd_percentile_bins:
            # Edges at percentiles of the GMD of the shots that will be binned
            g_all = f["gmd"][:, b0:b1][keep_mask]
            sel = np.isfinite(g_all)
            if gmd_min is not None:
                sel &= g_all >= gmd_min
            if mode == "energy":
                e_all = f[energy_key][:, b0:b1][keep_mask]
                sel &= (e_all >= e_edges_arr[0]) & (e_all < e_edges_arr[-1])
            gmd_edges = np.percentile(g_all[sel], np.linspace(0, 100, int(gmd_percentile_bins) + 1))
            gmd_edges[-1] = np.nextafter(gmd_edges[-1], np.inf)   # keep the largest shot in the last bin
        gmd_edges = np.asarray(gmd_edges, dtype=np.float64)
        n_gmd = len(gmd_edges) - 1

        log(f"input              : {input_h5}")
        log(f"config             : {config}")
        log(f"mode               : {mode}")
        log(f"trains             : {n_total} total, "
            f"{int(keep_mask.sum())} kept after bad-train filter + trim")
        log(f"bunches / train    : {m}, aggregating [{b0}, {b1})")
        log(f"GMD bins           : {n_gmd}  edges {gmd_edges}")
        if max_hits_e is not None or max_hits_i is not None:
            log(f"max hits / shot    : e {max_hits_e}, i {max_hits_i}")
        if config == 2:
            log(f"VLS ROI            : [{roi_min}, {roi_max}) = {n_pixels} pixels")
        if mode == "spectral" or config == 1:
            log(f"eTOF bins          : {n_tof} ({tof_edges[0]:.1f} .. {tof_edges[-1]:.1f} 100ps)")
            if config == 1:
                log(f"ion TOF bins       : {n_tof_i} ({ion_tof_edges[0]:.1f} .. {ion_tof_edges[-1]:.1f} 100ps)")
        else:
            log(f"TOF ROI            : [{tof_edges[0]:.1f}, {tof_edges[-1]:.1f}) 100ps (scalar D)")
        if mode == "time_resolved":
            log(f"Z bins             : {n_z}  edges {z_edges_arr}")
        if mode == "energy":
            log(f"energy bins        : {n_e} ({e_edges_arr[0]:.2f} .. {e_edges_arr[-1]:.2f} eV) from {energy_key}")
        if gmd_percentile_bins:
            log(f"GMD edges          : {int(gmd_percentile_bins)} percentile bins")

        # --- config 1 VLS: frame source, background, preprocessing -------
        vls_reader = None
        if cfg1_vls:
            if vls_raw_run is not None:
                import data_loading
                vls_reader = data_loading.RawVLSReader(vls_raw_run, f["tID"][...], m, vls_offset=vls_offset)
                vls_source = (f"raw files of run {vls_raw_run} ({len(vls_reader.paths)}), "
                              f"line b+{vls_reader.vls_offset} -> bunch b")
                n_px_full = vls_reader.n_pixels
            elif "vls" in f:
                vls_source = "combined vls"
                n_px_full = f["vls"].shape[2]
            else:
                raise RuntimeError("VLS_CROP_ROI is set but the file has no vls: set VLS_RAW_RUN")

            def read_vls(rows):
                rows = np.asarray(rows)
                if vls_reader is not None:
                    return vls_reader.read(rows)
                if rows[-1] - rows[0] + 1 == len(rows):
                    return f["vls"][rows[0]:rows[-1] + 1]
                return f["vls"][rows]

            if vls_background["type"] == "auto":
                bg0, bg1 = (int(q) for q in vls_background["roi"])
                ped_rows = np.flatnonzero(keep_mask)[::int(vls_bg_stride)]
                ped_sum = np.zeros(n_px_full)
                ped_n = np.zeros(n_px_full)
                if workers > 1 and len(ped_rows) > workers:
                    ptasks = [dict(input_h5=str(input_h5), rows=r, bg0=bg0, bg1=bg1, chunk=chunk_size,
                                   vls_raw_run=vls_raw_run, vls_offset=vls_offset)
                              for r in np.array_split(ped_rows, workers) if len(r)]
                    for ps, pn in _pool_map(_pedestal_part, ptasks, log):
                        ped_sum += ps
                        ped_n += pn
                else:
                    for i in range(0, len(ped_rows), chunk_size):
                        xb = read_vls(ped_rows[i:i + chunk_size])[:, bg0:bg1].reshape(-1, n_px_full)
                        ped_sum += np.nansum(xb, axis=0)
                        ped_n += np.isfinite(xb).sum(axis=0)
                vls_ped = ped_sum / np.maximum(ped_n, 1)
            elif vls_background["type"] == "none":
                vls_ped = np.zeros(n_px_full)
            else:
                raise ValueError(f"config 1 VLS background must be 'auto' or 'none', got {vls_background!r}")

            def vls_preprocess(x):
                # (n, m, n_px_full) raw frames -> (n, b1-b0, n_pixels) float64 and frame flags
                x = x[:, b0:b1]
                ok = np.isfinite(x[:, :, roi_min])
                a = x[:, :, roi_min:roi_max].astype(np.float64) - vls_ped[roi_min:roi_max]
                if cm_px is not None:
                    a -= np.median(x[:, :, cm_px] - vls_ped[cm_px], axis=-1, keepdims=True)
                a[..., bad_in_roi] = 0.0
                a[~ok] = 0.0
                return a, ok

            log(f"VLS                : {vls_source}; crop [{roi_min}, {roi_max}) = {n_pixels} pixels, "
                f"background {vls_background}, common mode {vls_cm_pixels}, {len(bad_in_roi)} bad pixels in the crop"
                + (f"; frames for {vls_reader.coverage:.1%} of the trains" if vls_reader is not None else ""))

        # --- resolve background (config 2 only) ------------------------
        background = None
        background_meta = None
        if config == 2:
            bg_type = background_spec["type"]
            if bg_type == "auto":
                bg_start, bg_end = background_spec["roi"]
                log(f"computing auto background over bunches [{bg_start}, {bg_end})...")
                background = _compute_auto_background(
                    f, keep_mask, int(bg_start), int(bg_end),
                    roi_min, roi_max, chunk_size,
                )
                background_meta = {"type": "auto",
                                   "roi": [int(bg_start), int(bg_end)]}
            elif bg_type == "array":
                bg_arr = np.load(background_spec["path"])
                if bg_arr.shape != (n_pixels,):
                    raise ValueError(
                        f"background array shape {bg_arr.shape} != "
                        f"VLS ROI ({n_pixels},)"
                    )
                background = bg_arr.astype(np.float64)
                background_meta = {"type": "array",
                                   "path": str(background_spec["path"])}
            elif bg_type == "none":
                background = np.zeros(n_pixels, dtype=np.float64)
                background_meta = {"type": "none"}
            else:
                raise ValueError(f"unknown background type: {bg_type!r}")

        # --- allocate aggregate sums -----------------------------------
        if mode == "spectral":
            agg_prefix = (n_gmd,)
            n_bins_total = n_gmd
        elif mode == "energy":
            agg_prefix = (n_e, n_gmd)
            n_bins_total = n_e * n_gmd
        else:
            agg_prefix = (n_gmd, n_z)
            n_bins_total = n_gmd * n_z

        # Always present (both configs).
        D_sum   = np.zeros(agg_prefix + (n_tof,),       dtype=np.float64)
        DtD_sum = np.zeros(agg_prefix + (n_tof, n_tof), dtype=np.float64)
        G_sum   = np.zeros(agg_prefix,                  dtype=np.float64)
        GtG_sum = np.zeros(agg_prefix,                  dtype=np.float64)
        DtG_sum = np.zeros(agg_prefix + (n_tof,),       dtype=np.float64)
        n_per_bin = np.zeros(agg_prefix, dtype=np.int64)

        # Config-specific.
        if config == 2:
            A_sum   = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)
            AtA_sum = np.zeros(agg_prefix + (n_pixels, n_pixels), dtype=np.float64)
            AtD_sum = np.zeros(agg_prefix + (n_pixels, n_tof),    dtype=np.float64)
            AtG_sum = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)
        else:
            C_sum   = np.zeros(agg_prefix + (n_tof_i,),           dtype=np.float64)
            CtC_sum = np.zeros(agg_prefix + (n_tof_i, n_tof_i),   dtype=np.float64)
            DtC_sum = np.zeros(agg_prefix + (n_tof, n_tof_i),     dtype=np.float64)
            CtG_sum = np.zeros(agg_prefix + (n_tof_i,),           dtype=np.float64)
            if cfg1_vls:
                A_sum   = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)
                AtA_sum = np.zeros(agg_prefix + (n_pixels, n_pixels), dtype=np.float64)
                AtD_sum = np.zeros(agg_prefix + (n_pixels, n_tof),    dtype=np.float64)
                AtC_sum = np.zeros(agg_prefix + (n_pixels, n_tof_i),  dtype=np.float64)
                AtG_sum = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)

        n_skipped_gmd_oor = 0
        n_skipped_gmd_nan = 0
        n_skipped_max_hits = 0
        n_skipped_z_oor   = 0
        n_skipped_z_nan   = 0
        n_skipped_e_oor   = 0
        n_skipped_e_nan   = 0
        n_skipped_vls     = 0

        if config == 1:
            log("accumulating (hit lists" + (", VLS" if cfg1_vls else "") + ")...")
            task = dict(
                input_h5=str(input_h5), keep_mask=keep_mask, b0=b0, b1=b1, mode=mode, n_gmd=n_gmd,
                gmd_edges=gmd_edges, e_edges=e_edges_arr, n_e=n_e, z_edges=z_edges_arr, n_z=n_z,
                energy_key=energy_key, n_cells=n_bins_total, n_tof=n_tof, n_tof_i=n_tof_i,
                tof_edges=np.asarray(tof_edges, np.float64), ion_tof_edges=np.asarray(ion_tof_edges, np.float64),
                max_hits_e=max_hits_e, max_hits_i=max_hits_i, vls_on=cfg1_vls, n_pixels=n_pixels,
                vls_raw_run=vls_raw_run, vls_offset=vls_offset,
                roi_min=roi_min if cfg1_vls else None, roi_max=roi_max if cfg1_vls else None,
                vls_ped=vls_ped if cfg1_vls else None, cm_px=cm_px if cfg1_vls else None,
                bad_in_roi=bad_in_roi if cfg1_vls else None,
            )
            chunks = [(cs, ce) for cs, ce in _iter_chunks(n_total, chunk_size) if keep_mask[cs:ce].any()]
            res = _run_cfg1(task, chunks, workers, log)
            shp = agg_prefix
            n_per_bin[...] = res["n"].reshape(shp)
            G_sum[...], GtG_sum[...] = res["G"].reshape(shp), res["GtG"].reshape(shp)
            D_sum[...], DtG_sum[...] = res["D"].reshape(shp + (n_tof,)), res["DtG"].reshape(shp + (n_tof,))
            DtD_sum[...] = res["DtD"].reshape(shp + (n_tof, n_tof))
            C_sum[...], CtG_sum[...] = res["C"].reshape(shp + (n_tof_i,)), res["CtG"].reshape(shp + (n_tof_i,))
            CtC_sum[...] = res["CtC"].reshape(shp + (n_tof_i, n_tof_i))
            DtC_sum[...] = res["DtC"].reshape(shp + (n_tof, n_tof_i))
            if cfg1_vls:
                A_sum[...], AtG_sum[...] = res["A"].reshape(shp + (n_pixels,)), res["AtG"].reshape(shp + (n_pixels,))
                AtA_sum[...] = res["AtA"].reshape(shp + (n_pixels, n_pixels))
                AtD_sum[...] = res["AtD"].reshape(shp + (n_pixels, n_tof))
                AtC_sum[...] = res["AtC"].reshape(shp + (n_pixels, n_tof_i))
            cn = res["counters"]
            n_skipped_gmd_nan += cn["gmd_nan"]
            n_skipped_gmd_oor += cn["gmd_oor"]
            n_skipped_max_hits += cn["max_hits"]
            n_skipped_z_nan += cn["z_nan"]
            n_skipped_z_oor += cn["z_oor"]
            n_skipped_e_nan += cn["e_nan"]
            n_skipped_e_oor += cn["e_oor"]
            n_skipped_vls += cn["vls"]
            del res
        else:
            log("accumulating...")
        # Config 2: dense loop over chunks (config 1 is done above)
        for s, e in (_iter_chunks(n_total, chunk_size) if config == 2 else ()):
            sub_mask = keep_mask[s:e]
            if not sub_mask.any():
                continue

            gmd_block = f["gmd"][s:e, b0:b1][sub_mask]                     # (n_keep, m)
            tof_block = f[e_tof_key][s:e, b0:b1][sub_mask]                 # (n_keep, m, max_hits)
            if config == 2:
                vls_block = f["vls"][s:e, b0:b1, roi_min:roi_max][sub_mask]  # (n_keep, m, n_pixels)
                A_block = vls_block.astype(np.float64) - background
            else:
                ion_tof_block = f["tofs_i"][s:e, b0:b1][sub_mask]          # (n_keep, m, max_hits_i)
                C_block = _histogram_tof_chunk(ion_tof_block, ion_tof_edges)
                if cfg1_vls:
                    A_block, vls_ok = vls_preprocess(read_vls(np.arange(s, e)[sub_mask]))
            if mode == "time_resolved":
                z_block = f["z"][s:e, b0:b1][sub_mask]
            if mode == "energy":
                e_block = f[energy_key][s:e, b0:b1][sub_mask]

            n_keep = gmd_block.shape[0]

            if mode == "spectral" or config == 1:
                # Config 1 (both modes) and config 2 spectral: full TOF
                # histogram. Config 2 TR collapses the eTOF to a scalar
                # count in the ROI below.
                D_block = _histogram_tof_chunk(tof_block, tof_edges)
            else:
                roi_lo, roi_hi = float(tof_edges[0]), float(tof_edges[-1])
                in_roi = (tof_block >= roi_lo) & (tof_block < roi_hi) & (tof_block != 0)
                D_block = in_roi.sum(axis=-1).astype(np.float64)[:, :, None]

            # GMD bin per shot.
            gmd_flat = gmd_block.ravel()
            gmd_bin  = np.digitize(gmd_flat, gmd_edges) - 1
            gmd_nan  = ~np.isfinite(gmd_flat)
            gmd_oor  = ~gmd_nan & ((gmd_bin < 0) | (gmd_bin >= n_gmd))
            valid    = ~gmd_nan & ~gmd_oor
            n_skipped_gmd_nan += int(gmd_nan.sum())
            n_skipped_gmd_oor += int(gmd_oor.sum())

            # Burst / clipped shots: hit count over the full TOF range.
            too_many = np.zeros_like(valid)
            if max_hits_e is not None:
                too_many |= (tof_block != 0).sum(axis=-1).ravel() > int(max_hits_e)
            if max_hits_i is not None and config == 1:
                too_many |= (ion_tof_block != 0).sum(axis=-1).ravel() > int(max_hits_i)
            n_skipped_max_hits += int((too_many & valid).sum())
            valid = valid & ~too_many
            if cfg1_vls:
                n_skipped_vls += int((valid & ~vls_ok.ravel()).sum())
                valid = valid & vls_ok.ravel()

            if mode == "spectral":
                bin_id = gmd_bin
            elif mode == "energy":
                e_flat = e_block.ravel()
                e_bin  = np.digitize(e_flat, e_edges_arr) - 1
                e_nan  = ~np.isfinite(e_flat)
                e_oor  = ~e_nan & ((e_bin < 0) | (e_bin >= n_e))
                n_skipped_e_nan += int((e_nan & valid).sum())
                n_skipped_e_oor += int((e_oor & valid).sum())
                valid = valid & ~e_nan & ~e_oor
                bin_id = e_bin * n_gmd + gmd_bin
            else:
                z_flat = z_block.ravel()
                z_bin  = np.digitize(z_flat, z_edges_arr) - 1
                z_nan  = ~np.isfinite(z_flat)
                z_oor  = ~z_nan & ((z_bin < 0) | (z_bin >= n_z))
                n_skipped_z_nan += int(z_nan.sum())
                n_skipped_z_oor += int(z_oor.sum())
                valid = valid & ~z_nan & ~z_oor
                bin_id = gmd_bin * n_z + z_bin

            D_flat = D_block.reshape(-1, n_tof)
            if config == 2:
                A_flat = A_block.reshape(-1, n_pixels)
            else:
                C_flat = C_block.reshape(-1, n_tof_i)
                if cfg1_vls:
                    A_flat = A_block.reshape(-1, n_pixels)

            # Shots of the chunk grouped by bin (stable sort keeps their order
            # within a bin), so each bin is a contiguous slice: one gather per
            # chunk instead of a boolean mask over the chunk per bin.
            order = np.flatnonzero(valid)
            order = order[np.argsort(bin_id[order], kind="stable")]
            bins_sorted, starts = np.unique(bin_id[order], return_index=True)
            ends = np.r_[starts[1:], len(order)]
            gmd_flat, D_flat = gmd_flat[order], D_flat[order]
            if config == 2:
                A_flat = A_flat[order]
            else:
                C_flat = C_flat[order]
                if cfg1_vls:
                    A_flat = A_flat[order]

            for b, a0, a1 in zip(bins_sorted, starts, ends):
                mask = slice(a0, a1)
                if mode == "spectral":
                    idx = (int(b),)
                elif mode == "energy":
                    idx = (int(b) // n_gmd, int(b) % n_gmd)
                else:
                    idx = (int(b) // n_z, int(b) % n_z)

                G_b = gmd_flat[mask]                # (n_b,)  per-shot scalar
                D_b = D_flat[mask]                  # (n_b, n_tof)
                n_b = G_b.shape[0]

                n_per_bin[idx] += n_b
                G_sum[idx]   += G_b.sum()
                GtG_sum[idx] += float(G_b @ G_b)
                D_sum[idx]   += D_b.sum(axis=0)
                DtD_sum[idx] += D_b.T @ D_b
                DtG_sum[idx] += D_b.T @ G_b

                if config == 2:
                    A_b = A_flat[mask]
                    A_sum[idx]   += A_b.sum(axis=0)
                    AtA_sum[idx] += A_b.T @ A_b
                    AtD_sum[idx] += A_b.T @ D_b
                    AtG_sum[idx] += A_b.T @ G_b
                else:
                    C_b = C_flat[mask]
                    C_sum[idx]   += C_b.sum(axis=0)
                    CtC_sum[idx] += C_b.T @ C_b
                    DtC_sum[idx] += D_b.T @ C_b
                    CtG_sum[idx] += C_b.T @ G_b
                    if cfg1_vls:
                        A_b = A_flat[mask]
                        A_sum[idx]   += A_b.sum(axis=0)
                        AtA_sum[idx] += A_b.T @ A_b
                        AtD_sum[idx] += A_b.T @ D_b
                        AtC_sum[idx] += A_b.T @ C_b
                        AtG_sum[idx] += A_b.T @ G_b

            log(f"  [{s:>6d}..{e:>6d})  trains={n_keep:>4d}  "
                f"binned={int(valid.sum()):>5d}")

        if vls_reader is not None:
            vls_reader.close()

    # --- means ---------------------------------------------------------
    safe_n = np.where(n_per_bin > 0, n_per_bin, 1).astype(np.float64)
    empty = n_per_bin == 0

    def _norm_vec(s):
        m = s / safe_n[..., None]
        m[empty] = np.nan
        return m

    def _norm_mat(s):
        m = s / safe_n[..., None, None]
        m[empty] = np.nan
        return m

    def _norm_scalar(s):
        m = s / safe_n
        m[empty] = np.nan
        return m

    means = {
        "D":   _norm_vec(D_sum),
        "DtD": _norm_mat(DtD_sum),
        "G":   _norm_scalar(G_sum),
        "GtG": _norm_scalar(GtG_sum),
        "DtG": _norm_vec(DtG_sum),
    }
    if config == 2:
        means.update({
            "A":   _norm_vec(A_sum),
            "AtA": _norm_mat(AtA_sum),
            "AtD": _norm_mat(AtD_sum),
            "AtG": _norm_vec(AtG_sum),
        })
    else:
        means.update({
            "C":   _norm_vec(C_sum),
            "CtC": _norm_mat(CtC_sum),
            "DtC": _norm_mat(DtC_sum),
            "CtG": _norm_vec(CtG_sum),
        })
        if cfg1_vls:
            means.update({
                "A":   _norm_vec(A_sum),
                "AtA": _norm_mat(AtA_sum),
                "AtD": _norm_mat(AtD_sum),
                "AtC": _norm_mat(AtC_sum),
                "AtG": _norm_vec(AtG_sum),
            })

    # --- save ----------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}")
    with h5py.File(output_h5, "w") as fout:
        for name, arr in means.items():
            fout.create_dataset(name, data=arr, compression="gzip")
        fout.create_dataset("n_per_bin", data=n_per_bin)
        fout.create_dataset("gmd_edges", data=gmd_edges)
        fout.create_dataset("tof_edges", data=tof_edges)
        if config == 2:
            fout.create_dataset("vls_pixels", data=np.arange(roi_min, roi_max, dtype=np.int64))
            fout.create_dataset("background", data=background)
        else:
            fout.create_dataset("ion_tof_edges", data=ion_tof_edges)
            if cfg1_vls:
                fout.create_dataset("vls_pixels", data=np.arange(roi_min, roi_max, dtype=np.int64))
                fout.create_dataset("background", data=vls_ped[roi_min:roi_max])
                fout.attrs["vls_crop_roi"] = np.asarray([roi_min, roi_max], dtype=np.int64)
                fout.attrs["vls_source"] = vls_source
                fout.attrs["vls_background_type"] = vls_background["type"]
                if vls_background["type"] == "auto":
                    fout.attrs["background_roi"] = np.asarray(vls_background["roi"], dtype=np.int64)
                    fout.attrs["vls_bg_stride"] = int(vls_bg_stride)
                if vls_cm_pixels is not None:
                    fout.attrs["vls_cm_pixels"] = np.asarray(vls_cm_pixels, dtype=np.int64)
                fout.attrs["vls_bad_pixels"] = np.asarray(bad, dtype=np.int64)
                fout.attrs["n_skipped_no_vls"] = int(n_skipped_vls)
        if mode == "time_resolved":
            fout.create_dataset("z_edges", data=z_edges_arr)
        if mode == "energy":
            fout.create_dataset("energy_edges", data=e_edges_arr)
            fout.attrs["energy_key"] = energy_key
            fout.attrs["n_skipped_energy_nan"] = int(n_skipped_e_nan)
            fout.attrs["n_skipped_energy_oor"] = int(n_skipped_e_oor)
        if gmd_percentile_bins:
            fout.attrs["gmd_percentile_bins"] = int(gmd_percentile_bins)
            if gmd_min is not None:
                fout.attrs["gmd_min"] = float(gmd_min)

        fout.attrs["input_h5"] = str(input_h5)
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)
        fout.attrs["mode"]            = mode
        fout.attrs["config"]          = int(config)
        fout.attrs["trim_start"]      = int(trim_start)
        fout.attrs["trim_end"]        = int(trim_end)
        fout.attrs["chunk_size"]      = int(chunk_size)
        fout.attrs["bunch_range"]     = np.asarray([b0, b1], dtype=np.int64)
        fout.attrs["n_skipped_gmd_oor"] = int(n_skipped_gmd_oor)
        fout.attrs["n_skipped_gmd_nan"] = int(n_skipped_gmd_nan)
        fout.attrs["n_skipped_max_hits"] = int(n_skipped_max_hits)
        if max_hits_e is not None:
            fout.attrs["max_hits_e"] = int(max_hits_e)
        if max_hits_i is not None:
            fout.attrs["max_hits_i"] = int(max_hits_i)
        if config == 2:
            fout.attrs["vls_crop_roi"]    = np.asarray([roi_min, roi_max], dtype=np.int64)
            fout.attrs["background_type"] = background_meta["type"]
            if "roi" in background_meta:
                fout.attrs["background_roi"] = np.asarray(
                    background_meta["roi"], dtype=np.int64
                )
            if "path" in background_meta:
                fout.attrs["background_path"] = background_meta["path"]
        if mode == "time_resolved":
            fout.attrs["n_skipped_z_oor"] = int(n_skipped_z_oor)
            fout.attrs["n_skipped_z_nan"] = int(n_skipped_z_nan)
            if config == 2:
                # Config-2 TR collapses eTOF to a scalar count; record
                # the ROI used. Config-1 TR keeps full TOF spectra.
                fout.attrs["tof_roi"] = np.asarray(tof_edges, dtype=np.float64)

    summary = (f"done: {int(n_per_bin.sum())} shots aggregated "
               f"across {n_bins_total} bins (config={config}, mode={mode})")
    if mode == "energy":
        summary += (f"; skipped {n_skipped_e_nan} without a photon energy, "
                    f"{n_skipped_e_oor} outside the energy edges")
    if mode == "time_resolved":
        summary += (f"; skipped {n_skipped_gmd_oor} GMD-oor, "
                    f"{n_skipped_z_oor} z-oor, "
                    f"{n_skipped_gmd_nan + n_skipped_z_nan} nan")
    else:
        summary += (f"; skipped {n_skipped_gmd_oor} GMD-oor, "
                    f"{n_skipped_gmd_nan} nan")
    summary += f", {n_skipped_max_hits} over max hits"
    if cfg1_vls:
        summary += f", {n_skipped_vls} without a VLS frame"
    log(summary)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Python module exposing CROP_ROI, BACKGROUND, GMD_EDGES, "
             "CONFIG, plus MODE-specific extras (TOF_EDGES, or "
             "Z_EDGES + TOF_ROI)",
    )
    parser.add_argument(
        "input_h5", type=Path, help="combined H5 file to process",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output H5 (default: <input>_aggregates.h5 next to the input)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="config 1: worker processes (default: WORKERS in the config, else 1)",
    )
    args = parser.parse_args(argv)

    cfg = _load_python_config(args.config_path)

    cfg_n = int(cfg.CONFIG)
    mode = getattr(cfg, "MODE", "spectral")
    base = {"time_resolved": "_aggregates_tr", "energy": "_aggregates_energy"}.get(mode, "_aggregates")
    user_suffix = getattr(cfg, "OUTPUT_SUFFIX", "") or ""
    out = args.output or args.input_h5.with_name(
        args.input_h5.stem + base + user_suffix + ".h5"
    )

    kwargs = dict(
        input_h5=args.input_h5,
        output_h5=out,
        gmd_edges=(None if getattr(cfg, "GMD_EDGES", None) is None
                   else np.asarray(cfg.GMD_EDGES, dtype=float)),
        gmd_percentile_bins=getattr(cfg, "GMD_PERCENTILE_BINS", None),
        gmd_min=getattr(cfg, "GMD_MIN", None),
        config=cfg_n,
        mode=mode,
        chunk_size=int(getattr(cfg, "CHUNK_SIZE", 200)),
        trim_start=int(getattr(cfg, "TRIM_START", 0)),
        trim_end=int(getattr(cfg, "TRIM_END", 0)),
        bunch_range=getattr(cfg, "BUNCH_RANGE", None),
        max_hits_e=getattr(cfg, "MAX_HITS_E", None),
        max_hits_i=getattr(cfg, "MAX_HITS_I", None),
        config_path=args.config_path,
    )

    if cfg_n == 2:
        background = cfg.BACKGROUND
        # Shorthand: tuple/list of length 2 is treated as an auto bunch ROI.
        if isinstance(background, (tuple, list)) and len(background) == 2:
            background = {"type": "auto", "roi": list(background)}
        kwargs["crop_roi"] = cfg.CROP_ROI
        kwargs["background_spec"] = background

    if cfg_n == 1:
        # Config 1 always uses full TOF + ion TOF spectra (both modes).
        kwargs["tof_edges"]     = np.asarray(cfg.TOF_EDGES, dtype=float)
        kwargs["ion_tof_edges"] = np.asarray(cfg.ION_TOF_EDGES, dtype=float)
        if mode == "time_resolved":
            kwargs["z_edges"] = np.asarray(cfg.Z_EDGES, dtype=float)
        if mode == "energy":
            kwargs["energy_edges"] = np.asarray(cfg.ENERGY_EDGES, dtype=float)
            kwargs["energy_key"] = getattr(cfg, "ENERGY_KEY", "photon_energy/measured")
        if getattr(cfg, "VLS_CROP_ROI", None) is not None:
            kwargs.update(
                vls_crop_roi=tuple(cfg.VLS_CROP_ROI),
                vls_background=getattr(cfg, "VLS_BACKGROUND", None),
                vls_cm_pixels=getattr(cfg, "VLS_CM_PIXELS", None),
                vls_bad_pixels=getattr(cfg, "VLS_BAD_PIXELS", None),
                vls_raw_run=getattr(cfg, "VLS_RAW_RUN", None),
                vls_offset=getattr(cfg, "VLS_OFFSET", None),
                vls_bg_stride=int(getattr(cfg, "VLS_BG_STRIDE", 10)),
            )
    else:
        if mode == "spectral":
            kwargs["tof_edges"] = np.asarray(cfg.TOF_EDGES, dtype=float)
        else:
            kwargs["z_edges"] = np.asarray(cfg.Z_EDGES, dtype=float)
            kwargs["tof_roi"] = tuple(cfg.TOF_ROI)
    kwargs["workers"] = args.workers if args.workers is not None else int(getattr(cfg, "WORKERS", 1))

    compute_aggregates(**kwargs)


if __name__ == "__main__":
    main()
