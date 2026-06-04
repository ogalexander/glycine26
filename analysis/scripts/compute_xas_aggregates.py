"""
Compute (nominal energy x GMD) aggregates for a photon-energy XAS scan.

This is the XAS-scan analogue of ``compute_aggregates.py``: it reads the
raw FLASH H5 files for a run (because the SDU/TDC streams are not used
for this analysis), detects the open / closed segments of the fast
shutter, assigns each open segment a nominal photon energy by section
index, and writes per-(energy, gmd_bin) aggregates of the VLS spectrum
and the GMD. The resulting H5 follows the ``AggregatesData`` layout
defined in ``compute_aggregates.py``.

The VLS is background-subtracted in two steps:

1. **Per-train background** — applied to *every* train of the run, right
   after cropping. For each train the mean spectrum over the non-signal
   bunches (``BG_BUNCH_RANGE``) is subtracted from every bunch of that
   train, removing per-train DC offset and slow dark drift before any
   section logic runs. Implemented via
   :meth:`ExperimentData.auto_subtract_background_trainwise`.
2. **Per-section background** — for each open shutter section, the 2D
   ``(m_bunches, n_pixels)`` mean spectrum over the trains of the
   *immediately preceding* closed shutter section (after step 1) is
   subtracted from every train of the open section.

Shots are then flattened over (train, signal_bunch), binned by GMD, and
accumulated into ``A``, ``AtA``, ``AtG``, ``G``, ``GtG``, ``n_per_bin``
arrays of shape ``(N_E, N_GMD, ...)``.

Config file
-----------
A Python module exposing the module-level names below. See
``analysis/configs/aggregates_example_xas.py``.

    RUN_NO              : int
    NOMINAL_ENERGIES    : 1D array-like of nominal photon energies (eV)
                          assigned to detected sections by index
    GMD_EDGES           : 1D array-like of GMD bin edges (uJ); use a
                          single bin [-inf, +inf] to disable GMD binning
    CROP_ROI            : (roi_min, roi_max) half-open VLS pixel ROI
    SIGNAL_BUNCH_RANGE  : (b0, b1) half-open bunch range that overlaps
                          the GMD acquisition window
    BG_BUNCH_RANGE      : (b0, b1) half-open bunch range used for the
                          per-train VLS background
    CONFIG              : 2 (xas_scan is config-2 only)
    MAX_FILES           : int or None — limit number of raw H5 files
    TRAIN_LENGTH        : int or None — bunches per train (default 100)
    SHUTTER_VALUE_PATH  : str — HDF5 path of the fast shutter value
    SHUTTER_INDEX_PATH  : str — HDF5 path of the fast shutter index
    TRAIN_RATE_HZ       : float — FLASH train rate (default 10)
    TRANSITION_TRIM_SECONDS : float — trim window around shutter moves
    FIRST_SECTION_STATE : "open" | "closed" — protocol prior
    GROUP_BY_ENERGY     : bool — when False, collapse the energy axis
                          to a single bin so the output is binned only
                          on GMD (default True)

CLI
---
    python compute_xas_aggregates.py CONFIG.py [-o OUT.h5]

Output H5 layout (means per bin; ``N_E = n_sections_used`` when
``GROUP_BY_ENERGY=True`` else ``N_E = 1``):
    /A                 (N_E, N_GMD, n_pixels)
    /AtA               (N_E, N_GMD, n_pixels, n_pixels)
    /AtG               (N_E, N_GMD, n_pixels)
    /G                 (N_E, N_GMD)
    /GtG               (N_E, N_GMD)
    /n_per_bin         (N_E, N_GMD)               int64
    /gmd_edges         (N_GMD+1,)
    /nominal_energies  (N_E,)                     eV (NaN when GROUP_BY_ENERGY=False)
    /vls_pixels        (n_pixels,)                source-pixel indices
    /section_bg        (N_E, m_bunches, n_pixels) preceding-closed mean
                                                  (averaged over sections
                                                  when GROUP_BY_ENERGY=False)
    attrs:             mode='xas_scan', config=2, run_no, group_by_energy,
                       signal_bunch_range, bg_bunch_range,
                       n_sections_detected, n_sections_used, etc.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

import data_loading
from compute_aggregates import load_python_config


__all__ = ["compute_xas_aggregates", "main"]


# Default fast-shutter HDF5 paths.
DEFAULT_SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
DEFAULT_SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _file_order_key(path_str: str):
    """Sort raw H5 files by numeric index in ..._fileNN_..."""
    name = Path(path_str).name
    if "_file" in name:
        tail = name.split("_file", 1)[1]
        num_str = tail.split("_", 1)[0]
        if num_str.isdigit():
            return (0, int(num_str), name)
    return (1, 0, name)


def _list_raw_h5_files(run_no: int, raw_dir: Path, max_files: Optional[int]):
    pattern = str(raw_dir / f"*run{run_no}*.h5")
    paths = sorted(glob.glob(pattern), key=_file_order_key)
    if not paths:
        raise FileNotFoundError(f"No raw H5 files matching {pattern!r}.")
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def _read_aligned_shutter(
    h5_paths,
    idx_path: str,
    val_path: str,
    master_tID: np.ndarray,
) -> np.ndarray:
    """
    Return a per-train shutter array aligned to ``master_tID``. Source
    indices and values are concatenated across files (raw H5 train IDs
    are sorted ascending within and across files when files are read in
    file-index order). Multi-dim shutter samples are reduced to a scalar
    per train by ``nanmean`` over trailing axes.
    """
    src_idx_parts = []
    src_val_parts = []
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if (idx_path not in f) or (val_path not in f):
                continue
            src_idx_parts.append(f[idx_path][...])
            v = f[val_path][...]
            if v.dtype.kind in ("S", "U", "O"):
                raise NotImplementedError(
                    "Non-numeric shutter values are not supported."
                )
            if v.ndim > 1:
                v = np.nanmean(v.astype(np.float64),
                               axis=tuple(range(1, v.ndim)))
            src_val_parts.append(v.astype(np.float64))
    if not src_idx_parts:
        raise RuntimeError(
            f"Shutter datasets {idx_path}, {val_path} not found in any "
            "raw H5 file for this run."
        )
    src_idx = np.concatenate(src_idx_parts)
    src_val = np.concatenate(src_val_parts)
    out = np.full(master_tID.shape, np.nan, dtype=np.float64)
    matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
    out[matched] = src_val[src_pos]
    return out


def _contiguous_true_blocks(mask: np.ndarray):
    """Return [(start, stop), ...] for contiguous True runs (half-open)."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    x = m.astype(np.int8)
    starts = np.where(np.diff(np.r_[0, x]) == 1)[0]
    ends   = np.where(np.diff(np.r_[x, 0]) == -1)[0] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _classify_shutter(
    shutter: np.ndarray,
    train_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Threshold a numeric shutter signal into ``(is_open, is_closed,
    finite)`` boolean masks. Auto-flips if the open group has lower mean
    VLS score than the closed group.
    """
    arr = np.asarray(shutter, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise RuntimeError("Not enough finite shutter samples.")
    lo, hi = np.nanpercentile(arr[finite], [10, 90])
    thr = 0.5 * (lo + hi)
    is_open = arr >= thr
    is_closed = arr < thr
    # Safety flip if shutter polarity is inverted.
    open_score = np.nanmean(train_score[is_open]) if np.any(is_open) else np.nan
    closed_score = np.nanmean(train_score[is_closed]) if np.any(is_closed) else np.nan
    if np.isfinite(open_score) and np.isfinite(closed_score) and (open_score < closed_score):
        is_open, is_closed = is_closed, is_open
    return is_open, is_closed, finite


def _detect_sections(
    shutter: np.ndarray,
    vls_score: np.ndarray,
    transition_trim_trains: int,
):
    """
    Detect contiguous open / closed shutter sections after applying a
    transition-trim window around every shutter state change.

    Returns
    -------
    open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask
    """
    is_open_raw, is_closed_raw, finite_sh = _classify_shutter(shutter, vls_score)
    valid_mask = finite_sh.copy()
    transition_idx = np.where(np.diff(is_open_raw.astype(np.int8)) != 0)[0] + 1
    for t in transition_idx:
        i0 = max(0, t - transition_trim_trains)
        i1 = min(valid_mask.size, t + transition_trim_trains + 1)
        valid_mask[i0:i1] = False
    open_blocks   = _contiguous_true_blocks(is_open_raw   & finite_sh)
    closed_blocks = _contiguous_true_blocks(is_closed_raw & finite_sh)
    return open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask


def _preceding_closed_indices(
    open_start: int,
    closed_blocks,
    first_closed_idx: np.ndarray,
    first_section_state: str,
    section_index: int,
) -> np.ndarray:
    """
    Indices of the closed-shutter block to use as background for the
    open block starting at ``open_start``.

    Selection rule:
    - First section, when the scan opens with the shutter open
      (``FIRST_SECTION_STATE == "open"``): use the first closed block.
    - Otherwise: use the closed block immediately preceding this open
      block.
    """
    if section_index == 0 and first_section_state == "open":
        return first_closed_idx
    prev_block = None
    for cs, ce in closed_blocks:
        if ce <= open_start:
            prev_block = (cs, ce)
        else:
            break
    if prev_block is None:
        return np.array([], dtype=np.int64)
    ps, pe = prev_block
    return np.arange(ps, pe, dtype=np.int64)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def compute_xas_aggregates(
    output_h5,
    *,
    run_no: int,
    nominal_energies,
    gmd_edges,
    crop_roi: Tuple[int, int],
    signal_bunch_range: Tuple[int, int],
    bg_bunch_range: Tuple[int, int],
    config: int = 2,
    raw_dir=None,
    max_files: Optional[int] = None,
    train_length: Optional[int] = None,
    shutter_index_path: str = DEFAULT_SHUTTER_INDEX_PATH,
    shutter_value_path: str = DEFAULT_SHUTTER_VALUE_PATH,
    train_rate_hz: float = 10.0,
    transition_trim_seconds: float = 3.0,
    first_section_state: str = "open",
    group_by_energy: bool = True,
    config_path=None,
    verbose: bool = True,
) -> None:
    """
    Build per-(nominal photon energy, GMD bin) VLS / GMD aggregates from
    a raw FLASH H5 photon-energy scan and write them to ``output_h5``.

    Parameters
    ----------
    output_h5 : str or Path
    run_no : int
        FLASH run number; raw H5 files are read from ``raw_dir``.
    nominal_energies : array-like
        Nominal photon energies (eV) assigned to detected sections by
        section index. If more sections are detected than energies are
        provided, the extras are dropped with a warning; if fewer
        sections are detected, only the leading subset of
        ``nominal_energies`` is used.
    gmd_edges : array-like
        GMD bin edges (uJ). Use a single bin
        ``[-inf, +inf]`` to disable GMD binning.
    crop_roi : (int, int)
        Half-open VLS pixel ROI applied before any background
        subtraction.
    signal_bunch_range : (int, int)
        Half-open bunch range overlapping the GMD acquisition window.
    bg_bunch_range : (int, int)
        Half-open bunch range used for the per-train VLS background
        (applied to every train of the run before section logic runs).
    config : int
        Must be 2 (xas_scan is config-2 only).
    raw_dir : Path, optional
        Override the raw-H5 directory. Defaults to ``config.RAW_H5_DIR``.
    max_files, train_length :
        Forwarded to ``data_loading.load_raw_h5``.
    shutter_index_path, shutter_value_path : str
        HDF5 paths to the fast shutter index / value datasets.
    train_rate_hz, transition_trim_seconds : float
        Trim window around every shutter state change.
    first_section_state : "open" | "closed"
        Protocol prior controlling background selection for section 0.
    group_by_energy : bool
        If True (default), the leading axis of every aggregate is the
        per-section nominal photon energy (``N_E``). If False, all open
        sections collapse onto a single bin of size 1 — the output is
        effectively only binned on GMD. Per-section closed-shutter
        background subtraction still runs unchanged in both modes; only
        the accumulator axis is reduced. The ``nominal_energies``
        dataset becomes ``[NaN]`` in the not-grouped case and
        ``section_bg`` becomes the mean of the per-section backgrounds.
    config_path : Path, optional
        Recorded as a provenance attribute.
    verbose : bool
    """
    output_h5 = Path(output_h5)
    if int(config) != 2:
        raise ValueError("xas_scan mode is config=2 only.")

    log = print if verbose else (lambda *a, **k: None)

    nominal_energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
    gmd_edges = np.asarray(gmd_edges, dtype=np.float64).ravel()
    n_gmd = len(gmd_edges) - 1
    if n_gmd < 1:
        raise ValueError("gmd_edges must contain at least 2 entries.")

    roi_min, roi_max = int(crop_roi[0]), int(crop_roi[1])
    n_pixels = roi_max - roi_min
    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])
    bg_b0, bg_b1   = int(bg_bunch_range[0]),     int(bg_bunch_range[1])

    if raw_dir is None:
        import config as path_config
        raw_dir = Path(path_config.RAW_H5_DIR)
    else:
        raw_dir = Path(raw_dir)

    # ----- load full run (gmd + vls) ----------------------------------
    log(f"run                : {run_no}")
    log(f"raw dir            : {raw_dir}")
    log(f"GMD bins           : {n_gmd}  edges {gmd_edges}")
    log(f"nominal energies   : {nominal_energies.size}  "
        f"({nominal_energies[0]:.2f} .. {nominal_energies[-1]:.2f} eV)")
    log(f"VLS ROI            : [{roi_min}, {roi_max}) = {n_pixels} pixels")
    log(f"signal bunches     : [{sig_b0}, {sig_b1})")
    log(f"bg bunches         : [{bg_b0}, {bg_b1})")

    data = data_loading.load_raw_h5(
        run_no, config=2, raw_dir=raw_dir,
        train_length=train_length, max_files=max_files,
    )
    if data.vls is None:
        raise RuntimeError("load_raw_h5 returned no VLS data for this run.")

    # Crop to ROI up front so all downstream arrays use the cropped
    # pixel axis. Then subtract a per-train background (mean over the
    # non-signal bunches of every train), so the section-level
    # closed-shutter average that follows is computed on already
    # baseline-corrected spectra.
    data = data.crop_vls(roi_min, roi_max)
    m = data.vls.shape[1]
    if not (0 <= sig_b0 < sig_b1 <= m):
        raise ValueError(f"signal_bunch_range {signal_bunch_range} not within [0, {m}].")
    if not (0 <= bg_b0 < bg_b1 <= m):
        raise ValueError(f"bg_bunch_range {bg_bunch_range} not within [0, {m}].")
    data = data.auto_subtract_background_trainwise((bg_b0, bg_b1))

    vls = np.asarray(data.vls, dtype=np.float64)   # (n_trains, m, n_pixels)
    gmd = np.asarray(data.gmd, dtype=np.float64)   # (n_trains, m)
    n_trains, m, n_pix_check = vls.shape
    if n_pix_check != n_pixels:
        raise RuntimeError(
            f"unexpected VLS pixel width {n_pix_check} after crop "
            f"(expected {n_pixels})."
        )

    # ----- read shutter, detect sections ------------------------------
    h5_paths = _list_raw_h5_files(run_no, raw_dir, max_files)
    shutter = _read_aligned_shutter(
        h5_paths, shutter_index_path, shutter_value_path, data.tID,
    )

    # Per-train VLS score for polarity-safety: mean over signal bunches
    # of the per-bunch integrated intensity. Robust to NaN-padded rows.
    train_score = np.nanmean(
        np.nansum(vls[:, sig_b0:sig_b1, :], axis=2),
        axis=1,
    )

    transition_trim_trains = int(round(transition_trim_seconds * train_rate_hz))
    open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask = (
        _detect_sections(shutter, train_score, transition_trim_trains)
    )
    if not open_blocks:
        raise RuntimeError("No open-shutter sections detected.")
    if not closed_blocks:
        raise RuntimeError("No closed-shutter sections detected.")

    first_section_state = str(first_section_state).strip().lower()
    if first_section_state not in ("open", "closed"):
        raise ValueError("first_section_state must be 'open' or 'closed'.")

    # Truncate the section list to the smaller of (detected, requested).
    n_detected = len(open_blocks)
    n_requested = nominal_energies.size
    n_e = min(n_detected, n_requested)
    if n_detected != n_requested:
        log(f"warning: detected {n_detected} open sections vs "
            f"{n_requested} nominal energies; using leading {n_e}.")
    open_blocks      = open_blocks[:n_e]
    energies         = nominal_energies[:n_e]
    first_closed_idx = np.arange(closed_blocks[0][0], closed_blocks[0][1],
                                 dtype=np.int64)

    # ----- allocate aggregates ----------------------------------------
    # Output leading axis: per-section when grouping by energy, single
    # collapsed bin otherwise. Per-section backgrounds are tracked at
    # the full per-section resolution and only collapsed at write time.
    n_e_out = n_e if group_by_energy else 1
    A_sum   = np.zeros((n_e_out, n_gmd, n_pixels),            dtype=np.float64)
    AtA_sum = np.zeros((n_e_out, n_gmd, n_pixels, n_pixels),  dtype=np.float64)
    AtG_sum = np.zeros((n_e_out, n_gmd, n_pixels),            dtype=np.float64)
    G_sum   = np.zeros((n_e_out, n_gmd),                      dtype=np.float64)
    GtG_sum = np.zeros((n_e_out, n_gmd),                      dtype=np.float64)
    n_per_bin = np.zeros((n_e_out, n_gmd),                    dtype=np.int64)
    section_bg = np.full((n_e, m, n_pixels), np.nan,          dtype=np.float64)

    n_skipped_gmd_oor = 0
    n_skipped_gmd_nan = 0
    n_skipped_nan_vls = 0

    log(f"group_by_energy    : {group_by_energy} "
        f"(output energy axis = {n_e_out})")
    log(f"accumulating over {n_e} sections...")

    # ----- main loop over open sections -------------------------------
    for i, (s, e) in enumerate(open_blocks):
        sec_open_mask = np.zeros(n_trains, dtype=bool)
        sec_open_mask[s:e] = True
        sec_open_mask &= is_open_raw & valid_mask
        sec_open_idx = np.where(sec_open_mask)[0]
        if sec_open_idx.size == 0:
            log(f"  section {i:>3d} (E={energies[i]:.2f} eV): "
                "no open trains after trim; skipping")
            continue

        # Per-section closed-shutter background (2D over bunches x pixels).
        sec_closed_idx = _preceding_closed_indices(
            s, closed_blocks, first_closed_idx,
            first_section_state, i,
        )
        if sec_closed_idx.size == 0:
            # Conservative fallback: use all closed trains.
            sec_closed_idx = np.where(is_closed_raw & valid_mask)[0]
        if sec_closed_idx.size == 0:
            raise RuntimeError(
                f"section {i}: no closed trains available for background."
            )

        # Per-section background: mean over the trains of the preceding
        # closed section. The per-train baseline has already been
        # removed by auto_subtract_background_trainwise above, so this
        # captures any residual shutter-state-dependent offset.
        bg2d = np.nanmean(vls[sec_closed_idx, :, :], axis=0)   # (m, n_pixels)
        section_bg[i] = bg2d

        # Apply the per-section background and keep only the signal bunches.
        A_block = (
            vls[sec_open_idx][:, sig_b0:sig_b1, :]
            - bg2d[None, sig_b0:sig_b1, :]
        )                                                       # (n_sec, n_sig, n_pixels)
        G_block = gmd[sec_open_idx, sig_b0:sig_b1]              # (n_sec, n_sig)

        # Flatten over (train, bunch) -> shots.
        A_flat = A_block.reshape(-1, n_pixels)
        G_flat = G_block.reshape(-1)

        # GMD bin per shot.
        gmd_bin = np.digitize(G_flat, gmd_edges) - 1
        gmd_nan = ~np.isfinite(G_flat)
        gmd_oor = ~gmd_nan & ((gmd_bin < 0) | (gmd_bin >= n_gmd))
        vls_ok  = np.all(np.isfinite(A_flat), axis=1)
        valid   = ~gmd_nan & ~gmd_oor & vls_ok

        n_skipped_gmd_nan += int(gmd_nan.sum())
        n_skipped_gmd_oor += int(gmd_oor.sum())
        n_skipped_nan_vls += int((~vls_ok & ~gmd_nan & ~gmd_oor).sum())

        e_idx = i if group_by_energy else 0

        for b in np.unique(gmd_bin[valid]):
            mask = (gmd_bin == b) & valid
            A_b = A_flat[mask]
            G_b = G_flat[mask]
            n_b = G_b.shape[0]

            n_per_bin[e_idx, b] += n_b
            G_sum[e_idx, b]   += G_b.sum()
            GtG_sum[e_idx, b] += float(G_b @ G_b)
            A_sum[e_idx, b]   += A_b.sum(axis=0)
            AtA_sum[e_idx, b] += A_b.T @ A_b
            AtG_sum[e_idx, b] += A_b.T @ G_b

        log(f"  section {i:>3d} (E={energies[i]:.2f} eV): "
            f"open trains={sec_open_idx.size:>4d}  "
            f"closed bg trains={sec_closed_idx.size:>4d}  "
            f"shots binned={int(valid.sum()):>6d}")

    # ----- means ------------------------------------------------------
    safe_n = np.where(n_per_bin > 0, n_per_bin, 1).astype(np.float64)
    empty = n_per_bin == 0

    def _norm_scalar(s):
        m = s / safe_n
        m[empty] = np.nan
        return m

    def _norm_vec(s):
        m = s / safe_n[..., None]
        m[empty] = np.nan
        return m

    def _norm_mat(s):
        m = s / safe_n[..., None, None]
        m[empty] = np.nan
        return m

    A_mean   = _norm_vec(A_sum)
    AtA_mean = _norm_mat(AtA_sum)
    AtG_mean = _norm_vec(AtG_sum)
    G_mean   = _norm_scalar(G_sum)
    GtG_mean = _norm_scalar(GtG_sum)

    if group_by_energy:
        energies_out   = energies
        section_bg_out = section_bg
    else:
        energies_out = np.array([np.nan], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            section_bg_out = np.nanmean(section_bg, axis=0, keepdims=True)

    # ----- write ------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("A",   data=A_mean,   compression="gzip")
        fout.create_dataset("AtA", data=AtA_mean, compression="gzip")
        fout.create_dataset("AtG", data=AtG_mean, compression="gzip")
        fout.create_dataset("G",   data=G_mean,   compression="gzip")
        fout.create_dataset("GtG", data=GtG_mean, compression="gzip")
        fout.create_dataset("n_per_bin", data=n_per_bin)
        fout.create_dataset("gmd_edges", data=gmd_edges)
        fout.create_dataset("nominal_energies", data=energies_out)
        fout.create_dataset("vls_pixels",
                            data=np.arange(roi_min, roi_max, dtype=np.int64))
        fout.create_dataset("section_bg", data=section_bg_out, compression="gzip")

        fout.attrs["mode"]                = "xas_scan"
        fout.attrs["config"]              = int(config)
        fout.attrs["run_no"]              = int(run_no)
        fout.attrs["raw_dir"]             = str(raw_dir)
        fout.attrs["n_sections_detected"] = int(n_detected)
        fout.attrs["n_sections_used"]     = int(n_e)
        fout.attrs["group_by_energy"]     = bool(group_by_energy)
        fout.attrs["vls_crop_roi"]        = np.asarray([roi_min, roi_max], dtype=np.int64)
        fout.attrs["signal_bunch_range"]  = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["bg_bunch_range"]      = np.asarray([bg_b0, bg_b1], dtype=np.int64)
        fout.attrs["train_rate_hz"]       = float(train_rate_hz)
        fout.attrs["transition_trim_seconds"] = float(transition_trim_seconds)
        fout.attrs["transition_trim_trains"]  = int(transition_trim_trains)
        fout.attrs["first_section_state"] = first_section_state
        fout.attrs["shutter_index_path"]  = shutter_index_path
        fout.attrs["shutter_value_path"]  = shutter_value_path
        fout.attrs["n_skipped_gmd_oor"]   = int(n_skipped_gmd_oor)
        fout.attrs["n_skipped_gmd_nan"]   = int(n_skipped_gmd_nan)
        fout.attrs["n_skipped_nan_vls"]   = int(n_skipped_nan_vls)
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    bin_label = "(energy x GMD)" if group_by_energy else "(GMD-only)"
    summary = (
        f"done: {int(n_per_bin.sum())} shots aggregated across "
        f"{n_e_out} x {n_gmd} = {n_e_out * n_gmd} {bin_label} bins "
        f"from {n_e} open section(s); "
        f"skipped {n_skipped_gmd_oor} GMD-oor, "
        f"{n_skipped_gmd_nan} GMD-nan, {n_skipped_nan_vls} VLS-nan."
    )
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
        help="Python module exposing RUN_NO, NOMINAL_ENERGIES, GMD_EDGES, "
             "CROP_ROI, SIGNAL_BUNCH_RANGE, BG_BUNCH_RANGE, CONFIG=2, "
             "MODE='xas_scan', plus optional shutter knobs.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output H5 (default: <COMBINED_DIR>/run<RUN_NO>_xas_aggregates.h5)",
    )
    args = parser.parse_args(argv)

    cfg = load_python_config(args.config_path)

    if int(getattr(cfg, "CONFIG", 2)) != 2:
        raise ValueError("compute_xas_aggregates requires CONFIG = 2.")
    mode = getattr(cfg, "MODE", "xas_scan")
    if mode != "xas_scan":
        raise ValueError(
            f"this entry point expects MODE='xas_scan', got {mode!r}."
        )

    run_no = int(cfg.RUN_NO)
    if args.output is None:
        import config as path_config
        out = Path(path_config.COMBINED_DIR) / f"run{run_no}_xas_aggregates.h5"
    else:
        out = args.output

    compute_xas_aggregates(
        output_h5=out,
        run_no=run_no,
        nominal_energies=np.asarray(cfg.NOMINAL_ENERGIES, dtype=float),
        gmd_edges=np.asarray(cfg.GMD_EDGES, dtype=float),
        crop_roi=tuple(cfg.CROP_ROI),
        signal_bunch_range=tuple(cfg.SIGNAL_BUNCH_RANGE),
        bg_bunch_range=tuple(cfg.BG_BUNCH_RANGE),
        config=int(getattr(cfg, "CONFIG", 2)),
        raw_dir=getattr(cfg, "RAW_DIR", None),
        max_files=getattr(cfg, "MAX_FILES", None),
        train_length=getattr(cfg, "TRAIN_LENGTH", None),
        shutter_index_path=getattr(cfg, "SHUTTER_INDEX_PATH",
                                   DEFAULT_SHUTTER_INDEX_PATH),
        shutter_value_path=getattr(cfg, "SHUTTER_VALUE_PATH",
                                   DEFAULT_SHUTTER_VALUE_PATH),
        train_rate_hz=float(getattr(cfg, "TRAIN_RATE_HZ", 10.0)),
        transition_trim_seconds=float(
            getattr(cfg, "TRANSITION_TRIM_SECONDS", 3.0)
        ),
        first_section_state=str(getattr(cfg, "FIRST_SECTION_STATE", "open")),
        group_by_energy=bool(getattr(cfg, "GROUP_BY_ENERGY", True)),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
