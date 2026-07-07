"""
Collect per-shot (VLS, GMD) for a photon-energy XAS scan.

Runs the same pipeline as ``compute_xas_aggregates.py`` — VLS pixel
crop, VLS bunch-axis roll to align with GMD, per-train background
subtraction, fast-shutter section detection, per-section closed-shutter
background subtraction, signal-bunch selection — but stops after
flattening (train x bunch -> shot).  No GMD binning is performed; the
raw shot-level data is written for flexible downstream analysis and
visualisation.

Output H5 layout (default: <processed/xas_static>/run<N>_static_xas.h5):
    /vls               (N_E, N_shots, n_pixels)  float64, NaN-padded
    /gmd               (N_E, N_shots)            float64, NaN-padded, hall
    /gmd_tunnel        (N_E, N_shots)            float64, NaN-padded
    /n_shots           (N_E,)                    int64
    /nominal_energies  (N_E,)                    float64  eV
    /vls_pixels        (n_pixels,)               int64    source-pixel indices
    /section_bg        (N_E, m_bunches, n_pixels) float64 per-section bg
    attrs:  mode='xas_static', config, run_no, vls_crop_roi,
            vls_bunch_roll, signal_bunch_range, bg_bunch_range,
            n_sections_detected, n_sections_used,
            transition_trim_trains, ...

Config file
-----------
Python module exposing: RUN_NO, NOMINAL_ENERGIES, CROP_ROI,
SIGNAL_BUNCH_RANGE, BG_BUNCH_RANGE, CONFIG=2, plus optional shutter /
trim knobs. No GMD_EDGES needed. ``MODE`` may be any of ``"xas"``,
``"xas_scan"``, or ``"xas_static"``. The same config file also drives
``compute_xas_aggregates.py``; that script consumes the extra
GMD-binning fields when present. See ``analysis/configs/xas_static/``
and ``analysis/configs/aggregates_example_xas.py``.

CLI
---
    python compute_static_xas.py CONFIG.py [-o OUT.h5]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

import data_loading

__all__ = ["compute_static_xas", "main"]

# ---------------------------------------------------------------------------
# Shutter HDF5 path defaults
# ---------------------------------------------------------------------------

_DEFAULT_SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
_DEFAULT_SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"

_GMD_TUNNEL_INDEX_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/index",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/tunnel/index",
)
_GMD_TUNNEL_VALUE_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/value",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/tunnel/value",
)

# Config MODE strings accepted by both this script and
# compute_xas_aggregates. The two pipelines share inputs and
# preprocessing knobs; a single config can drive either entry point.
# The *output* H5 still gets a distinguishing ``mode`` attribute set by
# whichever script wrote it (``xas_static`` here, ``xas_scan`` in
# compute_xas_aggregates).
ACCEPTED_XAS_MODES = ("xas", "xas_scan", "xas_static")

# ---------------------------------------------------------------------------
# Internal helpers  (self-contained — no imports from other analysis scripts)
# ---------------------------------------------------------------------------

def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("static_xas_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _normalize_run_numbers(run_no):
    if isinstance(run_no, str):
        parts = run_no.replace(",", " ").split()
        return [int(p) for p in parts]
    if np.isscalar(run_no):
        return [int(run_no)]
    return [int(r) for r in run_no]


def _run_label(run_numbers):
    if len(run_numbers) == 1:
        return f"run{run_numbers[0]}"
    return "runs" + "_".join(str(r) for r in run_numbers)


def _list_raw_h5_files_for_runs(run_numbers, raw_dir: Path, max_files: Optional[int]):
    paths = []
    for run in run_numbers:
        paths.extend(_list_raw_h5_files(run, raw_dir, max_files))
    return paths


def _concat_experiment_data(parts):
    if len(parts) == 1:
        return parts[0]

    def cat(name):
        arrays = [getattr(p, name) for p in parts]
        if arrays[0] is None:
            return None
        return np.concatenate(arrays, axis=0)

    return replace(
        parts[0],
        tID=cat("tID"),
        gmd=cat("gmd"),
        mpe=cat("mpe"),
        z=cat("z"),
        between_tdc_files=cat("between_tdc_files"),
        local_daq_running=cat("local_daq_running"),
        tofs_e=cat("tofs_e"),
        tofs_i=cat("tofs_i"),
        liq_tofs_e=cat("liq_tofs_e"),
        vls=cat("vls"),
        shutter=cat("shutter"),
        shot_mask=cat("shot_mask"),
    )


def _read_aligned_shutter(
    h5_paths,
    idx_path: str,
    val_path: str,
    master_tID: np.ndarray,
) -> np.ndarray:
    """
    Concatenate shutter index/value across files and align to
    ``master_tID``. Multi-dim samples are reduced via nanmean.
    """
    src_idx_parts, src_val_parts = [], []
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if (idx_path not in f) or (val_path not in f):
                continue
            src_idx_parts.append(f[idx_path][...])
            v = f[val_path][...]
            if v.dtype.kind in ("S", "U", "O"):
                raise NotImplementedError("Non-numeric shutter values are not supported.")
            if v.ndim > 1:
                v = np.nanmean(v.astype(np.float64), axis=tuple(range(1, v.ndim)))
            src_val_parts.append(v.astype(np.float64))
    if not src_idx_parts:
        raise RuntimeError(
            f"Shutter datasets {idx_path!r}, {val_path!r} not found in any raw H5 file."
        )
    src_idx = np.concatenate(src_idx_parts)
    src_val = np.concatenate(src_val_parts)
    out = np.full(master_tID.shape, np.nan, dtype=np.float64)
    matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
    out[matched] = src_val[src_pos]
    return out


def _read_aligned_pulse_gmd(
    h5_paths,
    index_candidates,
    value_candidates,
    master_tID: np.ndarray,
    train_length: int,
    *,
    channel: int = 0,
):
    """
    Concatenate pulse-resolved GMD index/value datasets and align to
    ``master_tID``. Missing trains and bunches are NaN-filled.

    Returns
    -------
    out, index_path, value_path
        ``out`` is ``None`` if no matching path pair was found.
    """
    index_path = value_path = None
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path is None:
                for p in index_candidates:
                    if p in f:
                        index_path = p
                        break
            if value_path is None:
                for p in value_candidates:
                    if p in f:
                        value_path = p
                        break
        if index_path is not None and value_path is not None:
            break

    if index_path is None or value_path is None:
        return None, index_path, value_path

    out = np.full((master_tID.shape[0], train_length), np.nan, dtype=np.float64)
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path not in f or value_path not in f:
                continue
            src_idx = f[index_path][...]
            val = f[value_path]
            if val.ndim == 3:
                m = min(val.shape[2], train_length)
                src_val = val[:, int(channel), :m]
            elif val.ndim == 2:
                m = min(val.shape[1], train_length)
                src_val = val[:, :m]
            elif val.ndim == 1:
                m = 1
                src_val = val[:, None]
            else:
                continue
            matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
            if matched.any():
                out[matched, :m] = np.asarray(src_val[src_pos], dtype=np.float64)
    return out, index_path, value_path


def _contiguous_true_blocks(mask: np.ndarray):
    """Return [(start, stop), ...] half-open intervals for True runs."""
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
    Threshold shutter signal into (is_open, is_closed, finite) masks.
    Auto-flips polarity if the open group has lower mean VLS score.
    """
    arr = np.asarray(shutter, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise RuntimeError("Not enough finite shutter samples to classify.")
    lo, hi = np.nanpercentile(arr[finite], [10, 90])
    thr = 0.5 * (lo + hi)
    is_open   = arr >= thr
    is_closed = arr < thr
    open_score   = np.nanmean(train_score[is_open])   if np.any(is_open)   else np.nan
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
    Detect contiguous open / closed sections after trimming trains
    around every shutter state change.

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
    Return train indices of the closed block to use as background for an
    open section.

    - section_index == 0 and first_section_state == "open": use the
      first closed block (which follows the first open section).
    - Otherwise: use the closed block immediately preceding open_start.
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_static_xas(
    output_h5,
    *,
    run_no,
    nominal_energies,
    crop_roi: Tuple[int, int],
    signal_bunch_range: Tuple[int, int],
    bg_bunch_range: Tuple[int, int],
    vls_bunch_roll: int = 0,
    config: int = 2,
    raw_dir=None,
    max_files: Optional[int] = None,
    train_length: Optional[int] = None,
    shutter_index_path: str = _DEFAULT_SHUTTER_INDEX_PATH,
    shutter_value_path: str = _DEFAULT_SHUTTER_VALUE_PATH,
    train_rate_hz: float = 10.0,
    transition_trim_seconds: float = 3.0,
    first_section_state: str = "open",
    config_path=None,
    verbose: bool = True,
) -> None:
    """
    Run the static-XAS shot-collection pipeline and write to ``output_h5``.

    Same processing as ``compute_xas_aggregates`` (pixel crop → per-train
    bg subtraction → shutter detection → per-section bg subtraction →
    signal-bunch selection → flatten to shots) but the result is stored
    as raw per-shot arrays rather than pre-binned aggregates.

    Parameters
    ----------
    output_h5 : str or Path
    run_no : int or sequence of int
    nominal_energies : array-like
        Energies (eV) assigned to detected sections by index.
    crop_roi : (int, int)
        Half-open VLS pixel ROI.
    signal_bunch_range : (int, int)
        Half-open bunch range for signal shots.
    bg_bunch_range : (int, int)
        Half-open bunch range for per-train baseline subtraction.
    vls_bunch_roll : int
        Cyclic shift (``np.roll``) applied to the VLS bunch axis right
        after the pixel crop, aligning the VLS bunch coordinate with
        the GMD bunch coordinate. ``signal_bunch_range`` and
        ``bg_bunch_range`` apply in the rolled frame.
    config : int
        Must be 2 (xas_static is config-2 only).
    raw_dir : Path, optional
        Overrides ``config.RAW_H5_DIR``.
    max_files, train_length :
        Forwarded to ``data_loading.load_raw_h5``.
    shutter_index_path, shutter_value_path : str
    train_rate_hz, transition_trim_seconds : float
    first_section_state : "open" | "closed"
    config_path : Path, optional  (recorded as a provenance attribute)
    verbose : bool
    """
    output_h5 = Path(output_h5)
    if int(config) != 2:
        raise ValueError("xas_static mode is config=2 only.")

    log = print if verbose else (lambda *a, **k: None)

    run_numbers = _normalize_run_numbers(run_no)
    if not run_numbers:
        raise ValueError("run_no must contain at least one run number.")

    nominal_energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
    roi_min, roi_max = int(crop_roi[0]), int(crop_roi[1])
    n_pixels = roi_max - roi_min
    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])
    bg_b0,  bg_b1  = int(bg_bunch_range[0]),     int(bg_bunch_range[1])
    vls_bunch_roll = int(vls_bunch_roll)

    first_section_state = str(first_section_state).strip().lower()
    if first_section_state not in ("open", "closed"):
        raise ValueError("first_section_state must be 'open' or 'closed'.")

    if raw_dir is None:
        import config as path_config
        raw_dir = Path(path_config.RAW_H5_DIR)
    else:
        raw_dir = Path(raw_dir)

    log(f"run                : {_run_label(run_numbers)}")
    log(f"raw dir            : {raw_dir}")
    log(f"nominal energies   : {nominal_energies.size}  "
        f"({nominal_energies[0]:.2f} .. {nominal_energies[-1]:.2f} eV)")
    log(f"VLS ROI            : [{roi_min}, {roi_max}) = {n_pixels} pixels")
    log(f"VLS bunch roll     : {vls_bunch_roll}")
    log(f"signal bunches     : [{sig_b0}, {sig_b1})")
    log(f"bg bunches         : [{bg_b0}, {bg_b1})")

    # ------------------------------------------------------------------
    # Load full run (gmd + vls), crop, per-train baseline subtraction
    # ------------------------------------------------------------------
    data = _concat_experiment_data(
        [
            data_loading.load_raw_h5(
                run, config=2, raw_dir=raw_dir,
                train_length=train_length, max_files=max_files,
            )
            for run in run_numbers
        ]
    )
    if data.vls is None:
        raise RuntimeError("load_raw_h5 returned no VLS data for this run.")

    data = data.crop_vls(roi_min, roi_max)
    data = data.roll_vls_bunches(vls_bunch_roll)
    m = data.vls.shape[1]
    if not (0 <= sig_b0 < sig_b1 <= m):
        raise ValueError(f"signal_bunch_range {signal_bunch_range} not within [0, {m}].")
    if not (0 <= bg_b0 < bg_b1 <= m):
        raise ValueError(f"bg_bunch_range {bg_bunch_range} not within [0, {m}].")
    data = data.auto_subtract_background_trainwise((bg_b0, bg_b1))

    vls = np.asarray(data.vls, dtype=np.float64)   # (n_trains, m, n_pixels)
    gmd = np.asarray(data.gmd, dtype=np.float64)   # (n_trains, m), hall GMD
    n_trains = vls.shape[0]

    h5_paths = _list_raw_h5_files_for_runs(run_numbers, raw_dir, max_files)
    gmd_tunnel, gmd_tunnel_index_path, gmd_tunnel_value_path = _read_aligned_pulse_gmd(
        h5_paths,
        _GMD_TUNNEL_INDEX_CANDIDATES,
        _GMD_TUNNEL_VALUE_CANDIDATES,
        data.tID,
        m,
        channel=0,
    )
    gmd_tunnel_present = gmd_tunnel is not None
    if gmd_tunnel is None:
        gmd_tunnel = np.full_like(gmd, np.nan, dtype=np.float64)
        log("GMD tunnel         : not found; /gmd_tunnel will be NaN-filled")
    else:
        log(f"GMD tunnel         : {gmd_tunnel_value_path}")

    # ------------------------------------------------------------------
    # Shutter section detection
    # ------------------------------------------------------------------
    shutter  = _read_aligned_shutter(
        h5_paths, shutter_index_path, shutter_value_path, data.tID,
    )
    train_score = np.nanmean(
        np.nansum(vls[:, sig_b0:sig_b1, :], axis=2), axis=1,
    )
    transition_trim_trains = int(round(transition_trim_seconds * train_rate_hz))
    open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask = (
        _detect_sections(shutter, train_score, transition_trim_trains)
    )
    if not open_blocks:
        raise RuntimeError("No open-shutter sections detected.")
    if not closed_blocks:
        raise RuntimeError("No closed-shutter sections detected.")

    n_detected  = len(open_blocks)
    n_e = min(n_detected, nominal_energies.size)
    if n_detected != nominal_energies.size:
        log(f"warning: detected {n_detected} open sections vs "
            f"{nominal_energies.size} nominal energies; using leading {n_e}.")
    open_blocks      = open_blocks[:n_e]
    energies         = nominal_energies[:n_e]
    first_closed_idx = np.arange(
        closed_blocks[0][0], closed_blocks[0][1], dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Collect shots per section
    # ------------------------------------------------------------------
    section_bg  = np.full((n_e, m, n_pixels), np.nan, dtype=np.float64)
    vls_shots: list[np.ndarray] = []
    gmd_shots: list[np.ndarray] = []
    gmd_tunnel_shots: list[np.ndarray] = []
    n_shots_arr = np.zeros(n_e, dtype=np.int64)

    log(f"collecting shots over {n_e} sections...")
    for i, (s, e) in enumerate(open_blocks):
        sec_open_mask = np.zeros(n_trains, dtype=bool)
        sec_open_mask[s:e] = True
        sec_open_mask &= is_open_raw & valid_mask
        sec_open_idx = np.where(sec_open_mask)[0]

        if sec_open_idx.size == 0:
            log(f"  section {i:>3d} (E={energies[i]:.2f} eV): no open trains; skipping")
            vls_shots.append(np.empty((0, n_pixels), dtype=np.float64))
            gmd_shots.append(np.empty(0, dtype=np.float64))
            gmd_tunnel_shots.append(np.empty(0, dtype=np.float64))
            continue

        sec_closed_idx = _preceding_closed_indices(
            s, closed_blocks, first_closed_idx, first_section_state, i,
        )
        if sec_closed_idx.size == 0:
            sec_closed_idx = np.where(is_closed_raw & valid_mask)[0]
        if sec_closed_idx.size == 0:
            raise RuntimeError(f"section {i}: no closed trains available for background.")

        bg2d = np.nanmean(vls[sec_closed_idx, :, :], axis=0)   # (m, n_pixels)
        section_bg[i] = bg2d

        A_block = (                                             # (n_sec, n_sig, n_pixels)
            vls[sec_open_idx][:, sig_b0:sig_b1, :]
            - bg2d[None, sig_b0:sig_b1, :]
        )
        G_block = gmd[sec_open_idx, sig_b0:sig_b1]              # (n_sec, n_sig)
        G_tunnel_block = gmd_tunnel[sec_open_idx, sig_b0:sig_b1]

        A_flat = A_block.reshape(-1, n_pixels)
        G_flat = G_block.reshape(-1)
        G_tunnel_flat = G_tunnel_block.reshape(-1)

        n_shots_arr[i] = A_flat.shape[0]
        vls_shots.append(A_flat)
        gmd_shots.append(G_flat)
        gmd_tunnel_shots.append(G_tunnel_flat)
        log(f"  section {i:>3d} (E={energies[i]:.2f} eV): "
            f"open trains={sec_open_idx.size:>4d}  "
            f"bg trains={sec_closed_idx.size:>4d}  "
            f"shots={A_flat.shape[0]:>6d}")

    # ------------------------------------------------------------------
    # NaN-pad to rectangular arrays
    # ------------------------------------------------------------------
    n_shots_max = int(n_shots_arr.max()) if n_e > 0 else 0
    vls_out = np.full((n_e, n_shots_max, n_pixels), np.nan, dtype=np.float64)
    gmd_out = np.full((n_e, n_shots_max),            np.nan, dtype=np.float64)
    gmd_tunnel_out = np.full((n_e, n_shots_max),     np.nan, dtype=np.float64)
    for i, (A, G, Gt) in enumerate(zip(vls_shots, gmd_shots, gmd_tunnel_shots)):
        n = A.shape[0]
        if n > 0:
            vls_out[i, :n] = A
            gmd_out[i, :n] = G
            gmd_tunnel_out[i, :n] = Gt

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}  (vls: {vls_out.shape}, gmd: {gmd_out.shape})")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("vls",              data=vls_out,     compression="gzip")
        fout.create_dataset("gmd",              data=gmd_out,     compression="gzip")
        fout.create_dataset("gmd_tunnel",       data=gmd_tunnel_out, compression="gzip")
        fout.create_dataset("n_shots",          data=n_shots_arr)
        fout.create_dataset("nominal_energies", data=energies)
        fout.create_dataset("vls_pixels",
                            data=np.arange(roi_min, roi_max, dtype=np.int64))
        fout.create_dataset("section_bg",       data=section_bg,  compression="gzip")

        fout.attrs["mode"]                    = "xas_static"
        fout.attrs["config"]                  = int(config)
        fout.attrs["run_no"]                  = (
            int(run_numbers[0]) if len(run_numbers) == 1
            else np.asarray(run_numbers, dtype=np.int64)
        )
        fout.attrs["raw_dir"]                 = str(raw_dir)
        fout.attrs["n_sections_detected"]     = int(n_detected)
        fout.attrs["n_sections_used"]         = int(n_e)
        fout.attrs["vls_crop_roi"]            = np.asarray([roi_min, roi_max], dtype=np.int64)
        fout.attrs["vls_bunch_roll"]          = int(vls_bunch_roll)
        fout.attrs["signal_bunch_range"]      = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["bg_bunch_range"]          = np.asarray([bg_b0, bg_b1],   dtype=np.int64)
        fout.attrs["train_rate_hz"]           = float(train_rate_hz)
        fout.attrs["transition_trim_seconds"] = float(transition_trim_seconds)
        fout.attrs["transition_trim_trains"]  = int(transition_trim_trains)
        fout.attrs["first_section_state"]     = first_section_state
        fout.attrs["shutter_index_path"]      = shutter_index_path
        fout.attrs["shutter_value_path"]      = shutter_value_path
        fout.attrs["gmd_tunnel_present"]      = bool(gmd_tunnel_present)
        if gmd_tunnel_index_path is not None:
            fout.attrs["gmd_tunnel_index_path"] = gmd_tunnel_index_path
        if gmd_tunnel_value_path is not None:
            fout.attrs["gmd_tunnel_value_path"] = gmd_tunnel_value_path
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    log(f"done: {int(n_shots_arr.sum())} shots across {n_e} energies.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-shot VLS/GMD for a static XAS scan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Config module exposing RUN_NO, NOMINAL_ENERGIES, CROP_ROI, "
             "SIGNAL_BUNCH_RANGE, BG_BUNCH_RANGE, CONFIG=2.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output H5 path (default: processed/xas_static/run<N>_static_xas.h5)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config_path)

    if int(getattr(cfg, "CONFIG", 2)) != 2:
        raise ValueError("compute_static_xas requires CONFIG = 2.")
    mode = getattr(cfg, "MODE", "xas")
    if mode not in ACCEPTED_XAS_MODES:
        raise ValueError(
            f"this entry point expects MODE in {ACCEPTED_XAS_MODES}, "
            f"got {mode!r}."
        )

    run_no = _normalize_run_numbers(cfg.RUN_NO)
    if args.output is None:
        import config as path_config
        out_dir = Path(path_config.COMBINED_DIR).parent / "xas_static"
        out = out_dir / f"{_run_label(run_no)}_static_xas.h5"
    else:
        out = args.output

    compute_static_xas(
        output_h5=out,
        run_no=run_no,
        nominal_energies=np.asarray(cfg.NOMINAL_ENERGIES, dtype=float),
        crop_roi=tuple(cfg.CROP_ROI),
        signal_bunch_range=tuple(cfg.SIGNAL_BUNCH_RANGE),
        bg_bunch_range=tuple(cfg.BG_BUNCH_RANGE),
        vls_bunch_roll=int(getattr(cfg, "VLS_BUNCH_ROLL", 0)),
        config=int(getattr(cfg, "CONFIG", 2)),
        raw_dir=getattr(cfg, "RAW_DIR", None),
        max_files=getattr(cfg, "MAX_FILES", None),
        train_length=getattr(cfg, "TRAIN_LENGTH", None),
        shutter_index_path=getattr(cfg, "SHUTTER_INDEX_PATH",
                                   _DEFAULT_SHUTTER_INDEX_PATH),
        shutter_value_path=getattr(cfg, "SHUTTER_VALUE_PATH",
                                   _DEFAULT_SHUTTER_VALUE_PATH),
        train_rate_hz=float(getattr(cfg, "TRAIN_RATE_HZ", 10.0)),
        transition_trim_seconds=float(getattr(cfg, "TRANSITION_TRIM_SECONDS", 3.0)),
        first_section_state=str(getattr(cfg, "FIRST_SECTION_STATE", "open")),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
