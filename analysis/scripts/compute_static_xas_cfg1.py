"""
Collect per-shot (TOF_e, TOF_i, GMD) for a photon-energy XAS scan in
config 1 (electron + ion TOF).

Reads a combined H5 file (built by ``write_h5.py``) which now carries
``/shutter`` per train, detects shutter open / closed sections, assigns
each open section a nominal photon energy, and flattens its signal-bunch
hits into a single (n_shots, max_hits) array per energy. Stops at
flattening — no GMD binning, no TOF histogramming, no background
subtraction.

Output H5 layout (default:
``<processed/xas_static>/run<N>_cfg1_static_xas.h5``)::

    /tofs_e            (N_E, N_shots, max_ecounts)  float32, zero-padded
    /tofs_i            (N_E, N_shots, max_icounts)  float32, zero-padded
    /gmd               (N_E, N_shots)               float64, NaN-padded
    /n_shots           (N_E,)                       int64
    /nominal_energies  (N_E,)                       float64  eV
    attrs:  mode='xas_static', config=1, run_no, signal_bunch_range,
            n_sections_detected, n_sections_used,
            transition_trim_trains, first_section_state, ...

Config file
-----------
Python module exposing: ``RUN_NO``, ``NOMINAL_ENERGIES``,
``SIGNAL_BUNCH_RANGE``, ``CONFIG = 1``, plus optional trim and
shutter-transition knobs and an optional ``INPUT_H5`` override.
``MODE`` may be any of ``"xas"``, ``"xas_scan"``, or ``"xas_static"``.

CLI
---
    python compute_static_xas_cfg1.py CONFIG.py [-o OUT.h5] [-i INPUT.h5]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

import data_loading

__all__ = ["compute_static_xas_cfg1", "main"]

ACCEPTED_XAS_MODES = ("xas", "xas_scan", "xas_static")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("static_xas_cfg1_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    """Threshold shutter signal into (is_open, is_closed, finite) masks.
    Auto-flips polarity if the open group has a lower mean train score."""
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
    train_score: np.ndarray,
    transition_trim_trains: int,
):
    """Detect contiguous open / closed sections after trimming trains
    around every shutter state change."""
    is_open_raw, is_closed_raw, finite_sh = _classify_shutter(shutter, train_score)
    valid_mask = finite_sh.copy()
    transition_idx = np.where(np.diff(is_open_raw.astype(np.int8)) != 0)[0] + 1
    for t in transition_idx:
        i0 = max(0, t - transition_trim_trains)
        i1 = min(valid_mask.size, t + transition_trim_trains + 1)
        valid_mask[i0:i1] = False
    open_blocks   = _contiguous_true_blocks(is_open_raw   & finite_sh)
    closed_blocks = _contiguous_true_blocks(is_closed_raw & finite_sh)
    return open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_static_xas_cfg1(
    output_h5,
    *,
    run_no: int,
    nominal_energies,
    signal_bunch_range: Tuple[int, int],
    input_h5=None,
    trim_start: int = 2,
    trim_end: int = 2,
    train_rate_hz: float = 10.0,
    transition_trim_seconds: float = 3.0,
    first_section_state: str = "open",
    config_path=None,
    verbose: bool = True,
) -> None:
    """
    Run the config-1 static-XAS shot-collection pipeline and write to
    ``output_h5``.

    Loads ``tofs_e``, ``tofs_i``, ``gmd``, ``tID``, and ``shutter`` from
    a combined H5 (via :func:`data_loading.load_data`), detects open /
    closed sections from the shutter signal, and flattens each open
    section's signal-bunch hits to a single (n_shots, max_hits) array
    per energy.

    Parameters
    ----------
    output_h5 : str or Path
    run_no : int
    nominal_energies : array-like
        Energies (eV) assigned to detected open sections by index.
    signal_bunch_range : (int, int)
        Half-open bunch range for signal shots.
    input_h5 : str or Path, optional
        Combined H5 path. Defaults to
        ``config.COMBINED_DIR / f"run{run_no}.h5"``.
    trim_start, trim_end : int
        Forwarded to :func:`data_loading.load_data`.
    train_rate_hz, transition_trim_seconds : float
    first_section_state : "open" | "closed"
    config_path : Path, optional  (recorded as a provenance attribute)
    verbose : bool
    """
    output_h5 = Path(output_h5)
    log = print if verbose else (lambda *a, **k: None)

    nominal_energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])

    first_section_state = str(first_section_state).strip().lower()
    if first_section_state not in ("open", "closed"):
        raise ValueError("first_section_state must be 'open' or 'closed'.")

    import config as path_config
    if input_h5 is None:
        input_h5 = Path(path_config.COMBINED_DIR) / f"run{run_no}.h5"
    else:
        input_h5 = Path(input_h5)
    if not input_h5.exists():
        raise FileNotFoundError(f"Combined H5 not found: {input_h5}")

    log(f"run                : {run_no}")
    log(f"input combined H5  : {input_h5}")
    log(f"nominal energies   : {nominal_energies.size}  "
        f"({nominal_energies[0]:.2f} .. {nominal_energies[-1]:.2f} eV)")
    log(f"signal bunches     : [{sig_b0}, {sig_b1})")

    # ------------------------------------------------------------------
    # Load combined H5
    # ------------------------------------------------------------------
    data = data_loading.load_data(
        str(input_h5), config=1,
        trim_start=trim_start, trim_end=trim_end,
    )
    if data.tofs_e is None or data.tofs_i is None:
        raise RuntimeError(
            f"Combined H5 {input_h5} is missing tofs_e or tofs_i — "
            "rebuild it with write_h5.py before running this script."
        )
    if data.shutter is None:
        raise RuntimeError(
            f"Combined H5 {input_h5} has no /shutter dataset — rebuild "
            "it with the current write_h5.py (which adds shutter to the "
            "combined-H5 schema)."
        )

    tofs_e  = np.asarray(data.tofs_e)              # (n_trains, m, max_ecounts)
    tofs_i  = np.asarray(data.tofs_i)              # (n_trains, m, max_icounts)
    gmd     = np.asarray(data.gmd, dtype=np.float64)  # (n_trains, m)
    shutter = np.asarray(data.shutter, dtype=np.float64)  # (n_trains,)

    n_trains, m, max_ecounts = tofs_e.shape
    _, _, max_icounts        = tofs_i.shape
    if not (0 <= sig_b0 < sig_b1 <= m):
        raise ValueError(f"signal_bunch_range {signal_bunch_range} not within [0, {m}].")
    log(f"loaded n_trains    : {n_trains}, m={m}, "
        f"max_ecounts={max_ecounts}, max_icounts={max_icounts}")

    # ------------------------------------------------------------------
    # Shutter section detection
    # ------------------------------------------------------------------
    # Per-train signal score: total non-zero electron hits in signal bunches.
    train_score = np.sum(
        tofs_e[:, sig_b0:sig_b1, :] > 0, axis=(1, 2),
    ).astype(np.float64)

    transition_trim_trains = int(round(transition_trim_seconds * train_rate_hz))
    open_blocks, closed_blocks, is_open_raw, _is_closed_raw, valid_mask = (
        _detect_sections(shutter, train_score, transition_trim_trains)
    )
    if not open_blocks:
        raise RuntimeError("No open-shutter sections detected.")
    if not closed_blocks:
        log("warning: no closed-shutter sections detected.")

    n_detected = len(open_blocks)
    n_e = min(n_detected, nominal_energies.size)
    if n_detected != nominal_energies.size:
        log(f"warning: detected {n_detected} open sections vs "
            f"{nominal_energies.size} nominal energies; using leading {n_e}.")
    open_blocks = open_blocks[:n_e]
    energies    = nominal_energies[:n_e]

    # ------------------------------------------------------------------
    # Collect shots per section
    # ------------------------------------------------------------------
    e_shots: list[np.ndarray] = []
    i_shots: list[np.ndarray] = []
    g_shots: list[np.ndarray] = []
    n_shots_arr = np.zeros(n_e, dtype=np.int64)

    log(f"collecting shots over {n_e} sections...")
    for i, (s, e) in enumerate(open_blocks):
        sec_mask = np.zeros(n_trains, dtype=bool)
        sec_mask[s:e] = True
        sec_mask &= is_open_raw & valid_mask
        sec_idx = np.where(sec_mask)[0]

        if sec_idx.size == 0:
            log(f"  section {i:>3d} (E={energies[i]:.2f} eV): no open trains; skipping")
            e_shots.append(np.empty((0, max_ecounts), dtype=tofs_e.dtype))
            i_shots.append(np.empty((0, max_icounts), dtype=tofs_i.dtype))
            g_shots.append(np.empty(0, dtype=np.float64))
            continue

        e_block = tofs_e[sec_idx][:, sig_b0:sig_b1, :]   # (n_sec, n_sig, max_ecounts)
        i_block = tofs_i[sec_idx][:, sig_b0:sig_b1, :]   # (n_sec, n_sig, max_icounts)
        g_block = gmd[sec_idx, sig_b0:sig_b1]            # (n_sec, n_sig)

        e_flat = e_block.reshape(-1, max_ecounts)
        i_flat = i_block.reshape(-1, max_icounts)
        g_flat = g_block.reshape(-1)

        n_shots_arr[i] = g_flat.shape[0]
        e_shots.append(e_flat)
        i_shots.append(i_flat)
        g_shots.append(g_flat)
        log(f"  section {i:>3d} (E={energies[i]:.2f} eV): "
            f"open trains={sec_idx.size:>4d}  shots={g_flat.shape[0]:>6d}")

    # ------------------------------------------------------------------
    # Pad to rectangular arrays
    # ------------------------------------------------------------------
    n_shots_max = int(n_shots_arr.max()) if n_e > 0 else 0
    tofs_e_out = np.zeros((n_e, n_shots_max, max_ecounts), dtype=np.float32)
    tofs_i_out = np.zeros((n_e, n_shots_max, max_icounts), dtype=np.float32)
    gmd_out    = np.full((n_e, n_shots_max),  np.nan,      dtype=np.float64)
    for k, (E, I, G) in enumerate(zip(e_shots, i_shots, g_shots)):
        n = G.shape[0]
        if n > 0:
            tofs_e_out[k, :n] = E
            tofs_i_out[k, :n] = I
            gmd_out[k, :n]    = G

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}  "
        f"(tofs_e: {tofs_e_out.shape}, tofs_i: {tofs_i_out.shape})")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("tofs_e",          data=tofs_e_out, compression="gzip")
        fout.create_dataset("tofs_i",          data=tofs_i_out, compression="gzip")
        fout.create_dataset("gmd",             data=gmd_out,    compression="gzip")
        fout.create_dataset("n_shots",         data=n_shots_arr)
        fout.create_dataset("nominal_energies", data=energies)

        fout.attrs["mode"]                    = "xas_static"
        fout.attrs["config"]                  = 1
        fout.attrs["run_no"]                  = int(run_no)
        fout.attrs["input_h5"]                = str(input_h5)
        fout.attrs["n_sections_detected"]     = int(n_detected)
        fout.attrs["n_sections_used"]         = int(n_e)
        fout.attrs["signal_bunch_range"]      = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["trim_start"]              = int(trim_start)
        fout.attrs["trim_end"]                = int(trim_end)
        fout.attrs["train_rate_hz"]           = float(train_rate_hz)
        fout.attrs["transition_trim_seconds"] = float(transition_trim_seconds)
        fout.attrs["transition_trim_trains"]  = int(transition_trim_trains)
        fout.attrs["first_section_state"]     = first_section_state
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    log(f"done: {int(n_shots_arr.sum())} shots across {n_e} energies.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-shot eTOF / iTOF / GMD for a config-1 static XAS scan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Config module exposing RUN_NO, NOMINAL_ENERGIES, "
             "SIGNAL_BUNCH_RANGE, CONFIG=1.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output H5 path (default: processed/xas_static/run<N>_cfg1_static_xas.h5)",
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=None,
        help="Combined H5 input path (default: COMBINED_DIR/run<N>.h5)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config_path)

    if int(getattr(cfg, "CONFIG", 1)) != 1:
        raise ValueError("compute_static_xas_cfg1 requires CONFIG = 1.")
    mode = getattr(cfg, "MODE", "xas")
    if mode not in ACCEPTED_XAS_MODES:
        raise ValueError(
            f"this entry point expects MODE in {ACCEPTED_XAS_MODES}, "
            f"got {mode!r}."
        )

    run_no = int(cfg.RUN_NO)
    if args.output is None:
        import config as path_config
        out_dir = Path(path_config.COMBINED_DIR).parent / "xas_static"
        out = out_dir / f"run{run_no}_cfg1_static_xas.h5"
    else:
        out = args.output

    input_h5 = args.input if args.input is not None else getattr(cfg, "INPUT_H5", None)

    compute_static_xas_cfg1(
        output_h5=out,
        run_no=run_no,
        nominal_energies=np.asarray(cfg.NOMINAL_ENERGIES, dtype=float),
        signal_bunch_range=tuple(cfg.SIGNAL_BUNCH_RANGE),
        input_h5=input_h5,
        trim_start=int(getattr(cfg, "TRIM_START", 2)),
        trim_end=int(getattr(cfg, "TRIM_END", 2)),
        train_rate_hz=float(getattr(cfg, "TRAIN_RATE_HZ", 10.0)),
        transition_trim_seconds=float(getattr(cfg, "TRANSITION_TRIM_SECONDS", 3.0)),
        first_section_state=str(getattr(cfg, "FIRST_SECTION_STATE", "open")),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
