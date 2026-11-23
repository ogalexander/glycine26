"""
Collect per-shot (eTOF, GMD) for a photon-energy config-1 scan.

This is the config-1 analogue of ``compute_static_xas.py`` in spirit:
the output is shot-resolved rather than pre-binned, so downstream
notebooks can choose their own TOF histogramming, GMD binning, ROI
integration, and normalisation strategy.

Preferred workflow
------------------
One combined config-1 H5 per nominal photon-energy step, i.e. ``INPUT_H5S``
in the config. For each file, all trains that survive the standard
``between_tdc_files == False`` filter and edge trimming are kept. The
selected train x bunch block is then flattened into a shot axis.

This script now assumes the per-energy files already correspond to
always-open acquisition. It does **not** do fast-shutter open/closed
section detection or closed-shutter background subtraction.

Output H5 layout (default: ``<processed/tof_static>/run<N>_static_tof.h5``):
    /tofs_e              (N_E, N_shots_max, max_hits)  uint32, zero-padded
        Per-shot electron TOF hit lists. A shot is one (train, bunch).
        Zero entries are padding, matching the combined-H5 convention.
    /gmd                 (N_E, N_shots_max)            float64, NaN-padded
        Per-shot GMD values aligned to ``tofs_e``.
    /train_id            (N_E, N_shots_max)            int64, -1 padded
        Train ID of each shot.
    /bunch_index         (N_E, N_shots_max)            int64, -1 padded
        Bunch index of each shot within the train.
    /n_shots             (N_E,)                        int64
        Number of valid shots for each nominal energy.
    /n_selected_trains   (N_E,)                        int64
        Number of trains contributing to each nominal energy.
    /nominal_energies    (N_E,)                        float64  eV
    /tof_edges           (n_tof + 1,)                 float64
        Recommended default histogram grid for notebooks.
    /signal_bunch_indices (n_sig,)                     int64
    /input_h5_paths      (N_E,) string                 only in INPUT_H5S mode

Config file
-----------
Python module exposing either:

    RUN_NO, INPUT_H5S, NOMINAL_ENERGIES, TOF_EDGES, SIGNAL_BUNCH_RANGE,
    CONFIG=1, MODE='tof_static'

or, for a single-energy file:

    RUN_NO, INPUT_H5, NOMINAL_ENERGIES, TOF_EDGES, SIGNAL_BUNCH_RANGE,
    CONFIG=1, MODE='tof_static'

If ``INPUT_H5`` is used, ``NOMINAL_ENERGIES`` must contain exactly one
energy, because there is no shutter-based sectioning in this script.

CLI
---
    python compute_static_tof.py CONFIG.py [-o OUT.h5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np

from compute_aggregates import (
    _build_keep_mask,
    _verify_h5_for_config,
    load_python_config,
)

__all__ = ["compute_static_tof", "main"]


ACCEPTED_TOF_STATIC_MODES = ("tof", "tof_static")


def _iter_index_chunks(indices: np.ndarray, chunk_size: int):
    """Yield sorted integer index chunks for h5py fancy indexing."""
    for s in range(0, int(indices.size), int(chunk_size)):
        yield indices[s:s + int(chunk_size)]


def _flatten_selected_trains(
    f: h5py.File,
    row_pos: np.ndarray,
    signal_bunch_range: Tuple[int, int],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten selected trains and bunches into a shot axis.

    Returns
    -------
    tofs_e : (n_shots, max_hits) uint32
    gmd : (n_shots,) float64
    train_id : (n_shots,) int64
    bunch_index : (n_shots,) int64
    """
    sig_b0, sig_b1 = signal_bunch_range
    n_sig = sig_b1 - sig_b0
    max_hits = int(f["tofs_e"].shape[2])
    n_trains = int(row_pos.size)
    n_shots = n_trains * n_sig

    tofs_e = np.zeros((n_shots, max_hits), dtype=f["tofs_e"].dtype)
    gmd = np.full(n_shots, np.nan, dtype=np.float64)
    train_id = np.full(n_shots, -1, dtype=np.int64)
    bunch_index = np.full(n_shots, -1, dtype=np.int64)

    write_pos = 0
    bunch_template = np.arange(sig_b0, sig_b1, dtype=np.int64)
    for pos_chunk in _iter_index_chunks(row_pos, chunk_size):
        if pos_chunk.size == 0:
            continue

        tof_chunk = f["tofs_e"][pos_chunk, sig_b0:sig_b1, :]
        gmd_chunk = f["gmd"][pos_chunk, sig_b0:sig_b1].astype(np.float64)
        tID_chunk = np.asarray(f["tID"][pos_chunk], dtype=np.int64)

        n_chunk_shots = int(pos_chunk.size) * n_sig
        sl = slice(write_pos, write_pos + n_chunk_shots)

        tofs_e[sl] = tof_chunk.reshape(n_chunk_shots, max_hits)
        gmd[sl] = gmd_chunk.reshape(n_chunk_shots)
        train_id[sl] = np.repeat(tID_chunk, n_sig)
        bunch_index[sl] = np.tile(bunch_template, int(pos_chunk.size))

        write_pos += n_chunk_shots

    return tofs_e, gmd, train_id, bunch_index


def _process_one_energy_h5(
    *,
    input_h5,
    signal_bunch_range: Tuple[int, int],
    trim_start: int,
    trim_end: int,
    chunk_size: int,
):
    """
    Process one combined config-1 H5 representing one nominal energy step.

    All kept trains are used; there is no shutter-based subsectioning.
    """
    input_h5 = Path(input_h5)
    sig_b0, sig_b1 = signal_bunch_range

    with h5py.File(input_h5, "r") as f:
        _verify_h5_for_config(f, 1)

        keep_mask = _build_keep_mask(f, int(trim_start), int(trim_end))
        if not keep_mask.any():
            raise RuntimeError(
                f"{input_h5}: no trains survive the bad-train + trim filters."
            )

        n_bunches = int(f["gmd"].shape[1])
        if not (0 <= sig_b0 < sig_b1 <= n_bunches):
            raise ValueError(
                f"signal_bunch_range {signal_bunch_range} not within [0, {n_bunches}] "
                f"for {input_h5}."
            )

        keep_pos = np.where(keep_mask)[0].astype(np.int64)
        tofs_e, gmd, train_id, bunch_index = _flatten_selected_trains(
            f,
            row_pos=keep_pos,
            signal_bunch_range=(sig_b0, sig_b1),
            chunk_size=int(chunk_size),
        )

        return {
            "tofs_e": tofs_e,
            "gmd": gmd,
            "train_id": train_id,
            "bunch_index": bunch_index,
            "n_shots": int(tofs_e.shape[0]),
            "n_selected_trains": int(keep_pos.size),
            "max_hits": int(tofs_e.shape[1]),
        }


def _compute_static_tof_single_file(
    output_h5,
    *,
    input_h5,
    run_no: int,
    nominal_energies: Sequence[float],
    tof_edges: Sequence[float],
    signal_bunch_range: Tuple[int, int],
    config: int = 1,
    trim_start: int = 2,
    trim_end: int = 2,
    chunk_size: int = 50,
    config_path=None,
    verbose: bool = True,
) -> None:
    """Single-file mode, valid only for a single nominal energy."""
    output_h5 = Path(output_h5)
    input_h5 = Path(input_h5)

    if int(config) != 1:
        raise ValueError("tof_static mode is config=1 only.")

    log = print if verbose else (lambda *a, **k: None)
    nominal_energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
    tof_edges = np.asarray(tof_edges, dtype=np.float64).ravel()

    if nominal_energies.size != 1:
        raise ValueError(
            "Single-file no-shutter mode requires exactly one nominal energy. "
            "For multi-energy scans, provide INPUT_H5S with one file per energy."
        )
    if tof_edges.ndim != 1 or tof_edges.size < 2:
        raise ValueError("tof_edges must be a 1D array with at least two entries.")
    if not np.all(np.diff(tof_edges) > 0):
        raise ValueError("tof_edges must be strictly increasing.")

    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])
    n_sig = sig_b1 - sig_b0

    log(f"run                : {run_no}")
    log(f"input h5           : {input_h5}")
    log(f"nominal energies   : {nominal_energies[0]:.2f} eV")
    log(f"signal bunches     : [{sig_b0}, {sig_b1}) = {n_sig} bunches")
    log(f"default eTOF bins  : {tof_edges.size - 1}  "
        f"([{tof_edges[0]:.1f}, {tof_edges[-1]:.1f}))")
    log(f"trim start / end   : {trim_start} / {trim_end}")
    log(f"shot chunk size    : {chunk_size}")

    result = _process_one_energy_h5(
        input_h5=input_h5,
        signal_bunch_range=(sig_b0, sig_b1),
        trim_start=trim_start,
        trim_end=trim_end,
        chunk_size=chunk_size,
    )

    signal_bunch_indices = np.arange(sig_b0, sig_b1, dtype=np.int64)
    n_shots = np.asarray([result["n_shots"]], dtype=np.int64)
    n_selected_trains = np.asarray([result["n_selected_trains"]], dtype=np.int64)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("tofs_e", data=result["tofs_e"][None, ...], compression="gzip")
        fout.create_dataset("gmd", data=result["gmd"][None, ...], compression="gzip")
        fout.create_dataset("train_id", data=result["train_id"][None, ...], compression="gzip")
        fout.create_dataset("bunch_index", data=result["bunch_index"][None, ...], compression="gzip")
        fout.create_dataset("n_shots", data=n_shots)
        fout.create_dataset("n_selected_trains", data=n_selected_trains)
        fout.create_dataset("nominal_energies", data=nominal_energies)
        fout.create_dataset("tof_edges", data=tof_edges)
        fout.create_dataset("signal_bunch_indices", data=signal_bunch_indices)

        fout.attrs["mode"] = "tof_static"
        fout.attrs["config"] = int(config)
        fout.attrs["run_no"] = int(run_no)
        fout.attrs["input_mode"] = "single_energy_h5"
        fout.attrs["input_h5"] = str(input_h5)
        fout.attrs["background_mode"] = "none"
        fout.attrs["signal_bunch_range"] = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["trim_start"] = int(trim_start)
        fout.attrs["trim_end"] = int(trim_end)
        fout.attrs["chunk_size"] = int(chunk_size)
        fout.attrs["max_hits"] = int(result["max_hits"])
        fout.attrs["tof_padding_value"] = 0
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    log(f"done: {int(n_shots.sum())} shots across 1 energy.")


def _compute_static_tof_multi_file(
    output_h5,
    *,
    input_h5s: Sequence[Path],
    run_no: int,
    nominal_energies: Sequence[float],
    tof_edges: Sequence[float],
    signal_bunch_range: Tuple[int, int],
    config: int = 1,
    trim_start: int = 2,
    trim_end: int = 2,
    chunk_size: int = 50,
    config_path=None,
    verbose: bool = True,
) -> None:
    """Preferred mode: one combined config-1 H5 per nominal energy."""
    output_h5 = Path(output_h5)
    if int(config) != 1:
        raise ValueError("tof_static mode is config=1 only.")

    log = print if verbose else (lambda *a, **k: None)
    nominal_energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
    tof_edges = np.asarray(tof_edges, dtype=np.float64).ravel()
    input_h5s = [Path(p) for p in input_h5s]

    if len(input_h5s) == 0:
        raise ValueError("input_h5s is empty.")
    if len(input_h5s) != nominal_energies.size:
        raise ValueError(
            f"INPUT_H5S has {len(input_h5s)} files but NOMINAL_ENERGIES has "
            f"{nominal_energies.size} entries."
        )
    if tof_edges.ndim != 1 or tof_edges.size < 2:
        raise ValueError("tof_edges must be a 1D array with at least two entries.")
    if not np.all(np.diff(tof_edges) > 0):
        raise ValueError("tof_edges must be strictly increasing.")

    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])
    n_sig = sig_b1 - sig_b0

    log(f"run                : {run_no}")
    log(f"input h5 count     : {len(input_h5s)}")
    log(f"nominal energies   : {nominal_energies.size}  "
        f"({nominal_energies[0]:.2f} .. {nominal_energies[-1]:.2f} eV)")
    log(f"signal bunches     : [{sig_b0}, {sig_b1}) = {n_sig} bunches")
    log(f"default eTOF bins  : {tof_edges.size - 1}  "
        f"([{tof_edges[0]:.1f}, {tof_edges[-1]:.1f}))")
    log(f"trim start / end   : {trim_start} / {trim_end}")
    log(f"shot chunk size    : {chunk_size}")

    per_energy = []
    max_hits = None
    for i, (energy, path) in enumerate(zip(nominal_energies, input_h5s)):
        log(f"  [{i + 1:>2d}/{len(input_h5s):>2d}] {energy:.2f} eV  <-  {path.name}")
        result = _process_one_energy_h5(
            input_h5=path,
            signal_bunch_range=(sig_b0, sig_b1),
            trim_start=trim_start,
            trim_end=trim_end,
            chunk_size=chunk_size,
        )
        per_energy.append(result)
        if max_hits is None:
            max_hits = int(result["max_hits"])
        elif int(result["max_hits"]) != max_hits:
            raise ValueError(
                f"{path}: max_hits={result['max_hits']} does not match earlier "
                f"file max_hits={max_hits}."
            )
        log(f"      trains={result['n_selected_trains']:>5d}  "
            f"shots={result['n_shots']:>7d}")

    n_shots = np.asarray([r["n_shots"] for r in per_energy], dtype=np.int64)
    n_selected_trains = np.asarray(
        [r["n_selected_trains"] for r in per_energy], dtype=np.int64
    )
    n_shots_max = int(n_shots.max()) if n_shots.size > 0 else 0

    tofs_out = np.zeros(
        (nominal_energies.size, n_shots_max, int(max_hits)),
        dtype=per_energy[0]["tofs_e"].dtype,
    )
    gmd_out = np.full((nominal_energies.size, n_shots_max), np.nan, dtype=np.float64)
    train_id_out = np.full(
        (nominal_energies.size, n_shots_max), -1, dtype=np.int64
    )
    bunch_index_out = np.full(
        (nominal_energies.size, n_shots_max), -1, dtype=np.int64
    )

    for i, result in enumerate(per_energy):
        n = int(result["n_shots"])
        if n == 0:
            continue
        tofs_out[i, :n] = result["tofs_e"]
        gmd_out[i, :n] = result["gmd"]
        train_id_out[i, :n] = result["train_id"]
        bunch_index_out[i, :n] = result["bunch_index"]

    signal_bunch_indices = np.arange(sig_b0, sig_b1, dtype=np.int64)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("tofs_e", data=tofs_out, compression="gzip")
        fout.create_dataset("gmd", data=gmd_out, compression="gzip")
        fout.create_dataset("train_id", data=train_id_out, compression="gzip")
        fout.create_dataset("bunch_index", data=bunch_index_out, compression="gzip")
        fout.create_dataset("n_shots", data=n_shots)
        fout.create_dataset("n_selected_trains", data=n_selected_trains)
        fout.create_dataset("nominal_energies", data=nominal_energies)
        fout.create_dataset("tof_edges", data=tof_edges)
        fout.create_dataset("signal_bunch_indices", data=signal_bunch_indices)
        fout.create_dataset(
            "input_h5_paths",
            data=np.asarray([str(p) for p in input_h5s], dtype=h5py.string_dtype()),
        )

        fout.attrs["mode"] = "tof_static"
        fout.attrs["config"] = int(config)
        fout.attrs["run_no"] = int(run_no)
        fout.attrs["input_mode"] = "per_energy_h5s"
        fout.attrs["n_input_h5"] = int(len(input_h5s))
        fout.attrs["background_mode"] = "none"
        fout.attrs["signal_bunch_range"] = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["trim_start"] = int(trim_start)
        fout.attrs["trim_end"] = int(trim_end)
        fout.attrs["chunk_size"] = int(chunk_size)
        fout.attrs["max_hits"] = int(max_hits)
        fout.attrs["tof_padding_value"] = 0
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    log(f"done: {int(n_shots.sum())} shots across {nominal_energies.size} energies.")


def compute_static_tof(
    output_h5,
    *,
    input_h5=None,
    input_h5s: Optional[Sequence[Path]] = None,
    run_no: int,
    nominal_energies: Sequence[float],
    tof_edges: Sequence[float],
    signal_bunch_range: Tuple[int, int],
    config: int = 1,
    trim_start: int = 2,
    trim_end: int = 2,
    chunk_size: int = 50,
    use_shutter: bool = False,
    config_path=None,
    verbose: bool = True,
) -> None:
    """
    Collect shot-resolved eTOF/GMD arrays for one-energy-per-file scans.

    ``use_shutter`` is kept as an explicit guard. This shot-resolved
    config-1 path expects always-open per-energy files.
    """
    if use_shutter:
        raise ValueError(
            "This shot-resolved compute_static_tof path expects always-open "
            "per-energy files. Set USE_SHUTTER = False."
        )

    if input_h5s is not None:
        _compute_static_tof_multi_file(
            output_h5=output_h5,
            input_h5s=input_h5s,
            run_no=run_no,
            nominal_energies=nominal_energies,
            tof_edges=tof_edges,
            signal_bunch_range=signal_bunch_range,
            config=config,
            trim_start=trim_start,
            trim_end=trim_end,
            chunk_size=chunk_size,
            config_path=config_path,
            verbose=verbose,
        )
        return

    if input_h5 is None:
        raise ValueError("Provide either input_h5 or input_h5s.")

    _compute_static_tof_single_file(
        output_h5=output_h5,
        input_h5=input_h5,
        run_no=run_no,
        nominal_energies=nominal_energies,
        tof_edges=tof_edges,
        signal_bunch_range=signal_bunch_range,
        config=config,
        trim_start=trim_start,
        trim_end=trim_end,
        chunk_size=chunk_size,
        config_path=config_path,
        verbose=verbose,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect shot-resolved eTOF/GMD arrays for a config-1 scan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Config module exposing RUN_NO plus either INPUT_H5 or INPUT_H5S, "
             "then NOMINAL_ENERGIES, TOF_EDGES, SIGNAL_BUNCH_RANGE, CONFIG=1.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output H5 path (default: processed/tof_static/run<N>_static_tof.h5)",
    )
    args = parser.parse_args(argv)

    cfg = load_python_config(args.config_path)

    if int(getattr(cfg, "CONFIG", 1)) != 1:
        raise ValueError("compute_static_tof requires CONFIG = 1.")
    mode = getattr(cfg, "MODE", "tof_static")
    if mode not in ACCEPTED_TOF_STATIC_MODES:
        raise ValueError(
            f"this entry point expects MODE in {ACCEPTED_TOF_STATIC_MODES}, got {mode!r}."
        )

    run_no = int(cfg.RUN_NO)
    input_h5 = getattr(cfg, "INPUT_H5", None)
    input_h5s = getattr(cfg, "INPUT_H5S", None)
    if args.output is None:
        import config as path_config
        out_dir = Path(path_config.COMBINED_DIR).parent / "tof_static"
        out = out_dir / f"run{run_no}_static_tof.h5"
    else:
        out = args.output

    if input_h5s is not None:
        resolved_input_h5s = [Path(p) for p in input_h5s]
        resolved_input_h5 = None
    else:
        resolved_input_h5s = None
        resolved_input_h5 = None if input_h5 is None else Path(input_h5)

    compute_static_tof(
        output_h5=out,
        input_h5=resolved_input_h5,
        input_h5s=resolved_input_h5s,
        run_no=run_no,
        nominal_energies=np.asarray(cfg.NOMINAL_ENERGIES, dtype=float),
        tof_edges=np.asarray(cfg.TOF_EDGES, dtype=float),
        signal_bunch_range=tuple(cfg.SIGNAL_BUNCH_RANGE),
        config=int(getattr(cfg, "CONFIG", 1)),
        trim_start=int(getattr(cfg, "TRIM_START", 2)),
        trim_end=int(getattr(cfg, "TRIM_END", 2)),
        chunk_size=int(getattr(cfg, "CHUNK_SIZE", 50)),
        use_shutter=bool(getattr(cfg, "USE_SHUTTER", False)),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
