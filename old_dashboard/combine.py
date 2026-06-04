"""
Callable wrappers around the config 1 and config 2 combine pipelines.

These functions provide callable wrappers for config-1/config-2 combine runs
so they can be triggered from the dashboard UI without hardcoded values.

Utility classes (DataChunk, TDCIterator, SDUIterator, h5Iterator) are
imported from the existing ``write_h5.py`` — no logic is duplicated.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


class _LineLogWriter:
    """File-like writer that forwards complete stdout lines to a logger."""

    def __init__(self, log: Callable[[str], None]):
        self._log = log
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._log(self._buf)
        self._buf = ""

# ---------------------------------------------------------------------------
# Resolve imports from the project tree
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))
sys.path.insert(
    0,
    str(_REPO_ROOT / "11022188" / "processed" / "analysis_tools" / "decoding_script"),
)

# ---------------------------------------------------------------------------
# Config 1 — electron + ion TOF, no VLS
# ---------------------------------------------------------------------------

def run_config1(
    run_no: int,
    measurement_name: str,
    output_path: Path | str,
    *,
    h5_folder: str,
    local_daq_folder: str,
    train_length: int = 400,
    max_e_per_bunch: int = 50,
    max_i_per_bunch: int = 120,
    folding_parameter: float = 9969.225,
    chunk_size: int = 1000,
    log: Callable[[str], None] = print,
) -> None:
    """
    Combine raw H5 + local DAQ streams into a single config-1 combined H5.

    Parameters
    ----------
    run_no : int
        Raw H5 run number (e.g. 48346).
    measurement_name : str
        Local DAQ subfolder / file-stem prefix (e.g. "delay_scan3").
    output_path : Path or str
        Where to write the output combined H5.
    h5_folder : str
        Directory that contains the raw ``FLASH2_USER1_*runXXXXX*.h5`` files.
    local_daq_folder : str
        Directory that contains the ``{measurement_name}_*.txt / .lst`` files.
    train_length : int
        Expected bunches per train (default 400).
    max_e_per_bunch : int
        Zero-padding limit for electron TOF hits (default 50).
    max_i_per_bunch : int
        Zero-padding limit for ion TOF hits (default 120).
    folding_parameter : float
        ns — period used to fold the TDC sweep into per-bunch lists.
    chunk_size : int
        H5 chunk size for I/O buffering (default 1000 trains).
    log : callable
        Function used for progress messages (default ``print``).
        Pass ``st.write`` or a similar callable to redirect output.
    """
    from write_h5 import (
        DataChunk,
        TDCIterator,
        SDUIterator,
        h5Iterator,
        extract_sdu_data_from_single_file,
    )

    output_path = Path(output_path)
    log(f"Measurement   : {measurement_name}  (run {run_no})")
    log(f"Local DAQ dir : {local_daq_folder}")
    log(f"Raw H5 dir    : {h5_folder}")
    log(f"Output        : {output_path}")

    if not Path(local_daq_folder).is_dir():
        raise FileNotFoundError(f"Local DAQ folder not found: {local_daq_folder}")
    if not Path(h5_folder).is_dir():
        raise FileNotFoundError(f"Raw H5 folder not found: {h5_folder}")

    files_in_folder = os.listdir(local_daq_folder)

    sdu_names = sorted(
        filter(re.compile(measurement_name + r"_\d{10}\.txt").match, files_in_folder)
    )
    sdu_fpaths = [local_daq_folder + "/" + n for n in sdu_names]
    if not sdu_fpaths:
        raise ValueError(f"No SDU .txt files in {local_daq_folder}")
    log(f"Found {len(sdu_fpaths)} SDU .txt files.")

    tdc_names = sorted(
        filter(re.compile(measurement_name + r"_\d{10}\.lst").match, files_in_folder)
    )
    tdc_fpaths = [local_daq_folder + "/" + n for n in tdc_names]
    if not tdc_fpaths:
        raise ValueError(f"No TDC .lst files in {local_daq_folder}")
    log(f"Found {len(tdc_fpaths)} TDC .lst files.")

    h5_paths = sorted(glob.glob(os.path.join(h5_folder, f"*run{run_no}*.h5")))
    if not h5_paths:
        raise ValueError(f"No raw H5 files for run {run_no} in {h5_folder}")
    log(f"Found {len(h5_paths)} raw H5 files for run {run_no}.")

    first_tID = int(extract_sdu_data_from_single_file(sdu_fpaths[0])[0][0])
    last_tID = int(extract_sdu_data_from_single_file(sdu_fpaths[-1])[0][-1])
    log(f"Measurement spans train IDs {first_tID} .. {last_tID} "
        f"({last_tID - first_tID + 1} trains).")

    first_h5_idx = None
    for path_idx, h5_path in enumerate(h5_paths):
        with h5py.File(h5_path, "r") as f:
            tids = f["/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index"]
            tids_max = int(np.max(tids[...]))
            if tids_max > first_tID:
                first_h5_idx = path_idx
                break

    if first_h5_idx is None:
        raise RuntimeError("None of the raw H5 files overlap with the SDU train-ID range.")

    sdu_it = SDUIterator(sdu_fpaths)
    tdc_it = TDCIterator(tdc_fpaths)

    gmd_it = h5Iterator(h5_paths[first_h5_idx:], [
        "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index",
        "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value",
    ])
    mpe_it = h5Iterator(h5_paths[first_h5_idx:], [
        "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/index",
        "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/value",
    ])
    hor_pos_it = h5Iterator(h5_paths[first_h5_idx:], [
        "/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/index",
        "/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/value",
    ])
    ver_pos_it = h5Iterator(h5_paths[first_h5_idx:], [
        "/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/index",
        "/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/value",
    ])

    next_tID_gmd, next_gmd = _advance_to(gmd_it, first_tID, "gmd", log)
    next_tID_mpe, next_mpe = _advance_to(mpe_it, first_tID, "mpe", log)
    next_tID_hor_pos, next_hor_pos = _advance_to(hor_pos_it, first_tID, "hor_pos", log)
    next_tID_ver_pos, next_ver_pos = _advance_to(ver_pos_it, first_tID, "ver_pos", log)
    next_tID_z, next_z, next_z_std = sdu_it.__next__()
    (next_tID_tdc, next_eventcounts_e, next_tofs_e,
     next_eventcounts_i, next_tofs_i) = tdc_it.__next__()

    data_len = last_tID - first_tID
    chunk = DataChunk(
        chunk_size, train_length,
        max_ecounts=max_e_per_bunch, max_icounts=max_i_per_bunch,
    )

    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as f_out:
        tID_dset               = f_out.create_dataset("tID",               (data_len,),                            dtype="double")
        data_flag_dset         = f_out.create_dataset("local_DAQ_running", (data_len,),                            dtype="bool")
        z_dset                 = f_out.create_dataset("z",                 (data_len, train_length),               dtype=np.float32)
        z_std_dset             = f_out.create_dataset("z_std",             (data_len, train_length),               dtype=np.float32)
        gmd_dset               = f_out.create_dataset("gmd",               (data_len, train_length),               dtype=np.float32)
        mpe_dset               = f_out.create_dataset("mpe",               (data_len,),                            dtype=np.float32)
        hor_pos_dset           = f_out.create_dataset("hor_pos",           (data_len,),                            dtype=np.float32)
        ver_pos_dset           = f_out.create_dataset("ver_pos",           (data_len,),                            dtype=np.float32)
        tofs_e_dset            = f_out.create_dataset(
            "tofs_e", (data_len, train_length, max_e_per_bunch),
            dtype=np.uint32, compression="gzip",
        )
        tofs_i_dset            = f_out.create_dataset(
            "tofs_i", (data_len, train_length, max_i_per_bunch),
            dtype=np.uint32, compression="gzip",
        )
        between_tdc_files_dset = f_out.create_dataset("between_tdc_files", (data_len,),                            dtype="bool")

        for tID in range(first_tID, last_tID + 1):
            # gmd
            if tID < next_tID_gmd:
                gmd = np.nan
            elif tID == next_tID_gmd:
                gmd = next_gmd[0]
                try:
                    next_tID_gmd, next_gmd = gmd_it.__next__()
                except StopIteration:
                    log(f"Stopped by gmd on tID {next_tID_gmd}")
                    break
            else:
                raise ValueError(f"gmd overshot: tID={tID}, next={next_tID_gmd}")

            # mpe
            if tID < next_tID_mpe:
                mpe = np.nan
            elif tID == next_tID_mpe:
                mpe = next_mpe
                try:
                    next_tID_mpe, next_mpe = mpe_it.__next__()
                except StopIteration:
                    next_tID_mpe = last_tID + 1
                    mpe = np.nan
            else:
                raise ValueError(f"mpe overshot: tID={tID}, next={next_tID_mpe}")

            # hor_pos
            if tID < next_tID_hor_pos:
                hor_pos = np.nan
            elif tID == next_tID_hor_pos:
                hor_pos = next_hor_pos
                try:
                    next_tID_hor_pos, next_hor_pos = hor_pos_it.__next__()
                except StopIteration:
                    next_tID_hor_pos = last_tID + 1
                    hor_pos = np.nan
            else:
                raise ValueError(f"hor_pos overshot: tID={tID}, next={next_tID_hor_pos}")

            # ver_pos
            if tID < next_tID_ver_pos:
                ver_pos = np.nan
            elif tID == next_tID_ver_pos:
                ver_pos = next_ver_pos
                try:
                    next_tID_ver_pos, next_ver_pos = ver_pos_it.__next__()
                except StopIteration:
                    next_tID_ver_pos = last_tID + 1
                    ver_pos = np.nan
            else:
                raise ValueError(f"ver_pos overshot: tID={tID}, next={next_tID_ver_pos}")

            # SDU (z)
            if tID < next_tID_z:
                z = np.nan
                z_std = np.nan
                is_data = False
            elif tID == next_tID_z:
                z = next_z
                z_std = next_z_std
                is_data = True
                try:
                    next_tID_z, next_z, next_z_std = sdu_it.__next__()
                except StopIteration:
                    log(f"Stopped by SDU on tID {next_tID_z}")
                    break
            else:
                raise ValueError(f"SDU overshot: tID={tID}, next={next_tID_z}")

            # TDC (tofs_e, tofs_i)
            if tID < next_tID_tdc:
                tofs_e = None
                tofs_i = None
                between_tdc_files = tdc_it.is_between_files()
            elif tID == next_tID_tdc:
                tofs_e_raw = next_tofs_e
                tofs_i_raw = next_tofs_i
                tofs_e = [
                    tofs_e_raw[
                        (tofs_e_raw > b * folding_parameter)
                        & (tofs_e_raw < (b + 1) * folding_parameter)
                    ] - b * folding_parameter
                    for b in range(train_length)
                ]
                tofs_i = [
                    tofs_i_raw[
                        (tofs_i_raw > b * folding_parameter)
                        & (tofs_i_raw < (b + 1) * folding_parameter)
                    ] - b * folding_parameter
                    for b in range(train_length)
                ]
                between_tdc_files = False
                try:
                    (next_tID_tdc, next_eventcounts_e, next_tofs_e,
                     next_eventcounts_i, next_tofs_i) = tdc_it.__next__()
                except StopIteration:
                    log(f"Stopped by TDC on tID {next_tID_tdc}")
                    break
            else:
                raise ValueError(f"TDC overshot: tID={tID}, next={next_tID_tdc}")

            chunk_full = chunk.add_row(
                is_data, tID, gmd, mpe, hor_pos, ver_pos,
                z, z_std, tofs_e, tofs_i, between_tdc_files,
            )
            if chunk_full:
                chunk.dump(
                    tID_dset, data_flag_dset, z_dset, z_std_dset,
                    gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
                    tofs_e_dset, tofs_i_dset, between_tdc_files_dset,
                )
                chunk.reset()

            if tID % 1000 == 0:
                log(f"  tID {tID}  ({tID - first_tID}/{last_tID - first_tID})")

        chunk.finish(
            tID_dset, data_flag_dset, z_dset, z_std_dset,
            gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
            tofs_e_dset, tofs_i_dset, between_tdc_files_dset,
        )

    log(f"Done — {output_path.name} written.")


# ---------------------------------------------------------------------------
# Config 2 — real combine via write_h5.py (liq eTOF + VLS)
# ---------------------------------------------------------------------------

def run_config2(
    run_no: int,
    measurement_name: str,
    output_path: Path | str,
    *,
    train_length: int = 100,
    chunk_size: int = 200,
    max_e_per_bunch: int = 50,
    n_vls_pixels: int = 1280,
    folding_parameter: float = 9969.225,
    log: Callable[[str], None] = print,
) -> None:
    """
    Run the real config-2 writer path from ``write_h5.py``.

    This follows the same train-ID alignment structure as config 1 and uses
    config-2 detector layout (liq eTOF + VLS) from raw streams.
    """
    import contextlib
    import io
    from write_h5 import main as write_h5_main

    output_path = Path(output_path)
    log(f"Measurement   : {measurement_name}  (run {run_no})")
    log("Config        : 2")
    log(f"Output        : {output_path}")

    stream = _LineLogWriter(log)
    with contextlib.redirect_stdout(stream):
        write_h5_main(
            config_no=2,
            measurement_name=measurement_name,
            run_no=int(run_no),
            output_path=output_path,
            train_length=int(train_length),
            chunk_size=int(chunk_size),
            max_ecounts=int(max_e_per_bunch),
            n_vls_pixels=int(n_vls_pixels),
            folding_parameter=float(folding_parameter),
        )

    stream.flush()


def run_aggregates(
    input_h5: Path | str,
    config_path: Path | str,
    output_path: Path | str | None = None,
    *,
    log: Callable[[str], None] = print,
) -> None:
    """
    Run the existing ``analysis/scripts/compute_aggregates.py`` flow.

    Parameters
    ----------
    input_h5 : Path or str
        Combined H5 file to aggregate.
    config_path : Path or str
        Python config module path under ``analysis/configs``.
    output_path : Path or str or None
        Optional output H5 path. If None, the script default is used.
    log : callable
        Progress sink.
    """
    import contextlib
    import io
    from compute_aggregates import main as compute_main

    input_h5 = Path(input_h5)
    config_path = Path(config_path)
    out = None if output_path is None else Path(output_path)

    if not input_h5.exists():
        raise FileNotFoundError(f"input H5 not found: {input_h5}")
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    argv = [str(config_path), str(input_h5)]
    if out is not None:
        argv.extend(["-o", str(out)])

    log(f"Aggregates config : {config_path}")
    log(f"Aggregates input  : {input_h5}")
    if out is not None:
        log(f"Aggregates output : {out}")

    stream = _LineLogWriter(log)
    with contextlib.redirect_stdout(stream):
        compute_main(argv)
    stream.flush()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _advance_to(
    it,
    target_tID: int,
    label: str,
    log: Callable[[str], None] = print,
) -> tuple:
    for t, v in it:
        if t >= target_tID:
            log(f"  {label} reached tID {t} (target {target_tID}).")
            return t, v
    raise RuntimeError(f"{label} iterator exhausted before reaching {target_tID}")
