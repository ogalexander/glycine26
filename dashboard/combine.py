"""
Callable wrappers around the config 1 and config 2 combine pipelines.

These functions contain the same alignment logic as ``write_h5_test_config1.py``
and ``write_h5_test_config2.py``, but accept parameters so they can be
called from the dashboard UI (or any other script) without hardcoded values.

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
# Config 2 — liquid-jet eTOF + VLS (synthetic index alignment)
# ---------------------------------------------------------------------------

_PER_BUNCH_1D_FIELDS = ("gmd", "z", "z_std")
_PER_BUNCH_3D_FIELDS = ("tofs_e",)
_RENAME_ON_COPY = {"tofs_e": "liq_tofs_e"}
_DROP_FROM_CONFIG1 = ("tofs_i",)
_VLS_DATASET = "/FL2/Support Infrastructure/Gotthard/images/value"


def run_config2(
    config1_h5: Path | str,
    vls_h5: Path | str,
    output_path: Path | str,
    *,
    log: Callable[[str], None] = print,
) -> None:
    """
    Build a synthetic config-2 combined H5 from a config-1 file and a VLS file.

    Alignment is by sequential index (not train ID) — the two runs share no
    train IDs.  Both stacks are truncated to min(n_config1, n_vls).

    Parameters
    ----------
    config1_h5 : Path or str
        Path to an existing config-1 combined H5 (e.g. ``test_config1.h5``).
    vls_h5 : Path or str
        Path to the raw Gotthard H5 file (``FLASH2_USER1_main_run…h5``).
    output_path : Path or str
        Where to write the synthetic config-2 combined H5.
    log : callable
        Progress sink (default ``print``).
    """
    config1_h5 = Path(config1_h5)
    vls_h5 = Path(vls_h5)
    output_path = Path(output_path)

    log(f"Config 1 source : {config1_h5}")
    log(f"VLS source      : {vls_h5}")
    log(f"Output          : {output_path}")

    if not config1_h5.exists():
        raise FileNotFoundError(f"{config1_h5} not found.")
    if not vls_h5.exists():
        raise FileNotFoundError(f"{vls_h5} not found.")

    with h5py.File(config1_h5, "r") as f1:
        n1 = f1["tID"].shape[0]
        log(f"  Config 1: {n1} trains.")
        cfg1 = {k: f1[k][...] for k in f1.keys() if k not in _DROP_FROM_CONFIG1}

    with h5py.File(vls_h5, "r") as f2:
        vls_arr = f2[_VLS_DATASET]
        n2, m_vls, _ = vls_arr.shape
        log(f"  VLS: {n2} trains × {m_vls} bunches.")
        vls = vls_arr[...].astype(np.float32)

    n = min(n1, n2)
    log(f"Aligning by index, truncating to n={n} trains, m={m_vls} bunches.")

    for k in cfg1:
        cfg1[k] = cfg1[k][:n]
    vls = vls[:n]

    for k in _PER_BUNCH_1D_FIELDS + _PER_BUNCH_3D_FIELDS:
        if k in cfg1:
            cfg1[k] = cfg1[k][:, :m_vls, ...]

    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as f_out:
        for src_key, arr in cfg1.items():
            dst_key = _RENAME_ON_COPY.get(src_key, src_key)
            f_out.create_dataset(
                dst_key, data=arr,
                compression="gzip" if arr.ndim >= 2 else None,
            )
        f_out.create_dataset("vls", data=vls, compression="gzip")

    log(f"Done — {output_path.name} written.")


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
