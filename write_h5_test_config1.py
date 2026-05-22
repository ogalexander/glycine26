"""
Produce ``test_config1.h5`` — the clean config 1 reference combined file
for the 11022188 (glycine 2026) beamtime.

This script orchestrates the alignment of three data streams by train ID:

1.  **Raw FLASH HDF5** (``run48346`` files) — gmd, mpe, hor_pos, ver_pos.
2.  **Local DAQ SDU ``.txt``** — z, z_std (delay stage position).
3.  **Local DAQ TDC ``.lst``** — tofs_e, tofs_i (binary MCS6A list-mode).

The heavy lifting (TDC binary decoding, SDU text parsing, train-ID
correction) is reused from the legacy ``write_h5.py`` to avoid duplicating
~700 lines of subtle decoder code.

Paths are taken from ``config.py``; no absolute paths appear here.

Output schema
-------------
The output contains the canonical schema keys plus four legacy "extras"
(``z_std``, ``hor_pos``, ``ver_pos``, ``local_DAQ_running``) that the
write_h5 pipeline naturally produces.

Run this from the project root::

    python write_h5_test_config1.py
"""

from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup — must happen *before* importing write_h5 or beamtime_scripts
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))
sys.path.insert(
    0,
    str(_REPO_ROOT / "11022188" / "processed" / "analysis_tools" / "decoding_script"),
)

import h5py as h5  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from write_h5 import (  # noqa: E402
    DataChunk,
    TDCIterator,
    SDUIterator,
    h5Iterator,
    extract_sdu_data_from_single_file,
)


# ---------------------------------------------------------------------------
# Measurement-specific constants for the config 1 test data
# ---------------------------------------------------------------------------

MEASUREMENT_NAME: str = "delay_scan3"
RUN_NO: int = 48346

TRAIN_LENGTH: int = 400          # bunches per train
MAX_E_PER_BUNCH: int = 50         # zero-padding limit for electron TOF
MAX_I_PER_BUNCH: int = 120        # zero-padding limit for ion TOF
FOLDING_PARAMETER: float = 9969.225  # ns — used to fold TDC sweep into bunches
CHUNK_SIZE: int = 1000


# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------

DATA_FOLDER: str = str(config.LOCAL_DAQ_DIR / MEASUREMENT_NAME)
H5_FOLDER:   str = str(config.RAW_H5_DIR)
OUTPUT_PATH: Path = config.COMBINED_DIR / "test_config1.h5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Decode + align all three streams, write the combined H5, print summary."""
    print(f"Measurement   : {MEASUREMENT_NAME}  (run {RUN_NO})")
    print(f"Local DAQ dir : {DATA_FOLDER}")
    print(f"Raw H5 dir    : {H5_FOLDER}")
    print(f"Output        : {OUTPUT_PATH}")
    print()

    if not Path(DATA_FOLDER).is_dir():
        raise FileNotFoundError(f"Local DAQ folder not found: {DATA_FOLDER}")
    if not Path(H5_FOLDER).is_dir():
        raise FileNotFoundError(f"Raw H5 folder not found: {H5_FOLDER}")

    # --- Locate SDU .txt files ------------------------------------------
    files_in_folder = os.listdir(DATA_FOLDER)
    sdu_names = sorted(
        filter(re.compile(MEASUREMENT_NAME + r"_\d{10}\.txt").match, files_in_folder)
    )
    sdu_fpaths = [DATA_FOLDER + "/" + n for n in sdu_names]
    if not sdu_fpaths:
        raise ValueError(f"No SDU .txt files in {DATA_FOLDER}")
    print(f"Found {len(sdu_fpaths)} SDU .txt files.")

    # --- Locate TDC .lst files ------------------------------------------
    tdc_names = sorted(
        filter(re.compile(MEASUREMENT_NAME + r"_\d{10}\.lst").match, files_in_folder)
    )
    tdc_fpaths = [DATA_FOLDER + "/" + n for n in tdc_names]
    if not tdc_fpaths:
        raise ValueError(f"No TDC .lst files in {DATA_FOLDER}")
    print(f"Found {len(tdc_fpaths)} TDC .lst files.")

    # --- Locate raw H5 files for this run -------------------------------
    h5_paths = sorted(glob.glob(os.path.join(H5_FOLDER, f"*run{RUN_NO}*.h5")))
    if not h5_paths:
        raise ValueError(f"No raw H5 files for run {RUN_NO} in {H5_FOLDER}")
    print(f"Found {len(h5_paths)} raw H5 files for run {RUN_NO}.")
    for p in h5_paths:
        print(f"  {p}")

    # --- Determine train-ID range of the measurement --------------------
    first_tID = int(extract_sdu_data_from_single_file(sdu_fpaths[0])[0][0])
    last_tID  = int(extract_sdu_data_from_single_file(sdu_fpaths[-1])[0][-1])
    print(f"\nMeasurement spans train IDs {first_tID} .. {last_tID}  "
          f"({last_tID - first_tID + 1} trains).")

    # --- Find the first raw H5 file whose train range overlaps ----------
    first_h5_idx = None
    for path_idx, h5_path in enumerate(h5_paths):
        with h5.File(h5_path, "r") as f:
            tids = f["/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index"]
            tids_max = int(np.max(tids[...]))
            print(f"  {Path(h5_path).name}: max train ID = {tids_max}")
            if tids_max > first_tID:
                first_h5_idx = path_idx
                break
    if first_h5_idx is None:
        raise RuntimeError("None of the raw H5 files overlap with the SDU train-ID range.")
    print(f"First raw H5 file with overlap: index {first_h5_idx}")

    # --- Iterators ------------------------------------------------------
    sdu_it = SDUIterator(sdu_fpaths)
    tdc_it = TDCIterator(tdc_fpaths)

    # Each h5Iterator pulls (index, value) pairs row-by-row from the raw H5.
    # The 8-channel axis on "Pulse resolved energy" is *not* 8 detector
    # channels; the first index is "Intensity per pulse" which is what we
    # treat as the gmd. The selection is done where we consume next_gmd.
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

    # --- Fast-forward each iterator to first_tID -----------------------
    next_tID_gmd, next_gmd = _advance_to(gmd_it, first_tID, "gmd")
    next_tID_mpe, next_mpe = _advance_to(mpe_it, first_tID, "mpe")
    next_tID_hor_pos, next_hor_pos = _advance_to(hor_pos_it, first_tID, "hor_pos")
    next_tID_ver_pos, next_ver_pos = _advance_to(ver_pos_it, first_tID, "ver_pos")
    next_tID_z, next_z, next_z_std = sdu_it.__next__()
    (next_tID_tdc, next_eventcounts_e, next_tofs_e,
     next_eventcounts_i, next_tofs_i) = tdc_it.__next__()

    # --- Output H5 layout (schema + legacy extras) ---------------------
    data_len = last_tID - first_tID
    chunk = DataChunk(
        CHUNK_SIZE, TRAIN_LENGTH,
        max_ecounts=MAX_E_PER_BUNCH, max_icounts=MAX_I_PER_BUNCH,
    )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with h5.File(OUTPUT_PATH, "w") as f_out:
        tID_dset               = f_out.create_dataset("tID",               (data_len,),              dtype="double")
        data_flag_dset         = f_out.create_dataset("local_DAQ_running", (data_len,),              dtype="bool")
        z_dset                 = f_out.create_dataset("z",                 (data_len, TRAIN_LENGTH), dtype=np.float32)
        z_std_dset             = f_out.create_dataset("z_std",             (data_len, TRAIN_LENGTH), dtype=np.float32)
        gmd_dset               = f_out.create_dataset("gmd",               (data_len, TRAIN_LENGTH), dtype=np.float32)
        mpe_dset               = f_out.create_dataset("mpe",               (data_len,),              dtype=np.float32)
        hor_pos_dset           = f_out.create_dataset("hor_pos",           (data_len,),              dtype=np.float32)
        ver_pos_dset           = f_out.create_dataset("ver_pos",           (data_len,),              dtype=np.float32)
        tofs_e_dset            = f_out.create_dataset(
            "tofs_e", (data_len, TRAIN_LENGTH, MAX_E_PER_BUNCH),
            dtype=np.uint32, compression="gzip",
        )
        tofs_i_dset            = f_out.create_dataset(
            "tofs_i", (data_len, TRAIN_LENGTH, MAX_I_PER_BUNCH),
            dtype=np.uint32, compression="gzip",
        )
        between_tdc_files_dset = f_out.create_dataset("between_tdc_files", (data_len,),              dtype="bool")

        # --- Main alignment loop --------------------------------------
        for tID in range(first_tID, last_tID + 1):

            # ---- gmd (per-bunch) ------------------------------------
            if tID < next_tID_gmd:
                gmd = np.nan
            elif tID == next_tID_gmd:
                # next_gmd has shape (8, TRAIN_LENGTH); index 0 is the
                # per-pulse intensity (the gmd we want).
                gmd = next_gmd[0]
                try:
                    next_tID_gmd, next_gmd = gmd_it.__next__()
                except StopIteration:
                    print(f"Stopped by gmd on tID {next_tID_gmd}")
                    break
            else:
                raise ValueError(f"gmd iterator overshot: tID={tID}, next={next_tID_gmd}")

            # ---- mean photon energy ---------------------------------
            if tID < next_tID_mpe:
                mpe = np.nan
            elif tID == next_tID_mpe:
                mpe = next_mpe
                try:
                    next_tID_mpe, next_mpe = mpe_it.__next__()
                except StopIteration:
                    print(f"MPE stopped on {next_tID_mpe}. Subsequent are NaN.")
                    next_tID_mpe = last_tID + 1
                    mpe = np.nan
            else:
                raise ValueError(f"mpe iterator overshot: tID={tID}, next={next_tID_mpe}")

            # ---- horizontal beam position ---------------------------
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

            # ---- vertical beam position -----------------------------
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

            # ---- SDU (delay stage z) --------------------------------
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
                    print(f"Stopped by SDU on tID {next_tID_z}")
                    break
            else:
                raise ValueError(f"SDU overshot: tID={tID}, next={next_tID_z}")

            # ---- TDC (electron + ion tofs) --------------------------
            if tID < next_tID_tdc:
                tofs_e = None
                tofs_i = None
                between_tdc_files = tdc_it.is_between_files()
            elif tID == next_tID_tdc:
                tofs_e_raw = next_tofs_e
                tofs_i_raw = next_tofs_i
                # Fold the raw sweep tofs into per-bunch lists.
                tofs_e = [
                    tofs_e_raw[(tofs_e_raw >  b * FOLDING_PARAMETER)
                             & (tofs_e_raw < (b + 1) * FOLDING_PARAMETER)]
                    - b * FOLDING_PARAMETER
                    for b in range(TRAIN_LENGTH)
                ]
                tofs_i = [
                    tofs_i_raw[(tofs_i_raw >  b * FOLDING_PARAMETER)
                             & (tofs_i_raw < (b + 1) * FOLDING_PARAMETER)]
                    - b * FOLDING_PARAMETER
                    for b in range(TRAIN_LENGTH)
                ]
                between_tdc_files = False
                try:
                    (next_tID_tdc, next_eventcounts_e, next_tofs_e,
                     next_eventcounts_i, next_tofs_i) = tdc_it.__next__()
                except StopIteration:
                    print(f"Stopped by TDC on tID {next_tID_tdc}")
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
                print(f"  tID {tID}  ({tID - first_tID}/{last_tID - first_tID})")

        chunk.finish(
            tID_dset, data_flag_dset, z_dset, z_std_dset,
            gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
            tofs_e_dset, tofs_i_dset, between_tdc_files_dset,
        )

    # --- Completion summary -----------------------------------------
    print(f"\n=== test_config1.h5 written successfully ===")
    with h5.File(OUTPUT_PATH, "r") as f_out:
        n_trains = f_out["tID"].shape[0]
        n_bunches = f_out["gmd"].shape[1]
        print(f"Trains written  : {n_trains}")
        print(f"Bunches/train   : {n_bunches}")
        print(f"Datasets:")
        for k in sorted(f_out.keys()):
            d = f_out[k]
            print(f"  /{k:<22} shape={d.shape!s:<24} dtype={d.dtype}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _advance_to(it: h5Iterator, target_tID: int, label: str) -> tuple:
    """
    Pull rows from an ``h5Iterator`` until the train ID reaches ``target_tID``.

    Parameters
    ----------
    it : h5Iterator
        Iterator yielding ``(tID, value)`` pairs from a raw-H5 dataset.
    target_tID : int
        Train ID at which to stop. The returned tuple is the first row at
        or above this train ID.
    label : str
        Human-readable name used for progress messages.

    Returns
    -------
    tuple
        ``(tID, value)`` for the first row at or above ``target_tID``.
    """
    for t, v in it:
        if t >= target_tID:
            print(f"  {label} reached tID {t} (target {target_tID}).")
            return t, v
    raise RuntimeError(f"{label} iterator exhausted before reaching {target_tID}")


if __name__ == "__main__":
    main()
