"""
Path management for the gas-vls beamtime analysis (FLASH beamtime 11024642).

All paths used by the writers, analysis scripts, and notebooks should be
imported from this module. This is the *only* place hardcoded absolute
paths are allowed to live, and the beamtime is identified in exactly one
place: ``BEAMTIME_ID`` / ``BEAMTIME_YEAR`` / ``BEAMLINE_HALL`` below.

Switching environments
----------------------
The active profile is selected via the environment variable ``FLASH_ENV``:

- ``local``  (default) — data reached through ``<repo root>/<BEAMTIME_ID>``.
                         On Maxwell this is a symlink to the beamtime folder
                         (``ln -s /asap3/flash/gpfs/fl24/2026/data/11024642
                         <repo root>/11024642``); on a workstation it can be
                         a local copy with the same layout.
- ``remote``           — absolute ``/asap3`` paths (no symlink needed, e.g.
                         for batch jobs on Maxwell).

Both profiles use the same folder layout below the data root.

POSIX example::

    FLASH_ENV=remote python write_h5.py my_scan 1 60513

Exported paths
--------------
- ``DATA_ROOT``     : root of the beamtime folder
- ``RAW_H5_DIR``    : raw HDF5 files from the FLASH DAQ
- ``LOCAL_DAQ_DIR`` : parent of the SDU / TDC trees
- ``SDU_DIR``       : SDU delay-stage ``.txt`` files (``LOCAL_DAQ_DIR / "delay_log"``)
- ``TDC_DIR``       : TDC ``.lst`` files (``LOCAL_DAQ_DIR / "TDC"``)
- ``COMBINED_DIR``  : output directory for combined H5 files (created if missing)
- ``DECODER_DIR``   : folder containing the ``beamtime_scripts_2021`` TDC
                      decoder package (added to ``sys.path`` by ``write_h5.py``)
- ``ANALYSIS_DIR``  : analysis/scripts folder (location of this file)
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Beamtime identity — change these (and only these) for a new beamtime.
# ---------------------------------------------------------------------------

BEAMTIME_ID:   str = "11024642"
BEAMTIME_YEAR: str = "2026"
BEAMLINE_HALL: str = "fl24"

# ---------------------------------------------------------------------------
# Bunch-axis layout for config 1 (gas arm) — working assumptions, to be
# confirmed during the beamtime. Used as defaults by write_h5.py and
# data_loading.load_raw_h5.
# ---------------------------------------------------------------------------

TRAIN_LENGTH: int = 101  # bunch slots per train in combined files
# Gotthard line (b + VLS_OFFSET) is stored as bunch b. With 100 pulses the
# frame has 105 lines: 0-1 dark, 2-101 = GMD bunches 0-99, 102-104 dark
# (runs 60545, 60559; see analysis/notebooks/vls_alignment.ipynb), so the
# 101-slot crop keeps every lit line plus one dark line in slot 100.
VLS_OFFSET:   int = 2

# TDC channel -> config 1 dataset. At this beamtime the electron TOF is on
# channel 2 and the ion TOF on channel 1 (checked on run 60538: ch 2 hits
# sit below ~1000 ns, ch 1 hits are the Ar+ / N2+ ... ion peaks at
# 1400-3200 ns). Beamtime 11022188 used electron = 1, ion = 2.
TDC_CHANNEL_E: int = 2   # -> tofs_e
TDC_CHANNEL_I: int = 1   # -> tofs_i

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

# `ANALYSIS_DIR` is anchored to this file's location so the rest of the
# paths can be expressed relative to the repo root, regardless of the
# current working directory.
ANALYSIS_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = ANALYSIS_DIR.parents[1]  # analysis/scripts -> analysis -> repo root

_ASAP3_ROOT: Path = Path(
    f"/asap3/flash/gpfs/{BEAMLINE_HALL}/{BEAMTIME_YEAR}/data/{BEAMTIME_ID}"
)

_DATA_ROOTS: dict[str, Path] = {
    "local":  _REPO_ROOT / BEAMTIME_ID,
    "remote": _ASAP3_ROOT,
}


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

FLASH_ENV: str = os.environ.get("FLASH_ENV", "local").lower()

if FLASH_ENV not in _DATA_ROOTS:
    raise ValueError(
        f"FLASH_ENV={FLASH_ENV!r} is not a known profile. "
        f"Expected one of {sorted(_DATA_ROOTS)}."
    )

DATA_ROOT:     Path = _DATA_ROOTS[FLASH_ENV]
RAW_H5_DIR:    Path = DATA_ROOT / "raw" / "hdf" / "online-0" / "fl2user1"
# Local-DAQ layout carried over from the previous beamtime; confirm once
# the first TDC / SDU files of this beamtime are copied over.
LOCAL_DAQ_DIR: Path = DATA_ROOT / "processed" / "local_DAQ"
SDU_DIR:       Path = LOCAL_DAQ_DIR / "delay_log"
TDC_DIR:       Path = LOCAL_DAQ_DIR / "TDC"
COMBINED_DIR:  Path = DATA_ROOT / "processed" / "combined"
DECODER_DIR:   Path = (DATA_ROOT / "processed" / "analysis_tools"
                       / "analysis_tools_2023" / "decoding_script")


# Ensure the combined-output directory exists. The raw / local-DAQ
# directories should already exist (they are read-only inputs); creating
# them here would hide a misconfigured profile. A read-only or absent
# data root (e.g. a workstation without the data) must not break imports.
try:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass


__all__ = [
    "BEAMTIME_ID",
    "BEAMTIME_YEAR",
    "BEAMLINE_HALL",
    "TRAIN_LENGTH",
    "VLS_OFFSET",
    "TDC_CHANNEL_E",
    "TDC_CHANNEL_I",
    "FLASH_ENV",
    "DATA_ROOT",
    "RAW_H5_DIR",
    "LOCAL_DAQ_DIR",
    "SDU_DIR",
    "TDC_DIR",
    "COMBINED_DIR",
    "DECODER_DIR",
    "ANALYSIS_DIR",
]
