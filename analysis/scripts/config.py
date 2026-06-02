"""
Path management for the glycine 2026 analysis project.

All paths used by the writers, analysis scripts, and notebooks should be
imported from this module. This is the *only* place hardcoded absolute
paths are allowed to live.

Switching environments
----------------------
The active profile is selected via the environment variable ``FLASH_ENV``:

- ``local``  (default) — paths on the user's workstation
- ``remote``           — paths on the FLASH online-cluster (TBD)

PowerShell example::

    $env:FLASH_ENV = "remote"
    python write_h5_test_config1.py

POSIX example::

    FLASH_ENV=remote python write_h5_test_config1.py

Remote paths must be filled in once the production location on the FLASH
online cluster is confirmed.

Exported paths
--------------
- ``DATA_ROOT``     : root of the 11022188 beamtime folder
- ``RAW_H5_DIR``    : raw HDF5 files from the FLASH DAQ
- ``LOCAL_DAQ_DIR`` : processed local DAQ data (.lst + .txt pairs)
- ``COMBINED_DIR``  : output directory for combined H5 files (created if missing)
- ``ANALYSIS_DIR``  : analysis/scripts folder (location of this file)
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

# `ANALYSIS_DIR` is anchored to this file's location so the rest of the
# paths can be expressed relative to the repo root, regardless of the
# current working directory.
ANALYSIS_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = ANALYSIS_DIR.parents[1]  # analysis/scripts -> analysis -> repo root


_PROFILES: dict[str, dict[str, Path]] = {
    "local": {
        "DATA_ROOT":     _REPO_ROOT / "11022188",
        "RAW_H5_DIR":    _REPO_ROOT / "11022188" / "raw" / "hdf" / "online-0" / "fl2user1",
        "LOCAL_DAQ_DIR": _REPO_ROOT / "11022188" / "processed" / "local_DAQ",
        "COMBINED_DIR":  _REPO_ROOT / "11022188" / "processed" / "combined",
    },
    # TODO: confirm the production data root on the FLASH online cluster
    # and fill in the four paths below. Until then `FLASH_ENV=remote` will
    # raise.
    "remote": {
        "DATA_ROOT":     Path("/asap3/flash/gpfs/fl24/2026/data/11022188"),
        "RAW_H5_DIR":    Path("/asap3/flash/gpfs/fl24/2026/data/11022188/raw/hdf/online-0/fl2user1"),
        "LOCAL_DAQ_DIR": Path("/asap3/flash/gpfs/fl24/2026/data/11022188/processed/local_DAQ"),
        "COMBINED_DIR":  Path("/asap3/flash/gpfs/fl24/2026/data/11022188/processed/combined"),
    },
}


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

FLASH_ENV: str = os.environ.get("FLASH_ENV", "local").lower()

if FLASH_ENV not in _PROFILES:
    raise ValueError(
        f"FLASH_ENV={FLASH_ENV!r} is not a known profile. "
        f"Expected one of {sorted(_PROFILES)}."
    )

_active = _PROFILES[FLASH_ENV]

DATA_ROOT:     Path = _active["DATA_ROOT"]
RAW_H5_DIR:    Path = _active["RAW_H5_DIR"]
LOCAL_DAQ_DIR: Path = _active["LOCAL_DAQ_DIR"]
COMBINED_DIR:  Path = _active["COMBINED_DIR"]


# Ensure the combined-output directory exists. The raw / local-DAQ
# directories should already exist (they are read-only inputs); creating
# them here would hide a misconfigured profile.
COMBINED_DIR.mkdir(parents=True, exist_ok=True)


__all__ = [
    "FLASH_ENV",
    "DATA_ROOT",
    "RAW_H5_DIR",
    "LOCAL_DAQ_DIR",
    "COMBINED_DIR",
    "ANALYSIS_DIR",
]
