"""
Config for ``compute_static_tof.py`` — shot-resolved config-1 eTOF scan.

Run 58873:
    279 eV -> 288 eV in 1 eV steps
    magnetic-bottle data (config 1)
    all per-energy files should be treated as always-open acquisition

Edit ``INPUT_H5S`` to point at the combined config-1 H5 files produced
by ``write_h5.py`` for this run, one file per photon energy step.

Run with:
    python analysis/scripts/compute_static_tof.py \
        analysis/configs/tof_static/run58873_magnetic_bottle.py [-o OUT.h5]
"""

from pathlib import Path

import numpy as np

import config as path_config


# Identify the scan.
RUN_NO = 58873
CONFIG = 1
MODE = "tof_static"

# Preferred workflow for this run:
# one combined config-1 H5 per photon energy step, listed in the same
# order as NOMINAL_ENERGIES below.
INPUT_H5S = [
Path(path_config.COMBINED_DIR) / "glycine_WL_scan_279eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_280eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_281eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_282eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_283eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_284eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_285eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_286eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_287eV.h5",
    Path(path_config.COMBINED_DIR) / "glycine_WL_scan_288eV.h5",
]

# Legacy alternative:
# if you ever have one combined H5 spanning the full scan, comment out
# INPUT_H5S above and set INPUT_H5 instead.
# INPUT_H5 = Path(path_config.COMBINED_DIR) / "EDIT_ME_run58873_config1.h5"

# Nominal photon energies (eV), matched one-to-one with INPUT_H5S above.
NOMINAL_ENERGIES = np.arange(279.0, 288.0 + 0.5, 1.0)

# Electron TOF histogram edges stored into the output H5 as the notebook's
# default plot grid. The writer itself keeps the raw per-shot hit lists.
TOF_EDGES = np.linspace(0.0, 40000.0, 2000)

# Half-open bunch range treated as signal.
SIGNAL_BUNCH_RANGE = (0, 101)

# Combined-H5 train filtering after removing between_tdc_files == True.
TRIM_START = 2
TRIM_END = 2

# Histogramming chunk size (trains per chunk). Lower this if memory is tight.
CHUNK_SIZE = 50

# Explicit guard: this shot-resolved path expects always-open per-energy files.
USE_SHUTTER = False
