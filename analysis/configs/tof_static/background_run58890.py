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
RUN_NO = 58890
CONFIG = 1
MODE = "xas_static"

# Preferred workflow for this run:
# one combined config-1 H5 per photon energy step, listed in the same
# order as NOMINAL_ENERGIES below.
INPUT_H5 = Path(path_config.COMBINED_DIR) / "background_run58890.h5"


# Legacy alternative:
# if you ever have one combined H5 spanning the full scan, comment out
# INPUT_H5S above and set INPUT_H5 instead.
# INPUT_H5 = Path(path_config.COMBINED_DIR) / "EDIT_ME_run58873_config1.h5"

# Nominal photon energies (eV), matched one-to-one with INPUT_H5S above.
NOMINAL_ENERGIES = np.array([
    271.0,
    271.5,
    272.0,
    272.5,
    273.0,
    273.5,
    274.0,
    274.5,
    275.0,
    271.0,
    271.5,
])

SECTION_SOURCE = "tid_starts"
SECTION_TID_STARTS = np.array([2661317895, 2661320027, 2661322085, 2661324143, 2661326326, 2661329013, 2661331805, 2661334468, 2661337192, 2661340026, 2661342671], dtype=np.int64)

# Half-open bunch range treated as signal.
SIGNAL_BUNCH_RANGE = (10, 40)

# Combined-H5 train filtering after removing between_tdc_files == True.
TRIM_START = 2
TRIM_END = 2

TRAIN_RATE_HZ = 10.0
TRANSITION_TRIM_SECONDS = 0.0
FIRST_SECTION_STATE = "open"


