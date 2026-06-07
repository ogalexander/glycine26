"""
Example config for ``compute_aggregates.py`` — config 1 spectral mode
(electron + ion TOF).

Config 1 datasets:
    * ``tofs_e``  — electron TOF (used as D)
    * ``tofs_i``  — ion TOF      (used as C)

No VLS in config 1; ``CROP_ROI`` and ``BACKGROUND`` are omitted.

Usage:

    python analysis/scripts/compute_aggregates.py \
        analysis/configs/aggregates_example_cfg1.py \
        <path-to-input.h5> [-o <output.h5>]
"""

import numpy as np


# Electron TOF histogram edges (100 ps). n_tof = len(TOF_EDGES) - 1.
TOF_EDGES     = np.linspace(3, 3000.0, 1200) # in unit of 100 ps !!

# Ion TOF histogram edges (100 ps). n_tof_i = len(ION_TOF_EDGES) - 1.
# Ions have much longer flight times; widen the window accordingly.
ION_TOF_EDGES = np.linspace(0, 40000.0, 500) # in unit of 100 ps !!

# GMD bin edges (µJ). Shots outside [first, last] are dropped.
GMD_EDGES = [0.0, 3.0, 6.0, 12.0, 24.0]

TRIM_END=40

CONFIG = 1
MODE   = "spectral"

CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END   = 3

# Appended to the default output filename before ".h5". Empty string
# keeps the conventional "<input>_aggregates.h5" name.
OUTPUT_SUFFIX = ""
