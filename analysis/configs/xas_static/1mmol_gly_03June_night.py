"""
Config for ``compute_static_xas.py`` — static XAS, config 2.
1 mM glycine, 3 June 2026 (night run).

Each open-shutter section is assigned a nominal photon energy by section
index. Per-shot (VLS, GMD) arrays are written to disk without any GMD
binning; binning is done in the notebook.

Run with:
    python analysis/scripts/compute_static_xas.py \
        analysis/configs/xas_static/1mmol_gly_03June_night.py [-o OUT.h5]
"""

import numpy as np

# Identify the scan.
RUN_NO  = 58793
CONFIG  = 2
MODE    = "xas_static"

# Read at most this many raw H5 files (None = all).
MAX_FILES = None

# Nominal photon energies (eV) assigned to detected sections by index.
NOMINAL_ENERGIES = np.array([
    *np.arange(270, 276, 0.5), 276, *np.arange(276, 283.5), 283, 283.5, 285,
])

# VLS pixel ROI (half-open). Applied before any background subtraction.
CROP_ROI = (450, 650)

# Bunch ranges (half-open).
#   SIGNAL_BUNCH_RANGE : bunches overlapping the GMD acquisition window.
#   BG_BUNCH_RANGE     : non-signal bunches used for per-train baseline.
SIGNAL_BUNCH_RANGE = (10, 40)
BG_BUNCH_RANGE     = (50, 100)

# Fast shutter dataset paths.
SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"

# Transition rejection around shutter moves.
TRAIN_RATE_HZ           = 10.0
TRANSITION_TRIM_SECONDS = 3.0

# Protocol prior: "open" means the scan begins with the shutter open,
# so section 0 uses the first closed block as its background.
FIRST_SECTION_STATE = "open"
