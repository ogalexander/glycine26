import numpy as np

# Identify the scan.
RUN_NO  = 58780
CONFIG  = 2
MODE    = "xas_static"

# Read at most this many raw H5 files for the run (None = all).
MAX_FILES = None

# Nominal photon energies (eV) assigned by section index. Use whatever
NOMINAL_ENERGIES = np.arange(270.0, 286.0 + 0.25, 0.5)

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

