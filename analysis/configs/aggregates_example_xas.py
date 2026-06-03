"""
Example config for ``compute_xas_aggregates.py`` — XAS scan, config 2.

Each open-shutter section in the run is assigned a nominal photon
energy by section index, and per-(energy, GMD bin) aggregates of the
VLS spectrum and GMD are written to disk. Run with:

    python analysis/scripts/compute_xas_aggregates.py \
        analysis/configs/aggregates_example_xas.py [-o OUT.h5]

When the raw H5 carries the nominal scan energies natively, replace the
NOMINAL_ENERGIES line below with a lookup of those values.
"""

import numpy as np


# Identify the scan.
RUN_NO  = 58780
CONFIG  = 2
MODE    = "xas_scan"

# Read at most this many raw H5 files for the run (None = all).
MAX_FILES = None

# Nominal photon energies (eV) assigned by section index. Use whatever
# scan grid was set on the beamline; here, 270 -> 286 eV in 0.5 eV
# steps (33 points).
NOMINAL_ENERGIES = np.arange(270.0, 286.0 + 0.25, 0.5)

# GMD bin edges (uJ). Use a single bin (e.g. [-np.inf, np.inf]) to
# disable GMD binning.
GMD_EDGES = [0.0, 0.5, 1.0, 2.0, 4.0]

# VLS pixel ROI (half-open). Applied before any background subtraction.
CROP_ROI = (500, 600)

# Bunch ranges (half-open) for the VLS.
#   SIGNAL_BUNCH_RANGE : bunches overlapping the GMD acquisition window;
#                        these contribute one shot each.
#   BG_BUNCH_RANGE     : non-signal bunches (no x-rays); their mean
#                        spectrum on each train is subtracted from every
#                        signal bunch of that train.
SIGNAL_BUNCH_RANGE = (0, 30)
BG_BUNCH_RANGE     = (30, 100)

# Fast shutter dataset paths.
SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"

# Transition rejection around shutter moves.
TRAIN_RATE_HZ           = 10.0
TRANSITION_TRIM_SECONDS = 3.0

# Protocol prior: which state the scan opens with. If "open", section 0
# uses the *first* closed block (which comes after it) as its
# background; otherwise the rule is always "use the preceding closed
# block."
FIRST_SECTION_STATE = "open"
