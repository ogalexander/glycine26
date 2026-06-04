"""
Config for ``compute_xas_aggregates.py`` — XAS scan, config 2.
First xas scan take (without sample) on 3 June 2026

Each open-shutter section in the run is assigned a nominal photon
energy by section index, and per-(energy, GMD bin) aggregates of the
VLS spectrum and GMD are written to disk. Run with:

    python analysis/scripts/compute_xas_aggregates.py \
        analysis/configs/aggregates_static_xas_no_sample_03June.py [-o OUT.h5]

When the raw H5 carries the nominal scan energies natively, replace the
NOMINAL_ENERGIES line below with a lookup of those values.
"""

import numpy as np


# Identify the scan.
RUN_NO  = 58793
CONFIG  = 2
MODE    = "xas_scan"

# Read at most this many raw H5 files for the run (None = all).
MAX_FILES = None

# Nominal photon energies (eV) assigned by section index. Use whatever
# scan grid was set on the beamline; here, 270 -> 286 eV in 0.5 eV
# steps (33 points).
# NOMINAL_ENERGIES = np.arange(270.0, 286.0 + 0.25, 0.5)
NOMINAL_ENERGIES = np.array([*np.arange(270, 276, 0.5), 276, *np.arange(276, 283.5, 0.5), 283, 283.5, 285])

# GMD bin edges (uJ). Use a single bin (e.g. [-np.inf, np.inf]) to
# disable GMD binning.
GMD_EDGES = [0.0, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]

# VLS pixel ROI (half-open). Applied before any background subtraction.
CROP_ROI = (450, 650)

# Bunch ranges (half-open) for the VLS.
#   SIGNAL_BUNCH_RANGE : bunches overlapping the GMD acquisition window;
#                        these contribute one shot each.
#   BG_BUNCH_RANGE     : non-signal bunches (no x-rays); their mean
#                        spectrum on each train is subtracted from every
#                        signal bunch of that train.
SIGNAL_BUNCH_RANGE = (10, 40)
BG_BUNCH_RANGE     = (50, 100)

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


# When True (default) the aggregates have a leading axis of length
# n_sections_used so they are indexed by per-section nominal photon
# energy. Set to False to collapse the energy axis to a single bin -
# useful when feeding the aggregates into the ADMM XAS solver, which
# wants one Cov(A,A) / Cov(A,G) pair per GMD bin built across the whole
# energy scan.
GROUP_BY_ENERGY = True