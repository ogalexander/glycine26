"""
Example config for ``compute_static_xas_cfg1.py`` — static XAS, config 1
(electron + ion TOF).

The shutter signal is now carried by the combined H5 (per-train
``/shutter`` dataset written by ``write_h5.py``), so this script reads
*only* the combined H5 — no separate raw-H5 lookup is needed.

Each open-shutter section is assigned a nominal photon energy by section
index. Per-shot (tofs_e, tofs_i, gmd) arrays are written to disk without
any GMD binning or TOF histogramming; downstream notebooks decide how to
bin / window the hits.

Run with::

    python analysis/scripts/compute_static_xas_cfg1.py \\
        analysis/configs/xas_static/example_cfg1.py [-o OUT.h5] [-i IN.h5]
"""
import numpy as np

# Identify the scan.
RUN_NO  = 48346       # placeholder — fill in real cfg-1 XAS run
CONFIG  = 1
MODE    = "xas_static"

# Optional override: combined H5 path. Default is COMBINED_DIR/run<N>.h5.
# INPUT_H5 = "/path/to/some_combined.h5"

# Nominal photon energies (eV) assigned to detected open sections by index.
NOMINAL_ENERGIES = np.array([*np.arange(270, 276, 0.5),
                             276,
                             *np.arange(276, 283.5, 0.5),
                             283, 283.5, 285])

# Signal bunch range (half-open). Bunches overlapping the GMD acquisition
# window — same convention as the cfg-2 script.
SIGNAL_BUNCH_RANGE = (10, 40)

# load_data trimming.
TRIM_START = 2
TRIM_END   = 2

# Transition rejection around shutter moves.
TRAIN_RATE_HZ           = 10.0
TRANSITION_TRIM_SECONDS = 3.0

# Protocol prior: "open" means the scan begins with the shutter open.
FIRST_SECTION_STATE = "open"

# Section detection. Default "shutter" thresholds the /shutter trace in
# the combined H5. Use "tid_starts" for scans where each local-DAQ file
# is one energy and the shutter readback is not a clean binary signal.
# SECTION_SOURCE = "tid_starts"
# SECTION_TID_STARTS = np.array([
#     2661317895,
#     2661320027,
#     2661322085,
# ])
