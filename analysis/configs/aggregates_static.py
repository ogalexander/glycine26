"""
Aggregates for static gas-phase measurements -- config 1, spectral mode.

Electron TOF (D) x ion TOF (C) moments per GMD bin.

Electron binning follows the undulator and delay scan configurations.
GMD bins are calculated automatically from the shot population.

TOF units: TDC units of 100 ps (10 units = 1 ns).

Analysis:
    analysis/notebooks/inspect_static_spectra.ipynb
"""

import numpy as np
import calibration

CONFIG = 1
MODE   = "spectral"

# Electron TOF bins.
# Approximately 5% relative KE width below 140 eV,
# and 1 eV bins above 140 eV.
_KE_EDGES = np.r_[
    np.geomspace(1, 140, int(np.ceil(np.log(140) / np.log(1.05))) + 1)[:-1],
    np.arange(140, 290.5, 1.0)
]

TOF_EDGES = np.unique(
    np.floor(10 * calibration.tof_of_ke(_KE_EDGES)[::-1]) + 0.5
)

# Ion TOF: 200-4500 ns, bins of 5 ns.
# Extended to include ethanolamine and glycine parent ions.
ION_TOF_EDGES = np.arange(2000, 45000 + 1, 50)

# Automatic GMD bins (uJ).
GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

# Bunch selection.
BUNCH_RANGE = (2, 100)

# Reject burst events.
MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 100
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = "static"

# Gotthard / VLS aggregation.
# VLS_CROP_ROI  = (250, 630)

# VLS_BACKGROUND = {"type": "auto", "roi": (100, 101)}
# VLS_CM_PIXELS  = (650, 800)
# VLS_BAD_PIXELS = [0, 1, 291, 292, 679, 680, 681, *range(832, 864)]
# VLS_RAW_RUN    = None