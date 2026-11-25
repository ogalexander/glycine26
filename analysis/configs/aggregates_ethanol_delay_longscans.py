"""
Ethanol delay scans at 268 eV.

Config 1: electron and ion TOF aggregates in GMD and delay bins.
TOF binning follows the undulator scan of run 60560.
"""

import numpy as np
import calibration

CONFIG = 1
MODE   = "time_resolved"

# Electron bins: 5% energy spacing below 140 eV, 1 eV above.
_KE_EDGES = np.r_[
    np.geomspace(1, 140, int(np.ceil(np.log(140) / np.log(1.05))) + 1)[:-1],
    np.arange(140, 290.5, 1.0)
]

TOF_EDGES = np.unique(
    np.floor(10 * calibration.tof_of_ke(_KE_EDGES)[::-1]) + 0.5
)

# Ion TOF bins (100 ps units).
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 100)


# Delay binning (stage units).
# 1000 stage units = 1 um = 0.349 fs.

# Long ethanol delay scan: -4.89 to 59.33 fs

# Z_EDGES = np.linspace(-14010, 170010, 65)       # 1.00 fs
# Z_EDGES = np.linspace(-14010, 170010, 44)     # 1.49 fs
Z_EDGES = np.linspace(-14010, 170010, 33)     # 2.01 fs
# Z_EDGES = np.linspace(-14010, 170010, 27)     # 2.47 fs

# GMD binning (uJ).
GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)

# VLS aggregation (disabled for initial delay analysis).
# VLS aggregation (same settings as the energy scan).
VLS_CROP_ROI   = (250, 630)
VLS_BACKGROUND = {"type": "auto", "roi": (100, 101)}
VLS_CM_PIXELS  = (650, 800)
VLS_BAD_PIXELS = [0, 1, 291, 292, 679, 680, 681, *range(832, 864)]
VLS_RAW_RUN    = None

# Remove burst shots.
MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 500
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""