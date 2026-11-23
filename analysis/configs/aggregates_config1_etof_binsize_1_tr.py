import numpy as np

# Calibration-style eTOF binning: 1 unit bins.
TOF_EDGES = np.arange(-0.5, 5000.5 + 1.0, 1.0)

# iTOF is not the focus here; keep moderate binning.
ION_TOF_EDGES = np.arange(0.0, 40000.0 + 20.0, 20.0)

# Keep same GMD bins as the existing 5 June / etof1 config.
GMD_EDGES = [0.0, 3.0, 6.0, 12.0, 24.0]

CONFIG = 1
MODE = "time_resolved"

# Stage-position / delay bin edges.
# Same as the previous time-resolved delay aggregate:
# first/last = -15000 -> 90000, 48 bins, width 2000.
Z_EDGES = np.arange(-15000.0, 90000.0 + 2000.0, 2000.0)

CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END = 3

OUTPUT_SUFFIX = ""