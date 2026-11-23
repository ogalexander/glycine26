import numpy as np

# Calibration-style eTOF binning: 1 unit bins.
TOF_EDGES = np.arange(0.0, 5000.0 + 1.0, 1.0)

# iTOF is not the focus here; keep moderate binning.
ION_TOF_EDGES = np.arange(0.0, 40000.0 + 20.0, 20.0)

# Keep same GMD bins as the existing 5 June config.
GMD_EDGES = [0.0, 3.0, 6.0, 12.0, 24.0]

CONFIG = 1
MODE = "spectral"

CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END = 3

OUTPUT_SUFFIX = ""
