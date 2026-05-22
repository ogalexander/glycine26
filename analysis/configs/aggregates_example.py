"""
Example config for ``compute_aggregates.py`` — tuned for test_config2.h5.

Copy this file, tweak the constants for your run, and pass it to the
script:

    python analysis/scripts/compute_aggregates.py \
        analysis/configs/aggregates_example.py \
        <path-to-input.h5> [-o <output.h5>]
"""

import numpy as np


# VLS pixel ROI (half-open). The VLS will be cropped to this slab
# before background subtraction and aggregation.
CROP_ROI = (380, 500)

# Background to subtract from the cropped VLS, one of:
#   {"type": "auto",  "roi": (bunch_start, bunch_end)}   – mean spectrum
#       over those bunch indices, averaged across all kept trains.
#   {"type": "array", "path": "background.npy"}          – pre-computed
#       spectrum matching the cropped pixel axis.
#   {"type": "none"}                                     – no subtraction.
BACKGROUND = {"type": "auto", "roi": (0, 5)}

# Per-shot eTOF histogram edges (ns). n_tof = len(TOF_EDGES) - 1.
TOF_EDGES = np.linspace(500.0, 1750.0, 126)

# GMD bin edges (µJ). Shots with GMD outside [first, last] are dropped.
GMD_EDGES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]

# H5 layout: 1 for tofs_e/tofs_i, 2 for liq_tofs_e/vls.
CONFIG = 2

# Iteration / quality knobs.
CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END   = 3
