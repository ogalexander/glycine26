"""
Example time-resolved config for ``compute_aggregates.py``.

Binning axes are GMD × stage z; the eTOF is reduced per shot to a scalar
count inside ``TOF_ROI`` (so ``n_tof = 1`` in the output). Output
defaults to ``<input>_aggregates_tr.h5`` to avoid clashing with the
spectral aggregate.

    python analysis/scripts/compute_aggregates.py \
        analysis/configs/aggregates_example_tr.py \
        <path-to-input.h5>
"""

import numpy as np


# Shared with the spectral config.
CROP_ROI   = (380, 500)
BACKGROUND = {"type": "auto", "roi": (0, 5)}
GMD_EDGES  = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
CONFIG     = 2

# Time-resolved mode.
MODE = "time_resolved"

# Stage-position (z / sdu) bin edges. The test-config 2 file has a roughly
# uniform stage scan from ~-15000 to ~+16500 arb. units.
Z_EDGES = np.linspace(-15000.0, 16500.0, 21)   # 20 bins

# Half-open TOF window (ns). Per-shot D_i is the number of non-zero
# hits whose TOF falls in [TOF_ROI[0], TOF_ROI[1]).
TOF_ROI = (800.0, 1200.0)

# Iteration / quality knobs.
CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END   = 3
