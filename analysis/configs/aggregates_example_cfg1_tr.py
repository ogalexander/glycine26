"""
Example time-resolved config for ``compute_aggregates.py`` — config 1
(electron + ion TOF), binned by GMD x stage z.

Both the electron and ion TOF spectra keep their full ``n_tof`` /
``n_tof_i`` resolution; only an extra ``N_Z`` axis is prepended to
every aggregate.  No ``TOF_ROI`` is used (that is a config-2 TR
shorthand for collapsing the eTOF to a scalar count).

Output defaults to ``<input>_aggregates_tr.h5``.

    python analysis/scripts/compute_aggregates.py \\
        analysis/configs/aggregates_example_cfg1_tr.py \\
        <path-to-input.h5>
"""

import numpy as np


# Electron TOF histogram edges (100 ps).
TOF_EDGES     = np.linspace(0.0, 40000.0, 2000)

# Ion TOF histogram edges (100 ps).
ION_TOF_EDGES = np.linspace(0.0, 40000.0, 2000)

# GMD bin edges (uJ). Shots outside [first, last] are dropped.
GMD_EDGES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]

CONFIG = 1
MODE   = "time_resolved"

# Stage-position (z / sdu) bin edges. Adjust to the actual scan range.
Z_EDGES = np.linspace(-15000.0, 16500.0, 21)   # 20 bins

CHUNK_SIZE = 200
TRIM_START = 3
TRIM_END   = 3

# Appended to the default output filename before ".h5". Empty string
# keeps the conventional "<input>_aggregates_tr.h5" name.
OUTPUT_SUFFIX = ""
