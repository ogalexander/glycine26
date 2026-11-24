"""
Aggregates for run 60538 (Argon_2807259185) -- config 1, spectral mode.
Electron TOF (D) x ion TOF (C) second moments per GMD bin, for the
e-e / e-i covariance check in
``analysis/notebooks/covariance_ee_ei.ipynb``.

Combined H5, written with the beamtime channel map (electron = TDC ch 2,
ion = ch 1; ``config.TDC_CHANNEL_E/I``) and the hit caps swapped to suit
it (electrons are the busier channel here):

    python write_h5.py Argon_2807259185 1 60538 --max-ecounts 120 --max-icounts 50 \
        -o 11024642/processed/combined/run60538_Argon_2807259185.h5

6057 trains x 101 slots, 100 FEL pulses per train (slot 100 has no GMD).

    python analysis/scripts/compute_aggregates.py \
        analysis/configs/aggregates_run60538_argon.py \
        11024642/processed/combined/run60538_Argon_2807259185.h5

TOF units are TDC units of 100 ps (10 units = 1 ns).
    electrons 500-10000   (50-1000 ns). Nothing but flat background below
                          ~875; the electron spectrum ends at ~9500. Later
                          hits include crosstalk from the ion detector
                          (fast electrons from the ion TOF reaching the
                          electron TOF ~25 ns after each ion hit), so the
                          window stops at 10000.
    ions      2000-34000  (200-3400 ns): the light-ion group at 2500-5000
                          and the main peaks at 14500-31000.
"""

import numpy as np

CONFIG = 1
MODE   = "spectral"

TOF_EDGES     = np.arange(500, 10000 + 1, 50)     # 190 bins of 5 ns
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 50)    # 640 bins of 5 ns

# Roughly equal-population GMD bins (uJ); median 0.15 uJ. Shots below
# 0.02 (GMD noise floor, incl. negative readings) and above 1.5 are dropped.
GMD_EDGES = [0.02, 0.04, 0.06, 0.085, 0.115, 0.15, 0.19, 0.24, 0.31, 0.43, 0.65, 1.5]

BUNCH_RANGE = (0, 100)      # all FEL pulses

# Burst events: 5 trains have one shot with >=120 e and one with >=50 i
# (both hit caps clipped). Otherwise the 99.99th percentiles are 31 e /
# 16 i per shot (hits anywhere in the bunch period).
MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 100
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
