"""
As ``aggregates_run60538_argon.py`` but only the first 20 pulses of each
train. The ion yield per uJ falls ~6x over the first ~50 bunches of run
60538 (electrons stay within ~10%), so this keeps the high-efficiency
part of the train for comparison.

    python analysis/scripts/compute_aggregates.py \
        analysis/configs/aggregates_run60538_argon_b0-20.py \
        11024642/processed/combined/run60538_Argon_2807259185.h5
"""

import numpy as np

CONFIG = 1
MODE   = "spectral"

TOF_EDGES     = np.arange(500, 10000 + 1, 50)
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 50)

GMD_EDGES = [0.02, 0.04, 0.06, 0.085, 0.115, 0.15, 0.19, 0.24, 0.31, 0.43, 0.65, 1.5]

BUNCH_RANGE = (0, 20)

MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 100
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = "_b0-20"
