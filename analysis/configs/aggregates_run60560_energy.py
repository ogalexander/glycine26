"""
Aggregates for run 60560 (Ethanol_undulator_scan_2808016357) -- config 1,
energy mode, same bins as run 60559 (aggregates_run60559_energy.py):
electron TOF (D) x ion TOF (C) moments in bins of the VLS-measured photon
energy x GMD.

Undulator scan from 265 eV in 0.5 eV nominal steps of ~10 min; the run was
still acquiring when this was set up, so the combined file is a live write
without VLS (whole run so far), and the photon energy is measured from the
raw Gotthard frames:

    python write_h5.py Ethanol_undulator_scan_2808016357 1 60560 --live --no-vls \
        --max-ecounts 120 --max-icounts 50 \
        -o 11024642/processed/combined/run60560_Ethanol_undulator_scan_2808016357_live.h5
    python analysis/scripts/photon_energy.py --raw-run 60560 \
        11024642/processed/combined/run60560_Ethanol_undulator_scan_2808016357_live.h5
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60560_energy.py \
        11024642/processed/combined/run60560_Ethanol_undulator_scan_2808016357_live.h5
    -> run60560_Ethanol_undulator_scan_2808016357_live_aggregates_energy.h5

Rerun all three to update as the run continues (photon_energy.py needs
--overwrite once the group exists). The pixel -> eV calibration is fitted
on this run's own nominal steps.

TOF units are TDC units of 100 ps (10 units = 1 ns); see the run 60559
config for the windows.
"""

import numpy as np

CONFIG = 1
MODE   = "energy"

TOF_EDGES     = np.arange(500, 10000 + 1, 50)     # 190 bins of 5 ns
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 100)   # 320 bins of 10 ns

# 0.5 eV bins centred on the integer and half-integer energies (the 0.5 eV
# nominal steps sit at bin centres); bins beyond the scan so far stay empty.
ENERGY_KEY   = "photon_energy/measured"
ENERGY_EDGES = np.arange(263.75, 283.25 + 0.01, 0.5)   # 39 bins, 264.0 ... 283.0

GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)

MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 100
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
