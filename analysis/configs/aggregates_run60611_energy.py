"""
Aggregates for run 60611 (AcH_undulator_scan_265-280eV_2809702542) -- config 1,
energy mode, same bins as runs 60559 / 60560 / 60586 / 60591: electron TOF (D) x
ion TOF (C) (+ VLS) moments in bins of the VLS-measured photon energy x GMD.

Undulator scan 265 -> 280 eV (AcH), local DAQ from 23:24 on 26 Nov (first
train 2809702542), raw run 60611 (first raw train 2809702577). Written with
the VLS in the combined file (~/slurm_60611.sh live|final):

    F=11024642/processed/combined/run60611_AcH_undulator_scan_265-280eV_2809702542.h5
    python write_h5.py AcH_undulator_scan_265-280eV_2809702542 1 60611 --live --live-margin 0 \
        --max-ecounts 120 --max-icounts 50 -o $F
    python analysis/scripts/photon_energy.py --workers 48 $F
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60611_energy.py $F --workers 48
    -> run60611_AcH_undulator_scan_265-280eV_2809702542_aggregates_energy.h5

run60611_..._live*.h5 are partial writes made while the scan was acquiring
(--live with the default margin). The pixel -> eV calibration is fitted on the
run's own nominal steps (those acquired so far, for a live write).

TOF units are TDC units of 100 ps (10 units = 1 ns); see the run 60559
config for the windows. Ion bins reach m/q ~50 (AcH+ m/q 44).
"""

import numpy as np

CONFIG = 1
MODE   = "energy"

import calibration   # analysis/scripts, on the path when run through compute_aggregates.py

# Electron bins defined in kinetic energy, as for run 60560: 1 eV from 140 to 290 eV, 5 % of the energy from 1 to
# 140 eV, snapped to half TDC ticks.
_KE_EDGES = np.r_[np.geomspace(1, 140, int(np.ceil(np.log(140) / np.log(1.05))) + 1)[:-1], np.arange(140, 290.5, 1.0)]
TOF_EDGES = np.unique(np.floor(10 * calibration.tof_of_ke(_KE_EDGES)[::-1]) + 0.5)
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 100)   # 320 bins of 10 ns

# Same 0.5 eV photon-energy bins as runs 60559 / 60560 / 60586 / 60591; bins outside 265-280 eV stay empty.
ENERGY_KEY   = "photon_energy/measured"
ENERGY_EDGES = np.arange(263.75, 283.25 + 0.01, 0.5)   # 39 bins, 264.0 ... 283.0

GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)

# VLS aggregates from the combined file's vls dataset.
VLS_CROP_ROI   = (250, 630)
VLS_BACKGROUND = {"type": "auto", "roi": (100, 101)}
VLS_CM_PIXELS  = (650, 800)
VLS_BAD_PIXELS = [0, 1, 291, 292, 679, 680, 681, *range(832, 864)]

MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 500
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
