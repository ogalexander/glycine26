"""
Aggregates for run 60560 (Ethanol_undulator_scan_2808016357) -- config 1,
energy mode, same bins as run 60559 (aggregates_run60559_energy.py):
electron TOF (D) x ion TOF (C) moments in bins of the VLS-measured photon
energy x GMD.

Undulator scan from 265 eV in 0.5 eV nominal steps of ~10 min (42 raw
files, ended 05:47 on 25 Nov). The combined file is written without VLS
(--live only for its read-only .lst snapshot, --live-margin 0 keeps every
train), and the photon energy is measured from the raw Gotthard frames:

    F=11024642/processed/combined/run60560_Ethanol_undulator_scan_2808016357.h5
    python write_h5.py Ethanol_undulator_scan_2808016357 1 60560 --live --live-margin 0 --no-vls \
        --max-ecounts 120 --max-icounts 50 -o $F
    python analysis/scripts/photon_energy.py --raw-run 60560 --workers 48 $F
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60560_energy.py $F --workers 48
    -> run60560_Ethanol_undulator_scan_2808016357_aggregates_energy.h5

(run on a upex node: ~/slurm_60560_final.sh). The *_live* files are
partial writes made while the run was acquiring. The pixel -> eV
calibration is fitted on this run's own nominal steps.

TOF units are TDC units of 100 ps (10 units = 1 ns); see the run 60559
config for the windows.
"""

import numpy as np

CONFIG = 1
MODE   = "energy"

import calibration   # analysis/scripts, on the path when run through compute_aggregates.py

# Electron bins defined in kinetic energy (calibration.py): 1 eV from 140 to 290 eV, 5 % of the energy from 1 to
# 140 eV. The TOF edges (TDC units, rising = falling energy) are their images snapped to half TDC ticks, so every bin
# holds a whole number of 0.1 ns steps (1-2 near 290 eV, where 1 eV is ~0.13 ns) and the energy width used for the
# Jacobian is that of the snapped bin.
_KE_EDGES = np.r_[np.geomspace(1, 140, int(np.ceil(np.log(140) / np.log(1.05))) + 1)[:-1], np.arange(140, 290.5, 1.0)]
TOF_EDGES = np.unique(np.floor(10 * calibration.tof_of_ke(_KE_EDGES)[::-1]) + 0.5)
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 100)   # 320 bins of 10 ns

# 0.5 eV bins centred on the integer and half-integer energies (the 0.5 eV
# nominal steps sit at bin centres); bins beyond the scan so far stay empty.
ENERGY_KEY   = "photon_energy/measured"
ENERGY_EDGES = np.arange(263.75, 283.25 + 0.01, 0.5)   # 39 bins, 264.0 ... 283.0

GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)

# VLS aggregates as for run 60559; the live file has no VLS, so the frames
# are read from the raw files of the run (line b + 2 -> bunch b).
VLS_CROP_ROI   = (250, 630)
VLS_BACKGROUND = {"type": "auto", "roi": (100, 101)}
VLS_CM_PIXELS  = (650, 800)
VLS_BAD_PIXELS = [0, 1, 291, 292, 679, 680, 681, *range(832, 864)]
VLS_RAW_RUN    = 60560

MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 500            # bigger chunks: fewer per-bin additions of the VLS matrices
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
