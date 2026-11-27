"""
Aggregates for run 60586 (Ethanol_undulator_scan_265-272eV_2808708850) -- config 1,
energy mode, same bins as runs 60559 / 60560 (aggregates_run60560_energy.py):
electron TOF (D) x ion TOF (C) moments in bins of the VLS-measured photon
energy x GMD.

Undulator scan from 265 eV in 0.5 eV nominal steps of ~11 min, started
19:48 on 25 Nov, planned to 275 eV and stopped at 272 eV (local DAQ 22:21,
raw run 22:22, 21 raw files). The local-DAQ files were renamed from
..._265-275eV_... to ..._265-272eV_... at the end; the SDU .txt appeared only
then. The combined file is written without VLS and the photon energy is
measured from the raw Gotthard frames (~/slurm_60586.sh live|final):

    F=11024642/processed/combined/run60586_Ethanol_undulator_scan_265-272eV_2808708850.h5
    python write_h5.py Ethanol_undulator_scan_265-272eV_2808708850 1 60586 --live --live-margin 0 --no-vls \
        --max-ecounts 120 --max-icounts 50 -o $F
    python analysis/scripts/photon_energy.py --raw-run 60586 --workers 48 $F
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60586_energy.py $F --workers 48
    -> run60586_Ethanol_undulator_scan_265-272eV_2808708850_aggregates_energy.h5

The run60586_Ethanol_undulator_scan_265-275eV_2808708850_live* files are a
partial write made while the run was acquiring (up to 271.5 eV, TDC-defined
trains, --live with the default margin). The pixel -> eV calibration is
fitted on this run's own nominal steps.

TOF units are TDC units of 100 ps (10 units = 1 ns); see the run 60559
config for the windows.
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

# Same 0.5 eV photon-energy bins as runs 60559 / 60560; bins outside 265-275 eV stay empty.
ENERGY_KEY   = "photon_energy/measured"
ENERGY_EDGES = np.arange(263.75, 283.25 + 0.01, 0.5)   # 39 bins, 264.0 ... 283.0

GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)

# VLS aggregates from the raw files of the run (line b + 2 -> bunch b).
VLS_CROP_ROI   = (250, 630)
VLS_BACKGROUND = {"type": "auto", "roi": (100, 101)}
VLS_CM_PIXELS  = (650, 800)
VLS_BAD_PIXELS = [0, 1, 291, 292, 679, 680, 681, *range(832, 864)]
VLS_RAW_RUN    = 60586

MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 500
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
