"""
Aggregates for runs 60611 + 60612 combined: the AcH undulator scan 265 -> 280 eV, parts 1 and 2 -- config 1,
energy mode, same bins as the per-run configs (aggregates_run60611_energy.py, aggregates_run60612_energy.py).

The combined file is the two final per-run files concatenated along the train axis
(analysis/scripts/concat_combined_h5.py). Each part keeps its own VLS pixel -> eV calibration (photon_energy/,
fitted on that run's nominal steps), so the measured energies of both parts are on the undulator (nominal) scale;
a single fit through both would mix their VLS offsets (~1.5 px = 0.09 eV apart at 270.5 eV). Part 1 covers
265-271 eV, part 2 270.5 eV upwards; 270.5 and 271 eV were taken in both. ~/slurm_concat.sh:

    C=11024642/processed/combined
    F=$C/run60611-60612_AcH_undulator_scan_265-280eV.h5
    python analysis/scripts/concat_combined_h5.py $F $C/run60611_AcH_undulator_scan_265-280eV_2809702542.h5 \
        $C/run60612_AcH_undulator_scan_part2_265-280eV_2809782740.h5
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60611_60612_energy.py $F --workers 48
    python analysis/scripts/compute_ready_counts.py ${F%.h5}_aggregates_energy.h5 --workers 16
    -> run60611-60612_AcH_undulator_scan_265-280eV_aggregates_energy.h5 (+ _ready.h5)

The GMD bins are percentiles of the combined shots, so they differ from those of the per-run files.
TOF units are TDC units of 100 ps (10 units = 1 ns); see the run 60559 config for the windows. Ion bins reach
m/q ~50 (AcH+ m/q 44).
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
