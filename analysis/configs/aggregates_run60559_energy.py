"""
Aggregates for run 60559 (Ethanol_undulator_scan_2807989445) -- config 1,
energy mode: electron TOF (D) x ion TOF (C) moments in bins of the
VLS-measured photon energy x GMD.

Undulator scan 265-282 eV in 1 eV nominal steps. The per-shot photon
energy comes from the VLS (``photon_energy/measured``), written into the
combined file first:

    python write_h5.py Ethanol_undulator_scan_2807989445 1 60559 --max-ecounts 120 --max-icounts 50 \
        -o 11024642/processed/combined/run60559_Ethanol_undulator_scan_2807989445.h5
    python analysis/scripts/photon_energy.py \
        11024642/processed/combined/run60559_Ethanol_undulator_scan_2807989445.h5
    python analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run60559_energy.py \
        11024642/processed/combined/run60559_Ethanol_undulator_scan_2807989445.h5
    -> run60559_Ethanol_undulator_scan_2807989445_aggregates_energy.h5

Measured energy: VLS peak centre per shot, calibrated against the nominal
steps (so on the undulator energy scale; ~0.06 eV per Gotthard pixel,
single-shot spread ~0.3 eV about nominal). Shots whose VLS signal is
below 5 sigma of the dark-line noise have no energy and are not binned
(about a third of the shots, mostly low GMD).

TOF units are TDC units of 100 ps (10 units = 1 ns).
    electrons 500-10000 (50-1000 ns), 5 ns bins: prompt peak ~90 ns,
                        photoline ~230-330 ns; later hits include ion-TOF
                        crosstalk, so the window stops at 1000 ns.
    ions      2000-34000 (200-3400 ns), 10 ns bins: H+ ~460 ns ... the
                        parent m/q 46 ~3250 ns (run 60538 calibration).
                        10 ns keeps CtC at 320 x 320 per bin.
"""

import numpy as np

CONFIG = 1
MODE   = "energy"

TOF_EDGES     = np.arange(500, 10000 + 1, 50)     # 190 bins of 5 ns
ION_TOF_EDGES = np.arange(2000, 34000 + 1, 100)   # 320 bins of 10 ns

# Measured photon energy (eV): 0.5 eV bins centred on the integer and
# half-integer energies (the nominal steps sit at bin centres).
ENERGY_KEY   = "photon_energy/measured"
ENERGY_EDGES = np.arange(263.75, 283.25 + 0.01, 0.5)   # 39 bins, 264.0 ... 283.0

# GMD: 5 bins at the 0/20/40/60/80/100th percentiles of the shots that are
# binned (those with a measured energy inside ENERGY_EDGES), after leaving
# out the GMD noise floor (incl. negative readings) below 0.02 uJ.
GMD_EDGES           = None
GMD_PERCENTILE_BINS = 5
GMD_MIN             = 0.02

BUNCH_RANGE = (0, 100)      # all FEL pulses; slot 100 holds no pulse

# Max 60 e / 27 i hits per shot in this run; same caps as run 60538.
MAX_HITS_E = 60
MAX_HITS_I = 30

CHUNK_SIZE = 100
TRIM_START = 3
TRIM_END   = 3

OUTPUT_SUFFIX = ""
