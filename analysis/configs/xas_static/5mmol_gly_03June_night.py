"""
Config for ``compute_static_xas.py`` — static XAS, config 2.
5 mM glycine, 3 June 2026 (night run).

Each open-shutter section is assigned a nominal photon energy by section
index. Per-shot (VLS, GMD) arrays are written to disk without any GMD
binning; binning is done in the notebook.

Run with:
    python analysis/scripts/compute_static_xas.py \
        analysis/configs/xas_static/5mmol_gly_03June_night.py [-o OUT.h5]
"""

# Identify the scan.
RUN_NO  = 58794
CONFIG  = 2
MODE    = "xas_static"

# Read at most this many raw H5 files (None = all).
MAX_FILES = None

# Section photon energies are derived from the median undulator
# ``set_wavelength_1`` value in each detected open section:
# E[eV] = 1239.841984 / wavelength[nm].
SECTION_ENERGY_SOURCE = "set_wavelength_1"
SECTION_ENERGY_ROUND_DECIMALS = 3
NOMINAL_ENERGIES = None

# VLS pixel ROI (half-open). Applied before any background subtraction.
CROP_ROI = (450, 650)

# Cyclic shift (np.roll) of the VLS bunch axis applied right after the
# pixel crop. VLS bunch 10 maps to raw GMD bunch 0 for this run.
VLS_BUNCH_ROLL = 0
GMD_BUNCH_START = 0

# Bunch ranges (half-open).
#   SIGNAL_BUNCH_RANGE : bunches overlapping the GMD acquisition window.
#   BG_BUNCH_RANGE     : non-signal bunches used for per-train baseline.
SIGNAL_BUNCH_RANGE = (10, 40)
BG_BUNCH_RANGE     = (50, 100)

# Fast shutter dataset paths.
SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"

# Transition rejection around shutter moves.
TRAIN_RATE_HZ           = 10.0
TRANSITION_TRIM_SECONDS = 3.0

# Protocol prior: "open" means the scan begins with the shutter open,
# so section 0 uses the first closed block as its background.
FIRST_SECTION_STATE = "open"

# ML feature names written to /ml/features. Edit this list before rerunning
# compute_static_xas.py if you want to save a smaller or reordered feature set.
ML_FEATURE_NAMES = [
    "gmd_hall_ch0_intensity",
    "gmd_hall_ch2_x",
    "gmd_hall_ch3_y",
    "gmd_hall_ch4_intensity_sigma",
    "gmd_tunnel_ch0_intensity",
    "gmd_tunnel_ch2_x",
    "gmd_tunnel_ch3_y",
    "gmd_tunnel_ch4_intensity_sigma",
    "gmd_hall_ch7_tss",
    "gmd_tunnel_ch7_tss",
    "set_wavelength_1",
    "undulator_gap_mean",
    "undulator_gap_std",
    "undulator_gap_slope",
    "gap_error",
    "attenuator_pressure",
    "attenuator_gas_type",
    "fl26_pgas1_bl_8_3_pressure_mbar",
    "fl26_pgas2_bl_8_3_pressure_mbar",
    "fl26_bl5_1_pressure",
    "fl26_hg0_pressure_mbar",
    "fl26_js0_pressure_mbar",
    "fl26_js3_1_pressure",
    "fl26_valve_bl1_closed",
    "fl26_valve_bl1_open",
    "qc_finite_vls",
]
