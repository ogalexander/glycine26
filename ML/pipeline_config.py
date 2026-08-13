"""Configuration for the processed-only static-XAS ML pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np


# User settings
RUN_NAME = ""

# Processed H5 inputs. These files must have been produced by the enriched
# compute_static_xas.py output contract with the /ml group present.
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT.parent / "11022188" / "processed" / "xas_static"
REFERENCE_H5 = PROCESSED_DIR / "reference_frame.h5"
SAMPLE_H5 = PROCESSED_DIR / "run58794_static_xas.h5"

# Optional limits for local development. Keep None for full training.
MAX_REFERENCE_SHOTS = None
MAX_SAMPLE_SHOTS = None

# Row filtering. GMD is a quality/binning feature here, not the absorbance
# normalization target. Extra ML-only QC/TSS gates are off by default so the
# absorbance plots match the traditional notebook filter stack.
VLS_PEAK_THRESHOLD = 1000.0
GMD_MIN_THRESHOLD = None
GMD_MAX_THRESHOLD = 10.0
REJECT_TSS_FLAGS = False
REQUIRE_QC_OK = False

# Match the traditional static-XAS notebook defaults for actual-energy binning.
CENTER_METHOD = "com"
CENTER_HALF_WINDOW = 5
CENTER_CHUNK_SIZE = 50_000
PROXY_PIXEL_BIN_WIDTH = 1.0
REJECT_DOUBLE_PEAKS = False
PEAKFIND_BASELINE_Q = 0.15
PEAKFIND_SG_WINDOW = 9
PEAKFIND_SG_POLY = 2
DOUBLE_PEAK_KWARGS = {
    "min_distance_px": 3,
    "max_distance_px": 17,
    "min_prom_sigma": 2.5,
    "min_height_sigma": 2.0,
    "min_rel_peak_height": 0.35,
    "min_second_ratio": 0.05,
    "min_bridge_drop": 0.015,
    "min_split_sigma": 0.75,
}

# Feature selection. leakage_check features include nominal energy and
# set-wavelength settings. They are useful for this multi-energy scan, but the
# pipeline also reports a zeroed-leakage sensitivity metric.
INCLUDE_LEAKAGE_FEATURES = True
RUN_LEAKAGE_ZEROING_CHECK = True

# Training feature override. Set to None to select by role:
#   predictor only, or predictor + leakage_check when INCLUDE_LEAKAGE_FEATURES=True.
# Saved role=candidate columns, such as filters/apertures/mirrors/vacuum/gas
# context, are not selected automatically. Add them explicitly here when
# training across runs where they vary.
# Set to an explicit list to train only on these saved /ml/features columns.
SELECTED_FEATURE_NAMES = None
# Example:
# SELECTED_FEATURE_NAMES = [
#     "gmd_hall_ch0_intensity",
#     "gmd_hall_ch2_x",
#     "gmd_hall_ch3_y",
#     "gmd_hall_ch4_intensity_sigma",
#     "undulator_k_mean",
#     "undulator_k_std",
#     "undulator_k_slope",
#     "attenuator_pressure",
#     "nominal_energy",
#     "fl26_pgas1_bl_8_3_pressure_mbar",
# ]

# Spectrum preprocessing for the reference target and transmitted spectra.
# Defaults follow the traditional notebook for absorbance: use processed VLS
# spectra directly, without per-shot baseline subtraction or area normalization.
BASELINE_QUANTILE = None
PREPROCESS_CLIP_NEGATIVE = False
AREA_NORMALIZE_SPECTRA = False
NORMALIZATION_EPS = 1e-12

# Random split policy for reference data.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

# PCA-score MLP.
SEED = 42
PCA_COMPONENTS = 20
HIDDEN_LAYERS = [64, 64, 32]
EPOCHS = 128
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 16

# Binning for ML absorbance summaries.
GMD_EDGES = np.arange(0.0, 11.0, 1.0)
NOMINAL_ENERGY_DECIMALS = 6
ABSORBANCE_EPS = 1e-9
ABSORBANCE_PIXEL_ROI = None
INTENSITY_CLIP_NEGATIVE = True

# Optional actual-energy conversion. Uses the same center-pixel and bin-width
# convention as the traditional static-XAS notebook.
MAKE_ACTUAL_ENERGY_ABSORBANCE = True
ACTUAL_ENERGY_PIXEL_BIN_WIDTH = PROXY_PIXEL_BIN_WIDTH
CALIBRATION_PIXEL = 524.0
CALIBRATION_ENERGY_EV = 284.2
PIXEL_TO_ENERGY_RANGE_EV = (260.0, 290.0)


_run_name = (
    os.environ.get("PIPELINE_RUN_NAME", "")
    or RUN_NAME
    or datetime.now().strftime("%Y%m%d_%H%M%S")
)

RESULTS_DIR = ROOT / "results" / _run_name
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_PATH = MODELS_DIR / "virtual_i0_pca_mlp.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PCA_PATH = MODELS_DIR / "pca.joblib"
PREPROCESSING_JSON = MODELS_DIR / "preprocessing.json"
FEATURE_MANIFEST_CSV = RESULTS_DIR / "feature_manifest.csv"

HELDOUT_REFERENCE_METRICS_CSV = METRICS_DIR / "heldout_reference_metrics.csv"
LEAKAGE_ABLATION_METRICS_CSV = METRICS_DIR / "leakage_ablation_metrics.csv"
BINNED_ABSORBANCE_METRICS_CSV = METRICS_DIR / "binned_absorbance_metrics.csv"
