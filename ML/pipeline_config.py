"""Configuration for hall-GMD full-spectrum virtual-reference ML."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT.parent / "11022188" / "processed" / "xas_static"

# Hall-chain inputs. The good H5 files define accepted shots and corrected VLS;
# the processed H5 files provide the corresponding /ml/features rows.
REFERENCE_RUN = 58780
SAMPLE_RUN = 58794
REFERENCE_GOOD_H5 = PROCESSED_DIR / f"run{REFERENCE_RUN}_good_static_xas.h5"
SAMPLE_GOOD_H5 = PROCESSED_DIR / f"run{SAMPLE_RUN}_good_static_xas.h5"
REFERENCE_PROCESSED_H5 = PROCESSED_DIR / f"run{REFERENCE_RUN}_static_xas.h5"
SAMPLE_PROCESSED_H5 = PROCESSED_DIR / f"run{SAMPLE_RUN}_static_xas.h5"

GMD_SOURCE = "hall"
CORRECTION_AXIS = "converted"

# Optional limits for local development. Keep None for full training.
MAX_REFERENCE_SHOTS = None
MAX_SAMPLE_SHOTS = None

# Final ML row filtering on top of the existing processed->good selection.
GMD_MIN_THRESHOLD = None
GMD_MAX_THRESHOLD = 10.0
REQUIRE_QC_OK = False

# Default hall-core model inputs. These are machine/source parameters; no
# sample-VLS-derived quantities are used as predictors.
SELECTED_FEATURE_NAMES = [
    "gmd_hall_ch0_intensity",
    "gmd_hall_ch2_x",
    "gmd_hall_ch3_y",
    "gmd_hall_ch4_intensity_sigma",
    "set_wavelength_1",
    "undulator_gap_mean",
    "undulator_gap_std",
    "undulator_gap_slope",
    "attenuator_pressure",
]

# Reference split. This is grouped by train ID, within nominal-energy groups
# when possible, instead of randomized by individual shots.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
NOMINAL_ENERGY_DECIMALS = 6

# PCA-score MLP for full corrected VLS spectra.
SEED = 42
PCA_COMPONENTS = 20
HIDDEN_LAYERS = [64, 64, 32]
EPOCHS = 128
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 16
MODEL_CLIP_NEGATIVE = True

# Absorbance summaries are computed only after per-shot virtual references are
# predicted, binned, and averaged.
GMD_EDGES = np.arange(1.0, 11.0, 1.0)
ABSORBANCE_EPS = 1e-9
ABSORBANCE_PIXEL_ROI = None
INTENSITY_CLIP_NEGATIVE = True
CONVERTED_ENERGY_DECIMALS = 6
BASELINE_ENERGY_RANGE = (271.0, 280.0)
BASELINE_EXCLUDE_WINDOWS = ((272.7, 273.4), (274.6, 275.4))

RUN_NAME = os.environ.get(
    "PIPELINE_RUN_NAME",
    "hall_virtual_reference_58780_to_58794",
)
RESULTS_DIR = ROOT / "results" / RUN_NAME
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_PATH = MODELS_DIR / "virtual_i0_pca_mlp.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PCA_PATH = MODELS_DIR / "pca.joblib"
CONFIG_JSON = RESULTS_DIR / "config.json"
FEATURE_MANIFEST_CSV = RESULTS_DIR / "feature_manifest.csv"
DROPPED_FEATURES_CSV = RESULTS_DIR / "dropped_features.csv"
FEATURE_COVERAGE_CSV = METRICS_DIR / "feature_coverage_reference_vs_sample.csv"
FILTER_SURVIVAL_CSV = METRICS_DIR / "filter_survival_counts.csv"
HELDOUT_REFERENCE_METRICS_CSV = METRICS_DIR / "heldout_reference_metrics.csv"
HELDOUT_REFERENCE_PER_SHOT_CSV = METRICS_DIR / "heldout_reference_per_shot_errors.csv"
BINNED_ABSORBANCE_NOMINAL_CSV = METRICS_DIR / "binned_ml_absorbance_nominal.csv"
BINNED_ABSORBANCE_CONVERTED_CSV = METRICS_DIR / "binned_ml_absorbance_converted.csv"
BASELINE_SLOPES_CSV = METRICS_DIR / "baseline_slope_checks.csv"
PREDICTIONS_H5 = RESULTS_DIR / "sample_virtual_reference_spectra.h5"
