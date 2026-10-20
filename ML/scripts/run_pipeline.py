#!/usr/bin/env python3
"""Run hall-GMD full-spectrum virtual-reference ML for static XAS."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", help="Results folder name")
    parser.add_argument("--reference-good", type=Path, default=None, help="Reference good H5")
    parser.add_argument("--sample-good", type=Path, default=None, help="Sample good H5")
    parser.add_argument(
        "--reference-processed",
        type=Path,
        default=None,
        help="Reference processed H5 with /ml/features",
    )
    parser.add_argument(
        "--sample-processed",
        type=Path,
        default=None,
        help="Sample processed H5 with /ml/features",
    )
    parser.add_argument("--max-reference-shots", type=int, default=None)
    parser.add_argument("--max-sample-shots", type=int, default=None)
    return parser.parse_args()


args = _parse_args()
if args.name:
    os.environ["PIPELINE_RUN_NAME"] = args.name

import pipeline_config as config
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

from static_xas_ml.absorbance import (
    baseline_slope_rows,
    bin_integrated_absorbance_by_energy,
    bin_integrated_absorbance_by_nominal,
)
from static_xas_ml.data import (
    assert_matching_feature_schema,
    load_good_ml_h5,
    write_prediction_h5,
)
from static_xas_ml.features import (
    build_feature_matrix,
    feature_coverage_rows,
    feature_manifest_rows,
    filter_table_rows,
    sanitize_selected_feature_columns,
    select_feature_columns,
)
from static_xas_ml.metrics import per_shot_errors, shape_error_table
from static_xas_ml.model import build_mlp, fit_pca, predict_spectra, set_global_seed, train_mlp
from static_xas_ml.plots import (
    plot_binned_absorbance,
    plot_error_histograms,
    plot_feature_coverage,
    plot_pca_explained,
    plot_predicted_examples,
)
from static_xas_ml.splits import make_reference_energy_block_split


def _mkdirs() -> None:
    for path in (config.RESULTS_DIR, config.MODELS_DIR, config.METRICS_DIR, config.PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def _postprocess_prediction(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    if config.MODEL_CLIP_NEGATIVE:
        spectra = np.clip(spectra, 0.0, None)
    return spectra.astype(np.float32)


def _mean_spectrum_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    mean_spectrum = np.nanmean(y_train, axis=0, keepdims=True).astype(np.float32)
    return _postprocess_prediction(np.repeat(mean_spectrum, int(n_rows), axis=0))


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(config.SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    set_global_seed(config.SEED)
    _mkdirs()

    reference_good = args.reference_good or config.REFERENCE_GOOD_H5
    sample_good = args.sample_good or config.SAMPLE_GOOD_H5
    reference_processed = args.reference_processed or config.REFERENCE_PROCESSED_H5
    sample_processed = args.sample_processed or config.SAMPLE_PROCESSED_H5
    max_reference = args.max_reference_shots or config.MAX_REFERENCE_SHOTS
    max_sample = args.max_sample_shots or config.MAX_SAMPLE_SHOTS

    _require_inputs([reference_good, sample_good, reference_processed, sample_processed])
    print(f"Pipeline run       : {config.RESULTS_DIR.name}")
    print(f"GMD source         : {config.GMD_SOURCE}")
    print(f"Correction axis    : {config.CORRECTION_AXIS}")
    print(f"Reference good     : {reference_good}")
    print(f"Reference processed: {reference_processed}")
    print(f"Sample good        : {sample_good}")
    print(f"Sample processed   : {sample_processed}")
    print(f"Results dir        : {config.RESULTS_DIR}")

    reference = load_good_ml_h5(
        reference_good,
        reference_processed,
        correction_axis=config.CORRECTION_AXIS,
        gmd_source=config.GMD_SOURCE,
        max_shots=max_reference,
    )
    sample = load_good_ml_h5(
        sample_good,
        sample_processed,
        correction_axis=config.CORRECTION_AXIS,
        gmd_source=config.GMD_SOURCE,
        max_shots=max_sample,
    )
    assert_matching_feature_schema(reference, sample)

    selected_columns = select_feature_columns(
        reference.feature_names,
        reference.feature_roles,
        selected_feature_names=config.SELECTED_FEATURE_NAMES,
    )
    selected_columns, dropped_features = sanitize_selected_feature_columns(
        reference,
        sample,
        selected_columns,
    )

    reference, ref_counts = filter_table_rows(
        reference,
        selected_columns=selected_columns,
        gmd_min_threshold=config.GMD_MIN_THRESHOLD,
        gmd_max_threshold=config.GMD_MAX_THRESHOLD,
        require_qc_ok=config.REQUIRE_QC_OK,
    )
    sample, sample_counts = filter_table_rows(
        sample,
        selected_columns=selected_columns,
        gmd_min_threshold=config.GMD_MIN_THRESHOLD,
        gmd_max_threshold=config.GMD_MAX_THRESHOLD,
        require_qc_ok=config.REQUIRE_QC_OK,
    )
    selected_columns, dropped_after_filter = sanitize_selected_feature_columns(
        reference,
        sample,
        selected_columns,
    )
    dropped_features.extend(dropped_after_filter)
    pd.DataFrame(
        [
            {"run_type": "reference", **ref_counts},
            {"run_type": "sample", **sample_counts},
        ]
    ).to_csv(config.FILTER_SURVIVAL_CSV, index=False)
    print(f"Reference kept     : {reference.n_rows:,} / {ref_counts['input_rows']:,}")
    print(f"Sample kept        : {sample.n_rows:,} / {sample_counts['input_rows']:,}")

    train_mask, val_mask, test_mask, split_policy = make_reference_energy_block_split(
        reference,
        train_fraction=config.TRAIN_FRACTION,
        validation_fraction=config.VALIDATION_FRACTION,
        test_fraction=config.TEST_FRACTION,
        energy_decimals=config.NOMINAL_ENERGY_DECIMALS,
    )
    x_ref = build_feature_matrix(reference, selected_columns)
    x_sample = build_feature_matrix(sample, selected_columns)
    y_ref = reference.spectra.astype(np.float32)
    y_sample = sample.spectra.astype(np.float32)

    x_train, x_val, x_test = x_ref[train_mask], x_ref[val_mask], x_ref[test_mask]
    y_train, y_val, y_test = y_ref[train_mask], y_ref[val_mask], y_ref[test_mask]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    pca, y_train_scores = fit_pca(
        y_train,
        n_components=config.PCA_COMPONENTS,
        seed=config.SEED,
    )
    y_val_scores = pca.transform(y_val).astype(np.float32)

    mlp = build_mlp(
        input_dim=x_train_scaled.shape[1],
        output_dim=y_train_scores.shape[1],
        hidden_layers=config.HIDDEN_LAYERS,
        learning_rate=config.LEARNING_RATE,
        seed=config.SEED,
    )
    train_mlp(
        mlp,
        x_train_scaled,
        y_train_scores,
        x_val_scaled,
        y_val_scores,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.EARLY_STOPPING_PATIENCE,
    )

    y_val_pred = predict_spectra(
        mlp,
        scaler,
        pca,
        x_val,
        clip_negative=config.MODEL_CLIP_NEGATIVE,
    )
    y_test_pred = predict_spectra(
        mlp,
        scaler,
        pca,
        x_test,
        clip_negative=config.MODEL_CLIP_NEGATIVE,
    )
    val_mean_baseline = _mean_spectrum_baseline(y_train, y_val.shape[0])
    test_mean_baseline = _mean_spectrum_baseline(y_train, y_test.shape[0])
    val_pca_recon = _postprocess_prediction(pca.inverse_transform(y_val_scores))
    y_test_scores = pca.transform(y_test).astype(np.float32)
    test_pca_recon = _postprocess_prediction(pca.inverse_transform(y_test_scores))

    metrics_rows = [
        shape_error_table(
            y_val,
            y_val_pred,
            pixel_axis=reference.vls_pixels,
            label="validation_pca_mlp",
        ),
        shape_error_table(
            y_val,
            val_mean_baseline,
            pixel_axis=reference.vls_pixels,
            label="validation_mean_spectrum_baseline",
        ),
        shape_error_table(
            y_val,
            val_pca_recon,
            pixel_axis=reference.vls_pixels,
            label="validation_pca_reconstruction_floor",
        ),
        shape_error_table(
            y_test,
            y_test_pred,
            pixel_axis=reference.vls_pixels,
            label="test_pca_mlp",
        ),
        shape_error_table(
            y_test,
            test_mean_baseline,
            pixel_axis=reference.vls_pixels,
            label="test_mean_spectrum_baseline",
        ),
        shape_error_table(
            y_test,
            test_pca_recon,
            pixel_axis=reference.vls_pixels,
            label="test_pca_reconstruction_floor",
        ),
    ]
    pd.DataFrame(metrics_rows).to_csv(config.HELDOUT_REFERENCE_METRICS_CSV, index=False)

    test_errors = per_shot_errors(y_test, y_test_pred, pixel_axis=reference.vls_pixels)
    pd.DataFrame(
        {
            **test_errors,
            "train_id": reference.train_id[test_mask],
            "nominal_energy": reference.nominal_energy[test_mask],
            "gmd": reference.gmd[test_mask],
            "section_index": reference.section_index[test_mask],
        }
    ).to_csv(config.HELDOUT_REFERENCE_PER_SHOT_CSV, index=False)

    coverage = pd.DataFrame(
        feature_coverage_rows(
            reference,
            sample,
            selected_columns,
            reference_mask=train_mask,
        )
    )
    coverage.to_csv(config.FEATURE_COVERAGE_CSV, index=False)

    dropped_df = pd.DataFrame(dropped_features, columns=["feature_name", "reason"])
    dropped_df.to_csv(config.DROPPED_FEATURES_CSV, index=False)
    manifest = pd.DataFrame(
        feature_manifest_rows(
            reference.feature_names,
            reference.feature_units,
            reference.feature_roles,
            selected_columns,
            dropped_features=dropped_features,
        )
    )
    manifest.to_csv(config.FEATURE_MANIFEST_CSV, index=False)

    mlp.save(config.MODEL_PATH)
    dump(scaler, config.SCALER_PATH)
    dump(pca, config.PCA_PATH)
    _write_json(
        config.CONFIG_JSON,
        {
            "reference_run": config.REFERENCE_RUN,
            "sample_run": config.SAMPLE_RUN,
            "reference_good_h5": reference_good,
            "sample_good_h5": sample_good,
            "reference_processed_h5": reference_processed,
            "sample_processed_h5": sample_processed,
            "gmd_source": config.GMD_SOURCE,
            "correction_axis": config.CORRECTION_AXIS,
            "selected_feature_names": [reference.feature_names[i] for i in selected_columns],
            "dropped_features": dropped_features,
            "split_policy": split_policy,
            "train_rows": int(np.sum(train_mask)),
            "validation_rows": int(np.sum(val_mask)),
            "test_rows": int(np.sum(test_mask)),
            "pca_requested_components": config.PCA_COMPONENTS,
            "pca_components": int(pca.n_components_),
            "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            "gmd_edges": np.asarray(config.GMD_EDGES, dtype=float),
            "absorbance_order": "predict_per_shot_then_bin_mean_spectra_integrate_then_log10_ratio",
            "model_clip_negative": config.MODEL_CLIP_NEGATIVE,
            "intensity_clip_negative": config.INTENSITY_CLIP_NEGATIVE,
        },
    )

    pred_i0_sample = predict_spectra(
        mlp,
        scaler,
        pca,
        x_sample,
        clip_negative=config.MODEL_CLIP_NEGATIVE,
    )
    write_prediction_h5(
        config.PREDICTIONS_H5,
        sample=sample,
        predicted_i0_spectra=pred_i0_sample,
    )

    nominal_df = bin_integrated_absorbance_by_nominal(
        y_sample,
        pred_i0_sample,
        sample.nominal_energy,
        sample.gmd,
        pixel_axis=sample.vls_pixels,
        gmd_edges=config.GMD_EDGES,
        pixel_roi=config.ABSORBANCE_PIXEL_ROI,
        clip_negative=config.INTENSITY_CLIP_NEGATIVE,
        eps=config.ABSORBANCE_EPS,
        energy_decimals=config.NOMINAL_ENERGY_DECIMALS,
    )
    nominal_df.to_csv(config.BINNED_ABSORBANCE_NOMINAL_CSV, index=False)

    converted_energy = np.array(sample.converted_energy_group, copy=True)
    missing_converted_group = ~np.isfinite(converted_energy)
    converted_energy[missing_converted_group] = sample.converted_energy[missing_converted_group]
    converted_df = bin_integrated_absorbance_by_energy(
        y_sample,
        pred_i0_sample,
        converted_energy,
        sample.gmd,
        pixel_axis=sample.vls_pixels,
        gmd_edges=config.GMD_EDGES,
        binning="converted_energy",
        x_label="Converted photon energy (eV)",
        energy_column="converted_energy_eV",
        pixel_roi=config.ABSORBANCE_PIXEL_ROI,
        clip_negative=config.INTENSITY_CLIP_NEGATIVE,
        eps=config.ABSORBANCE_EPS,
        energy_decimals=config.CONVERTED_ENERGY_DECIMALS,
    )
    converted_df.to_csv(config.BINNED_ABSORBANCE_CONVERTED_CSV, index=False)

    baseline_rows = baseline_slope_rows(
        nominal_df,
        energy_range=config.BASELINE_ENERGY_RANGE,
        exclude_windows=config.BASELINE_EXCLUDE_WINDOWS,
    ) + baseline_slope_rows(
        converted_df,
        energy_range=config.BASELINE_ENERGY_RANGE,
        exclude_windows=config.BASELINE_EXCLUDE_WINDOWS,
    )
    pd.DataFrame(baseline_rows).to_csv(config.BASELINE_SLOPES_CSV, index=False)

    plot_pca_explained(pca, config.PLOTS_DIR / "pca_explained_variance.png")
    plot_predicted_examples(
        reference.vls_pixels,
        y_test,
        y_test_pred,
        config.PLOTS_DIR / "heldout_reference_predicted_vs_true.png",
    )
    plot_error_histograms(test_errors, config.PLOTS_DIR / "heldout_reference_error_histograms.png")
    plot_feature_coverage(coverage, config.PLOTS_DIR / "feature_coverage_reference_vs_sample.png")
    plot_binned_absorbance(
        nominal_df,
        config.PLOTS_DIR / "nominal_energy_binned_ml_absorbance.png",
        title="Hall-GMD binned ML absorbance vs nominal photon energy",
    )
    plot_binned_absorbance(
        converted_df,
        config.PLOTS_DIR / "converted_energy_binned_ml_absorbance.png",
        title="Hall-GMD binned ML absorbance vs converted photon energy",
    )

    print("\nPipeline complete.")
    print(f"Model              : {config.MODEL_PATH}")
    print(f"Predicted spectra  : {config.PREDICTIONS_H5}")
    print(f"Held-out metrics   : {config.HELDOUT_REFERENCE_METRICS_CSV}")
    print(f"Nominal absorbance : {config.BINNED_ABSORBANCE_NOMINAL_CSV}")
    print(f"Converted absorbance: {config.BINNED_ABSORBANCE_CONVERTED_CSV}")


if __name__ == "__main__":
    main()
