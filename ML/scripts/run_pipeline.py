#!/usr/bin/env python3
"""Run the processed-only glycine static-XAS ML pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", help="Results folder name; default is timestamp")
    parser.add_argument("--reference", type=Path, default=None, help="Reference processed H5")
    parser.add_argument("--sample", type=Path, default=None, help="Sample processed H5")
    return parser.parse_args()


args = _parse_args()
if args.name:
    os.environ["PIPELINE_RUN_NAME"] = args.name

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

import pipeline_config as config
from static_xas_ml.absorbance import (
    bin_absorbance_by_actual_pixel,
    bin_absorbance_by_nominal,
    shape_absorbance,
)
from static_xas_ml.data import (
    assert_matching_feature_schema,
    build_feature_matrix,
    feature_manifest_rows,
    filter_shots,
    load_processed_ml_h5,
    make_reference_split,
    sanitize_selected_feature_columns,
    select_feature_columns,
    subset_table,
)
from static_xas_ml.metrics import per_shot_errors, shape_error_table
from static_xas_ml.model import build_mlp, fit_pca, predict_shapes, set_global_seed, train_mlp
from static_xas_ml.plots import (
    plot_binned_absorbance,
    plot_error_histograms,
    plot_pca_explained,
    plot_predicted_examples,
)
from static_xas_ml.preprocessing import preprocess_spectra, renormalize_shapes


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _valid_shape_mask(shapes: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(shapes), axis=1)


def _load_pixel_to_energy():
    module_path = ROOT.parent / "analysis" / "scripts" / "gotthard_energy_pixel_map.py"
    spec = importlib.util.spec_from_file_location("gotthard_energy_pixel_map", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.SpectrometerConfig()
    cal = mod.CalibrationPoint(
        pixel=float(config.CALIBRATION_PIXEL),
        energy_eV=float(config.CALIBRATION_ENERGY_EV),
    )
    e_min, e_max = config.PIXEL_TO_ENERGY_RANGE_EV

    def convert(pixels: np.ndarray) -> np.ndarray:
        return mod.pixel_to_energy(
            pixels,
            cfg,
            cal,
            e_min=float(e_min),
            e_max=float(e_max),
        )

    return convert


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(config.SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    set_global_seed(config.SEED)

    for path in (config.RESULTS_DIR, config.MODELS_DIR, config.METRICS_DIR, config.PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

    reference_path = args.reference or config.REFERENCE_H5
    sample_path = args.sample or config.SAMPLE_H5
    print(f"Pipeline run : {config.RESULTS_DIR.name}")
    print(f"Reference    : {reference_path}")
    print(f"Sample       : {sample_path}")
    print(f"Results dir  : {config.RESULTS_DIR}")

    reference = load_processed_ml_h5(reference_path, max_shots=config.MAX_REFERENCE_SHOTS)
    sample = load_processed_ml_h5(sample_path, max_shots=config.MAX_SAMPLE_SHOTS)
    assert_matching_feature_schema(reference, sample)

    selected_columns = select_feature_columns(
        reference.feature_names,
        reference.feature_roles,
        include_leakage=config.INCLUDE_LEAKAGE_FEATURES,
        selected_feature_names=config.SELECTED_FEATURE_NAMES,
    )
    selected_columns, dropped_features = sanitize_selected_feature_columns(
        reference, sample, selected_columns,
    )

    ref_filtered, ref_counts = filter_shots(
        reference,
        selected_columns=selected_columns,
        vls_peak_threshold=config.VLS_PEAK_THRESHOLD,
        gmd_min_threshold=config.GMD_MIN_THRESHOLD,
        gmd_max_threshold=config.GMD_MAX_THRESHOLD,
        require_qc_ok=config.REQUIRE_QC_OK,
        reject_tss_flags=config.REJECT_TSS_FLAGS,
    )
    sample_filtered, sample_counts = filter_shots(
        sample,
        selected_columns=selected_columns,
        vls_peak_threshold=config.VLS_PEAK_THRESHOLD,
        gmd_min_threshold=config.GMD_MIN_THRESHOLD,
        gmd_max_threshold=config.GMD_MAX_THRESHOLD,
        require_qc_ok=config.REQUIRE_QC_OK,
        reject_tss_flags=config.REJECT_TSS_FLAGS,
    )
    pd.DataFrame(
        [
            {"run_type": "reference", **ref_counts},
            {"run_type": "sample", **sample_counts},
        ]
    ).to_csv(config.METRICS_DIR / "filter_survival_counts.csv", index=False)
    print(f"Reference kept: {ref_counts['kept_rows']} / {ref_counts['input_rows']}")
    print(f"Sample kept   : {sample_counts['kept_rows']} / {sample_counts['input_rows']}")

    ref_shapes, ref_areas, _ = preprocess_spectra(
        ref_filtered.spectra,
        baseline_quantile=config.BASELINE_QUANTILE,
        clip_negative=config.CLIP_NEGATIVE,
        eps=config.NORMALIZATION_EPS,
    )
    ref_shape_ok = _valid_shape_mask(ref_shapes)
    ref_filtered = subset_table(ref_filtered, ref_shape_ok)
    ref_shapes = ref_shapes[ref_shape_ok]
    ref_areas = ref_areas[ref_shape_ok]

    sample_shapes, sample_areas, _ = preprocess_spectra(
        sample_filtered.spectra,
        baseline_quantile=config.BASELINE_QUANTILE,
        clip_negative=config.CLIP_NEGATIVE,
        eps=config.NORMALIZATION_EPS,
    )
    sample_shape_ok = _valid_shape_mask(sample_shapes)
    sample_filtered = subset_table(sample_filtered, sample_shape_ok)
    sample_shapes = sample_shapes[sample_shape_ok]
    sample_areas = sample_areas[sample_shape_ok]

    manifest = pd.DataFrame(
        feature_manifest_rows(
            reference.feature_names,
            reference.feature_units,
            reference.feature_roles,
            selected_columns,
        )
    )
    if dropped_features:
        drop_df = pd.DataFrame({"dropped_feature_reason": dropped_features})
        drop_df.to_csv(config.RESULTS_DIR / "dropped_features.csv", index=False)
    manifest.to_csv(config.FEATURE_MANIFEST_CSV, index=False)

    x_ref = build_feature_matrix(ref_filtered, selected_columns)
    x_sample = build_feature_matrix(sample_filtered, selected_columns)
    train_mask, val_mask, split_policy = make_reference_split(
        ref_filtered,
        validation_fraction=config.VALIDATION_FRACTION,
        min_runs_for_run_split=config.SPLIT_MIN_RUNS_FOR_RUN_SPLIT,
    )
    x_train, x_val = x_ref[train_mask], x_ref[val_mask]
    y_train, y_val = ref_shapes[train_mask], ref_shapes[val_mask]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    pca, y_train_scores = fit_pca(
        y_train,
        variance=config.PCA_VARIANCE,
        max_components=config.PCA_MAX_COMPONENTS,
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

    y_pred = predict_shapes(
        mlp, scaler, pca, x_val, eps=config.NORMALIZATION_EPS,
    )
    mean_baseline = np.tile(np.nanmean(y_train, axis=0, keepdims=True), (y_val.shape[0], 1))
    mean_baseline = renormalize_shapes(mean_baseline, eps=config.NORMALIZATION_EPS)
    pca_recon = renormalize_shapes(
        pca.inverse_transform(y_val_scores),
        eps=config.NORMALIZATION_EPS,
    )
    metrics_rows = [
        shape_error_table(y_val, y_pred, pixel_axis=ref_filtered.vls_pixels, label="pca_mlp"),
        shape_error_table(
            y_val, mean_baseline, pixel_axis=ref_filtered.vls_pixels, label="mean_shape_baseline",
        ),
        shape_error_table(
            y_val, pca_recon, pixel_axis=ref_filtered.vls_pixels, label="pca_reconstruction_floor",
        ),
    ]
    pd.DataFrame(metrics_rows).to_csv(config.HELDOUT_REFERENCE_METRICS_CSV, index=False)

    leakage_rows = []
    selected_roles = np.asarray(reference.feature_roles)[selected_columns]
    leakage_local = np.where(selected_roles == "leakage_check")[0]
    if config.RUN_LEAKAGE_ZEROING_CHECK and leakage_local.size:
        x_val_zeroed = np.array(x_val, copy=True)
        train_mean = np.nanmean(x_train, axis=0)
        x_val_zeroed[:, leakage_local] = train_mean[leakage_local]
        y_zeroed = predict_shapes(mlp, scaler, pca, x_val_zeroed, eps=config.NORMALIZATION_EPS)
        leakage_rows.append(
            shape_error_table(
                y_val,
                y_zeroed,
                pixel_axis=ref_filtered.vls_pixels,
                label="selected_leakage_features_zeroed",
            )
        )
    else:
        leakage_rows.append({"label": "not_run", "n_shots": int(y_val.shape[0])})
    pd.DataFrame(leakage_rows).to_csv(config.LEAKAGE_ABLATION_METRICS_CSV, index=False)

    mlp.save(config.MODEL_PATH)
    dump(scaler, config.SCALER_PATH)
    dump(pca, config.PCA_PATH)
    _write_json(
        config.PREPROCESSING_JSON,
        {
            "baseline_quantile": config.BASELINE_QUANTILE,
            "clip_negative": config.CLIP_NEGATIVE,
            "normalization_eps": config.NORMALIZATION_EPS,
            "split_policy": split_policy,
            "pca_components": int(pca.n_components_),
            "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            "selected_feature_names": [reference.feature_names[i] for i in selected_columns],
            "dropped_features": dropped_features,
            "reference_h5": str(reference_path),
            "sample_h5": str(sample_path),
        },
    )

    pred_i0_sample = predict_shapes(
        mlp, scaler, pca, x_sample, eps=config.NORMALIZATION_EPS,
    )
    absorbance_pixel = shape_absorbance(
        sample_shapes,
        pred_i0_sample,
        eps=config.ABSORBANCE_EPS,
    )
    nominal_df = bin_absorbance_by_nominal(
        absorbance_pixel,
        sample_filtered.nominal_energy,
        sample_filtered.gmd,
        pixel_axis=sample_filtered.vls_pixels,
        gmd_edges=config.GMD_EDGES,
        pixel_roi=config.ABSORBANCE_PIXEL_ROI,
        energy_decimals=config.NOMINAL_ENERGY_DECIMALS,
    )
    absorbance_tables = [nominal_df]

    if config.MAKE_ACTUAL_ENERGY_ABSORBANCE:
        actual_df = bin_absorbance_by_actual_pixel(
            absorbance_pixel,
            sample_shapes,
            sample_filtered.gmd,
            pixel_axis=sample_filtered.vls_pixels,
            gmd_edges=config.GMD_EDGES,
            pixel_bin_width=config.ACTUAL_ENERGY_PIXEL_BIN_WIDTH,
            pixel_to_energy=_load_pixel_to_energy(),
            pixel_roi=config.ABSORBANCE_PIXEL_ROI,
        )
        absorbance_tables.append(actual_df)

    binned_df = pd.concat(absorbance_tables, ignore_index=True)
    binned_df.to_csv(config.BINNED_ABSORBANCE_METRICS_CSV, index=False)
    np.savez_compressed(
        config.RESULTS_DIR / "sample_ml_absorbance_arrays.npz",
        absorbance_pixel=absorbance_pixel,
        transmitted_shape=sample_shapes,
        predicted_i0_shape=pred_i0_sample,
        nominal_energy=sample_filtered.nominal_energy,
        gmd=sample_filtered.gmd,
        train_id=sample_filtered.train_id,
        vls_bunch_index=sample_filtered.vls_bunch_index,
        vls_pixels=sample_filtered.vls_pixels,
        sample_areas=sample_areas,
        reference_train_areas=ref_areas[train_mask],
    )

    errors = per_shot_errors(y_val, y_pred, pixel_axis=ref_filtered.vls_pixels)
    plot_pca_explained(pca, config.PLOTS_DIR / "pca_explained_variance.png")
    plot_predicted_examples(
        ref_filtered.vls_pixels,
        y_val,
        y_pred,
        config.PLOTS_DIR / "heldout_reference_predicted_vs_true.png",
    )
    plot_error_histograms(errors, config.PLOTS_DIR / "heldout_reference_error_histograms.png")
    plot_binned_absorbance(
        nominal_df,
        config.PLOTS_DIR / "nominal_energy_binned_ml_absorbance.png",
        title="Nominal-energy binned ML absorbance",
    )
    if config.MAKE_ACTUAL_ENERGY_ABSORBANCE and len(absorbance_tables) > 1:
        plot_binned_absorbance(
            absorbance_tables[1],
            config.PLOTS_DIR / "actual_energy_binned_ml_absorbance.png",
            title="Actual-energy binned ML absorbance",
        )

    print("\nPipeline complete.")
    print(f"Model              : {config.MODEL_PATH}")
    print(f"Held-out metrics   : {config.HELDOUT_REFERENCE_METRICS_CSV}")
    print(f"Absorbance metrics : {config.BINNED_ABSORBANCE_METRICS_CSV}")


if __name__ == "__main__":
    main()
