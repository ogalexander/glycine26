#!/usr/bin/env python3
"""Build lightweight filtered NPZ caches from enriched processed H5 files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import pipeline_config as config
from static_xas_ml.data import (
    assert_matching_feature_schema,
    build_feature_matrix,
    filter_shots,
    load_processed_ml_h5,
    sanitize_selected_feature_columns,
    select_feature_columns,
)
from static_xas_ml.preprocessing import preprocess_spectra


def _build_one(path: Path, out_path: Path, *, max_shots: int | None, selected_columns):
    table = load_processed_ml_h5(path, max_shots=max_shots)
    filtered, counts = filter_shots(
        table,
        selected_columns=selected_columns,
        vls_peak_threshold=config.VLS_PEAK_THRESHOLD,
        gmd_min_threshold=config.GMD_MIN_THRESHOLD,
        gmd_max_threshold=config.GMD_MAX_THRESHOLD,
        require_qc_ok=config.REQUIRE_QC_OK,
        reject_tss_flags=config.REJECT_TSS_FLAGS,
    )
    shapes, areas, baselines = preprocess_spectra(
        filtered.spectra,
        baseline_quantile=config.BASELINE_QUANTILE,
        clip_negative=config.CLIP_NEGATIVE,
        eps=config.NORMALIZATION_EPS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=build_feature_matrix(filtered, selected_columns),
        spectra=filtered.spectra,
        shapes=shapes,
        areas=areas,
        baselines=baselines,
        gmd=filtered.gmd,
        nominal_energy=filtered.nominal_energy,
        train_id=filtered.train_id,
        run_id=filtered.run_id,
        vls_bunch_index=filtered.vls_bunch_index,
        gmd_bunch_index=filtered.gmd_bunch_index,
        vls_pixels=filtered.vls_pixels,
        selected_columns=np.asarray(selected_columns, dtype=int),
        counts=np.asarray([counts["input_rows"], counts["kept_rows"], counts["dropped_rows"]]),
    )
    print(f"{path.name}: {counts['kept_rows']} kept / {counts['input_rows']} input -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=config.REFERENCE_H5)
    parser.add_argument("--sample", type=Path, default=config.SAMPLE_H5)
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "datasets")
    args = parser.parse_args()

    reference = load_processed_ml_h5(args.reference, max_shots=config.MAX_REFERENCE_SHOTS)
    sample = load_processed_ml_h5(args.sample, max_shots=config.MAX_SAMPLE_SHOTS)
    assert_matching_feature_schema(reference, sample)
    selected_columns = select_feature_columns(
        reference.feature_names,
        reference.feature_roles,
        include_leakage=config.INCLUDE_LEAKAGE_FEATURES,
        selected_feature_names=config.SELECTED_FEATURE_NAMES,
    )
    selected_columns, dropped = sanitize_selected_feature_columns(reference, sample, selected_columns)
    for message in dropped:
        print(f"dropped feature: {message}")

    _build_one(
        args.reference,
        args.out_dir / "reference_filtered.npz",
        max_shots=config.MAX_REFERENCE_SHOTS,
        selected_columns=selected_columns,
    )
    _build_one(
        args.sample,
        args.out_dir / "sample_filtered.npz",
        max_shots=config.MAX_SAMPLE_SHOTS,
        selected_columns=selected_columns,
    )


if __name__ == "__main__":
    main()
