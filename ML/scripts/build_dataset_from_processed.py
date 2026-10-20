#!/usr/bin/env python3
"""Build shot-aligned good-ML NPZ caches for the virtual-reference pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import pipeline_config as config
from static_xas_ml.data import assert_matching_feature_schema, load_good_ml_h5
from static_xas_ml.features import (
    build_feature_matrix,
    filter_table_rows,
    sanitize_selected_feature_columns,
    select_feature_columns,
)


def _write_one(path: Path, table, selected_columns: np.ndarray, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=build_feature_matrix(table, selected_columns),
        spectra=table.spectra,
        trans=table.trans,
        gmd=table.gmd,
        nominal_energy=table.nominal_energy,
        converted_energy=table.converted_energy,
        converted_energy_group=table.converted_energy_group,
        converted_proxy_bin_index=table.converted_proxy_bin_index,
        train_id=table.train_id,
        run_id=table.run_id,
        section_index=table.section_index,
        source_shot_index=table.source_shot_index,
        vls_bunch_index=table.vls_bunch_index,
        gmd_bunch_index=table.gmd_bunch_index,
        vls_pixels=table.vls_pixels,
        selected_columns=np.asarray(selected_columns, dtype=int),
        counts=np.asarray([counts["input_rows"], counts["kept_rows"], counts["dropped_rows"]]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-good", type=Path, default=config.REFERENCE_GOOD_H5)
    parser.add_argument("--sample-good", type=Path, default=config.SAMPLE_GOOD_H5)
    parser.add_argument("--reference-processed", type=Path, default=config.REFERENCE_PROCESSED_H5)
    parser.add_argument("--sample-processed", type=Path, default=config.SAMPLE_PROCESSED_H5)
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "datasets")
    parser.add_argument("--max-reference-shots", type=int, default=config.MAX_REFERENCE_SHOTS)
    parser.add_argument("--max-sample-shots", type=int, default=config.MAX_SAMPLE_SHOTS)
    args = parser.parse_args()

    reference = load_good_ml_h5(
        args.reference_good,
        args.reference_processed,
        correction_axis=config.CORRECTION_AXIS,
        gmd_source=config.GMD_SOURCE,
        max_shots=args.max_reference_shots,
    )
    sample = load_good_ml_h5(
        args.sample_good,
        args.sample_processed,
        correction_axis=config.CORRECTION_AXIS,
        gmd_source=config.GMD_SOURCE,
        max_shots=args.max_sample_shots,
    )
    assert_matching_feature_schema(reference, sample)
    selected_columns = select_feature_columns(
        reference.feature_names,
        reference.feature_roles,
        selected_feature_names=config.SELECTED_FEATURE_NAMES,
    )
    selected_columns, dropped = sanitize_selected_feature_columns(reference, sample, selected_columns)
    for row in dropped:
        print(f"dropped feature: {row['feature_name']}: {row['reason']}")

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
    for row in dropped_after_filter:
        print(f"dropped after filter: {row['feature_name']}: {row['reason']}")
    _write_one(args.out_dir / "reference_good_ml.npz", reference, selected_columns, ref_counts)
    _write_one(args.out_dir / "sample_good_ml.npz", sample, selected_columns, sample_counts)
    print(f"reference: {ref_counts['kept_rows']} kept / {ref_counts['input_rows']} input")
    print(f"sample   : {sample_counts['kept_rows']} kept / {sample_counts['input_rows']} input")
    print(f"wrote    : {args.out_dir}")


if __name__ == "__main__":
    main()
