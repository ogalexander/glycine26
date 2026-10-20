"""Feature selection, QC filtering, and coverage checks for static-XAS ML."""

from __future__ import annotations

import numpy as np

from .data import GoodMLTable, subset_table


DEFAULT_HALL_CORE_FEATURES = [
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

QC_ROLES = {"qc"}


def select_feature_columns(
    feature_names: list[str],
    feature_roles: list[str] | None = None,
    *,
    selected_feature_names: list[str] | None = None,
    default_feature_names: list[str] | None = None,
) -> np.ndarray:
    """Return feature indices for the requested model inputs."""
    names = list(feature_names)
    by_name = {name: i for i, name in enumerate(names)}
    requested = list(default_feature_names or DEFAULT_HALL_CORE_FEATURES)
    if selected_feature_names is not None:
        requested = list(selected_feature_names)
        if not requested:
            raise ValueError("selected_feature_names cannot be empty.")

    missing = [name for name in requested if name not in by_name]
    if missing:
        available = ", ".join(names)
        raise ValueError(
            "Requested training feature(s) not present: "
            + ", ".join(missing)
            + f". Available features: {available}"
        )

    columns = np.asarray([by_name[name] for name in requested], dtype=int)
    if feature_roles is not None:
        roles = np.asarray(feature_roles)
        qc_selected = [names[i] for i in columns if roles[i] in QC_ROLES]
        if qc_selected:
            raise ValueError(
                "QC/status features cannot be selected as predictors: "
                + ", ".join(qc_selected)
            )
    return columns


def build_feature_matrix(table: GoodMLTable, selected_columns: np.ndarray) -> np.ndarray:
    return table.features[:, np.asarray(selected_columns, dtype=int)].astype(np.float32)


def sanitize_selected_feature_columns(
    reference: GoodMLTable,
    sample: GoodMLTable,
    selected_columns: np.ndarray,
    *,
    min_reference_std: float = 1e-12,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Drop selected columns that are unusable in either run."""
    selected_columns = np.asarray(selected_columns, dtype=int)
    keep: list[int] = []
    dropped: list[dict[str, str]] = []
    for col in selected_columns:
        name = reference.feature_names[int(col)]
        ref = np.asarray(reference.features[:, int(col)], dtype=np.float64)
        sam = np.asarray(sample.features[:, int(col)], dtype=np.float64)
        ref_finite = ref[np.isfinite(ref)]
        sam_finite = sam[np.isfinite(sam)]
        reason = ""
        if ref_finite.size == 0:
            reason = "no finite values in reference run"
        elif sam_finite.size == 0:
            reason = "no finite values in sample run"
        elif np.nanstd(ref_finite) <= float(min_reference_std):
            reason = "constant in reference training source"
        elif np.allclose(np.concatenate([ref_finite, sam_finite]), 0.0):
            reason = "all finite values are zero"

        if reason:
            dropped.append({"feature_name": name, "reason": reason})
        else:
            keep.append(int(col))

    if not keep:
        raise ValueError("All selected feature columns were dropped as unusable.")
    return np.asarray(keep, dtype=int), dropped


def feature_manifest_rows(
    feature_names: list[str],
    feature_units: list[str],
    feature_roles: list[str],
    selected_columns: np.ndarray,
    dropped_features: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    selected = set(int(i) for i in selected_columns)
    dropped_by_name = {
        str(row["feature_name"]): str(row["reason"]) for row in (dropped_features or [])
    }
    rows: list[dict[str, object]] = []
    for i, (name, unit, role) in enumerate(zip(feature_names, feature_units, feature_roles)):
        rows.append(
            {
                "index": i,
                "feature_name": name,
                "unit": unit,
                "role": role,
                "selected_for_model": i in selected,
                "dropped_reason": dropped_by_name.get(name, ""),
            }
        )
    return rows


def filter_table_rows(
    table: GoodMLTable,
    *,
    selected_columns: np.ndarray,
    gmd_min_threshold: float | None = None,
    gmd_max_threshold: float | None = None,
    require_qc_ok: bool = False,
) -> tuple[GoodMLTable, dict[str, int]]:
    """Apply final ML row QC on already accepted good shots."""
    n0 = table.n_rows
    mask = np.ones(n0, dtype=bool)
    finite_spectrum = np.all(np.isfinite(table.spectra), axis=1)
    finite_trans = np.isfinite(table.trans)
    finite_gmd = np.isfinite(table.gmd)
    finite_energy = np.isfinite(table.nominal_energy)
    finite_features = np.all(np.isfinite(build_feature_matrix(table, selected_columns)), axis=1)
    finite_all = finite_spectrum & finite_trans & finite_gmd & finite_energy & finite_features
    mask &= finite_all

    if require_qc_ok:
        mask &= table.qc_ok
    if gmd_min_threshold is not None:
        mask &= table.gmd >= float(gmd_min_threshold)
    if gmd_max_threshold is not None:
        mask &= table.gmd <= float(gmd_max_threshold)

    counts = {
        "input_rows": int(n0),
        "finite_spectrum_rows": int(np.sum(finite_spectrum)),
        "finite_trans_rows": int(np.sum(finite_trans)),
        "finite_gmd_rows": int(np.sum(finite_gmd)),
        "finite_energy_rows": int(np.sum(finite_energy)),
        "finite_selected_feature_rows": int(np.sum(finite_features)),
        "kept_rows": int(np.sum(mask)),
        "dropped_rows": int(n0 - np.sum(mask)),
    }
    return subset_table(table, mask), counts


def feature_coverage_rows(
    reference: GoodMLTable,
    sample: GoodMLTable,
    selected_columns: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    inner_percentiles: tuple[float, float] = (1.0, 99.0),
) -> list[dict[str, object]]:
    """Compare sample feature support against the reference training support."""
    if reference_mask is None:
        ref_rows = np.ones(reference.n_rows, dtype=bool)
    else:
        ref_rows = np.asarray(reference_mask, dtype=bool)
        if ref_rows.shape != (reference.n_rows,):
            raise ValueError("reference_mask must have one value per reference row.")

    lo_pct, hi_pct = inner_percentiles
    rows: list[dict[str, object]] = []
    for col in np.asarray(selected_columns, dtype=int):
        name = reference.feature_names[int(col)]
        ref = np.asarray(reference.features[ref_rows, int(col)], dtype=np.float64)
        sam = np.asarray(sample.features[:, int(col)], dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        sam = sam[np.isfinite(sam)]
        if ref.size == 0 or sam.size == 0:
            rows.append(
                {
                    "feature_name": name,
                    "reference_n": int(ref.size),
                    "sample_n": int(sam.size),
                    "sample_fraction_outside_reference_minmax": np.nan,
                    "sample_fraction_outside_reference_p01_p99": np.nan,
                }
            )
            continue
        ref_min = float(np.nanmin(ref))
        ref_max = float(np.nanmax(ref))
        ref_p_lo = float(np.nanpercentile(ref, lo_pct))
        ref_p_hi = float(np.nanpercentile(ref, hi_pct))
        rows.append(
            {
                "feature_name": name,
                "reference_n": int(ref.size),
                "sample_n": int(sam.size),
                "reference_min": ref_min,
                "reference_p01": ref_p_lo,
                "reference_median": float(np.nanmedian(ref)),
                "reference_p99": ref_p_hi,
                "reference_max": ref_max,
                "sample_min": float(np.nanmin(sam)),
                "sample_median": float(np.nanmedian(sam)),
                "sample_max": float(np.nanmax(sam)),
                "sample_fraction_outside_reference_minmax": float(
                    np.mean((sam < ref_min) | (sam > ref_max))
                ),
                "sample_fraction_outside_reference_p01_p99": float(
                    np.mean((sam < ref_p_lo) | (sam > ref_p_hi))
                ),
            }
        )
    return rows
