"""Load and filter enriched processed static-XAS H5 files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class MLShotTable:
    path: Path
    spectra: np.ndarray
    gmd: np.ndarray
    gmd_tunnel: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    feature_units: list[str]
    feature_roles: list[str]
    qc_ok: np.ndarray
    train_id: np.ndarray
    run_id: np.ndarray
    section_index: np.ndarray
    vls_bunch_index: np.ndarray
    gmd_bunch_index: np.ndarray
    nominal_energy: np.ndarray
    vls_pixels: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.spectra.shape[0])


def _decode_strings(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _flatten_by_n_shots(dataset, n_shots: np.ndarray, *, max_rows: int | None = None):
    parts = []
    total = 0
    for i, n in enumerate(n_shots.astype(int)):
        if n <= 0:
            continue
        take = n if max_rows is None else min(n, max_rows - total)
        if take <= 0:
            break
        parts.append(dataset[i, :take])
        total += take
        if max_rows is not None and total >= max_rows:
            break
    if not parts:
        shape = (0,) + dataset.shape[2:]
        return np.empty(shape, dtype=dataset.dtype)
    return np.concatenate(parts, axis=0)


def load_processed_ml_h5(path: str | Path, *, max_shots: int | None = None) -> MLShotTable:
    """Load one enriched processed H5 file and flatten finite shot slots only."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "ml" not in h5:
            raise KeyError(
                f"{path} has no /ml group. Rerun analysis/scripts/compute_static_xas.py "
                "with the enriched ML output support before using this pipeline."
            )
        ml = h5["ml"]
        n_shots = np.asarray(h5["n_shots"][...], dtype=np.int64)
        feature_names = _decode_strings(ml["feature_names"][...])
        feature_units = _decode_strings(ml["feature_units"][...])
        feature_roles = _decode_strings(ml["feature_roles"][...])

        spectra = _flatten_by_n_shots(h5["vls"], n_shots, max_rows=max_shots).astype(np.float32)
        gmd = _flatten_by_n_shots(h5["gmd"], n_shots, max_rows=max_shots).astype(np.float32)
        if "gmd_tunnel" in h5:
            gmd_tunnel = _flatten_by_n_shots(
                h5["gmd_tunnel"], n_shots, max_rows=max_shots,
            ).astype(np.float32)
        else:
            gmd_tunnel = np.full_like(gmd, np.nan, dtype=np.float32)
        features = _flatten_by_n_shots(
            ml["features"], n_shots, max_rows=max_shots,
        ).astype(np.float32)
        qc_ok = _flatten_by_n_shots(ml["qc_ok"], n_shots, max_rows=max_shots).astype(bool)
        train_id = _flatten_by_n_shots(ml["train_id"], n_shots, max_rows=max_shots).astype(np.int64)
        run_id = _flatten_by_n_shots(ml["run_id"], n_shots, max_rows=max_shots).astype(np.int64)
        section_index = _flatten_by_n_shots(
            ml["section_index"], n_shots, max_rows=max_shots,
        ).astype(np.int32)
        vls_bunch_index = _flatten_by_n_shots(
            ml["vls_bunch_index"], n_shots, max_rows=max_shots,
        ).astype(np.int16)
        gmd_bunch_index = _flatten_by_n_shots(
            ml["gmd_bunch_index"], n_shots, max_rows=max_shots,
        ).astype(np.int16)
        nominal_energy = _flatten_by_n_shots(
            ml["nominal_energy"], n_shots, max_rows=max_shots,
        ).astype(np.float64)
        vls_pixels = np.asarray(h5["vls_pixels"][...], dtype=np.float64)

    return MLShotTable(
        path=path,
        spectra=spectra,
        gmd=gmd,
        gmd_tunnel=gmd_tunnel,
        features=features,
        feature_names=feature_names,
        feature_units=feature_units,
        feature_roles=feature_roles,
        qc_ok=qc_ok,
        train_id=train_id,
        run_id=run_id,
        section_index=section_index,
        vls_bunch_index=vls_bunch_index,
        gmd_bunch_index=gmd_bunch_index,
        nominal_energy=nominal_energy,
        vls_pixels=vls_pixels,
    )


def subset_table(table: MLShotTable, mask: np.ndarray) -> MLShotTable:
    mask = np.asarray(mask, dtype=bool)
    return replace(
        table,
        spectra=table.spectra[mask],
        gmd=table.gmd[mask],
        gmd_tunnel=table.gmd_tunnel[mask],
        features=table.features[mask],
        qc_ok=table.qc_ok[mask],
        train_id=table.train_id[mask],
        run_id=table.run_id[mask],
        section_index=table.section_index[mask],
        vls_bunch_index=table.vls_bunch_index[mask],
        gmd_bunch_index=table.gmd_bunch_index[mask],
        nominal_energy=table.nominal_energy[mask],
    )


def select_feature_columns(
    feature_names: list[str],
    feature_roles: list[str],
    *,
    include_leakage: bool,
    selected_feature_names: list[str] | None = None,
) -> np.ndarray:
    if selected_feature_names is not None:
        by_name = {name: i for i, name in enumerate(feature_names)}
        missing = [name for name in selected_feature_names if name not in by_name]
        if missing:
            available = ", ".join(feature_names)
            raise ValueError(
                "Requested training feature(s) not present in processed H5: "
                + ", ".join(missing)
                + f". Available features: {available}"
            )
        if len(selected_feature_names) == 0:
            raise ValueError("selected_feature_names cannot be an empty list.")
        return np.asarray([by_name[name] for name in selected_feature_names], dtype=int)

    roles = np.asarray(feature_roles)
    keep = roles == "predictor"
    if include_leakage:
        keep |= roles == "leakage_check"
    if not np.any(keep):
        raise ValueError("No predictor feature columns were selected.")
    return np.where(keep)[0]


def feature_manifest_rows(
    feature_names: list[str],
    feature_units: list[str],
    feature_roles: list[str],
    selected_columns: np.ndarray,
) -> list[dict[str, object]]:
    selected = set(int(i) for i in selected_columns)
    rows = []
    for i, (name, unit, role) in enumerate(zip(feature_names, feature_units, feature_roles)):
        rows.append(
            {
                "index": i,
                "feature_name": name,
                "unit": unit,
                "role": role,
                "selected_for_model": i in selected,
            }
        )
    return rows


def build_feature_matrix(table: MLShotTable, selected_columns: np.ndarray) -> np.ndarray:
    return table.features[:, np.asarray(selected_columns, dtype=int)].astype(np.float32)


def sanitize_selected_feature_columns(
    reference: MLShotTable,
    sample: MLShotTable,
    selected_columns: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Drop selected columns that cannot be used robustly in both files.

    Optional channels stay present in /ml/features, but if a channel is absent
    or OPIS is zero/constant it is excluded from the model input.
    """
    selected_columns = np.asarray(selected_columns, dtype=int)
    keep = []
    dropped = []
    for col in selected_columns:
        name = reference.feature_names[int(col)]
        ref = reference.features[:, int(col)]
        sam = sample.features[:, int(col)]
        ref_finite = ref[np.isfinite(ref)]
        sam_finite = sam[np.isfinite(sam)]
        if ref_finite.size == 0 or sam_finite.size == 0:
            dropped.append(f"{name}: no finite values in reference or sample")
            continue
        if np.nanstd(ref_finite) <= 1e-12:
            dropped.append(f"{name}: constant in reference/training source")
            continue
        combined = np.concatenate([ref_finite, sam_finite])
        if np.allclose(combined, 0.0):
            dropped.append(f"{name}: all zero")
            continue
        keep.append(int(col))
    if not keep:
        raise ValueError("All selected feature columns were dropped as unusable.")
    return np.asarray(keep, dtype=int), dropped


def filter_shots(
    table: MLShotTable,
    *,
    selected_columns: np.ndarray,
    vls_peak_threshold: float | None,
    gmd_min_threshold: float | None,
    gmd_max_threshold: float | None,
    require_qc_ok: bool,
    reject_tss_flags: bool,
) -> tuple[MLShotTable, dict[str, int]]:
    """Apply row-level QC with inclusive GMD min/max thresholds."""
    n0 = table.n_rows
    mask = np.ones(n0, dtype=bool)
    finite_spectrum = np.all(np.isfinite(table.spectra), axis=1)
    finite_gmd = np.isfinite(table.gmd)
    finite_features = np.all(np.isfinite(build_feature_matrix(table, selected_columns)), axis=1)
    mask &= finite_spectrum & finite_gmd & finite_features

    if require_qc_ok:
        mask &= table.qc_ok

    if vls_peak_threshold is not None:
        peak = np.full(n0, np.nan, dtype=np.float64)
        ok = finite_spectrum
        if np.any(ok):
            peak[ok] = np.max(table.spectra[ok], axis=1)
        mask &= peak >= float(vls_peak_threshold)

    if gmd_min_threshold is not None:
        mask &= table.gmd >= float(gmd_min_threshold)
    if gmd_max_threshold is not None:
        mask &= table.gmd <= float(gmd_max_threshold)

    if reject_tss_flags:
        names = list(table.feature_names)
        for name in ("gmd_hall_ch7_tss", "gmd_tunnel_ch7_tss"):
            if name in names:
                values = table.features[:, names.index(name)]
                mask &= (~np.isfinite(values)) | (values == 0)

    counts = {
        "input_rows": int(n0),
        "kept_rows": int(mask.sum()),
        "dropped_rows": int(n0 - mask.sum()),
    }
    return subset_table(table, mask), counts


def make_reference_split(
    table: MLShotTable,
    *,
    validation_fraction: float,
    min_runs_for_run_split: int = 2,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split by run_id when possible, otherwise by contiguous train/time order."""
    if table.n_rows == 0:
        raise ValueError("Cannot split an empty table.")
    val_fraction = float(validation_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1.")

    unique_runs = np.unique(table.run_id[table.run_id >= 0])
    if unique_runs.size >= int(min_runs_for_run_split):
        n_val_runs = max(1, int(round(unique_runs.size * val_fraction)))
        val_runs = set(unique_runs[-n_val_runs:].tolist())
        val_mask = np.array([run in val_runs for run in table.run_id], dtype=bool)
        train_mask = ~val_mask
        if np.any(train_mask) and np.any(val_mask):
            return train_mask, val_mask, "run_id"

    order = np.lexsort((table.vls_bunch_index, table.train_id))
    n_val = max(1, int(round(table.n_rows * val_fraction)))
    val_idx = order[-n_val:]
    val_mask = np.zeros(table.n_rows, dtype=bool)
    val_mask[val_idx] = True
    train_mask = ~val_mask
    return train_mask, val_mask, "contiguous_train_block"


def assert_matching_feature_schema(reference: MLShotTable, sample: MLShotTable) -> None:
    if reference.feature_names != sample.feature_names:
        raise ValueError("Reference and sample feature_names do not match.")
    if reference.feature_roles != sample.feature_roles:
        raise ValueError("Reference and sample feature_roles do not match.")
