"""Shot-aligned good-H5 loaders for static-XAS virtual-reference ML."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class GoodMLTable:
    """One row per accepted static-XAS good shot."""

    good_h5: Path
    processed_h5: Path
    correction_axis: str
    gmd_source: str
    spectra: np.ndarray
    trans: np.ndarray
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
    source_shot_index: np.ndarray
    vls_bunch_index: np.ndarray
    gmd_bunch_index: np.ndarray
    nominal_energy: np.ndarray
    converted_energy: np.ndarray
    converted_energy_group: np.ndarray
    converted_proxy_bin_index: np.ndarray
    vls_pixels: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_pixels(self) -> int:
        return int(self.spectra.shape[1]) if self.spectra.ndim == 2 else 0


def _decode_strings(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _require_dataset(h5: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    if name not in h5:
        raise KeyError(f"Required dataset {name!r} is missing from {h5.file.filename}.")
    obj = h5[name]
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(f"{name!r} in {h5.file.filename} is not a dataset.")
    return obj


def _axis_dataset(prefix: str, correction_axis: str) -> str:
    axis = correction_axis.lower()
    if axis not in {"converted", "nominal"}:
        raise ValueError("correction_axis must be 'converted' or 'nominal'.")
    return f"{prefix}_{axis}"


def _validate_gmd_source(gmd_source: str) -> str:
    source = gmd_source.lower()
    if source not in {"hall", "tunnel"}:
        raise ValueError("gmd_source must be 'hall' or 'tunnel'.")
    return source


def _read_indexed(dataset: h5py.Dataset, section: int, indices: np.ndarray) -> np.ndarray:
    """Read one section at arbitrary shot indices while preserving input order."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if idx.size == 0:
        return np.empty((0,) + dataset.shape[2:], dtype=dataset.dtype)
    if np.any(idx < 0) or np.any(idx >= dataset.shape[1]):
        raise IndexError(
            f"source_shot_index out of bounds for {dataset.name}: "
            f"valid 0..{dataset.shape[1] - 1}, got {idx.min()}..{idx.max()}."
        )
    order = np.argsort(idx)
    sorted_idx = idx[order]
    if np.any(np.diff(sorted_idx) == 0):
        raise ValueError("source_shot_index contains duplicate entries within a section.")
    values_sorted = dataset[section, sorted_idx, ...]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return values_sorted[inverse]


def _section_values(dataset: h5py.Dataset, section: int, count: int) -> np.ndarray:
    return dataset[section, :count, ...]


def _empty_array(shape_tail: tuple[int, ...], dtype) -> np.ndarray:
    return np.empty((0,) + shape_tail, dtype=dtype)


def _concat(parts: list[np.ndarray], shape_tail: tuple[int, ...], dtype) -> np.ndarray:
    if not parts:
        return _empty_array(shape_tail, dtype)
    return np.concatenate(parts, axis=0)


def load_good_ml_h5(
    good_h5: str | Path,
    processed_h5: str | Path,
    *,
    correction_axis: str = "converted",
    gmd_source: str = "hall",
    max_shots: int | None = None,
    validate_alignment: bool = True,
) -> GoodMLTable:
    """
    Load accepted good shots and join their processed-H5 ML features.

    The good H5 is the authority for the accepted-shot list and corrected VLS
    spectra. ``qc/source_shot_index`` maps each good row back to the source row
    in the processed H5, where the ML feature matrix lives.
    """
    good_path = Path(good_h5)
    processed_path = Path(processed_h5)
    source = _validate_gmd_source(gmd_source)
    spectra_name = _axis_dataset("vls_intercept_corrected", correction_axis)
    trans_name = _axis_dataset("trans_intercept_corrected", correction_axis)

    spectra_parts: list[np.ndarray] = []
    trans_parts: list[np.ndarray] = []
    gmd_parts: list[np.ndarray] = []
    gmd_tunnel_parts: list[np.ndarray] = []
    features_parts: list[np.ndarray] = []
    qc_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    section_parts: list[np.ndarray] = []
    source_index_parts: list[np.ndarray] = []
    vls_bunch_parts: list[np.ndarray] = []
    gmd_bunch_parts: list[np.ndarray] = []
    nominal_parts: list[np.ndarray] = []
    converted_parts: list[np.ndarray] = []
    converted_group_parts: list[np.ndarray] = []
    converted_proxy_parts: list[np.ndarray] = []

    with h5py.File(good_path, "r") as good, h5py.File(processed_path, "r") as processed:
        if "ml" not in processed:
            raise KeyError(
                f"{processed_path} has no /ml group. Rerun raw processing with "
                "analysis/scripts/compute_static_xas.py before ML training."
            )
        if "ml/features" not in processed:
            raise KeyError(f"{processed_path} has no /ml/features dataset.")
        if "qc/source_shot_index" not in good:
            raise KeyError(
                f"{good_path} has no /qc/source_shot_index. Rerun the processed->good step."
            )

        n_good_by_section = np.asarray(_require_dataset(good, "n_shots")[...], dtype=np.int64)
        n_sections = int(n_good_by_section.size)
        feature_ds = _require_dataset(processed, "ml/features")
        feature_names = _decode_strings(_require_dataset(processed, "ml/feature_names")[...])
        feature_units = _decode_strings(_require_dataset(processed, "ml/feature_units")[...])
        feature_roles = _decode_strings(_require_dataset(processed, "ml/feature_roles")[...])
        vls_pixels = np.asarray(_require_dataset(good, "vls_pixels")[...], dtype=np.float64)
        n_pixels = int(vls_pixels.size)
        n_features = int(feature_ds.shape[2])

        spectra_ds = _require_dataset(good, spectra_name)
        trans_ds = _require_dataset(good, trans_name)
        gmd_ds = _require_dataset(good, "gmd" if source == "hall" else "gmd_tunnel")
        gmd_tunnel_ds = good["gmd_tunnel"] if "gmd_tunnel" in good else None
        source_index_ds = _require_dataset(good, "qc/source_shot_index")
        qc_ok_ds = _require_dataset(good, "ml/qc_ok")
        good_train_ds = _require_dataset(good, "ml/train_id")
        good_vls_bunch_ds = _require_dataset(good, "ml/vls_bunch_index")
        good_gmd_bunch_ds = _require_dataset(good, "ml/gmd_bunch_index")
        good_run_ds = _require_dataset(good, "ml/run_id")
        good_section_ds = _require_dataset(good, "ml/section_index")
        good_nominal_ds = _require_dataset(good, "ml/nominal_energy")
        converted_ds = good["ml/converted_energy"] if "ml/converted_energy" in good else None
        converted_group_ds = (
            good["ml/converted_energy_group"] if "ml/converted_energy_group" in good else None
        )
        converted_proxy_ds = (
            good["ml/converted_proxy_bin_index"]
            if "ml/converted_proxy_bin_index" in good
            else None
        )

        processed_train_ds = _require_dataset(processed, "ml/train_id")
        processed_vls_bunch_ds = _require_dataset(processed, "ml/vls_bunch_index")
        processed_gmd_bunch_ds = _require_dataset(processed, "ml/gmd_bunch_index")

        if spectra_ds.shape[:2] != trans_ds.shape[:2]:
            raise ValueError("Corrected spectrum and transmission datasets have incompatible shapes.")
        if spectra_ds.shape[2] != n_pixels:
            raise ValueError("Corrected spectra pixel dimension does not match /vls_pixels.")
        if feature_ds.shape[0] < n_sections:
            raise ValueError("Processed H5 has fewer sections than the good H5.")

        total = 0
        for section, n_available in enumerate(n_good_by_section):
            if max_shots is not None and total >= int(max_shots):
                break
            n_take = int(n_available)
            if n_take <= 0:
                continue
            if max_shots is not None:
                n_take = min(n_take, int(max_shots) - total)
            if n_take <= 0:
                continue

            source_index = np.asarray(
                _section_values(source_index_ds, section, n_take), dtype=np.int64,
            )
            features = _read_indexed(feature_ds, section, source_index).astype(np.float32)

            if validate_alignment:
                good_train = np.asarray(_section_values(good_train_ds, section, n_take))
                proc_train = np.asarray(_read_indexed(processed_train_ds, section, source_index))
                good_vls_bunch = np.asarray(_section_values(good_vls_bunch_ds, section, n_take))
                proc_vls_bunch = np.asarray(
                    _read_indexed(processed_vls_bunch_ds, section, source_index),
                )
                good_gmd_bunch = np.asarray(_section_values(good_gmd_bunch_ds, section, n_take))
                proc_gmd_bunch = np.asarray(
                    _read_indexed(processed_gmd_bunch_ds, section, source_index),
                )
                if not np.array_equal(good_train, proc_train):
                    raise ValueError(
                        f"train_id mismatch between good and processed H5 in section {section}."
                    )
                if not np.array_equal(good_vls_bunch, proc_vls_bunch):
                    raise ValueError(
                        f"vls_bunch_index mismatch between good and processed H5 in section {section}."
                    )
                if not np.array_equal(good_gmd_bunch, proc_gmd_bunch):
                    raise ValueError(
                        f"gmd_bunch_index mismatch between good and processed H5 in section {section}."
                    )

            spectra_parts.append(_section_values(spectra_ds, section, n_take).astype(np.float32))
            trans_parts.append(_section_values(trans_ds, section, n_take).astype(np.float32))
            gmd_parts.append(_section_values(gmd_ds, section, n_take).astype(np.float32))
            if gmd_tunnel_ds is None:
                gmd_tunnel_parts.append(np.full(n_take, np.nan, dtype=np.float32))
            else:
                gmd_tunnel_parts.append(
                    _section_values(gmd_tunnel_ds, section, n_take).astype(np.float32),
                )
            features_parts.append(features)
            qc_parts.append(_section_values(qc_ok_ds, section, n_take).astype(bool))
            train_parts.append(_section_values(good_train_ds, section, n_take).astype(np.int64))
            run_parts.append(_section_values(good_run_ds, section, n_take).astype(np.int64))
            section_parts.append(_section_values(good_section_ds, section, n_take).astype(np.int32))
            source_index_parts.append(source_index.astype(np.int64))
            vls_bunch_parts.append(
                _section_values(good_vls_bunch_ds, section, n_take).astype(np.int16),
            )
            gmd_bunch_parts.append(
                _section_values(good_gmd_bunch_ds, section, n_take).astype(np.int16),
            )
            nominal_parts.append(_section_values(good_nominal_ds, section, n_take).astype(np.float64))
            if converted_ds is None:
                converted_parts.append(np.full(n_take, np.nan, dtype=np.float64))
            else:
                converted_parts.append(
                    _section_values(converted_ds, section, n_take).astype(np.float64),
                )
            if converted_group_ds is None:
                converted_group_parts.append(np.full(n_take, np.nan, dtype=np.float64))
            else:
                converted_group_parts.append(
                    _section_values(converted_group_ds, section, n_take).astype(np.float64),
                )
            if converted_proxy_ds is None:
                converted_proxy_parts.append(np.full(n_take, -1, dtype=np.int32))
            else:
                converted_proxy_parts.append(
                    _section_values(converted_proxy_ds, section, n_take).astype(np.int32),
                )
            total += n_take

    return GoodMLTable(
        good_h5=good_path,
        processed_h5=processed_path,
        correction_axis=correction_axis.lower(),
        gmd_source=source,
        spectra=_concat(spectra_parts, (n_pixels,), np.float32),
        trans=_concat(trans_parts, (), np.float32),
        gmd=_concat(gmd_parts, (), np.float32),
        gmd_tunnel=_concat(gmd_tunnel_parts, (), np.float32),
        features=_concat(features_parts, (n_features,), np.float32),
        feature_names=feature_names,
        feature_units=feature_units,
        feature_roles=feature_roles,
        qc_ok=_concat(qc_parts, (), bool),
        train_id=_concat(train_parts, (), np.int64),
        run_id=_concat(run_parts, (), np.int64),
        section_index=_concat(section_parts, (), np.int32),
        source_shot_index=_concat(source_index_parts, (), np.int64),
        vls_bunch_index=_concat(vls_bunch_parts, (), np.int16),
        gmd_bunch_index=_concat(gmd_bunch_parts, (), np.int16),
        nominal_energy=_concat(nominal_parts, (), np.float64),
        converted_energy=_concat(converted_parts, (), np.float64),
        converted_energy_group=_concat(converted_group_parts, (), np.float64),
        converted_proxy_bin_index=_concat(converted_proxy_parts, (), np.int32),
        vls_pixels=vls_pixels,
    )


def subset_table(table: GoodMLTable, mask: np.ndarray) -> GoodMLTable:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (table.n_rows,):
        raise ValueError("mask must have one value per table row.")
    return replace(
        table,
        spectra=table.spectra[mask],
        trans=table.trans[mask],
        gmd=table.gmd[mask],
        gmd_tunnel=table.gmd_tunnel[mask],
        features=table.features[mask],
        qc_ok=table.qc_ok[mask],
        train_id=table.train_id[mask],
        run_id=table.run_id[mask],
        section_index=table.section_index[mask],
        source_shot_index=table.source_shot_index[mask],
        vls_bunch_index=table.vls_bunch_index[mask],
        gmd_bunch_index=table.gmd_bunch_index[mask],
        nominal_energy=table.nominal_energy[mask],
        converted_energy=table.converted_energy[mask],
        converted_energy_group=table.converted_energy_group[mask],
        converted_proxy_bin_index=table.converted_proxy_bin_index[mask],
    )


def assert_matching_feature_schema(reference: GoodMLTable, sample: GoodMLTable) -> None:
    if reference.feature_names != sample.feature_names:
        raise ValueError("Reference and sample feature_names do not match.")
    if reference.feature_roles != sample.feature_roles:
        raise ValueError("Reference and sample feature_roles do not match.")
    if reference.feature_units != sample.feature_units:
        raise ValueError("Reference and sample feature_units do not match.")
    if not np.array_equal(reference.vls_pixels, sample.vls_pixels):
        raise ValueError("Reference and sample VLS pixel axes do not match.")


def write_prediction_h5(
    path: str | Path,
    *,
    sample: GoodMLTable,
    predicted_i0_spectra: np.ndarray,
) -> None:
    """Write shot-level sample transmitted spectra and predicted virtual references."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predicted = np.asarray(predicted_i0_spectra, dtype=np.float32)
    if predicted.shape != sample.spectra.shape:
        raise ValueError("predicted_i0_spectra must have the same shape as sample.spectra.")
    chunks = (min(max(1, sample.n_rows), 4096), sample.n_pixels)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["mode"] = "static_xas_ml_virtual_reference"
        h5.attrs["gmd_source"] = sample.gmd_source
        h5.attrs["correction_axis"] = sample.correction_axis
        h5.attrs["sample_good_h5"] = str(sample.good_h5)
        h5.attrs["sample_processed_h5"] = str(sample.processed_h5)
        h5.create_dataset("vls_pixels", data=sample.vls_pixels)
        h5.create_dataset(
            "sample_transmitted_spectrum",
            data=sample.spectra.astype(np.float32),
            compression="gzip",
            chunks=chunks,
        )
        h5.create_dataset(
            "predicted_virtual_reference_spectrum",
            data=predicted,
            compression="gzip",
            chunks=chunks,
        )
        meta = h5.create_group("shot_metadata")
        meta.create_dataset("gmd", data=sample.gmd.astype(np.float32), compression="gzip")
        meta.create_dataset("nominal_energy", data=sample.nominal_energy, compression="gzip")
        meta.create_dataset("converted_energy", data=sample.converted_energy, compression="gzip")
        meta.create_dataset(
            "converted_energy_group",
            data=sample.converted_energy_group,
            compression="gzip",
        )
        meta.create_dataset(
            "converted_proxy_bin_index",
            data=sample.converted_proxy_bin_index,
            compression="gzip",
        )
        meta.create_dataset("train_id", data=sample.train_id, compression="gzip")
        meta.create_dataset("run_id", data=sample.run_id, compression="gzip")
        meta.create_dataset("section_index", data=sample.section_index, compression="gzip")
        meta.create_dataset("source_shot_index", data=sample.source_shot_index, compression="gzip")
        meta.create_dataset("vls_bunch_index", data=sample.vls_bunch_index, compression="gzip")
        meta.create_dataset("gmd_bunch_index", data=sample.gmd_bunch_index, compression="gzip")
