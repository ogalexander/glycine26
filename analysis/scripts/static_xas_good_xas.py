"""Good-H5 downstream helpers for static XAS.

This module starts from ``run*_good_static_xas.h5``.  The weak-VLS rejection is
assumed to have happened upstream.  The converted-energy axis follows the old
difference-notebook logic: shots are grouped by VLS center proxy-pixel bins,
then each occupied proxy bin is assigned the mean stored converted energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np


GMD_SOURCE_TO_DATASET = {
    "hall": "gmd",
    "tunnel": "gmd_tunnel",
}


@dataclass(frozen=True)
class GoodXASConfig:
    """Configuration for good-H5 GMD-binned XAS."""

    input_h5: str | Path
    energy_window_ev: Sequence[float] = (271.0, 272.0)
    unit_scale_axis: str = "converted"  # "converted" or "nominal"
    unit_reference_window_ev: Optional[Sequence[float]] = None
    gmd_edges: Sequence[float] = tuple(np.arange(1.0, 11.0, 1.0))
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    nominal_group_decimals: int = 3
    proxy_pixel_bin_width: int = 1
    correction_mode: str = "per_axis_intercept"  # "none" or "per_axis_intercept"
    gmd_source: str = "hall"  # "hall" or "tunnel"


@dataclass(frozen=True)
class GoodShots:
    """Flattened shot table read from a good static-XAS H5."""

    nominal_energy: np.ndarray
    converted_energy: np.ndarray
    center_pixel: np.ndarray
    peak_height: np.ndarray
    peak_snr: np.ndarray
    peak_prominence: np.ndarray
    gmd: np.ndarray
    trans_raw: np.ndarray
    section_index: np.ndarray
    good_shot_index: np.ndarray
    source_shot_index: np.ndarray
    vls_pixels: np.ndarray
    attrs: dict[str, Any]


@dataclass(frozen=True)
class LinearFits:
    """Free-intercept linear fit of integrated VLS versus GMD."""

    intercept: np.ndarray
    slope: np.ndarray
    r: np.ndarray
    r2: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class UnitScale:
    """Single VLS-to-uJ scale from a through-origin fit."""

    axis: str
    window_ev: tuple[float, float]
    vls_per_uJ: float
    n_shots: int
    source: str


@dataclass(frozen=True)
class BinnedXAS:
    """Summed GMD/VLS state binned by photon energy and GMD."""

    x: np.ndarray
    counts: np.ndarray
    gmd_sum: np.ndarray
    trans_sum: np.ndarray
    gmd_sumsq: np.ndarray
    trans_sumsq: np.ndarray
    gmd_trans_sum: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    x_label: str
    metadata: dict[str, Any]

    @property
    def od(self) -> np.ndarray:
        return log10_ratio(self.gmd_sum, self.trans_sum)

    @property
    def od_sem(self) -> np.ndarray:
        return delta_log_ratio_sem(
            self.gmd_sum,
            self.trans_sum,
            self.gmd_sumsq,
            self.trans_sumsq,
            self.gmd_trans_sum,
            self.counts,
        )


@dataclass(frozen=True)
class GoodXASResult:
    """Computed state for good-H5 static XAS."""

    config: GoodXASConfig
    path: Path
    shots: GoodShots
    keep_mask: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    nominal: BinnedXAS
    converted: BinnedXAS
    nominal_fits: LinearFits
    converted_fits: LinearFits
    unit_scale: UnitScale
    nominal_group_index: np.ndarray
    nominal_group_energy: np.ndarray
    center_rel: np.ndarray
    proxy_group_index: np.ndarray
    converted_row_index: np.ndarray
    converted_proxy_group_indices: np.ndarray
    trans_nominal_uJ: np.ndarray
    trans_converted_uJ: np.ndarray


@dataclass(frozen=True)
class BootstrapErrors:
    """Shot-level bootstrap OD uncertainty aligned to a GoodXASResult."""

    nominal_od_sigma: np.ndarray
    converted_od_sigma: np.ndarray
    nominal_counts: np.ndarray
    converted_counts: np.ndarray
    n_bootstrap: int
    seed: int
    min_shots_per_bin: int
    batch_size: int


@dataclass(frozen=True)
class CorrelationWindow:
    """Shot-level GMD/VLS correlation around one energy."""

    axis: str
    window_ev: tuple[float, float]
    energy: np.ndarray
    gmd: np.ndarray
    trans_uJ: np.ndarray
    fit: dict[str, float]


@dataclass(frozen=True)
class FeatureVLSStats:
    """Mean spectrum and shot-level feature distributions for one energy region."""

    axis: str
    center_ev: float
    half_width_ev: float
    window_ev: tuple[float, float]
    vls_dataset: str
    n_shots: int
    vls_pixels: np.ndarray
    mean_vls: np.ndarray
    std_vls: np.ndarray
    energy: np.ndarray
    nominal_energy: np.ndarray
    converted_energy: np.ndarray
    center_pixel: np.ndarray
    peak_height: np.ndarray
    peak_snr: np.ndarray
    peak_prominence: np.ndarray
    integrated_vls: np.ndarray
    gmd: np.ndarray


def compute_good_xas(config: GoodXASConfig) -> GoodXASResult:
    """Compute nominal and converted/proxy GMD-binned XAS from a good H5."""

    _validate_config(config)
    path = Path(config.input_h5)
    shots = read_good_shots(path, gmd_source=config.gmd_source)
    keep = _base_keep_mask(shots, config)
    gmd_edges = _validate_edges(np.asarray(config.gmd_edges, dtype=np.float64))
    gbin, in_gmd = gmd_bin_indices(shots.gmd, gmd_edges)
    keep_gmd = keep & in_gmd
    gmd_labels = format_gmd_labels(gmd_edges, shots.gmd[keep_gmd])

    nominal_values = np.round(shots.nominal_energy, int(config.nominal_group_decimals))
    nominal_energy = np.unique(nominal_values[np.isfinite(nominal_values) & keep_gmd])
    nominal_energy.sort()
    nominal_group = _assign_nominal_groups(nominal_values, nominal_energy)

    center_rel = _center_pixel_to_relative(shots.center_pixel, shots.vls_pixels)
    proxy_group = _proxy_groups(center_rel, shots.vls_pixels.size, int(config.proxy_pixel_bin_width))
    proxy_valid = keep_gmd & (proxy_group >= 0) & np.isfinite(shots.converted_energy)
    n_proxy = int(np.ceil(shots.vls_pixels.size / int(config.proxy_pixel_bin_width)))

    nominal_fits = fit_by_group(
        nominal_energy.size,
        nominal_group,
        shots.gmd,
        shots.trans_raw,
        keep_gmd,
    )
    converted_fits = fit_by_group(
        n_proxy,
        proxy_group,
        shots.gmd,
        shots.trans_raw,
        proxy_valid,
    )

    if config.correction_mode == "per_axis_intercept":
        trans_nominal = apply_group_intercepts(shots.trans_raw, nominal_group, nominal_fits.intercept)
        trans_converted = apply_group_intercepts(shots.trans_raw, proxy_group, converted_fits.intercept)
    else:
        trans_nominal = shots.trans_raw.astype(np.float64, copy=True)
        trans_converted = shots.trans_raw.astype(np.float64, copy=True)

    unit_scale = resolve_unit_scale(config, shots, keep_gmd, trans_nominal, trans_converted)
    trans_nominal_uJ = trans_nominal / unit_scale.vls_per_uJ
    trans_converted_uJ = trans_converted / unit_scale.vls_per_uJ

    proxy_energy_sum = np.zeros(n_proxy, dtype=np.float64)
    proxy_energy_count = np.zeros(n_proxy, dtype=np.float64)
    np.add.at(proxy_energy_sum, proxy_group[proxy_valid], shots.converted_energy[proxy_valid])
    np.add.at(proxy_energy_count, proxy_group[proxy_valid], 1.0)
    proxy_energy = np.full(n_proxy, np.nan, dtype=np.float64)
    np.divide(proxy_energy_sum, proxy_energy_count, out=proxy_energy, where=proxy_energy_count > 0)

    has_proxy = (proxy_energy_count > 0) & np.isfinite(proxy_energy)
    proxy_indices = np.where(has_proxy)[0]
    order = np.argsort(proxy_energy[proxy_indices])
    converted_proxy_indices = proxy_indices[order]
    converted_energy = proxy_energy[converted_proxy_indices]
    proxy_to_row = np.full(n_proxy, -1, dtype=np.int64)
    proxy_to_row[converted_proxy_indices] = np.arange(converted_proxy_indices.size, dtype=np.int64)
    converted_row = np.full(proxy_group.shape, -1, dtype=np.int64)
    valid_proxy_group = (proxy_group >= 0) & (proxy_group < n_proxy)
    converted_row[valid_proxy_group] = proxy_to_row[proxy_group[valid_proxy_group]]

    nominal_grid = new_grid(nominal_energy.size, gmd_edges.size - 1)
    converted_grid = new_grid(converted_energy.size, gmd_edges.size - 1)
    add_to_grid(nominal_grid, nominal_group, gbin, keep_gmd, shots.gmd, trans_nominal_uJ)
    add_to_grid(converted_grid, converted_row, gbin, keep_gmd, shots.gmd, trans_converted_uJ)

    common_metadata = {
        "correction_mode": config.correction_mode,
        "unit_scale_axis": unit_scale.axis,
        "unit_reference_window_ev": unit_scale.window_ev,
        "vls_per_uJ": unit_scale.vls_per_uJ,
        "gmd_source": _normalize_gmd_source(config.gmd_source),
        "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
    }
    nominal = make_binned(
        nominal_energy,
        nominal_grid,
        gmd_edges,
        gmd_labels,
        "nominal photon energy (eV)",
        {
            **common_metadata,
            "mode": "nominal",
            "nominal_group_decimals": int(config.nominal_group_decimals),
        },
    )
    converted = make_binned(
        converted_energy,
        converted_grid,
        gmd_edges,
        gmd_labels,
        "converted photon energy (eV)",
        {
            **common_metadata,
            "mode": "converted_proxy_pixel",
            "proxy_pixel_bin_width": int(config.proxy_pixel_bin_width),
            "proxy_group_indices": converted_proxy_indices,
        },
    )
    return GoodXASResult(
        config=config,
        path=path,
        shots=shots,
        keep_mask=keep_gmd,
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        nominal=nominal,
        converted=converted,
        nominal_fits=nominal_fits,
        converted_fits=converted_fits,
        unit_scale=unit_scale,
        nominal_group_index=nominal_group,
        nominal_group_energy=nominal_energy,
        center_rel=center_rel,
        proxy_group_index=proxy_group,
        converted_row_index=converted_row,
        converted_proxy_group_indices=converted_proxy_indices,
        trans_nominal_uJ=trans_nominal_uJ,
        trans_converted_uJ=trans_converted_uJ,
    )


def compute_bootstrap_errors(
    result: GoodXASResult,
    *,
    n_bootstrap: int = 500,
    seed: int = 123,
    min_shots_per_bin: int = 100,
    batch_size: int = 50,
) -> BootstrapErrors:
    """Compute shot-level bootstrap OD sigma for nominal and converted bins."""

    if int(n_bootstrap) < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if int(min_shots_per_bin) < 2:
        raise ValueError("min_shots_per_bin must be at least 2")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    gbin, in_gmd = gmd_bin_indices(result.shots.gmd, result.gmd_edges)
    keep = result.keep_mask & in_gmd
    nominal_sigma, nominal_counts = bootstrap_sigma_grid(
        result.shots.gmd,
        result.trans_nominal_uJ,
        result.nominal_group_index,
        gbin,
        keep,
        result.nominal.counts.shape,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )
    converted_sigma, converted_counts = bootstrap_sigma_grid(
        result.shots.gmd,
        result.trans_converted_uJ,
        result.converted_row_index,
        gbin,
        keep,
        result.converted.counts.shape,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed) + 1,
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )
    return BootstrapErrors(
        nominal_od_sigma=nominal_sigma,
        converted_od_sigma=converted_sigma,
        nominal_counts=nominal_counts,
        converted_counts=converted_counts,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        min_shots_per_bin=int(min_shots_per_bin),
        batch_size=int(batch_size),
    )


def read_good_shots(path: str | Path, *, gmd_source: str = "hall") -> GoodShots:
    """Read a good static-XAS H5 into flattened shot arrays."""

    gmd_source_norm = _normalize_gmd_source(gmd_source)
    gmd_dataset = _gmd_dataset_for_source(gmd_source_norm)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with h5py.File(p, "r") as h5:
        required = [
            "n_shots",
            "nominal_energies",
            gmd_dataset,
            "energy_vls_calibrated",
            "qc/peak_center_pixel",
            "vls_pixels",
        ]
        missing = [name for name in required if name not in h5]
        if missing:
            message = f"{p} is missing required good-H5 datasets for gmd_source={gmd_source_norm!r}: {missing}"
            if gmd_source_norm == "tunnel" and gmd_dataset in missing:
                message += (
                    ". Dataset /gmd_tunnel is written by process_processed_data_xas_static.py/"
                    "static_xas_process_good.py when the source run*_static_xas.h5 already contains /gmd_tunnel; "
                    "re-run the processed-to-good step, not raw compute_static_xas.py, in that case."
                )
            raise KeyError(message)

        n_shots = h5["n_shots"][...].astype(np.int64)
        nominal_energies = h5["nominal_energies"][...].astype(np.float64)
        parts: dict[str, list[np.ndarray]] = {
            "nominal_energy": [],
            "converted_energy": [],
            "center_pixel": [],
            "peak_height": [],
            "peak_snr": [],
            "peak_prominence": [],
            "gmd": [],
            "trans_raw": [],
            "section_index": [],
            "good_shot_index": [],
            "source_shot_index": [],
        }
        for ie, n in enumerate(n_shots.astype(int)):
            if n <= 0:
                continue
            sl = slice(0, int(n))
            if "ml/nominal_energy" in h5:
                nominal = h5["ml/nominal_energy"][ie, sl].astype(np.float64)
            else:
                nominal = np.full(int(n), nominal_energies[ie], dtype=np.float64)
            if "trans_raw" in h5:
                trans_raw = h5["trans_raw"][ie, sl].astype(np.float64)
            else:
                trans_raw = np.nansum(h5["vls"][ie, sl, :].astype(np.float64), axis=1)
            if "qc/source_shot_index" in h5:
                source_shot_index = h5["qc/source_shot_index"][ie, sl].astype(np.int64)
            else:
                source_shot_index = np.arange(int(n), dtype=np.int64)

            parts["nominal_energy"].append(nominal)
            parts["converted_energy"].append(h5["energy_vls_calibrated"][ie, sl].astype(np.float64))
            parts["center_pixel"].append(h5["qc/peak_center_pixel"][ie, sl].astype(np.float64))
            parts["peak_height"].append(_read_optional_2d(h5, "qc/peak_height", ie, sl, int(n), np.nan))
            parts["peak_snr"].append(_read_optional_2d(h5, "qc/peak_snr", ie, sl, int(n), np.nan))
            parts["peak_prominence"].append(_read_optional_2d(h5, "qc/peak_prominence", ie, sl, int(n), np.nan))
            parts["gmd"].append(h5[gmd_dataset][ie, sl].astype(np.float64))
            parts["trans_raw"].append(trans_raw)
            parts["section_index"].append(np.full(int(n), ie, dtype=np.int32))
            parts["good_shot_index"].append(np.arange(int(n), dtype=np.int64))
            parts["source_shot_index"].append(source_shot_index)

        flattened: dict[str, np.ndarray] = {}
        for name, chunks in parts.items():
            if chunks:
                flattened[name] = np.concatenate(chunks)
            else:
                dtype = np.int64 if name in {"section_index", "good_shot_index", "source_shot_index"} else np.float64
                flattened[name] = np.empty(0, dtype=dtype)

        attrs = dict(h5.attrs)
        attrs["selected_gmd_source"] = gmd_source_norm
        attrs["selected_gmd_dataset"] = f"/{gmd_dataset}"

        return GoodShots(
            vls_pixels=h5["vls_pixels"][...].astype(np.float64),
            attrs=attrs,
            **flattened,
        )


def resolve_unit_scale(
    config: GoodXASConfig,
    shots: GoodShots,
    keep: np.ndarray,
    trans_nominal: np.ndarray,
    trans_converted: np.ndarray,
) -> UnitScale:
    """Compute one VLS/uJ scale from a selected energy window."""

    axis = str(config.unit_scale_axis)
    window = _normal_window(config.unit_reference_window_ev or config.energy_window_ev)
    if axis == "converted":
        energy = shots.converted_energy
        trans = trans_converted
        source = "converted energy + converted proxy-bin intercept-corrected VLS"
    elif axis == "nominal":
        energy = shots.nominal_energy
        trans = trans_nominal
        source = "nominal energy + nominal intercept-corrected VLS"
    else:
        raise ValueError("unit_scale_axis must be 'converted' or 'nominal'")

    rows = keep & _window_mask(energy, window) & np.isfinite(trans)
    den = float(np.nansum(shots.gmd[rows] * shots.gmd[rows]))
    num = float(np.nansum(shots.gmd[rows] * trans[rows]))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0 or num <= 0.0:
        raise ValueError(
            "Cannot compute VLS/uJ scale: "
            f"axis={axis}, window={window[0]:g}-{window[1]:g} eV, "
            f"sum(GMD*VLS)={num:.6g}, sum(GMD^2)={den:.6g}"
        )
    return UnitScale(axis=axis, window_ev=window, vls_per_uJ=num / den, n_shots=int(np.sum(rows)), source=source)


def correlation_window(
    result: GoodXASResult,
    *,
    axis: str = "converted",
    center_ev: float = 273.0,
    half_width_ev: float = 0.05,
) -> CorrelationWindow:
    """Return shot-level GMD/VLS correlation around one energy."""

    if float(half_width_ev) <= 0:
        raise ValueError("half_width_ev must be positive")
    lo = float(center_ev) - float(half_width_ev)
    hi = float(center_ev) + float(half_width_ev)
    if axis == "converted":
        energy = result.shots.converted_energy
        trans = result.trans_converted_uJ
    elif axis == "nominal":
        energy = result.shots.nominal_energy
        trans = result.trans_nominal_uJ
    else:
        raise ValueError("axis must be 'converted' or 'nominal'")
    keep = result.keep_mask & _window_mask(energy, (lo, hi)) & np.isfinite(trans)
    g = result.shots.gmd[keep]
    t = trans[keep]
    e = energy[keep]
    fit = fit_line(g, t)
    return CorrelationWindow(axis=axis, window_ev=(lo, hi), energy=e, gmd=g, trans_uJ=t, fit=fit)


def feature_vls_stats(
    result: GoodXASResult,
    *,
    axis: str = "converted",
    center_ev: float = 272.0,
    half_width_ev: float = 0.05,
    vls_mode: str = "axis_corrected",
    max_shots: Optional[int] = None,
    seed: int = 123,
    chunk_size: int = 5000,
) -> FeatureVLSStats:
    """Read selected good-H5 spectra and summarize one feature-energy region."""

    if float(half_width_ev) <= 0:
        raise ValueError("half_width_ev must be positive")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    axis = str(axis).lower()
    energy = _energy_for_axis(result, axis)
    lo = float(center_ev) - float(half_width_ev)
    hi = float(center_ev) + float(half_width_ev)
    keep = result.keep_mask & _window_mask(energy, (lo, hi))
    selected = np.flatnonzero(keep)
    if max_shots is not None and selected.size > int(max_shots):
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(selected, size=int(max_shots), replace=False))

    vls_dataset = _resolve_feature_vls_dataset(axis, vls_mode)
    vls_pixels = result.shots.vls_pixels.astype(np.float64, copy=True)
    n_pix = int(vls_pixels.size)
    n_selected = int(selected.size)
    sum_y = np.zeros(n_pix, dtype=np.float64)
    sum_y2 = np.zeros(n_pix, dtype=np.float64)
    count_y = np.zeros(n_pix, dtype=np.float64)
    integrated = np.full(n_selected, np.nan, dtype=np.float64)
    peak_height = np.full(n_selected, np.nan, dtype=np.float64)

    if n_selected:
        with h5py.File(result.path, "r") as h5:
            if vls_dataset not in h5:
                raise KeyError(f"{result.path} is missing VLS dataset {vls_dataset!r}")
            ds = h5[vls_dataset]
            for section in np.unique(result.shots.section_index[selected]):
                pos = np.flatnonzero(result.shots.section_index[selected] == section)
                rows = result.shots.good_shot_index[selected[pos]].astype(np.int64)
                order = np.argsort(rows)
                rows = rows[order]
                pos = pos[order]
                for start in range(0, rows.size, int(chunk_size)):
                    stop = min(start + int(chunk_size), rows.size)
                    block = ds[int(section), rows[start:stop], :].astype(np.float64)
                    finite = np.isfinite(block)
                    block_sum = np.where(finite, block, 0.0)
                    sum_y += np.sum(block_sum, axis=0)
                    sum_y2 += np.sum(block_sum * block_sum, axis=0)
                    count_y += np.sum(finite, axis=0)
                    has_finite = np.any(finite, axis=1)
                    row_integral = np.full(block.shape[0], np.nan, dtype=np.float64)
                    row_peak = np.full(block.shape[0], np.nan, dtype=np.float64)
                    row_integral[has_finite] = np.sum(block_sum[has_finite], axis=1)
                    row_peak[has_finite] = np.max(np.where(finite[has_finite], block[has_finite], -np.inf), axis=1)
                    integrated[pos[start:stop]] = row_integral
                    peak_height[pos[start:stop]] = row_peak

    mean_vls = np.full(n_pix, np.nan, dtype=np.float64)
    std_vls = np.full(n_pix, np.nan, dtype=np.float64)
    if n_selected:
        np.divide(sum_y, count_y, out=mean_vls, where=count_y > 0)
        second = np.full(n_pix, np.nan, dtype=np.float64)
        np.divide(sum_y2, count_y, out=second, where=count_y > 0)
        variance = np.maximum(second - mean_vls * mean_vls, 0.0)
        std_vls = np.sqrt(variance)

    return FeatureVLSStats(
        axis=axis,
        center_ev=float(center_ev),
        half_width_ev=float(half_width_ev),
        window_ev=(lo, hi),
        vls_dataset=vls_dataset,
        n_shots=n_selected,
        vls_pixels=vls_pixels,
        mean_vls=mean_vls,
        std_vls=std_vls,
        energy=energy[selected].astype(np.float64, copy=True),
        nominal_energy=result.shots.nominal_energy[selected].astype(np.float64, copy=True),
        converted_energy=result.shots.converted_energy[selected].astype(np.float64, copy=True),
        center_pixel=result.shots.center_pixel[selected].astype(np.float64, copy=True),
        peak_height=peak_height,
        peak_snr=result.shots.peak_snr[selected].astype(np.float64, copy=True),
        peak_prominence=result.shots.peak_prominence[selected].astype(np.float64, copy=True),
        integrated_vls=integrated,
        gmd=result.shots.gmd[selected].astype(np.float64, copy=True),
    )


def fit_by_group(n_groups: int, group: np.ndarray, gmd: np.ndarray, trans: np.ndarray, keep: np.ndarray) -> LinearFits:
    """Fit trans = intercept + slope * GMD independently for each group."""

    n_groups = int(n_groups)
    n = np.zeros(n_groups, dtype=np.int64)
    sx = np.zeros(n_groups, dtype=np.float64)
    sy = np.zeros(n_groups, dtype=np.float64)
    sxx = np.zeros(n_groups, dtype=np.float64)
    syy = np.zeros(n_groups, dtype=np.float64)
    sxy = np.zeros(n_groups, dtype=np.float64)

    valid = keep & (group >= 0) & (group < n_groups) & np.isfinite(gmd) & np.isfinite(trans)
    if np.any(valid):
        idx = group[valid].astype(np.int64)
        g = gmd[valid].astype(np.float64, copy=False)
        t = trans[valid].astype(np.float64, copy=False)
        np.add.at(n, idx, 1)
        np.add.at(sx, idx, g)
        np.add.at(sy, idx, t)
        np.add.at(sxx, idx, g * g)
        np.add.at(syy, idx, t * t)
        np.add.at(sxy, idx, g * t)

    nf = n.astype(np.float64)
    denom = nf * sxx - sx * sx
    intercept = np.full(n_groups, np.nan, dtype=np.float64)
    slope = np.full(n_groups, np.nan, dtype=np.float64)
    r = np.full(n_groups, np.nan, dtype=np.float64)
    fit_ok = (n >= 2) & np.isfinite(denom) & (denom != 0.0)
    slope[fit_ok] = (nf[fit_ok] * sxy[fit_ok] - sx[fit_ok] * sy[fit_ok]) / denom[fit_ok]
    intercept[fit_ok] = (sy[fit_ok] - slope[fit_ok] * sx[fit_ok]) / nf[fit_ok]

    r_den = np.sqrt((nf * sxx - sx * sx) * (nf * syy - sy * sy))
    r_ok = (n >= 2) & np.isfinite(r_den) & (r_den > 0.0)
    r[r_ok] = (nf[r_ok] * sxy[r_ok] - sx[r_ok] * sy[r_ok]) / r_den[r_ok]
    return LinearFits(intercept=intercept, slope=slope, r=r, r2=r * r, n=n)


def apply_group_intercepts(trans: np.ndarray, group: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Subtract intercept[group], falling back to zero if the fit is unavailable."""

    out = np.asarray(trans, dtype=np.float64).copy()
    valid = (group >= 0) & (group < intercept.size)
    offsets = np.zeros(out.shape, dtype=np.float64)
    offsets[valid] = intercept[group[valid]]
    offsets[~np.isfinite(offsets)] = 0.0
    return out - offsets


def new_grid(n_x: int, n_gmd: int) -> dict[str, np.ndarray]:
    """Create accumulator arrays for binned XAS."""

    return {
        "counts": np.zeros((int(n_x), int(n_gmd)), dtype=np.int64),
        "gmd_sum": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "trans_sum": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "gmd_sumsq": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "trans_sumsq": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "gmd_trans_sum": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
    }


def add_to_grid(
    grid: dict[str, np.ndarray],
    xbin: np.ndarray,
    gbin: np.ndarray,
    keep: np.ndarray,
    gmd: np.ndarray,
    trans: np.ndarray,
) -> None:
    """Accumulate shot-level GMD and transmitted VLS into x/GMD bins."""

    valid = (
        keep
        & (xbin >= 0)
        & (xbin < grid["counts"].shape[0])
        & (gbin >= 0)
        & (gbin < grid["counts"].shape[1])
        & np.isfinite(gmd)
        & np.isfinite(trans)
    )
    if not np.any(valid):
        return
    idx = (xbin[valid].astype(np.int64), gbin[valid].astype(np.int64))
    g = gmd[valid].astype(np.float64, copy=False)
    t = trans[valid].astype(np.float64, copy=False)
    np.add.at(grid["counts"], idx, 1)
    np.add.at(grid["gmd_sum"], idx, g)
    np.add.at(grid["trans_sum"], idx, t)
    np.add.at(grid["gmd_sumsq"], idx, g * g)
    np.add.at(grid["trans_sumsq"], idx, t * t)
    np.add.at(grid["gmd_trans_sum"], idx, g * t)


def make_binned(
    x: np.ndarray,
    grid: dict[str, np.ndarray],
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    x_label: str,
    metadata: dict[str, Any],
) -> BinnedXAS:
    """Build a BinnedXAS object from accumulator arrays."""

    return BinnedXAS(
        x=np.asarray(x, dtype=np.float64),
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        gmd_labels=list(gmd_labels),
        x_label=x_label,
        metadata=dict(metadata),
        **grid,
    )


def bootstrap_sigma_grid(
    gmd: np.ndarray,
    trans: np.ndarray,
    xbin: np.ndarray,
    gbin: np.ndarray,
    keep: np.ndarray,
    shape: tuple[int, int],
    *,
    n_bootstrap: int,
    seed: int,
    min_shots_per_bin: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap OD uncertainty independently inside each x/GMD cell."""

    rng = np.random.default_rng(int(seed))
    n_x, n_g = (int(shape[0]), int(shape[1]))
    sigma = np.full((n_x, n_g), np.nan, dtype=np.float64)
    counts = np.zeros((n_x, n_g), dtype=np.int64)
    flat = xbin.astype(np.int64) * n_g + gbin.astype(np.int64)
    valid = (
        keep
        & (xbin >= 0)
        & (xbin < n_x)
        & (gbin >= 0)
        & (gbin < n_g)
        & np.isfinite(gmd)
        & np.isfinite(trans)
    )
    if not np.any(valid):
        return sigma, counts

    flat_v = flat[valid]
    g_v = gmd[valid].astype(np.float64, copy=False)
    t_v = trans[valid].astype(np.float64, copy=False)
    order = np.argsort(flat_v)
    flat_v = flat_v[order]
    g_v = g_v[order]
    t_v = t_v[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_v)) + 1]
    stops = np.r_[starts[1:], flat_v.size]
    for start, stop in zip(starts, stops):
        cell = int(flat_v[start])
        ix = cell // n_g
        ig = cell % n_g
        g = g_v[start:stop]
        t = t_v[start:stop]
        good = np.isfinite(g) & np.isfinite(t)
        g = g[good]
        t = t[good]
        counts[ix, ig] = g.size
        if g.size < int(min_shots_per_bin):
            continue
        boot = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
        write = 0
        for _ in range(0, int(n_bootstrap), int(batch_size)):
            n_batch = min(int(batch_size), int(n_bootstrap) - write)
            sample_idx = rng.integers(0, g.size, size=(n_batch, g.size), endpoint=False)
            boot[write : write + n_batch] = log10_ratio(np.sum(g[sample_idx], axis=1), np.sum(t[sample_idx], axis=1))
            write += n_batch
        finite = boot[np.isfinite(boot)]
        if finite.size >= 2:
            sigma[ix, ig] = float(np.std(finite, ddof=1))
    return sigma, counts


def print_good_xas_summary(result: GoodXASResult) -> None:
    """Print a compact summary for a good-H5 XAS result."""

    print(f"input H5            : {result.path}")
    print(
        "GMD source          : "
        f"{_normalize_gmd_source(result.config.gmd_source)} ({_gmd_dataset_for_source(result.config.gmd_source)})"
    )
    print(f"stored good shots   : {result.shots.gmd.size:,}")
    print(f"accepted shots      : {int(np.sum(result.keep_mask)):,}")
    print(f"nominal bins        : {result.nominal.x.size:,}")
    print(f"converted bins      : {result.converted.x.size:,}")
    print(f"proxy pixel width   : {result.config.proxy_pixel_bin_width}")
    print(f"unit scale axis     : {result.unit_scale.axis}")
    print(f"unit scale window   : {result.unit_scale.window_ev[0]:g}-{result.unit_scale.window_ev[1]:g} eV")
    print(f"unit scale shots    : {result.unit_scale.n_shots:,}")
    print(f"VLS per uJ          : {result.unit_scale.vls_per_uJ:.6g}")
    _print_count_summary("nominal", result.nominal)
    _print_count_summary("converted", result.converted)


def print_bootstrap_summary(bootstrap: BootstrapErrors) -> None:
    """Print a compact bootstrap uncertainty summary."""

    print(f"bootstrap draws     : {bootstrap.n_bootstrap:,}")
    print(f"bootstrap seed      : {bootstrap.seed}")
    print(f"bootstrap min shots : {bootstrap.min_shots_per_bin:,}")
    print(f"nominal boot bins   : {int(np.sum(np.isfinite(bootstrap.nominal_od_sigma)))}/{bootstrap.nominal_od_sigma.size} finite")
    print(f"converted boot bins : {int(np.sum(np.isfinite(bootstrap.converted_od_sigma)))}/{bootstrap.converted_od_sigma.size} finite")


def print_gmd_difference_summary(result: GoodXASResult, *, bin_a: int = 0, bin_b: int = -1) -> None:
    """Print resolved GMD-bin labels for the plotted difference."""

    ia = resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = resolve_bin_index(bin_b, len(result.gmd_labels))
    print(f"Delta OD direction  : bin {ib} - bin {ia}")
    print(f"bin A               : {result.gmd_labels[ia]}")
    print(f"bin B               : {result.gmd_labels[ib]}")


def plot_unit_scale_matplotlib(
    result: GoodXASResult,
    *,
    axis: Optional[str] = None,
    xlim: Optional[Sequence[float]] = None,
    raw_ylim: Optional[Sequence[float]] = None,
    scaled_ylim: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
):
    """Plot the single-slope VLS/uJ convention used by the good-H5 analysis."""

    import matplotlib.pyplot as plt

    axis = result.unit_scale.axis if axis is None else str(axis).lower()
    if axis == "converted":
        binned = result.converted
    elif axis == "nominal":
        binned = result.nominal
    else:
        raise ValueError("axis must be 'converted' or 'nominal'")

    counts = binned.counts.astype(np.float64)
    mean_gmd = safe_divide(binned.gmd_sum, counts)
    mean_trans_uJ = safe_divide(binned.trans_sum, counts)
    mean_trans_raw = mean_trans_uJ * result.unit_scale.vls_per_uJ
    energy = np.repeat(binned.x[:, None], counts.shape[1], axis=1)

    valid = (counts > 0) & np.isfinite(mean_gmd) & np.isfinite(mean_trans_uJ) & np.isfinite(energy)
    x = mean_gmd[valid]
    y_raw = mean_trans_raw[valid]
    y_scaled = mean_trans_uJ[valid]
    c = energy[valid]
    sizes = np.clip(np.sqrt(counts[valid]) * 2.0, 10.0, 80.0)
    is_ref = _window_mask(c, result.unit_scale.window_ev)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    if x.size == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No finite unit-scale bins", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.25)
        return fig, axes

    x_line = np.linspace(0.0, float(np.nanmax(x)) * 1.05, 200)
    sc = axes[0].scatter(x, y_raw, c=c, s=sizes, cmap="viridis", alpha=0.72, edgecolors="none")
    if np.any(is_ref):
        axes[0].scatter(
            x[is_ref],
            y_raw[is_ref],
            s=sizes[is_ref] + 18.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="reference window",
        )
    axes[0].plot(
        x_line,
        result.unit_scale.vls_per_uJ * x_line,
        color="black",
        lw=1.4,
        label=f"fit: VLS = {result.unit_scale.vls_per_uJ:.4g} * GMD",
    )
    axes[0].set_xlabel("mean GMD / uJ")
    axes[0].set_ylabel("mean integrated VLS")
    axes[0].set_title("Before unit conversion")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(x, y_scaled, c=c, s=sizes, cmap="viridis", alpha=0.72, edgecolors="none")
    if np.any(is_ref):
        axes[1].scatter(
            x[is_ref],
            y_scaled[is_ref],
            s=sizes[is_ref] + 18.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="reference window",
        )
    axes[1].plot(x_line, x_line, color="black", lw=1.4, label="after conversion: VLS in uJ = GMD")
    axes[1].set_xlabel("mean GMD / uJ")
    axes[1].set_ylabel("mean VLS in uJ")
    axes[1].set_title("After single-slope unit conversion")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label(f"{axis} photon energy (eV)")
    if title is None:
        title = f"VLS/uJ unit convention ({result.unit_scale.source})"
    fig.suptitle(title)
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if raw_ylim is not None:
        axes[0].set_ylim(*raw_ylim)
    if scaled_ylim is not None:
        axes[1].set_ylim(*scaled_ylim)
    return fig, axes


def plot_xas_with_gmd_difference_matplotlib(
    result: GoodXASResult,
    *,
    bootstrap: Optional[BootstrapErrors] = None,
    bin_a: int = 0,
    bin_b: int = -1,
    xlim: Optional[Sequence[float]] = None,
    xas_ylim: Optional[Sequence[float]] = None,
    delta_ylim: Optional[Sequence[float]] = None,
    nominal_xlim: Optional[Sequence[float]] = None,
    xas_nominal_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    xas_converted_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    share_xaxis: bool = True,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot nominal and converted GMD-binned XAS plus GMD-bin differences."""

    import matplotlib.pyplot as plt

    if nominal_xlim is None:
        nominal_xlim = xlim
    if converted_xlim is None:
        converted_xlim = xlim
    if xas_nominal_ylim is None:
        xas_nominal_ylim = xas_ylim
    if xas_converted_ylim is None:
        xas_converted_ylim = xas_ylim
    if delta_nominal_ylim is None:
        delta_nominal_ylim = delta_ylim
    if delta_converted_ylim is None:
        delta_converted_ylim = delta_ylim
    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim) if share_xaxis else None
    nominal_error = bootstrap.nominal_od_sigma if bootstrap is not None else result.nominal.od_sem
    converted_error = bootstrap.converted_od_sigma if bootstrap is not None else result.converted.od_sem
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.5), sharex=share_xaxis, constrained_layout=True)
    _plot_binned_od(
        axes[0],
        result.nominal,
        nominal_error,
        title="Nominal energy: GMD-binned XAS",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=xas_nominal_ylim,
        show_errorbars=show_errorbars,
        show_legend=True,
    )
    _plot_delta_od(
        axes[1],
        result.nominal,
        nominal_error,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Nominal energy: GMD-bin difference",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=delta_nominal_ylim,
        show_errorbars=show_errorbars,
    )
    _plot_binned_od(
        axes[2],
        result.converted,
        converted_error,
        title="Converted/proxy energy: GMD-binned XAS",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=xas_converted_ylim,
        show_errorbars=show_errorbars,
        show_legend=False,
    )
    _plot_delta_od(
        axes[3],
        result.converted,
        converted_error,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Converted/proxy energy: GMD-bin difference",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=delta_converted_ylim,
        show_errorbars=show_errorbars,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("photon energy (eV)")
    if title is None:
        error_label = "bootstrap" if bootstrap is not None else "SEM"
        title = (
            "Good static XAS and GMD-bin difference "
            f"({result.config.correction_mode}, {result.unit_scale.axis} unit scale, {error_label} errorbars)"
        )
    fig.suptitle(title)
    return fig, axes


def plot_xas_with_gmd_difference_plotly(
    result: GoodXASResult,
    *,
    bootstrap: Optional[BootstrapErrors] = None,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    xas_nominal_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    xas_converted_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot full OD and Delta OD in four Plotly rows with independent x-axes."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_xas_with_gmd_difference_plotly") from exc

    ia = resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = resolve_bin_index(bin_b, len(result.gmd_labels))
    diff_label = f"{result.gmd_labels[ib]} - {result.gmd_labels[ia]}"
    if title is None:
        error_label = "bootstrap" if bootstrap is not None else "SEM"
        title = (
            "Good static XAS and GMD-bin difference "
            f"({result.config.correction_mode}, {result.unit_scale.axis} unit scale, {error_label} errorbars)"
        )

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Nominal energy: GMD-binned XAS",
            "Nominal energy: GMD-bin difference",
            "Converted/proxy energy: GMD-binned XAS",
            "Converted/proxy energy: GMD-bin difference",
        ),
        shared_xaxes=False,
        vertical_spacing=0.075,
    )
    colors = [_rgba_to_hex(c) for c in _colors(len(result.gmd_labels))]
    error_by_binned = {
        id(result.nominal): bootstrap.nominal_od_sigma if bootstrap is not None else result.nominal.od_sem,
        id(result.converted): bootstrap.converted_od_sigma if bootstrap is not None else result.converted.od_sem,
    }

    for row, binned, showlegend in ((1, result.nominal, True), (3, result.converted, False)):
        order = finite_x_order(binned.x)
        x = binned.x[order]
        y = binned.od[order]
        err = error_by_binned[id(binned)][order]
        for ig, label in enumerate(result.gmd_labels):
            error_y = {"type": "data", "array": err[:, ig], "visible": True} if show_errorbars else None
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[:, ig],
                    error_y=error_y,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=showlegend,
                    marker={"size": 5, "color": colors[ig]},
                    line={"width": 1.4, "color": colors[ig]},
                ),
                row=row,
                col=1,
            )

    for row, binned in ((2, result.nominal), (4, result.converted)):
        x, delta, sigma, _, _ = delta_od_for_bins(
            binned,
            error_by_binned[id(binned)],
            bin_a=bin_a,
            bin_b=bin_b,
        )
        error_y = {"type": "data", "array": sigma, "visible": True} if show_errorbars else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                error_y=error_y,
                mode="lines+markers",
                name=diff_label,
                legendgroup="delta",
                showlegend=(row == 2),
                marker={"size": 6, "color": "#1f77b4"},
                line={"width": 1.8, "color": "#1f77b4"},
            ),
            row=row,
            col=1,
        )
        if x.size:
            fig.add_trace(
                go.Scatter(
                    x=[float(np.nanmin(x)), float(np.nanmax(x))],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="zero",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"width": 1.0, "color": "rgba(0,0,0,0.45)", "dash": "dot"},
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="nominal photon energy (eV)", range=nominal_xlim, row=2, col=1)
    fig.update_xaxes(title_text="converted photon energy (eV)", range=converted_xlim, row=4, col=1)
    fig.update_yaxes(title_text="OD", range=xas_nominal_ylim, row=1, col=1)
    fig.update_yaxes(title_text="Delta OD", range=delta_nominal_ylim, row=2, col=1)
    fig.update_yaxes(title_text="OD", range=xas_converted_ylim, row=3, col=1)
    fig.update_yaxes(title_text="Delta OD", range=delta_converted_ylim, row=4, col=1)
    fig.update_layout(
        title=title,
        height=980,
        width=980,
        template="plotly_white",
        legend_title_text="GMD bin",
        margin={"l": 75, "r": 25, "t": 100, "b": 60},
    )
    return fig


def plot_correlation_hexbin(
    corr: CorrelationWindow,
    *,
    ax=None,
    gridsize: int = 70,
    cmap: str = "viridis",
    mincnt: int = 1,
):
    """Plot GMD versus corrected VLS as a density map with fitted line."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    else:
        fig = ax.figure
    if corr.gmd.size == 0:
        ax.text(0.5, 0.5, "No shots in correlation window", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("GMD (uJ)")
        ax.set_ylabel("intercept-corrected VLS (uJ)")
        ax.set_title(f"{corr.axis} {corr.window_ev[0]:.3f}-{corr.window_ev[1]:.3f} eV: N=0")
        ax.grid(alpha=0.2)
        return fig, ax
    hb = ax.hexbin(corr.gmd, corr.trans_uJ, gridsize=int(gridsize), bins="log", mincnt=int(mincnt), cmap=cmap)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    fit = corr.fit
    if np.isfinite(fit.get("slope", np.nan)) and np.isfinite(fit.get("intercept", np.nan)):
        x = np.linspace(float(np.nanmin(corr.gmd)), float(np.nanmax(corr.gmd)), 200)
        ax.plot(x, fit["intercept"] + fit["slope"] * x, color="tab:red", lw=2.0, label="linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("GMD (uJ)")
    ax.set_ylabel("intercept-corrected VLS (uJ)")
    ax.set_title(
        f"{corr.axis} {corr.window_ev[0]:.3f}-{corr.window_ev[1]:.3f} eV: "
        f"N={int(fit.get('n', 0)):,}, R^2={fit.get('r2', np.nan):.3f}"
    )
    ax.grid(alpha=0.2)
    return fig, ax


def feature_summary_table(feature: FeatureVLSStats):
    """Return a compact summary table for one feature-energy selection."""

    import pandas as pd

    rows = [
        ("axis", feature.axis),
        ("energy window (eV)", f"{feature.window_ev[0]:.4g}-{feature.window_ev[1]:.4g}"),
        ("VLS dataset", feature.vls_dataset),
        ("shots", f"{feature.n_shots:,}"),
        ("selected energy mean ± std (eV)", _mean_std_text(feature.energy, precision=5)),
        ("nominal energies", _unique_range_text(feature.nominal_energy, precision=5)),
        ("converted energy mean ± std (eV)", _mean_std_text(feature.converted_energy, precision=5)),
        ("peak center mean ± std (pixel)", _mean_std_text(feature.center_pixel, precision=3)),
        ("peak height median [16,84]", _percentile_text(feature.peak_height, precision=4)),
        ("peak prominence median [16,84]", _percentile_text(feature.peak_prominence, precision=4)),
        ("peak SNR median [16,84]", _percentile_text(feature.peak_snr, precision=3)),
        ("integrated VLS median [16,84]", _percentile_text(feature.integrated_vls, precision=4)),
        ("GMD mean ± std (uJ)", _mean_std_text(feature.gmd, precision=4)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def feature_summary_comparison(features: Sequence[FeatureVLSStats], labels: Optional[Sequence[str]] = None):
    """Return side-by-side summary columns for multiple feature selections."""

    import pandas as pd

    features = list(features)
    labels = _feature_labels(features, labels)
    merged: Optional[pd.DataFrame] = None
    for feature, label in zip(features, labels):
        table = feature_summary_table(feature).rename(columns={"value": label})
        merged = table if merged is None else merged.merge(table, on="metric", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["metric"])
    return merged


def plot_feature_vls_mean(
    feature: FeatureVLSStats,
    *,
    ax=None,
    color: str = "tab:blue",
    std_alpha: float = 0.22,
):
    """Plot mean VLS trace with a ±1 std shaded band."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    else:
        fig = ax.figure

    if feature.n_shots == 0:
        ax.text(0.5, 0.5, "No shots in selected feature window", ha="center", va="center", transform=ax.transAxes)
    else:
        x = feature.vls_pixels
        y = feature.mean_vls
        s = feature.std_vls
        ax.plot(x, y, lw=1.6, color=color, label="mean VLS")
        ax.fill_between(x, y - s, y + s, color=color, alpha=float(std_alpha), linewidth=0, label="±1 std")
        center = np.nanmedian(feature.center_pixel)
        if np.isfinite(center):
            ax.axvline(center, color="0.2", lw=1.0, ls=":", label="median peak center")
        ax.legend(frameon=False)

    ax.set_title(_feature_title(feature, "Mean VLS trace"))
    ax.set_xlabel("VLS pixel")
    ax.set_ylabel("VLS intensity")
    ax.grid(alpha=0.25)
    return fig, ax


def plot_feature_vls_mean_comparison(
    features: Sequence[FeatureVLSStats],
    *,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    std_alpha: float = 0.14,
):
    """Overlay mean VLS traces with ±1 std bands for multiple selections."""

    import matplotlib.pyplot as plt

    features = list(features)
    labels = _feature_labels(features, labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    else:
        fig = ax.figure

    colors = _colors(max(len(features), 1))
    for feature, label, color in zip(features, labels, colors):
        if feature.n_shots == 0:
            continue
        x = feature.vls_pixels
        y = feature.mean_vls
        s = feature.std_vls
        ax.plot(x, y, lw=1.7, color=color, label=f"{label}; N={feature.n_shots:,}")
        ax.fill_between(x, y - s, y + s, color=color, alpha=float(std_alpha), linewidth=0)
        center = np.nanmedian(feature.center_pixel)
        if np.isfinite(center):
            ax.axvline(center, color=color, lw=0.9, ls=":", alpha=0.8)
    if not any(feature.n_shots > 0 for feature in features):
        ax.text(0.5, 0.5, "No shots in selected feature windows", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Mean VLS trace comparison")
    ax.set_xlabel("VLS pixel")
    ax.set_ylabel("VLS intensity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return fig, ax


def plot_feature_peak_distribution(feature: FeatureVLSStats, *, bins: int = 80):
    """Plot concise distributions for the selected feature-energy region."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    _hist_or_empty(axes[0], feature.energy, bins=bins, xlabel=f"{feature.axis} energy (eV)")
    _hist_or_empty(axes[1], feature.center_pixel, bins=bins, xlabel="VLS peak center pixel")
    _hist_or_empty(axes[2], feature.peak_height, bins=bins, xlabel="peak height")
    _hist_or_empty(axes[3], feature.peak_snr, bins=bins, xlabel="peak SNR")
    fig.suptitle(_feature_title(feature, "Feature-region distributions"))
    return fig, axes


def plot_feature_distribution_comparison(
    features: Sequence[FeatureVLSStats],
    *,
    labels: Optional[Sequence[str]] = None,
    bins: int = 80,
    density: bool = True,
):
    """Overlay key feature distributions for multiple feature selections."""

    import matplotlib.pyplot as plt

    features = list(features)
    labels = _feature_labels(features, labels)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    specs = [
        ("selected energy (eV)", [feature.energy for feature in features]),
        ("VLS peak center pixel", [feature.center_pixel for feature in features]),
        ("peak height", [feature.peak_height for feature in features]),
        ("peak SNR", [feature.peak_snr for feature in features]),
    ]
    colors = _colors(max(len(features), 1))
    for ax, (xlabel, arrays) in zip(axes, specs):
        plotted = False
        for values, label, color, feature in zip(arrays, labels, colors, features):
            vals = _finite_values(values)
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=int(bins),
                histtype="step",
                density=bool(density),
                linewidth=1.6,
                color=color,
                label=f"{label}; N={feature.n_shots:,}",
            )
            ax.axvline(np.nanmedian(vals), color=color, lw=0.9, ls=":")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density" if density else "shots")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Feature-region distribution comparison")
    return fig, axes


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y = intercept + slope * x and return basic diagnostics."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    out = {"n": float(x.size), "intercept": np.nan, "slope": np.nan, "r": np.nan, "r2": np.nan}
    if x.size < 2:
        return out
    sx = float(np.sum(x))
    sy = float(np.sum(y))
    sxx = float(np.sum(x * x))
    syy = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    n = float(x.size)
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return out
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    rden = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    r = np.nan if rden <= 0.0 else (n * sxy - sx * sy) / rden
    return {"n": float(x.size), "intercept": float(intercept), "slope": float(slope), "r": float(r), "r2": float(r * r)}


def gmd_bin_indices(gmd: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign GMD values to bins."""

    values = np.asarray(gmd, dtype=np.float64)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = idx.astype(np.int64)
    idx[values == edges[-1]] = edges.size - 2
    in_range = np.isfinite(values) & (idx >= 0) & (idx < edges.size - 1)
    return idx, in_range


def format_gmd_labels(edges: np.ndarray, accepted_gmd: np.ndarray) -> list[str]:
    """Format GMD bin labels with empirical bin means."""

    labels = []
    vals = np.asarray(accepted_gmd, dtype=np.float64)
    for ig, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        close = "]" if ig == edges.size - 2 else ")"
        sel = (vals >= lo) & (vals < hi)
        if ig == edges.size - 2:
            sel = (vals >= lo) & (vals <= hi)
        if np.any(sel):
            labels.append(f"{lo:.2g}-{hi:.2g}{close} uJ; mean {np.nanmean(vals[sel]):.2g}")
        else:
            labels.append(f"{lo:.2g}-{hi:.2g}{close} uJ")
    return labels


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Finite divide that returns NaN when denominator is zero or invalid."""

    out = np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den != 0))
    return out


def log10_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute log10(num / den), returning NaN for nonpositive ratios."""

    ratio = safe_divide(num, den)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.log10(ratio[valid])
    return out


def delta_log_ratio_sem(
    sum_g: np.ndarray,
    sum_t: np.ndarray,
    sum_g2: np.ndarray,
    sum_t2: np.ndarray,
    sum_gt: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Delta-method SEM for log10(sum(GMD) / sum(trans))."""

    n = np.asarray(counts, dtype=np.float64)
    mean_g = safe_divide(sum_g, n)
    mean_t = safe_divide(sum_t, n)
    mean_g2 = safe_divide(sum_g2, n)
    mean_t2 = safe_divide(sum_t2, n)
    mean_gt = safe_divide(sum_gt, n)
    var_g = np.maximum(mean_g2 - mean_g**2, 0.0)
    var_t = np.maximum(mean_t2 - mean_t**2, 0.0)
    cov_gt = mean_gt - mean_g * mean_t
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (1.0 / np.log(10.0) ** 2) * (
            var_g / (n * mean_g**2) + var_t / (n * mean_t**2) - 2.0 * cov_gt / (n * mean_g * mean_t)
        )
    sem = np.sqrt(np.maximum(var, 0.0))
    sem[~np.isfinite(sem)] = np.nan
    return sem


def delta_od_for_bins(
    binned: BinnedXAS,
    od_error: np.ndarray,
    *,
    bin_a: int,
    bin_b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return x, Delta OD, and propagated sigma for two GMD bins."""

    ia = resolve_bin_index(bin_a, binned.counts.shape[1])
    ib = resolve_bin_index(bin_b, binned.counts.shape[1])
    if ia == ib:
        raise ValueError("bin_a and bin_b resolve to the same GMD bin")
    order = finite_x_order(binned.x)
    x = binned.x[order]
    od = binned.od[order]
    err = np.asarray(od_error, dtype=np.float64)[order]
    delta = od[:, ib] - od[:, ia]
    sigma = np.sqrt(err[:, ib] ** 2 + err[:, ia] ** 2)
    valid = np.isfinite(x) & np.isfinite(delta)
    return x[valid], delta[valid], sigma[valid], ia, ib


def resolve_bin_index(index: int, n_bins: int) -> int:
    """Resolve negative GMD bin indices."""

    idx = int(index)
    if idx < 0:
        idx += int(n_bins)
    if idx < 0 or idx >= int(n_bins):
        raise IndexError(f"GMD bin index {index} is out of range for {n_bins} bins")
    return idx


def finite_x_order(x: np.ndarray) -> np.ndarray:
    """Return sorted indices for finite x values."""

    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    return np.where(finite)[0][np.argsort(x[finite])]


def _plot_binned_od(
    ax,
    binned: BinnedXAS,
    od_error: np.ndarray,
    *,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
    show_legend: bool,
) -> None:
    order = finite_x_order(binned.x)
    x = binned.x[order]
    y = binned.od[order]
    err = np.asarray(od_error, dtype=np.float64)[order]
    colors = _colors(y.shape[1])
    for ig, color in enumerate(colors):
        valid = np.isfinite(x) & np.isfinite(y[:, ig])
        if not np.any(valid):
            continue
        if show_errorbars:
            ax.errorbar(x[valid], y[valid, ig], yerr=err[valid, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
        else:
            ax.plot(x[valid], y[valid, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
    ax.set_title(title)
    ax.set_ylabel("OD = log10[sum(GMD) / sum(VLS in uJ)]")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(title="GMD bin", frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_delta_od(
    ax,
    binned: BinnedXAS,
    od_error: np.ndarray,
    *,
    bin_a: int,
    bin_b: int,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
) -> None:
    x, delta, sigma, ia, ib = delta_od_for_bins(binned, od_error, bin_a=bin_a, bin_b=bin_b)
    label = f"{binned.gmd_labels[ib]} - {binned.gmd_labels[ia]}"
    if show_errorbars:
        ax.errorbar(x, delta, yerr=sigma, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    else:
        ax.plot(x, delta, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    ax.axhline(0.0, color="0.25", lw=0.9, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Delta OD = OD[B] - OD[A]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _read_optional_2d(
    h5: h5py.File,
    name: str,
    section_index: int,
    shot_slice: slice,
    n: int,
    fill_value: float,
) -> np.ndarray:
    if name in h5:
        return h5[name][section_index, shot_slice].astype(np.float64)
    return np.full(int(n), fill_value, dtype=np.float64)


def _energy_for_axis(result: GoodXASResult, axis: str) -> np.ndarray:
    if axis == "converted":
        return result.shots.converted_energy
    if axis == "nominal":
        return result.shots.nominal_energy
    raise ValueError("axis must be 'converted' or 'nominal'")


def _resolve_feature_vls_dataset(axis: str, vls_mode: str) -> str:
    mode = str(vls_mode).lower()
    if mode == "axis_corrected":
        return "vls_intercept_corrected_converted" if axis == "converted" else "vls_intercept_corrected_nominal"
    if mode in {"raw", "uncorrected"}:
        return "vls"
    if mode in {"corrected", "nominal_corrected"}:
        return "vls_intercept_corrected_nominal"
    if mode == "converted_corrected":
        return "vls_intercept_corrected_converted"
    raise ValueError(
        "vls_mode must be 'axis_corrected', 'raw', 'nominal_corrected', "
        "'converted_corrected', or 'corrected'"
    )


def _feature_title(feature: FeatureVLSStats, prefix: str) -> str:
    return (
        f"{prefix}: {feature.axis} {feature.window_ev[0]:.3f}-{feature.window_ev[1]:.3f} eV, "
        f"N={feature.n_shots:,}"
    )


def _feature_labels(features: Sequence[FeatureVLSStats], labels: Optional[Sequence[str]]) -> list[str]:
    features = list(features)
    if labels is None:
        return [f"{feature.axis} {feature.center_ev:g} eV" for feature in features]
    labels = [str(label) for label in labels]
    if len(labels) != len(features):
        raise ValueError("labels must have the same length as features")
    return labels


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean_std_text(values: np.ndarray, *, precision: int = 4) -> str:
    vals = _finite_values(values)
    if vals.size == 0:
        return "n/a"
    fmt = f"{{:.{int(precision)}g}}"
    return f"{fmt.format(float(np.mean(vals)))} ± {fmt.format(float(np.std(vals)))}"


def _percentile_text(values: np.ndarray, *, precision: int = 4) -> str:
    vals = _finite_values(values)
    if vals.size == 0:
        return "n/a"
    q16, q50, q84 = np.nanpercentile(vals, [16, 50, 84])
    fmt = f"{{:.{int(precision)}g}}"
    return f"{fmt.format(float(q50))} [{fmt.format(float(q16))}, {fmt.format(float(q84))}]"


def _unique_range_text(values: np.ndarray, *, precision: int = 5) -> str:
    vals = np.unique(_finite_values(values))
    if vals.size == 0:
        return "n/a"
    fmt = f"{{:.{int(precision)}g}}"
    if vals.size == 1:
        return f"1 value: {fmt.format(float(vals[0]))}"
    return f"{vals.size:,} values: {fmt.format(float(vals[0]))}-{fmt.format(float(vals[-1]))}"


def _hist_or_empty(ax, values: np.ndarray, *, bins: int, xlabel: str) -> None:
    vals = _finite_values(values)
    if vals.size:
        ax.hist(vals, bins=int(bins), color="tab:blue", alpha=0.8)
        q16, q50, q84 = np.nanpercentile(vals, [16, 50, 84])
        ax.axvline(q50, color="0.15", lw=1.0, label="median")
        ax.axvspan(q16, q84, color="0.2", alpha=0.12, linewidth=0, label="16-84%")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("shots")
    ax.grid(alpha=0.25)


def _validate_config(config: GoodXASConfig) -> None:
    _normalize_gmd_source(config.gmd_source)
    if config.unit_scale_axis not in {"nominal", "converted"}:
        raise ValueError("unit_scale_axis must be 'nominal' or 'converted'")
    if config.correction_mode not in {"none", "per_axis_intercept"}:
        raise ValueError("correction_mode must be 'none' or 'per_axis_intercept'")
    if int(config.proxy_pixel_bin_width) <= 0:
        raise ValueError("proxy_pixel_bin_width must be positive")
    _normal_window(config.energy_window_ev)
    _normal_window(config.unit_reference_window_ev or config.energy_window_ev)


def _normalize_gmd_source(gmd_source: str) -> str:
    source = str(gmd_source).strip().lower()
    if source not in GMD_SOURCE_TO_DATASET:
        allowed = ", ".join(sorted(GMD_SOURCE_TO_DATASET))
        raise ValueError(f"gmd_source must be one of {{{allowed}}}; got {gmd_source!r}")
    return source


def _gmd_dataset_for_source(gmd_source: str) -> str:
    return GMD_SOURCE_TO_DATASET[_normalize_gmd_source(gmd_source)]


def _base_keep_mask(shots: GoodShots, config: GoodXASConfig) -> np.ndarray:
    keep = (
        np.isfinite(shots.nominal_energy)
        & np.isfinite(shots.converted_energy)
        & np.isfinite(shots.center_pixel)
        & np.isfinite(shots.gmd)
        & np.isfinite(shots.trans_raw)
    )
    if config.gmd_min_threshold is not None:
        keep &= shots.gmd >= float(config.gmd_min_threshold)
    if config.gmd_max_threshold is not None:
        keep &= shots.gmd <= float(config.gmd_max_threshold)
    return keep


def _assign_nominal_groups(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    group = np.full(values.shape, -1, dtype=np.int64)
    if centers.size == 0:
        return group
    idx = np.searchsorted(centers, values)
    valid = (idx >= 0) & (idx < centers.size)
    group[valid & (centers[idx.clip(0, centers.size - 1)] == values)] = idx[valid & (centers[idx.clip(0, centers.size - 1)] == values)]
    return group


def _center_pixel_to_relative(center_pixel: np.ndarray, vls_pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(vls_pixels, dtype=np.float64)
    rel = np.arange(pixels.size, dtype=np.float64)
    if pixels.size == 0:
        return np.full(center_pixel.shape, np.nan, dtype=np.float64)
    if pixels[0] <= pixels[-1]:
        return np.interp(center_pixel, pixels, rel, left=np.nan, right=np.nan)
    return np.interp(center_pixel, pixels[::-1], rel[::-1], left=np.nan, right=np.nan)


def _proxy_groups(center_rel: np.ndarray, n_pixels: int, width: int) -> np.ndarray:
    group = np.full(center_rel.shape, -1, dtype=np.int64)
    valid = np.isfinite(center_rel)
    rel = np.clip(np.round(center_rel[valid]).astype(np.int64), 0, int(n_pixels) - 1)
    group[valid] = rel // int(width)
    return group


def _validate_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("GMD edges must be a 1D array with at least two values")
    if not np.all(np.isfinite(edges)):
        raise ValueError("GMD edges must be finite")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("GMD edges must be strictly increasing")
    return edges


def _normal_window(window: Sequence[float]) -> tuple[float, float]:
    if len(window) != 2:
        raise ValueError("energy window must contain exactly two values")
    lo, hi = float(window[0]), float(window[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("energy window values must be finite")
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        raise ValueError("energy window must have nonzero width")
    return lo, hi


def _window_mask(values: np.ndarray, window: Sequence[float]) -> np.ndarray:
    lo, hi = _normal_window(window)
    values = np.asarray(values, dtype=np.float64)
    return np.isfinite(values) & (values >= lo) & (values <= hi)


def _resolve_shared_xlim(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
    if a is None:
        return None if b is None else _normal_window(b)
    if b is None:
        return _normal_window(a)
    aa = _normal_window(a)
    bb = _normal_window(b)
    return min(aa[0], bb[0]), max(aa[1], bb[1])


def _colors(n: int):
    import matplotlib.pyplot as plt

    return plt.cm.inferno(np.linspace(0.08, 0.92, int(n)))


def _rgba_to_hex(rgba) -> str:
    vals = np.asarray(rgba[:3], dtype=np.float64)
    vals = np.clip(np.round(vals * 255), 0, 255).astype(int)
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _print_count_summary(name: str, binned: BinnedXAS) -> None:
    counts = binned.counts[np.isfinite(binned.od)]
    if counts.size:
        print(f"{name} shots/bin    : {np.min(counts):,} / {np.median(counts):,.0f} / {np.max(counts):,} min/median/max")
    else:
        print(f"{name} shots/bin    : no finite OD bins")
