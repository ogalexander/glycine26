"""Build a good static-XAS H5 from the processed static-XAS H5.

This is the processed-data QC layer.  It intentionally reports VLS/GMD
correlation before and after the weak-VLS peak filter.  Nominal diagnostics are
grouped by section energy.  Converted diagnostics follow the difference
notebook convention: group shots by VLS center proxy-pixel bins, then attach a
converted-energy coordinate to each occupied proxy bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd


GMD_SOURCE_TO_CHANNEL_DATASET = {
    "hall": "ml/gmd_hall_channels",
    "tunnel": "ml/gmd_tunnel_channels",
}

GMD_SOURCE_TO_TOP_DATASET = {
    "hall": "gmd",
    "tunnel": "gmd_tunnel",
}


@dataclass(frozen=True)
class ProcessGoodConfig:
    """Configuration for processed static-XAS QC and good-H5 writing."""

    input_h5: str | Path
    output_h5: Optional[str | Path] = None
    use_ml_qc_ok: bool = True
    chunk_size: int = 20_000
    energy_group_decimals: int = 3
    proxy_pixel_bin_width: int = 1
    energy_calibration_poly_order: int = 1
    energy_calibration_min_sections: int = 4
    peak_snr_threshold: float = 5.0
    prominence_inner_px: int = 5
    prominence_outer_px: int = 20
    prominence_baseline_percentile: float = 20.0
    sigma_half_window_px: int = 5
    center_half_window_px: int = 5
    min_side_pixels: int = 5
    gmd_source: str = "hall"  # "hall" or "tunnel"


@dataclass(frozen=True)
class StaticInfo:
    """Metadata needed from the processed H5."""

    path: Path
    attrs: dict[str, Any]
    n_energy: int
    n_pixels: int
    max_shots: int
    n_shots: np.ndarray
    stored_energy_eV: np.ndarray
    section_energy_eV: np.ndarray
    section_energy_raw_eV: np.ndarray
    section_energy_source_values: np.ndarray
    section_energy_source: str
    section_energy_source_unit: str
    section_energy_conversion: str
    energy_group_decimals: int
    group_energies_eV: np.ndarray
    section_to_group: np.ndarray
    vls_pixels: np.ndarray
    sigma_raw: np.ndarray


@dataclass(frozen=True)
class FitTable:
    """Free-intercept VLS/GMD fit table."""

    x_eV: np.ndarray
    n: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray
    r: np.ndarray
    r2: np.ndarray
    proxy_group: Optional[np.ndarray] = None

    def dataframe(self) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {
            "energy_eV": self.x_eV,
            "n": self.n,
            "intercept": self.intercept,
            "slope": self.slope,
            "r": self.r,
            "r2": self.r2,
        }
        if self.proxy_group is not None:
            data = {"proxy_group": self.proxy_group, **data}
        return pd.DataFrame(data)


@dataclass(frozen=True)
class ProcessGoodResult:
    """Computed QC state for processed-to-good static XAS."""

    config: ProcessGoodConfig
    info: StaticInfo
    counts_df: pd.DataFrame
    converted_counts_df: pd.DataFrame
    fit_nominal_before: FitTable
    fit_nominal_after: FitTable
    fit_converted_before: FitTable
    fit_converted_after: FitTable
    fit_converted_full_before: FitTable
    fit_converted_full_after: FitTable
    section_center_pixel: np.ndarray
    section_center_n: np.ndarray
    section_center_p16: np.ndarray
    section_center_p84: np.ndarray
    energy_calibration_coeff: np.ndarray
    energy_calibration_order: int
    energy_calibration_table: pd.DataFrame
    proxy_energy_eV: np.ndarray
    occupied_proxy_groups: np.ndarray
    peak_snr: np.ndarray
    prominence: np.ndarray


def compute_processed_qc(config: ProcessGoodConfig) -> ProcessGoodResult:
    """Run finite/QC, weak-VLS filtering, calibration, and corr diagnostics."""

    _validate_config(config)
    info = read_static_info(config.input_h5, group_decimals=config.energy_group_decimals)
    _validate_gmd_source_available(info.path, config)
    n_groups = info.group_energies_eV.size
    n_proxy = int(np.ceil(info.n_pixels / int(config.proxy_pixel_bin_width)))

    nominal_before = new_fit_accumulator(n_groups)
    nominal_after = new_fit_accumulator(n_groups)
    counts = {
        "recorded": info.n_shots.astype(np.int64).copy(),
        "basic": np.zeros(info.n_energy, dtype=np.int64),
        "center_valid": np.zeros(info.n_energy, dtype=np.int64),
        "peak_keep": np.zeros(info.n_energy, dtype=np.int64),
    }
    center_by_section: list[list[np.ndarray]] = [[] for _ in range(info.n_energy)]
    peak_snr_all: list[np.ndarray] = []
    prominence_all: list[np.ndarray] = []

    for ie, _start, _stop, vls, gmd, qc_ok in iter_energy_chunks(info.path, info, config):
        group = int(info.section_to_group[ie])
        basic = basic_qc_mask(vls, gmd, qc_ok, config)
        counts["basic"][ie] += int(np.sum(basic))
        if group >= 0:
            update_fit_accumulator_by_index(
                nominal_before,
                np.full(gmd.shape, group, dtype=np.int64),
                gmd,
                integrated_vls(vls),
                basic,
            )

        peak = peak_metrics(vls, info.sigma_raw, info.vls_pixels, config)
        center_valid = basic & np.isfinite(peak["center_pixel"])
        keep = center_valid & (peak["peak_snr"] > float(config.peak_snr_threshold))
        counts["center_valid"][ie] += int(np.sum(center_valid))
        counts["peak_keep"][ie] += int(np.sum(keep))
        if np.any(keep):
            center_by_section[ie].append(peak["center_pixel"][keep].astype(np.float64))
            peak_snr_all.append(peak["peak_snr"][keep].astype(np.float64))
            prominence_all.append(peak["prominence"][keep].astype(np.float64))
            if group >= 0:
                update_fit_accumulator_by_index(
                    nominal_after,
                    np.full(gmd.shape, group, dtype=np.int64),
                    gmd,
                    integrated_vls(vls),
                    keep,
                )

    section_center_pixel, section_center_n, section_center_p16, section_center_p84 = _summarize_section_centers(
        center_by_section, info.n_energy
    )
    coeff, order, cal_table = fit_vls_pixel_energy_calibration(info, section_center_pixel, section_center_n, config)

    converted_before = new_fit_accumulator(n_proxy)
    converted_after = new_fit_accumulator(n_proxy)
    proxy_energy_sum = np.zeros(n_proxy, dtype=np.float64)
    proxy_energy_count = np.zeros(n_proxy, dtype=np.float64)
    converted_basic_counts = np.zeros(n_proxy, dtype=np.int64)
    converted_keep_counts = np.zeros(n_proxy, dtype=np.int64)

    for _ie, _start, _stop, vls, gmd, qc_ok in iter_energy_chunks(info.path, info, config):
        basic = basic_qc_mask(vls, gmd, qc_ok, config)
        peak = peak_metrics(vls, info.sigma_raw, info.vls_pixels, config)
        center_valid = basic & np.isfinite(peak["center_pixel"])
        keep = center_valid & (peak["peak_snr"] > float(config.peak_snr_threshold))
        proxy_group = proxy_groups_from_center_pixel(peak["center_pixel"], info.vls_pixels, config.proxy_pixel_bin_width)
        trans = integrated_vls(vls)
        shot_energy = np.polyval(coeff, peak["center_pixel"])

        update_fit_accumulator_by_index(converted_before, proxy_group, gmd, trans, center_valid)
        update_fit_accumulator_by_index(converted_after, proxy_group, gmd, trans, keep)
        _add_counts(converted_basic_counts, proxy_group, center_valid)
        _add_counts(converted_keep_counts, proxy_group, keep)

        energy_ok = center_valid & (proxy_group >= 0) & (proxy_group < n_proxy) & np.isfinite(shot_energy)
        if np.any(energy_ok):
            np.add.at(proxy_energy_sum, proxy_group[energy_ok], shot_energy[energy_ok])
            np.add.at(proxy_energy_count, proxy_group[energy_ok], 1.0)

    proxy_energy = np.full(n_proxy, np.nan, dtype=np.float64)
    np.divide(proxy_energy_sum, proxy_energy_count, out=proxy_energy, where=proxy_energy_count > 0)
    occupied = np.where((proxy_energy_count > 0) & np.isfinite(proxy_energy))[0]
    occupied = occupied[np.argsort(proxy_energy[occupied])]

    full_x = proxy_energy.copy()
    fit_converted_full_before = finish_fit_accumulator(converted_before, full_x, proxy_group=np.arange(n_proxy))
    fit_converted_full_after = finish_fit_accumulator(converted_after, full_x, proxy_group=np.arange(n_proxy))
    fit_converted_before = subset_fit_table(fit_converted_full_before, occupied)
    fit_converted_after = subset_fit_table(fit_converted_full_after, occupied)

    counts_df = pd.DataFrame(
        {
            "section": np.arange(info.n_energy),
            "section_energy_eV": info.section_energy_eV,
            "nominal_group": info.section_to_group,
            "recorded": counts["recorded"],
            "basic": counts["basic"],
            "center_valid": counts["center_valid"],
            "peak_keep": counts["peak_keep"],
            "basic_fraction": _safe_divide(counts["basic"], counts["recorded"]),
            "peak_fraction_of_basic": _safe_divide(counts["peak_keep"], counts["basic"]),
            "center_pixel": section_center_pixel,
            "center_n": section_center_n,
        }
    )
    converted_counts_df = pd.DataFrame(
        {
            "proxy_group": occupied,
            "energy_eV": proxy_energy[occupied],
            "basic": converted_basic_counts[occupied],
            "peak_keep": converted_keep_counts[occupied],
            "peak_fraction_of_basic": _safe_divide(converted_keep_counts[occupied], converted_basic_counts[occupied]),
        }
    )
    return ProcessGoodResult(
        config=config,
        info=info,
        counts_df=counts_df,
        converted_counts_df=converted_counts_df,
        fit_nominal_before=finish_fit_accumulator(nominal_before, info.group_energies_eV),
        fit_nominal_after=finish_fit_accumulator(nominal_after, info.group_energies_eV),
        fit_converted_before=fit_converted_before,
        fit_converted_after=fit_converted_after,
        fit_converted_full_before=fit_converted_full_before,
        fit_converted_full_after=fit_converted_full_after,
        section_center_pixel=section_center_pixel,
        section_center_n=section_center_n,
        section_center_p16=section_center_p16,
        section_center_p84=section_center_p84,
        energy_calibration_coeff=coeff,
        energy_calibration_order=order,
        energy_calibration_table=cal_table,
        proxy_energy_eV=proxy_energy,
        occupied_proxy_groups=occupied,
        peak_snr=np.concatenate(peak_snr_all) if peak_snr_all else np.empty(0, dtype=np.float64),
        prominence=np.concatenate(prominence_all) if prominence_all else np.empty(0, dtype=np.float64),
    )


def save_good_static_xas(result: ProcessGoodResult, output_h5: Optional[str | Path] = None) -> Path:
    """Write the good static-XAS H5 selected by a ProcessGoodResult."""

    info = result.info
    config = result.config
    gmd_source = _normalize_gmd_source(config.gmd_source)
    processing_gmd_dataset = _gmd_channel_dataset_for_source(gmd_source)
    path = Path(output_h5) if output_h5 is not None else _default_output_path(config)
    good_counts = result.counts_df["peak_keep"].to_numpy(dtype=np.int64)
    max_good = int(good_counts.max()) if good_counts.size else 0
    n_sections = info.n_energy
    n_pix = info.n_pixels
    fill_f = np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    nominal_after = result.fit_nominal_after
    converted_after_full = result.fit_converted_full_after
    section_intercept = np.full(n_sections, np.nan, dtype=np.float64)
    section_slope = np.full(n_sections, np.nan, dtype=np.float64)
    for ie, group in enumerate(info.section_to_group):
        if group >= 0:
            section_intercept[ie] = nominal_after.intercept[int(group)]
            section_slope[ie] = nominal_after.slope[int(group)]

    with h5py.File(info.path, "r") as src, h5py.File(path, "w") as out:
        out.create_dataset("vls", shape=(n_sections, max_good, n_pix), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("vls_intercept_corrected", shape=(n_sections, max_good, n_pix), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("vls_intercept_corrected_nominal", shape=(n_sections, max_good, n_pix), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("vls_intercept_corrected_converted", shape=(n_sections, max_good, n_pix), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("gmd", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("gmd_tunnel", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("trans_raw", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("trans_intercept_corrected", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("trans_intercept_corrected_nominal", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("trans_intercept_corrected_converted", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("energy_vls_calibrated", shape=(n_sections, max_good), dtype=np.float64, fillvalue=fill_f, compression="gzip")
        out.create_dataset("n_shots", data=good_counts)
        out.create_dataset("nominal_energies", data=info.section_energy_eV)
        out.create_dataset("vls_pixels", data=info.vls_pixels.astype(np.float64))

        write_energy_group(out, info, result)
        qc_grp = write_qc_group(out, result, max_good)
        write_fits_group(out, result, section_intercept, section_slope)
        ml = create_ml_group(out, src, n_sections, max_good)
        if "noise" in src:
            src.copy("noise", out)

        write_pos = np.zeros(n_sections, dtype=np.int64)
        for ie, start, stop, vls, gmd, qc_ok in iter_energy_chunks(info.path, info, config):
            basic = basic_qc_mask(vls, gmd, qc_ok, config)
            peak = peak_metrics(vls, info.sigma_raw, info.vls_pixels, config)
            good = basic & np.isfinite(peak["center_pixel"]) & (peak["peak_snr"] > float(config.peak_snr_threshold))
            if not np.any(good):
                continue

            local_indices = np.arange(start, stop, dtype=np.int64)
            j0 = int(write_pos[ie])
            j1 = j0 + int(np.sum(good))
            nominal_group = int(info.section_to_group[ie])
            nominal_offset = section_intercept[ie]
            if not np.isfinite(nominal_offset):
                nominal_offset = 0.0
            nominal_group_energy = info.group_energies_eV[nominal_group] if nominal_group >= 0 else np.nan

            proxy_group = proxy_groups_from_center_pixel(peak["center_pixel"], info.vls_pixels, config.proxy_pixel_bin_width)
            proxy_good = proxy_group[good]
            converted_offset = np.zeros(proxy_good.shape, dtype=np.float64)
            converted_group_energy = np.full(proxy_good.shape, np.nan, dtype=np.float64)
            valid_proxy = (proxy_good >= 0) & (proxy_good < converted_after_full.intercept.size)
            converted_offset[valid_proxy] = converted_after_full.intercept[proxy_good[valid_proxy]]
            converted_offset[~np.isfinite(converted_offset)] = 0.0
            converted_group_energy[valid_proxy] = result.proxy_energy_eV[proxy_good[valid_proxy]]

            shot_energy = np.polyval(result.energy_calibration_coeff, peak["center_pixel"][good])
            vv_good = vls[good]
            trans = integrated_vls(vv_good)
            nominal_vls = vv_good - nominal_offset / n_pix
            converted_vls = vv_good - converted_offset[:, None] / n_pix

            out["vls"][ie, j0:j1, :] = vv_good
            out["vls_intercept_corrected"][ie, j0:j1, :] = nominal_vls
            out["vls_intercept_corrected_nominal"][ie, j0:j1, :] = nominal_vls
            out["vls_intercept_corrected_converted"][ie, j0:j1, :] = converted_vls
            out["gmd"][ie, j0:j1] = _read_gmd_values_for_output(src, "hall", ie, start, stop, good)
            out["gmd_tunnel"][ie, j0:j1] = _read_gmd_values_for_output(src, "tunnel", ie, start, stop, good)
            out["trans_raw"][ie, j0:j1] = trans
            out["trans_intercept_corrected"][ie, j0:j1] = trans - nominal_offset
            out["trans_intercept_corrected_nominal"][ie, j0:j1] = trans - nominal_offset
            out["trans_intercept_corrected_converted"][ie, j0:j1] = trans - converted_offset
            out["energy_vls_calibrated"][ie, j0:j1] = shot_energy

            qc_grp["peak_prominence"][ie, j0:j1] = peak["prominence"][good]
            qc_grp["peak_snr"][ie, j0:j1] = peak["peak_snr"][good]
            qc_grp["peak_index"][ie, j0:j1] = peak["peak_index"][good]
            qc_grp["peak_center_pixel"][ie, j0:j1] = peak["center_pixel"][good]
            qc_grp["source_shot_index"][ie, j0:j1] = local_indices[good]

            ml["nominal_energy"][ie, j0:j1] = info.section_energy_eV[ie]
            ml["section_energy_group"][ie, j0:j1] = nominal_group_energy
            ml["converted_energy"][ie, j0:j1] = shot_energy
            ml["converted_energy_group"][ie, j0:j1] = converted_group_energy
            ml["converted_proxy_bin_index"][ie, j0:j1] = proxy_good
            ml["gmd_hall_channels"][ie, j0:j1, :] = src["ml/gmd_hall_channels"][ie, local_indices[good], :]
            if "ml/gmd_tunnel_channels" in src and "gmd_tunnel_channels" in ml:
                ml["gmd_tunnel_channels"][ie, j0:j1, :] = src["ml/gmd_tunnel_channels"][ie, local_indices[good], :]
            ml["qc_ok"][ie, j0:j1] = True
            for name in ["train_id", "run_id", "section_index", "vls_bunch_index", "gmd_bunch_index"]:
                if f"ml/{name}" in src and name in ml:
                    ml[name][ie, j0:j1] = src[f"ml/{name}"][ie, local_indices[good]]
            write_pos[ie] = j1

        if not np.array_equal(write_pos, good_counts):
            raise RuntimeError(f"Written counts do not match QC counts: {write_pos} vs {good_counts}")

        out.attrs["mode"] = "xas_static_good"
        out.attrs["source_h5"] = str(info.path)
        out.attrs["run_no"] = int(info.attrs.get("run_no", -1))
        out.attrs["gmd_source"] = "ml/gmd_hall_channels[..., 0]"
        out.attrs["gmd_tunnel_source"] = "ml/gmd_tunnel_channels[..., 0]"
        out.attrs["processing_gmd_source"] = gmd_source
        out.attrs["processing_gmd_dataset"] = f"{processing_gmd_dataset}[..., 0]"
        out.attrs["description"] = (
            "Processed static-XAS shots passing finite/QC and weak-VLS peak-SNR filtering. "
            "Nominal and proxy-pixel converted energy-axis corrections are stored."
        )
        out.attrs["section_energy_source"] = info.section_energy_source
        out.attrs["energy_group_decimals"] = int(info.energy_group_decimals)
        out.attrs["proxy_pixel_bin_width"] = int(config.proxy_pixel_bin_width)
        for key in ["vls_crop_roi", "signal_bunch_range", "bg_bunch_range", "gmd_bunch_start", "gmd_signal_bunch_range"]:
            if key in info.attrs:
                out.attrs[key] = info.attrs[key]

    return path


def read_static_info(path: str | Path, *, group_decimals: int = 3) -> StaticInfo:
    """Read metadata and energy grouping from the processed static-XAS H5."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with h5py.File(p, "r") as h5:
        required = ["vls", "n_shots", "nominal_energies", "vls_pixels", "ml/gmd_hall_channels", "noise/vls_bg_sigma_global"]
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"Missing required datasets in {p}: {missing}")
        attrs = dict(h5.attrs)
        n_shots = h5["n_shots"][...].astype(np.int64)
        stored_energy = h5["nominal_energies"][...].astype(np.float64)
        if "section_energy_raw_eV" in h5:
            section_raw = h5["section_energy_raw_eV"][...].astype(np.float64)
        else:
            section_raw = stored_energy.copy()
        if "section_energy_source_values" in h5:
            source_values = h5["section_energy_source_values"][...].astype(np.float64)
        else:
            source_values = np.full_like(stored_energy, np.nan, dtype=np.float64)
        section_energy = np.where(np.isfinite(section_raw), section_raw, stored_energy)
        rounded = np.round(section_energy, int(group_decimals))
        finite = np.isfinite(rounded)
        group_energy = np.unique(rounded[finite])
        group_energy.sort()
        section_to_group = np.full(rounded.shape, -1, dtype=np.int32)
        for ig, energy in enumerate(group_energy):
            section_to_group[rounded == energy] = ig
        return StaticInfo(
            path=p,
            attrs=attrs,
            n_energy=int(stored_energy.shape[0]),
            n_pixels=int(h5["vls"].shape[2]),
            max_shots=int(h5["vls"].shape[1]),
            n_shots=n_shots,
            stored_energy_eV=stored_energy,
            section_energy_eV=section_energy,
            section_energy_raw_eV=section_raw,
            section_energy_source_values=source_values,
            section_energy_source=str(attrs.get("section_energy_source", "nominal_energies")),
            section_energy_source_unit=str(attrs.get("section_energy_source_unit", "eV")),
            section_energy_conversion=str(attrs.get("section_energy_conversion", "stored /nominal_energies")),
            energy_group_decimals=int(group_decimals),
            group_energies_eV=group_energy,
            section_to_group=section_to_group,
            vls_pixels=h5["vls_pixels"][...].astype(np.float64),
            sigma_raw=h5["noise/vls_bg_sigma_global"][...].astype(np.float64),
        )


def iter_energy_chunks(path: Path, info: StaticInfo, config: ProcessGoodConfig):
    """Yield VLS, selected GMD, and optional ML QC in chunks."""

    gmd_dataset = _gmd_channel_dataset_for_source(config.gmd_source)
    with h5py.File(path, "r") as h5:
        for ie in range(info.n_energy):
            n = int(info.n_shots[ie])
            for start in range(0, n, int(config.chunk_size)):
                stop = min(start + int(config.chunk_size), n)
                vls = h5["vls"][ie, start:stop, :].astype(np.float64)
                gmd = h5[gmd_dataset][ie, start:stop, 0].astype(np.float64)
                if "ml/qc_ok" in h5:
                    qc_ok = h5["ml/qc_ok"][ie, start:stop].astype(bool)
                else:
                    qc_ok = np.ones(stop - start, dtype=bool)
                yield ie, start, stop, vls, gmd, qc_ok


def basic_qc_mask(vls: np.ndarray, gmd: np.ndarray, qc_ok: np.ndarray, config: ProcessGoodConfig) -> np.ndarray:
    """Finite-data and optional upstream QC mask before weak-VLS filtering."""

    keep = np.isfinite(gmd) & np.all(np.isfinite(vls), axis=1)
    if config.use_ml_qc_ok and qc_ok is not None:
        keep &= qc_ok.astype(bool)
    return keep


def integrated_vls(vls: np.ndarray) -> np.ndarray:
    """Integrated VLS intensity per shot."""

    return np.nansum(np.asarray(vls, dtype=np.float64), axis=1)


def peak_metrics(spectra: np.ndarray, sigma_raw: np.ndarray, pixels: np.ndarray, config: ProcessGoodConfig) -> dict[str, np.ndarray]:
    """Compute peak prominence, raw-noise SNR, and local COM center."""

    spectra = np.asarray(spectra, dtype=np.float64)
    sigma_raw = np.asarray(sigma_raw, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    n, n_pix = spectra.shape
    peak_index = np.full(n, -1, dtype=np.int32)
    peak_height = np.full(n, np.nan, dtype=np.float64)
    local_bg = np.full(n, np.nan, dtype=np.float64)
    prominence = np.full(n, np.nan, dtype=np.float64)
    sigma_near_peak = np.full(n, np.nan, dtype=np.float64)
    peak_snr = np.full(n, np.nan, dtype=np.float64)
    center_pixel = np.full(n, np.nan, dtype=np.float64)

    inner = int(config.prominence_inner_px)
    outer = int(config.prominence_outer_px)
    min_side = int(config.min_side_pixels)
    for i, y in enumerate(spectra):
        if not np.all(np.isfinite(y)):
            continue
        p0 = int(np.nanargmax(y))
        left = y[max(0, p0 - outer) : max(0, p0 - inner)]
        right = y[min(n_pix, p0 + inner) : min(n_pix, p0 + outer)]
        if left.size < min_side or right.size < min_side:
            continue
        lb = np.nanpercentile(left, float(config.prominence_baseline_percentile))
        rb = np.nanpercentile(right, float(config.prominence_baseline_percentile))
        bg = max(lb, rb)
        s0 = max(0, p0 - int(config.sigma_half_window_px))
        s1 = min(n_pix, p0 + int(config.sigma_half_window_px) + 1)
        sig = np.nanmedian(sigma_raw[s0:s1])

        c0 = max(0, p0 - int(config.center_half_window_px))
        c1 = min(n_pix, p0 + int(config.center_half_window_px) + 1)
        yw = y[c0:c1]
        xw = pixels[c0:c1]
        baseline = np.nanpercentile(yw, 20)
        weights = np.clip(yw - baseline, 0.0, None)
        if np.sum(weights) > 0:
            center_pixel[i] = float(np.sum(xw * weights) / np.sum(weights))

        peak_index[i] = p0
        peak_height[i] = y[p0]
        local_bg[i] = bg
        prominence[i] = y[p0] - bg
        sigma_near_peak[i] = sig
        if np.isfinite(sig) and sig > 0:
            peak_snr[i] = prominence[i] / sig

    return {
        "peak_index": peak_index,
        "peak_height": peak_height,
        "local_bg": local_bg,
        "prominence": prominence,
        "sigma_near_peak": sigma_near_peak,
        "peak_snr": peak_snr,
        "center_pixel": center_pixel,
    }


def fit_vls_pixel_energy_calibration(
    info: StaticInfo,
    section_center_pixel: np.ndarray,
    section_center_n: np.ndarray,
    config: ProcessGoodConfig,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    """Fit converted energy from section median VLS center to section energy."""

    x = np.asarray(section_center_pixel, dtype=np.float64)
    y = np.asarray(info.section_energy_eV, dtype=np.float64)
    w = np.sqrt(np.maximum(np.asarray(section_center_n, dtype=np.float64), 1.0))
    ok = np.isfinite(x) & np.isfinite(y) & (section_center_n > 0)
    n_fit = int(np.sum(ok))
    if n_fit < int(config.energy_calibration_min_sections):
        raise RuntimeError(
            f"Need at least {config.energy_calibration_min_sections} section centers for energy calibration; got {n_fit}."
        )
    order = min(int(config.energy_calibration_poly_order), n_fit - 1)
    coeff = np.polyfit(x[ok], y[ok], deg=order, w=w[ok])
    fit_energy = np.full_like(y, np.nan, dtype=np.float64)
    fit_energy[ok] = np.polyval(coeff, x[ok])
    table = pd.DataFrame(
        {
            "section": np.arange(info.n_energy),
            "section_energy_eV": y,
            "center_pixel": x,
            "n_center": section_center_n,
            "fit_energy_eV": fit_energy,
            "residual_eV": y - fit_energy,
        }
    )
    return coeff, order, table


def new_fit_accumulator(n_groups: int) -> dict[str, np.ndarray]:
    """Create linear-fit sufficient statistics."""

    n = int(n_groups)
    return {
        "n": np.zeros(n, dtype=np.int64),
        "sum_x": np.zeros(n, dtype=np.float64),
        "sum_y": np.zeros(n, dtype=np.float64),
        "sum_xx": np.zeros(n, dtype=np.float64),
        "sum_yy": np.zeros(n, dtype=np.float64),
        "sum_xy": np.zeros(n, dtype=np.float64),
    }


def update_fit_accumulator_by_index(
    acc: dict[str, np.ndarray],
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    keep: np.ndarray,
) -> None:
    """Update grouped linear-fit accumulators."""

    idx = np.asarray(idx, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = keep & (idx >= 0) & (idx < acc["n"].size) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    ii = idx[valid]
    xx = x[valid]
    yy = y[valid]
    np.add.at(acc["n"], ii, 1)
    np.add.at(acc["sum_x"], ii, xx)
    np.add.at(acc["sum_y"], ii, yy)
    np.add.at(acc["sum_xx"], ii, xx * xx)
    np.add.at(acc["sum_yy"], ii, yy * yy)
    np.add.at(acc["sum_xy"], ii, xx * yy)


def finish_fit_accumulator(acc: dict[str, np.ndarray], x_eV: np.ndarray, *, proxy_group: Optional[np.ndarray] = None) -> FitTable:
    """Turn grouped linear-fit sufficient statistics into fit parameters."""

    n = acc["n"].astype(np.float64)
    denom = n * acc["sum_xx"] - acc["sum_x"] ** 2
    slope = np.full(n.shape, np.nan, dtype=np.float64)
    intercept = np.full(n.shape, np.nan, dtype=np.float64)
    r = np.full(n.shape, np.nan, dtype=np.float64)
    valid = (n > 1) & np.isfinite(denom) & (denom != 0.0)
    slope[valid] = (n[valid] * acc["sum_xy"][valid] - acc["sum_x"][valid] * acc["sum_y"][valid]) / denom[valid]
    intercept[valid] = (acc["sum_y"][valid] - slope[valid] * acc["sum_x"][valid]) / n[valid]
    r_den = np.sqrt((n * acc["sum_xx"] - acc["sum_x"] ** 2) * (n * acc["sum_yy"] - acc["sum_y"] ** 2))
    r_valid = (n > 1) & np.isfinite(r_den) & (r_den > 0.0)
    r[r_valid] = (n[r_valid] * acc["sum_xy"][r_valid] - acc["sum_x"][r_valid] * acc["sum_y"][r_valid]) / r_den[r_valid]
    return FitTable(
        x_eV=np.asarray(x_eV, dtype=np.float64),
        n=acc["n"].astype(np.int64),
        intercept=intercept,
        slope=slope,
        r=r,
        r2=r * r,
        proxy_group=None if proxy_group is None else np.asarray(proxy_group, dtype=np.int64),
    )


def subset_fit_table(table: FitTable, rows: np.ndarray) -> FitTable:
    """Select rows from a FitTable."""

    rows = np.asarray(rows, dtype=np.int64)
    return FitTable(
        x_eV=table.x_eV[rows],
        n=table.n[rows],
        intercept=table.intercept[rows],
        slope=table.slope[rows],
        r=table.r[rows],
        r2=table.r2[rows],
        proxy_group=None if table.proxy_group is None else table.proxy_group[rows],
    )


def proxy_groups_from_center_pixel(center_pixel: np.ndarray, vls_pixels: np.ndarray, width: int) -> np.ndarray:
    """Assign center pixels to relative proxy-pixel bins."""

    rel = center_pixel_to_relative(center_pixel, vls_pixels)
    out = np.full(rel.shape, -1, dtype=np.int64)
    valid = np.isfinite(rel)
    if not np.any(valid):
        return out
    n_pix = int(np.asarray(vls_pixels).size)
    pix = np.clip(np.round(rel[valid]).astype(np.int64), 0, n_pix - 1)
    out[valid] = pix // int(width)
    return out


def center_pixel_to_relative(center_pixel: np.ndarray, vls_pixels: np.ndarray) -> np.ndarray:
    """Map absolute VLS pixel coordinates back to relative ROI index."""

    pixels = np.asarray(vls_pixels, dtype=np.float64)
    rel = np.arange(pixels.size, dtype=np.float64)
    if pixels.size == 0:
        return np.full(np.shape(center_pixel), np.nan, dtype=np.float64)
    if pixels[0] <= pixels[-1]:
        return np.interp(center_pixel, pixels, rel, left=np.nan, right=np.nan)
    return np.interp(center_pixel, pixels[::-1], rel[::-1], left=np.nan, right=np.nan)


def print_processed_qc_summary(result: ProcessGoodResult) -> None:
    """Print a compact summary of the processed-QC result."""

    info = result.info
    print(f"input H5            : {info.path}")
    print(
        "GMD source          : "
        f"{_normalize_gmd_source(result.config.gmd_source)} "
        f"({_gmd_channel_dataset_for_source(result.config.gmd_source)}[..., 0])"
    )
    print(f"sections            : {info.n_energy}")
    print(f"energy source       : {info.section_energy_source} [{info.section_energy_source_unit}]")
    print(f"nominal bins        : {result.fit_nominal_after.x_eV.size:,}")
    print(f"converted bins      : {result.fit_converted_after.x_eV.size:,}")
    print(f"proxy pixel width   : {result.config.proxy_pixel_bin_width}")
    print(f"recorded shots      : {int(np.sum(result.counts_df['recorded'])):,}")
    print(f"basic shots         : {int(np.sum(result.counts_df['basic'])):,}")
    print(f"peak-filter kept    : {int(np.sum(result.counts_df['peak_keep'])):,}")


def plot_peak_filter_corr_summary(result: ProcessGoodResult, *, xlim: Optional[tuple[float, float]] = None):
    """Plot concise counts and before/after correlation diagnostics."""

    import matplotlib.pyplot as plt

    counts = result.counts_df
    nb = result.fit_nominal_before.dataframe()
    na = result.fit_nominal_after.dataframe()
    cb = result.fit_converted_before.dataframe()
    ca = result.fit_converted_after.dataframe()

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), constrained_layout=True)
    axes[0].plot(counts["section_energy_eV"], counts["basic"], "o-", label="before peak filter")
    axes[0].plot(counts["section_energy_eV"], counts["peak_keep"], "o-", label="after peak filter")
    axes[0].set_ylabel("shots")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(nb["energy_eV"], nb["r2"], "o--", ms=4, alpha=0.65, label="nominal before")
    axes[1].plot(na["energy_eV"], na["r2"], "o-", ms=4, label="nominal after")
    axes[1].scatter(cb["energy_eV"], cb["r2"], s=10, alpha=0.35, label="converted before")
    axes[1].scatter(ca["energy_eV"], ca["r2"], s=10, alpha=0.75, label="converted after")
    axes[1].set_xlabel("photon energy (eV)")
    axes[1].set_ylabel("R² of integrated VLS vs GMD")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.25)
    if xlim is not None:
        axes[1].set_xlim(*xlim)
    return fig, axes


def plot_energy_calibration_summary(result: ProcessGoodResult):
    """Plot compact VLS pixel-to-energy calibration diagnostics."""

    import matplotlib.pyplot as plt

    table = result.energy_calibration_table
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    axes[0].plot(table["center_pixel"], table["section_energy_eV"], "o", label="section median")
    finite = np.isfinite(table["center_pixel"].to_numpy(dtype=np.float64))
    if np.any(finite):
        xx = np.linspace(float(np.nanmin(table.loc[finite, "center_pixel"])), float(np.nanmax(table.loc[finite, "center_pixel"])), 250)
        axes[0].plot(xx, np.polyval(result.energy_calibration_coeff, xx), "-", label="fit")
    axes[0].set_xlabel("VLS peak COM pixel")
    axes[0].set_ylabel("section energy (eV)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color="0.25", lw=0.9, ls=":")
    axes[1].plot(table["section_energy_eV"], table["residual_eV"], "o-")
    axes[1].set_xlabel("section energy (eV)")
    axes[1].set_ylabel("fit residual (eV)")
    axes[1].grid(alpha=0.25)
    return fig, axes


def plot_single_energy_correlation_colormap(
    result: ProcessGoodResult,
    *,
    energy_eV: float = 273.0,
    axis: str = "nominal",
    stage: str = "after_peak_filter",
    gridsize: int = 75,
    cmap: str = "viridis",
    mincnt: int = 1,
):
    """Plot one-energy GMD/VLS correlation density with a fitted line."""

    import matplotlib.pyplot as plt

    corr = extract_single_energy_correlation(result, energy_eV=energy_eV, axis=axis, stage=stage)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    if corr["gmd"].size == 0:
        ax.text(0.5, 0.5, "No shots in selected energy bin", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("GMD (uJ)")
        ax.set_ylabel("integrated VLS")
        ax.set_title(corr["title"])
        ax.grid(alpha=0.2)
        return fig, ax

    hb = ax.hexbin(corr["gmd"], corr["trans"], gridsize=int(gridsize), bins="log", mincnt=int(mincnt), cmap=cmap)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    fit = corr["fit"]
    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        x = np.linspace(float(np.nanmin(corr["gmd"])), float(np.nanmax(corr["gmd"])), 250)
        ax.plot(x, fit["intercept"] + fit["slope"] * x, color="tab:red", lw=2.0, label="linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("GMD (uJ)")
    ax.set_ylabel("integrated VLS")
    ax.set_title(f"{corr['title']}: N={int(fit['n']):,}, R²={fit['r2']:.3f}")
    ax.grid(alpha=0.2)
    return fig, ax


def extract_single_energy_correlation(
    result: ProcessGoodResult,
    *,
    energy_eV: float = 273.0,
    axis: str = "nominal",
    stage: str = "after_peak_filter",
) -> dict[str, Any]:
    """Extract GMD and integrated VLS shots for one displayed energy bin."""

    axis = str(axis)
    stage = str(stage)
    if stage not in {"before_peak_filter", "after_peak_filter"}:
        raise ValueError("stage must be 'before_peak_filter' or 'after_peak_filter'")

    gmd_parts: list[np.ndarray] = []
    trans_parts: list[np.ndarray] = []
    info = result.info
    cfg = result.config

    if axis == "nominal":
        if result.fit_nominal_after.x_eV.size == 0:
            selected_group = -1
            selected_energy = np.nan
        else:
            selected_group = int(np.nanargmin(np.abs(result.fit_nominal_after.x_eV - float(energy_eV))))
            selected_energy = float(result.fit_nominal_after.x_eV[selected_group])
        selected_sections = np.where(info.section_to_group == selected_group)[0]
        for ie, _start, _stop, vls, gmd, qc_ok in iter_energy_chunks(info.path, info, cfg):
            if ie not in selected_sections:
                continue
            basic = basic_qc_mask(vls, gmd, qc_ok, cfg)
            if stage == "after_peak_filter":
                peak = peak_metrics(vls, info.sigma_raw, info.vls_pixels, cfg)
                keep = basic & np.isfinite(peak["center_pixel"]) & (peak["peak_snr"] > float(cfg.peak_snr_threshold))
            else:
                keep = basic
            if np.any(keep):
                gmd_parts.append(gmd[keep].astype(np.float64))
                trans_parts.append(integrated_vls(vls[keep]))
        label = f"nominal {selected_energy:.3f} eV, {stage.replace('_', ' ')}"

    elif axis == "converted":
        occupied = result.occupied_proxy_groups
        energies = result.proxy_energy_eV[occupied]
        if energies.size == 0:
            selected_proxy = -1
            selected_energy = np.nan
        else:
            row = int(np.nanargmin(np.abs(energies - float(energy_eV))))
            selected_proxy = int(occupied[row])
            selected_energy = float(energies[row])
        for _ie, _start, _stop, vls, gmd, qc_ok in iter_energy_chunks(info.path, info, cfg):
            basic = basic_qc_mask(vls, gmd, qc_ok, cfg)
            peak = peak_metrics(vls, info.sigma_raw, info.vls_pixels, cfg)
            proxy_group = proxy_groups_from_center_pixel(peak["center_pixel"], info.vls_pixels, cfg.proxy_pixel_bin_width)
            center_valid = basic & np.isfinite(peak["center_pixel"]) & (proxy_group == selected_proxy)
            if stage == "after_peak_filter":
                keep = center_valid & (peak["peak_snr"] > float(cfg.peak_snr_threshold))
            else:
                keep = center_valid
            if np.any(keep):
                gmd_parts.append(gmd[keep].astype(np.float64))
                trans_parts.append(integrated_vls(vls[keep]))
        label = f"converted {selected_energy:.3f} eV proxy bin, {stage.replace('_', ' ')}"
    else:
        raise ValueError("axis must be 'nominal' or 'converted'")

    gmd_arr = np.concatenate(gmd_parts) if gmd_parts else np.empty(0, dtype=np.float64)
    trans_arr = np.concatenate(trans_parts) if trans_parts else np.empty(0, dtype=np.float64)
    return {
        "axis": axis,
        "stage": stage,
        "requested_energy_eV": float(energy_eV),
        "selected_energy_eV": selected_energy,
        "gmd": gmd_arr,
        "trans": trans_arr,
        "fit": fit_line(gmd_arr, trans_arr),
        "title": label,
    }


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y = intercept + slope * x and return compact diagnostics."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    out = {"n": float(x.size), "intercept": np.nan, "slope": np.nan, "r": np.nan, "r2": np.nan}
    if x.size < 2:
        return out
    n = float(x.size)
    sx = float(np.sum(x))
    sy = float(np.sum(y))
    sxx = float(np.sum(x * x))
    syy = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return out
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    r_den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    r = np.nan if r_den <= 0.0 else (n * sxy - sx * sy) / r_den
    return {"n": float(x.size), "intercept": float(intercept), "slope": float(slope), "r": float(r), "r2": float(r * r)}


def write_energy_group(out: h5py.File, info: StaticInfo, result: ProcessGoodResult) -> None:
    energy = out.create_group("energy")
    energy.create_dataset("section_energy_eV", data=info.section_energy_eV)
    energy.create_dataset("stored_nominal_energy_eV", data=info.stored_energy_eV)
    energy.create_dataset("section_energy_raw_eV", data=info.section_energy_raw_eV)
    energy.create_dataset("section_energy_source_values", data=info.section_energy_source_values)
    energy.create_dataset("group_energy_eV", data=info.group_energies_eV)
    energy.create_dataset("nominal_group_energy_eV", data=info.group_energies_eV)
    energy.create_dataset("section_to_group", data=info.section_to_group.astype(np.int32))
    energy.create_dataset("section_to_nominal_group", data=info.section_to_group.astype(np.int32))
    energy.create_dataset("converted_proxy_group", data=result.occupied_proxy_groups.astype(np.int64))
    energy.create_dataset("converted_proxy_group_energy_eV", data=result.proxy_energy_eV[result.occupied_proxy_groups])
    energy.create_dataset("vls_pixel_to_energy_coefficients", data=result.energy_calibration_coeff)
    energy.create_dataset("section_center_pixel", data=result.section_center_pixel)
    energy.create_dataset("section_center_n", data=result.section_center_n)
    energy.create_dataset("section_center_fit_energy_eV", data=result.energy_calibration_table["fit_energy_eV"].to_numpy(dtype=np.float64))
    energy.create_dataset("section_center_residual_eV", data=result.energy_calibration_table["residual_eV"].to_numpy(dtype=np.float64))
    energy.attrs["section_energy_source"] = info.section_energy_source
    energy.attrs["section_energy_source_unit"] = info.section_energy_source_unit
    energy.attrs["section_energy_conversion"] = info.section_energy_conversion
    energy.attrs["group_decimals"] = int(info.energy_group_decimals)
    energy.attrs["proxy_pixel_bin_width"] = int(result.config.proxy_pixel_bin_width)
    energy.attrs["vls_calibration_order"] = int(result.energy_calibration_order)
    energy.attrs["vls_calibration_x"] = "peak COM pixel after weak-VLS peak-SNR filter"
    energy.attrs["vls_calibration_y"] = "section_energy_eV"
    energy.attrs["converted_grouping"] = "shot-level VLS-COM proxy-pixel bins"


def write_qc_group(out: h5py.File, result: ProcessGoodResult, max_good: int) -> h5py.Group:
    info = result.info
    cfg = result.config
    qc = out.create_group("qc")
    qc.create_dataset("peak_prominence", shape=(info.n_energy, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    qc.create_dataset("peak_snr", shape=(info.n_energy, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    qc.create_dataset("peak_index", shape=(info.n_energy, max_good), dtype=np.int32, fillvalue=-1, compression="gzip")
    qc.create_dataset("peak_center_pixel", shape=(info.n_energy, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    qc.create_dataset("source_shot_index", shape=(info.n_energy, max_good), dtype=np.int64, fillvalue=-1, compression="gzip")
    str_dtype = h5py.string_dtype(encoding="utf-8")
    qc.create_dataset("counts_table", data=result.counts_df.to_numpy())
    qc.create_dataset("counts_columns", data=np.asarray(result.counts_df.columns, dtype=object), dtype=str_dtype)
    qc.create_dataset("converted_counts_table", data=result.converted_counts_df.to_numpy())
    qc.create_dataset("converted_counts_columns", data=np.asarray(result.converted_counts_df.columns, dtype=object), dtype=str_dtype)
    qc.attrs["peak_snr_threshold"] = float(cfg.peak_snr_threshold)
    qc.attrs["prominence_inner_px"] = int(cfg.prominence_inner_px)
    qc.attrs["prominence_outer_px"] = int(cfg.prominence_outer_px)
    qc.attrs["prominence_baseline_percentile"] = float(cfg.prominence_baseline_percentile)
    qc.attrs["center_half_window_px"] = int(cfg.center_half_window_px)
    return qc


def write_fits_group(out: h5py.File, result: ProcessGoodResult, section_intercept: np.ndarray, section_slope: np.ndarray) -> None:
    fits = out.create_group("fits")
    write_fit_table(fits, "nominal_before_peak_filter", result.fit_nominal_before)
    nominal_grp = write_fit_table(fits, "nominal", result.fit_nominal_after)
    write_fit_table(fits, "converted_before_peak_filter", result.fit_converted_before)
    write_fit_table(fits, "converted", result.fit_converted_after)
    nominal_after = result.fit_nominal_after
    fits.create_dataset("group_energy_eV", data=nominal_after.x_eV)
    for name in ["n", "intercept", "slope", "r", "r2"]:
        fits.create_dataset(name, data=getattr(nominal_after, name))
    fits.create_dataset("section_intercept", data=section_intercept)
    fits.create_dataset("section_slope", data=section_slope)
    nominal_grp.create_dataset("section_intercept", data=section_intercept)
    nominal_grp.create_dataset("section_slope", data=section_slope)
    fits.attrs["model"] = "integrated VLS = intercept(E_group) + slope(E_group) * selected GMD"
    fits.attrs["gmd_source"] = _normalize_gmd_source(result.config.gmd_source)
    fits.attrs["gmd_dataset"] = f"{_gmd_channel_dataset_for_source(result.config.gmd_source)}[..., 0]"
    fits.attrs["nominal_intercept_correction"] = "vls_corrected_nominal = vls - intercept(section_energy_group) / n_pixels"
    fits.attrs["converted_intercept_correction"] = "vls_corrected_converted = vls - intercept(converted_proxy_pixel_bin) / n_pixels"


def write_fit_table(parent: h5py.Group, name: str, table: FitTable) -> h5py.Group:
    grp = parent.create_group(name)
    grp.create_dataset("group_energy_eV", data=table.x_eV)
    if table.proxy_group is not None:
        grp.create_dataset("proxy_group", data=table.proxy_group)
    for column in ["n", "intercept", "slope", "r", "r2"]:
        grp.create_dataset(column, data=getattr(table, column))
    return grp


def create_ml_group(out: h5py.File, src: h5py.File, n_sections: int, max_good: int) -> h5py.Group:
    ml = out.create_group("ml")
    ml.create_dataset("nominal_energy", shape=(n_sections, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    ml.create_dataset("section_energy_group", shape=(n_sections, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    ml.create_dataset("converted_energy", shape=(n_sections, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    ml.create_dataset("converted_energy_group", shape=(n_sections, max_good), dtype=np.float64, fillvalue=np.nan, compression="gzip")
    ml.create_dataset("converted_proxy_bin_index", shape=(n_sections, max_good), dtype=np.int32, fillvalue=-1, compression="gzip")
    ml.create_dataset("gmd_hall_channels", shape=(n_sections, max_good, 8), dtype=np.float32, fillvalue=np.nan, compression="gzip")
    if "ml/gmd_tunnel_channels" in src:
        ml.create_dataset("gmd_tunnel_channels", shape=(n_sections, max_good, 8), dtype=np.float32, fillvalue=np.nan, compression="gzip")
    ml.create_dataset("qc_ok", shape=(n_sections, max_good), dtype=bool, fillvalue=False, compression="gzip")
    for name in ["train_id", "run_id", "section_index", "vls_bunch_index", "gmd_bunch_index"]:
        if f"ml/{name}" in src:
            ml.create_dataset(name, shape=(n_sections, max_good), dtype=src[f"ml/{name}"].dtype, fillvalue=-1, compression="gzip")
    return ml


def _summarize_section_centers(center_by_section: list[list[np.ndarray]], n_sections: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.full(n_sections, np.nan, dtype=np.float64)
    n = np.zeros(n_sections, dtype=np.int64)
    p16 = np.full(n_sections, np.nan, dtype=np.float64)
    p84 = np.full(n_sections, np.nan, dtype=np.float64)
    for ie, chunks in enumerate(center_by_section):
        if not chunks:
            continue
        vals = np.concatenate(chunks)
        vals = vals[np.isfinite(vals)]
        n[ie] = vals.size
        if vals.size:
            center[ie] = float(np.nanmedian(vals))
            p16[ie] = float(np.nanpercentile(vals, 16))
            p84[ie] = float(np.nanpercentile(vals, 84))
    return center, n, p16, p84


def _add_counts(counts: np.ndarray, idx: np.ndarray, keep: np.ndarray) -> None:
    valid = keep & (idx >= 0) & (idx < counts.size)
    if np.any(valid):
        np.add.at(counts, idx[valid], 1)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den != 0))
    return out


def _default_output_path(config: ProcessGoodConfig) -> Path:
    if config.output_h5 is not None:
        return Path(config.output_h5)
    source = _normalize_gmd_source(config.gmd_source)
    input_path = Path(config.input_h5)
    name = input_path.name.replace("_static_xas.h5", "_good_static_xas.h5")
    if source == "tunnel":
        if name.endswith(".h5"):
            name = f"{name[:-3]}_tunnel.h5"
        else:
            name = f"{name}_tunnel"
    return input_path.with_name(name)


def _validate_config(config: ProcessGoodConfig) -> None:
    _normalize_gmd_source(config.gmd_source)
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if int(config.proxy_pixel_bin_width) <= 0:
        raise ValueError("proxy_pixel_bin_width must be positive")
    if float(config.peak_snr_threshold) <= 0:
        raise ValueError("peak_snr_threshold must be positive")
    if int(config.prominence_outer_px) <= int(config.prominence_inner_px):
        raise ValueError("prominence_outer_px must be larger than prominence_inner_px")


def _normalize_gmd_source(gmd_source: str) -> str:
    source = str(gmd_source).strip().lower()
    if source not in GMD_SOURCE_TO_CHANNEL_DATASET:
        allowed = ", ".join(sorted(GMD_SOURCE_TO_CHANNEL_DATASET))
        raise ValueError(f"gmd_source must be one of {{{allowed}}}; got {gmd_source!r}")
    return source


def _gmd_channel_dataset_for_source(gmd_source: str) -> str:
    return GMD_SOURCE_TO_CHANNEL_DATASET[_normalize_gmd_source(gmd_source)]


def _gmd_top_dataset_for_source(gmd_source: str) -> str:
    return GMD_SOURCE_TO_TOP_DATASET[_normalize_gmd_source(gmd_source)]


def _validate_gmd_source_available(path: Path, config: ProcessGoodConfig) -> None:
    source = _normalize_gmd_source(config.gmd_source)
    dataset = _gmd_channel_dataset_for_source(source)
    with h5py.File(path, "r") as h5:
        if dataset in h5:
            return
    message = f"{path} is missing required GMD dataset {dataset!r} for gmd_source={source!r}"
    if source == "tunnel":
        message += (
            ". The raw-to-processed run must contain ml/gmd_tunnel_channels; "
            "if the source run*_static_xas.h5 already has tunnel GMD, re-run only "
            "the processed-to-good step with PROCESS_GMD_SOURCE='tunnel'."
        )
    raise KeyError(message)


def _read_gmd_values_for_output(
    h5: h5py.File,
    gmd_source: str,
    section_index: int,
    start: int,
    stop: int,
    keep: np.ndarray,
) -> np.ndarray:
    source = _normalize_gmd_source(gmd_source)
    channel_dataset = _gmd_channel_dataset_for_source(source)
    if channel_dataset in h5:
        return h5[channel_dataset][section_index, start:stop, 0].astype(np.float64)[keep]
    top_dataset = _gmd_top_dataset_for_source(source)
    if top_dataset in h5:
        return h5[top_dataset][section_index, start:stop].astype(np.float64)[keep]
    return np.full(int(np.sum(keep)), np.nan, dtype=np.float64)
