"""
Collect per-shot (VLS, GMD) for a photon-energy XAS scan.

Runs the same pipeline as ``compute_xas_aggregates.py`` — VLS pixel
crop, VLS bunch-axis roll to align with GMD, per-train background
subtraction, fast-shutter section detection, per-section closed-shutter
background subtraction, signal-bunch selection — but stops after
flattening (train x bunch -> shot).  No GMD binning is performed; the
raw shot-level data is written for flexible downstream analysis and
visualisation.

Output H5 layout (default: <processed/xas_static>/run<N>_static_xas.h5):
    /vls               (N_E, N_shots, n_pixels)  float64, NaN-padded
    /gmd               (N_E, N_shots)            float64, NaN-padded, mapped hall ch0
    /gmd_tunnel        (N_E, N_shots)            float64, NaN-padded, mapped tunnel ch0
    /n_shots           (N_E,)                    int64
    /nominal_energies  (N_E,)                    float64  eV
    /vls_pixels        (n_pixels,)               int64    source-pixel indices
    /section_bg        (N_E, m_bunches, n_pixels) float64 per-section bg
    /noise/*           global raw-background VLS noise, one value per VLS pixel
    /ml/*              compact per-shot identity, GMD-channel, feature,
                       and QC datasets aligned to the /vls flatten order
    attrs:  mode='xas_static', config, run_no, vls_crop_roi,
            vls_bunch_roll, signal_bunch_range, bg_bunch_range,
            n_sections_detected, n_sections_used,
            transition_trim_trains, ...

Config file
-----------
Python module exposing: RUN_NO, CROP_ROI, SIGNAL_BUNCH_RANGE,
BG_BUNCH_RANGE, CONFIG=2, plus either NOMINAL_ENERGIES or
SECTION_ENERGY_SOURCE. No GMD_EDGES needed. ``MODE`` may be any of
``"xas"``, ``"xas_scan"``, or ``"xas_static"``. The same config file
also drives ``compute_xas_aggregates.py``; that script consumes the
extra GMD-binning fields when present. See ``analysis/configs/xas_static/``
and ``analysis/configs/aggregates_example_xas.py``.

CLI
---
    python compute_static_xas.py CONFIG.py [-o OUT.h5]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

import data_loading

__all__ = ["compute_static_xas", "main"]

# ---------------------------------------------------------------------------
# Shutter HDF5 path defaults
# ---------------------------------------------------------------------------

_DEFAULT_SHUTTER_INDEX_PATH = "/FL2/Beamlines/Fast Shutter/shutter/index"
_DEFAULT_SHUTTER_VALUE_PATH = "/FL2/Beamlines/Fast Shutter/shutter/value"
_MAD_TO_SIGMA_NORMAL = 1.4826
_HC_EV_NM = 1239.8419843320026

_GMD_TUNNEL_INDEX_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/index",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/tunnel/index",
)
_GMD_TUNNEL_VALUE_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/value",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/tunnel/value",
)
_GMD_HALL_INDEX_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/hall/index",
)
_GMD_HALL_VALUE_CANDIDATES = (
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value",
    "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/hall/value",
)

_UNDULATOR_K_PATHS = [
    f"/Electron Diagnostic/Undulator setting/SASE{i:02d} k value"
    for i in range(2, 14)
]
_UNDULATOR_GAP_PATH_CANDIDATES = {
    i: (
        f"/Electron Diagnostic/Undulator setting/SASE{i:02d} gap",
        f"/FL2/Electron Diagnostic/Undulator setting/SASE{i:02d} gap",
    )
    for i in range(2, 14)
}
_SET_WAVELENGTH_1 = "/Electron Diagnostic/Undulator setting/set wavelength 1"
_SET_WAVELENGTH_2 = "/Electron Diagnostic/Undulator setting/set wavelength 2"
_GAP_ERROR = "/Electron Diagnostic/Undulator setting/gap error"
_ATTENUATOR_PRESSURE = "/FL2/Beamlines/Attenuator/pressure"
_OPIS_MEAN_PHOTON_ENERGY = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy"
_OPIS_MEAN_WAVELENGTH = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean wavelength"
_OPIS_NUMBER_ANALYSED_BUNCH = (
    "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/number of analysed bunch"
)

_SLOW_SCALAR_FEATURE_PATHS = {
    "attenuator_gas_type": "/FL2/Beamlines/Attenuator/gas type",
    "filter_position_1": "/FL2/Beamlines/Filters/position filter 1",
    "filter_position_2": "/FL2/Beamlines/Filters/position filter 2",
    "filter_fundamental_transmission_wheel_1": (
        "/FL2/Beamlines/Filters/fundamental transmission filter wheel 1"
    ),
    "filter_fundamental_transmission_wheel_2": (
        "/FL2/Beamlines/Filters/fundamental transmission filter wheel 2"
    ),
    "filter_third_harmonic_transmission_wheel_1": (
        "/FL2/Beamlines/Filters/3rd harmonic transmission filter wheel 1"
    ),
    "filter_third_harmonic_transmission_wheel_2": (
        "/FL2/Beamlines/Filters/3rd harmonic transmission filter wheel 2"
    ),
    "tunnel_aperture1_horizontal": (
        "/FL2/Beamlines/Tunnel Apertures/position aperture1 horizontal"
    ),
    "tunnel_aperture1_vertical": (
        "/FL2/Beamlines/Tunnel Apertures/position aperture1 vertical"
    ),
    "tunnel_aperture2_horizontal": (
        "/FL2/Beamlines/Tunnel Apertures/position aperture2 horizontal"
    ),
    "tunnel_aperture2_vertical": (
        "/FL2/Beamlines/Tunnel Apertures/position aperture2 vertical"
    ),
    "hall_aperture3_horizontal": (
        "/FL2/Beamlines/Hall Apertures/position aperture3 horizontal"
    ),
    "hall_aperture3_vertical": (
        "/FL2/Beamlines/Hall Apertures/position aperture3 vertical"
    ),
    "hall_aperture4_horizontal": (
        "/FL2/Beamlines/Hall Apertures/position aperture4 horizontal"
    ),
    "hall_aperture4_vertical": (
        "/FL2/Beamlines/Hall Apertures/position aperture4 vertical"
    ),
    "hall_aperture5_horizontal": (
        "/FL2/Beamlines/Hall Apertures/position aperture5 horizontal"
    ),
    "hall_aperture5_vertical": (
        "/FL2/Beamlines/Hall Apertures/position aperture5 vertical"
    ),
    "hall_aperture6_horizontal": (
        "/FL2/Beamlines/Hall Apertures/position aperture6 horizontal"
    ),
    "hall_aperture6_vertical": (
        "/FL2/Beamlines/Hall Apertures/position aperture6 vertical"
    ),
    "tunnel_mirror1_horizontal": "/FL2/Beamlines/Tunnel Mirrors/position Mirror1 horizontal",
    "tunnel_mirror1_vertical": "/FL2/Beamlines/Tunnel Mirrors/position Mirror1 vertical",
    "tunnel_mirror1_roll": "/FL2/Beamlines/Tunnel Mirrors/position Mirror1 roll",
    "tunnel_mirror1_rotation": "/FL2/Beamlines/Tunnel Mirrors/position Mirror1 rotation",
    "tunnel_mirror2_horizontal": "/FL2/Beamlines/Tunnel Mirrors/position Mirror2 horizontal",
    "tunnel_mirror2_vertical": "/FL2/Beamlines/Tunnel Mirrors/position Mirror2 vertical",
    "tunnel_mirror2_roll": "/FL2/Beamlines/Tunnel Mirrors/position Mirror2 roll",
    "tunnel_mirror2_rotation": "/FL2/Beamlines/Tunnel Mirrors/position Mirror2 rotation",
    "hall_mirror_fl20m3_horizontal": (
        "/FL2/Beamlines/Hall Mirrors/position FL20M3 horizontal"
    ),
    "hall_mirror_fl20m3_vertical": (
        "/FL2/Beamlines/Hall Mirrors/position FL20M3 vertical"
    ),
    "hall_mirror_fl20m3_rotation": (
        "/FL2/Beamlines/Hall Mirrors/position  FL20M3 rotation"
    ),
    "hall_mirror_kaos_premirror_vertical": (
        "/FL2/Beamlines/Hall Mirrors/position KAOS premirror vertical"
    ),
    "fl26_pgas1_bl_8_3_pressure_mbar": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/"
        "OPCUA.fbPGas1_BL_8_3.fPressureMBar"
    ),
    "fl26_pgas2_bl_8_3_pressure_mbar": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/"
        "OPCUA.fbPGas2_BL_8_3.fPressureMBar"
    ),
    "fl26_bl5_1_pressure": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BL5/"
        "OPCUA.stUHVG_BL_5_1.fPressure"
    ),
    "fl26_hg0_pressure_mbar": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.HG0/"
        "OPCUA.fbUHVG_HG_0_1.fPressureMBar"
    ),
    "fl26_js0_pressure_mbar": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.JS0/"
        "OPCUA.fbUHVG_JS_0_1.fPressureMBar"
    ),
    "fl26_js3_1_pressure": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.JS3/"
        "OPCUA.stUHVG_JS_3_1.fPressure"
    ),
    "fl26_valve_bl1_closed": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/"
        "OPCUA.fbV_BL_1.bClosed"
    ),
    "fl26_valve_bl1_open": (
        "/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/"
        "OPCUA.fbV_BL_1.bOpen"
    ),
}

# Edit this list to control which compact ML features are written into
# processed H5 /ml/features. Removing a name here means the feature will not be
# available for local ML training unless compute_static_xas.py is rerun.
ML_FEATURE_SPECS = [
    {"name": "gmd_hall_ch0_intensity", "unit": "au", "role": "predictor"},
    {"name": "gmd_hall_ch2_x", "unit": "mm", "role": "predictor"},
    {"name": "gmd_hall_ch3_y", "unit": "mm", "role": "predictor"},
    {"name": "gmd_hall_ch4_intensity_sigma", "unit": "au", "role": "predictor"},
    {"name": "gmd_tunnel_ch0_intensity", "unit": "au", "role": "predictor"},
    {"name": "gmd_tunnel_ch2_x", "unit": "mm", "role": "predictor"},
    {"name": "gmd_tunnel_ch3_y", "unit": "mm", "role": "predictor"},
    {"name": "gmd_tunnel_ch4_intensity_sigma", "unit": "au", "role": "predictor"},
    {"name": "undulator_gap_mean", "unit": "mm", "role": "candidate"},
    {"name": "undulator_gap_std", "unit": "mm", "role": "candidate"},
    {"name": "undulator_gap_slope", "unit": "mm/segment", "role": "candidate"},
    {"name": "undulator_k_mean", "unit": "", "role": "predictor"},
    {"name": "undulator_k_std", "unit": "", "role": "predictor"},
    {"name": "undulator_k_slope", "unit": "", "role": "predictor"},
    {"name": "attenuator_pressure", "unit": "mbar", "role": "predictor"},
    {"name": "attenuator_gas_type", "unit": "code", "role": "candidate"},
    {"name": "opis_mean_photon_energy", "unit": "eV", "role": "predictor"},
    {"name": "opis_mean_wavelength", "unit": "nm", "role": "candidate"},
    {"name": "opis_number_analysed_bunch", "unit": "count", "role": "qc"},
    {"name": "nominal_energy", "unit": "eV", "role": "leakage_check"},
    {"name": "set_wavelength_1", "unit": "nm", "role": "leakage_check"},
    {"name": "set_wavelength_2", "unit": "nm", "role": "leakage_check"},
    {"name": "filter_position_1", "unit": "code", "role": "candidate"},
    {"name": "filter_position_2", "unit": "code", "role": "candidate"},
    {"name": "filter_fundamental_transmission_wheel_1", "unit": "code", "role": "candidate"},
    {"name": "filter_fundamental_transmission_wheel_2", "unit": "code", "role": "candidate"},
    {"name": "filter_third_harmonic_transmission_wheel_1", "unit": "code", "role": "candidate"},
    {"name": "filter_third_harmonic_transmission_wheel_2", "unit": "code", "role": "candidate"},
    {"name": "tunnel_aperture1_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_aperture1_vertical", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_aperture2_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_aperture2_vertical", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture3_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture3_vertical", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture4_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture4_vertical", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture5_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture5_vertical", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture6_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "hall_aperture6_vertical", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_mirror1_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_mirror1_vertical", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_mirror1_roll", "unit": "mrad", "role": "candidate"},
    {"name": "tunnel_mirror1_rotation", "unit": "mrad", "role": "candidate"},
    {"name": "tunnel_mirror2_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_mirror2_vertical", "unit": "mm", "role": "candidate"},
    {"name": "tunnel_mirror2_roll", "unit": "mrad", "role": "candidate"},
    {"name": "tunnel_mirror2_rotation", "unit": "mrad", "role": "candidate"},
    {"name": "hall_mirror_fl20m3_horizontal", "unit": "mm", "role": "candidate"},
    {"name": "hall_mirror_fl20m3_vertical", "unit": "mm", "role": "candidate"},
    {"name": "hall_mirror_fl20m3_rotation", "unit": "mrad", "role": "candidate"},
    {"name": "hall_mirror_kaos_premirror_vertical", "unit": "raw", "role": "candidate"},
    {"name": "fl26_pgas1_bl_8_3_pressure_mbar", "unit": "mbar", "role": "candidate"},
    {"name": "fl26_pgas2_bl_8_3_pressure_mbar", "unit": "mbar", "role": "candidate"},
    {"name": "fl26_bl5_1_pressure", "unit": "pressure", "role": "candidate"},
    {"name": "fl26_hg0_pressure_mbar", "unit": "mbar", "role": "candidate"},
    {"name": "fl26_js0_pressure_mbar", "unit": "mbar", "role": "candidate"},
    {"name": "fl26_js3_1_pressure", "unit": "pressure", "role": "candidate"},
    {"name": "fl26_valve_bl1_closed", "unit": "bool", "role": "qc"},
    {"name": "fl26_valve_bl1_open", "unit": "bool", "role": "qc"},
    {"name": "gmd_hall_ch7_tss", "unit": "", "role": "qc"},
    {"name": "gmd_tunnel_ch7_tss", "unit": "", "role": "qc"},
    {"name": "gap_error", "unit": "", "role": "qc"},
    {"name": "shutter_value", "unit": "", "role": "qc"},
    {"name": "qc_finite_vls", "unit": "bool", "role": "qc"},
]

# Config MODE strings accepted by both this script and
# compute_xas_aggregates. The two pipelines share inputs and
# preprocessing knobs; a single config can drive either entry point.
# The *output* H5 still gets a distinguishing ``mode`` attribute set by
# whichever script wrote it (``xas_static`` here, ``xas_scan`` in
# compute_xas_aggregates).
ACCEPTED_XAS_MODES = ("xas", "xas_scan", "xas_static")

# ---------------------------------------------------------------------------
# Internal helpers  (self-contained — no imports from other analysis scripts)
# ---------------------------------------------------------------------------

def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("static_xas_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _file_order_key(path_str: str):
    """Sort raw H5 files by numeric index in ..._fileNN_..."""
    name = Path(path_str).name
    if "_file" in name:
        tail = name.split("_file", 1)[1]
        num_str = tail.split("_", 1)[0]
        if num_str.isdigit():
            return (0, int(num_str), name)
    return (1, 0, name)


def _list_raw_h5_files(run_no: int, raw_dir: Path, max_files: Optional[int]):
    pattern = str(raw_dir / f"*run{run_no}*.h5")
    paths = sorted(glob.glob(pattern), key=_file_order_key)
    if not paths:
        raise FileNotFoundError(f"No raw H5 files matching {pattern!r}.")
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def _normalize_run_numbers(run_no):
    if isinstance(run_no, str):
        parts = run_no.replace(",", " ").split()
        return [int(p) for p in parts]
    if np.isscalar(run_no):
        return [int(run_no)]
    return [int(r) for r in run_no]


def _run_label(run_numbers):
    if len(run_numbers) == 1:
        return f"run{run_numbers[0]}"
    return "runs" + "_".join(str(r) for r in run_numbers)


def _list_raw_h5_files_for_runs(run_numbers, raw_dir: Path, max_files: Optional[int]):
    paths = []
    for run in run_numbers:
        paths.extend(_list_raw_h5_files(run, raw_dir, max_files))
    return paths


def _concat_experiment_data(parts):
    if len(parts) == 1:
        return parts[0]

    def cat(name):
        arrays = [getattr(p, name) for p in parts]
        if arrays[0] is None:
            return None
        return np.concatenate(arrays, axis=0)

    return replace(
        parts[0],
        tID=cat("tID"),
        gmd=cat("gmd"),
        mpe=cat("mpe"),
        z=cat("z"),
        between_tdc_files=cat("between_tdc_files"),
        tofs_e=cat("tofs_e"),
        tofs_i=cat("tofs_i"),
        liq_tofs_e=cat("liq_tofs_e"),
        vls=cat("vls"),
        shutter=cat("shutter"),
        shot_mask=cat("shot_mask"),
    )


def _read_aligned_shutter(
    h5_paths,
    idx_path: str,
    val_path: str,
    master_tID: np.ndarray,
) -> np.ndarray:
    """
    Concatenate shutter index/value across files and align to
    ``master_tID``. Multi-dim samples are reduced via nanmean.
    """
    src_idx_parts, src_val_parts = [], []
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if (idx_path not in f) or (val_path not in f):
                continue
            src_idx_parts.append(f[idx_path][...])
            v = f[val_path][...]
            if v.dtype.kind in ("S", "U", "O"):
                raise NotImplementedError("Non-numeric shutter values are not supported.")
            if v.ndim > 1:
                v = np.nanmean(v.astype(np.float64), axis=tuple(range(1, v.ndim)))
            src_val_parts.append(v.astype(np.float64))
    if not src_idx_parts:
        raise RuntimeError(
            f"Shutter datasets {idx_path!r}, {val_path!r} not found in any raw H5 file."
        )
    src_idx = np.concatenate(src_idx_parts)
    src_val = np.concatenate(src_val_parts)
    out = np.full(master_tID.shape, np.nan, dtype=np.float64)
    matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
    out[matched] = src_val[src_pos]
    return out


def _read_aligned_pulse_gmd(
    h5_paths,
    index_candidates,
    value_candidates,
    master_tID: np.ndarray,
    train_length: int,
    *,
    channel: int = 0,
):
    """
    Concatenate pulse-resolved GMD index/value datasets and align to
    ``master_tID``. Missing trains and bunches are NaN-filled.

    Returns
    -------
    out, index_path, value_path
        ``out`` is ``None`` if no matching path pair was found.
    """
    index_path = value_path = None
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path is None:
                for p in index_candidates:
                    if p in f:
                        index_path = p
                        break
            if value_path is None:
                for p in value_candidates:
                    if p in f:
                        value_path = p
                        break
        if index_path is not None and value_path is not None:
            break

    if index_path is None or value_path is None:
        return None, index_path, value_path

    out = np.full((master_tID.shape[0], train_length), np.nan, dtype=np.float64)
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path not in f or value_path not in f:
                continue
            src_idx = f[index_path][...]
            val = f[value_path]
            if val.ndim == 3:
                m = min(val.shape[2], train_length)
                src_val = val[:, int(channel), :m]
            elif val.ndim == 2:
                m = min(val.shape[1], train_length)
                src_val = val[:, :m]
            elif val.ndim == 1:
                m = 1
                src_val = val[:, None]
            else:
                continue
            matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
            if matched.any():
                out[matched, :m] = np.asarray(src_val[src_pos], dtype=np.float64)
    return out, index_path, value_path


def _read_aligned_pulse_gmd_channels(
    h5_paths,
    index_candidates,
    value_candidates,
    master_tID: np.ndarray,
    train_length: int,
    *,
    n_channels: int = 8,
):
    """
    Align all pulse-resolved GMD channels to ``master_tID``.

    The returned array has shape ``(n_master_trains, train_length, n_channels)``
    so the second axis can be sliced by raw GMD bunch number.
    """
    index_path = value_path = None
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path is None:
                for p in index_candidates:
                    if p in f:
                        index_path = p
                        break
            if value_path is None:
                for p in value_candidates:
                    if p in f:
                        value_path = p
                        break
        if index_path is not None and value_path is not None:
            break

    if index_path is None or value_path is None:
        return None, index_path, value_path

    out = np.full(
        (master_tID.shape[0], int(train_length), int(n_channels)),
        np.nan,
        dtype=np.float32,
    )
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if index_path not in f or value_path not in f:
                continue
            src_idx = f[index_path][...]
            val = f[value_path]
            if val.ndim == 3:
                n_ch = min(val.shape[1], n_channels)
                m = min(val.shape[2], train_length)
                src_val = np.asarray(val[:, :n_ch, :m], dtype=np.float32)
                src_val = np.moveaxis(src_val, 1, 2)  # train, bunch, channel
            elif val.ndim == 2:
                n_ch = 1
                m = min(val.shape[1], train_length)
                src_val = np.asarray(val[:, :m, None], dtype=np.float32)
            elif val.ndim == 1:
                n_ch = 1
                m = 1
                src_val = np.asarray(val[:, None, None], dtype=np.float32)
            else:
                continue
            matched, src_pos = data_loading._align_by_tID(src_idx, master_tID)
            if matched.any():
                out[matched, :m, :n_ch] = src_val[src_pos]
    return out, index_path, value_path


def _read_previous_value_scalar(
    h5_paths,
    base_path: str,
    master_tID: np.ndarray,
) -> np.ndarray:
    """
    Align sparse train-indexed scalar data by previous-value hold.

    Missing paths or non-scalar values return an all-NaN vector.
    """
    idx_parts, val_parts = [], []
    idx_path = base_path.rstrip("/") + "/index"
    val_path = base_path.rstrip("/") + "/value"
    for fp in h5_paths:
        with h5py.File(fp, "r") as f:
            if idx_path not in f or val_path not in f:
                continue
            v = np.asarray(f[val_path][...])
            if v.ndim != 1 or v.dtype.kind not in ("b", "i", "u", "f"):
                continue
            idx_parts.append(np.asarray(f[idx_path][...]).reshape(-1))
            val_parts.append(v.astype(np.float64))
    out = np.full(master_tID.shape, np.nan, dtype=np.float64)
    if not idx_parts:
        return out
    src_idx = np.concatenate(idx_parts)
    src_val = np.concatenate(val_parts)
    order = np.argsort(src_idx, kind="mergesort")
    src_idx = src_idx[order]
    src_val = src_val[order]
    pos = np.searchsorted(src_idx, master_tID, side="right") - 1
    valid = pos >= 0
    out[valid] = src_val[pos[valid]]
    return out


def _read_previous_value_scalar_candidates(
    h5_paths,
    base_paths,
    master_tID: np.ndarray,
) -> np.ndarray:
    """Read the first candidate scalar path that exists and has finite data."""
    for base_path in base_paths:
        values = _read_previous_value_scalar(h5_paths, base_path, master_tID)
        if np.any(np.isfinite(values)):
            return values
    return np.full(master_tID.shape, np.nan, dtype=np.float64)


def _safe_linear_slope(y: np.ndarray) -> np.ndarray:
    """Slope over the second axis, NaN when fewer than two values are finite."""
    x = np.arange(y.shape[1], dtype=np.float64)
    x = x - np.nanmean(x)
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(y)
    for i in range(y.shape[0]):
        ok = finite[i]
        if np.sum(ok) < 2:
            continue
        xi = x[ok]
        yi = y[i, ok]
        denom = np.sum((xi - np.mean(xi)) ** 2)
        if denom > 0:
            out[i] = np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / denom
    return out


def _global_vls_background_noise(
    vls: np.ndarray,
    bg_b0: int,
    bg_b1: int,
) -> dict[str, np.ndarray | int | float]:
    """Robust global raw-background noise per cropped VLS pixel.

    The input must be raw VLS after crop/optional bunch roll and before any
    background subtraction.  All background bunches from all trains are pooled.
    """

    arr = np.asarray(vls)
    if arr.ndim != 3:
        raise ValueError("vls must have shape (n_trains, n_bunches, n_pixels)")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("vls must be numeric")
    n_bunches = arr.shape[1]
    if not (0 <= int(bg_b0) < int(bg_b1) <= n_bunches):
        raise ValueError(
            f"background bunch range [{bg_b0}, {bg_b1}) is outside [0, {n_bunches})"
        )

    bg = arr[:, int(bg_b0):int(bg_b1), :]
    flat = bg.reshape(-1, arr.shape[2])
    median = np.nanmedian(flat, axis=0)
    abs_dev = np.abs(flat - median[None, :])
    mad = np.nanmedian(abs_dev, axis=0)
    sigma = _MAD_TO_SIGMA_NORMAL * mad
    n_finite = np.sum(np.isfinite(flat), axis=0).astype(np.int64)
    return {
        "median": median.astype(np.float64, copy=False),
        "mad": mad.astype(np.float64, copy=False),
        "sigma": sigma.astype(np.float64, copy=False),
        "n_finite": n_finite,
        "n_background_bunches": int(bg_b1) - int(bg_b0),
        "n_trains": int(arr.shape[0]),
        "mad_scale": float(_MAD_TO_SIGMA_NORMAL),
    }


def _resolve_ml_feature_specs(feature_names: Optional[list[str]] = None) -> list[dict[str, str]]:
    """Return ML feature specs in the requested order."""
    if feature_names is None:
        return [dict(spec) for spec in ML_FEATURE_SPECS]
    by_name = {spec["name"]: spec for spec in ML_FEATURE_SPECS}
    missing = [name for name in feature_names if name not in by_name]
    if missing:
        available = ", ".join(by_name)
        raise ValueError(
            "Unknown ML feature name(s): "
            + ", ".join(missing)
            + f". Available names: {available}"
        )
    return [dict(by_name[name]) for name in feature_names]


def _build_feature_block(
    *,
    nominal_energy: float,
    shutter_value: np.ndarray,
    gmd_hall_block: np.ndarray,
    gmd_tunnel_block: np.ndarray,
    train_scalar: dict[str, np.ndarray],
    finite_vls: np.ndarray,
    feature_specs: list[dict[str, str]],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Create the compact ML feature matrix for one flattened section."""
    n_shots = gmd_hall_block.shape[0]
    k_cols = [train_scalar[name] for name in train_scalar if name.startswith("sase_k_")]
    k_values = np.column_stack(k_cols) if k_cols else np.empty((n_shots, 0), dtype=np.float64)
    k_mean = np.nanmean(k_values, axis=1) if k_values.size else np.full(n_shots, np.nan)
    k_std = np.nanstd(k_values, axis=1) if k_values.size else np.full(n_shots, np.nan)
    k_slope = _safe_linear_slope(k_values) if k_values.size else np.full(n_shots, np.nan)
    gap_cols = [train_scalar[name] for name in train_scalar if name.startswith("sase_gap_")]
    gap_values = (
        np.column_stack(gap_cols) if gap_cols else np.empty((n_shots, 0), dtype=np.float64)
    )
    gap_mean = np.nanmean(gap_values, axis=1) if gap_values.size else np.full(n_shots, np.nan)
    gap_std = np.nanstd(gap_values, axis=1) if gap_values.size else np.full(n_shots, np.nan)
    gap_slope = _safe_linear_slope(gap_values) if gap_values.size else np.full(n_shots, np.nan)

    available = {
        "gmd_hall_ch0_intensity": gmd_hall_block[:, 0],
        "gmd_hall_ch2_x": gmd_hall_block[:, 2],
        "gmd_hall_ch3_y": gmd_hall_block[:, 3],
        "gmd_hall_ch4_intensity_sigma": gmd_hall_block[:, 4],
        "gmd_tunnel_ch0_intensity": gmd_tunnel_block[:, 0],
        "gmd_tunnel_ch2_x": gmd_tunnel_block[:, 2],
        "gmd_tunnel_ch3_y": gmd_tunnel_block[:, 3],
        "gmd_tunnel_ch4_intensity_sigma": gmd_tunnel_block[:, 4],
        "undulator_gap_mean": gap_mean,
        "undulator_gap_std": gap_std,
        "undulator_gap_slope": gap_slope,
        "undulator_k_mean": k_mean,
        "undulator_k_std": k_std,
        "undulator_k_slope": k_slope,
        "attenuator_pressure": train_scalar["attenuator_pressure"],
        "opis_mean_photon_energy": train_scalar["opis_mean_photon_energy"],
        "nominal_energy": np.full(n_shots, float(nominal_energy), dtype=np.float64),
        "set_wavelength_1": train_scalar["set_wavelength_1"],
        "set_wavelength_2": train_scalar["set_wavelength_2"],
        "gmd_hall_ch7_tss": gmd_hall_block[:, 7],
        "gmd_tunnel_ch7_tss": gmd_tunnel_block[:, 7],
        "gap_error": train_scalar["gap_error"],
        "shutter_value": shutter_value,
        "qc_finite_vls": finite_vls.astype(np.float64),
    }
    for name in _SLOW_SCALAR_FEATURE_PATHS:
        if name in train_scalar:
            available[name] = train_scalar[name]
    names = [spec["name"] for spec in feature_specs]
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"ML feature specs reference unavailable columns: {missing}")
    units = [spec.get("unit", "") for spec in feature_specs]
    roles = [spec.get("role", "predictor") for spec in feature_specs]
    cols = [available[name] for name in names]
    if cols:
        features = np.column_stack(cols).astype(np.float32)
    else:
        features = np.empty((n_shots, 0), dtype=np.float32)
    return features, names, units, roles


def _contiguous_true_blocks(mask: np.ndarray):
    """Return [(start, stop), ...] half-open intervals for True runs."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    x = m.astype(np.int8)
    starts = np.where(np.diff(np.r_[0, x]) == 1)[0]
    ends   = np.where(np.diff(np.r_[x, 0]) == -1)[0] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _classify_shutter(
    shutter: np.ndarray,
    train_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Threshold shutter signal into (is_open, is_closed, finite) masks.
    Auto-flips polarity if the open group has lower mean VLS score.
    """
    arr = np.asarray(shutter, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise RuntimeError("Not enough finite shutter samples to classify.")
    lo, hi = np.nanpercentile(arr[finite], [10, 90])
    thr = 0.5 * (lo + hi)
    is_open   = arr >= thr
    is_closed = arr < thr
    open_score   = np.nanmean(train_score[is_open])   if np.any(is_open)   else np.nan
    closed_score = np.nanmean(train_score[is_closed]) if np.any(is_closed) else np.nan
    if np.isfinite(open_score) and np.isfinite(closed_score) and (open_score < closed_score):
        is_open, is_closed = is_closed, is_open
    return is_open, is_closed, finite


def _detect_sections(
    shutter: np.ndarray,
    vls_score: np.ndarray,
    transition_trim_trains: int,
):
    """
    Detect contiguous open / closed sections after trimming trains
    around every shutter state change.

    Returns
    -------
    open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask
    """
    is_open_raw, is_closed_raw, finite_sh = _classify_shutter(shutter, vls_score)
    valid_mask = finite_sh.copy()
    transition_idx = np.where(np.diff(is_open_raw.astype(np.int8)) != 0)[0] + 1
    for t in transition_idx:
        i0 = max(0, t - transition_trim_trains)
        i1 = min(valid_mask.size, t + transition_trim_trains + 1)
        valid_mask[i0:i1] = False
    open_blocks   = _contiguous_true_blocks(is_open_raw   & finite_sh)
    closed_blocks = _contiguous_true_blocks(is_closed_raw & finite_sh)
    return open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask


def _preceding_closed_indices(
    open_start: int,
    closed_blocks,
    first_closed_idx: np.ndarray,
    first_section_state: str,
    section_index: int,
) -> np.ndarray:
    """
    Return train indices of the closed block to use as background for an
    open section.

    - section_index == 0 and first_section_state == "open": use the
      first closed block (which follows the first open section).
    - Otherwise: use the closed block immediately preceding open_start.
    """
    if section_index == 0 and first_section_state == "open":
        return first_closed_idx
    prev_block = None
    for cs, ce in closed_blocks:
        if ce <= open_start:
            prev_block = (cs, ce)
        else:
            break
    if prev_block is None:
        return np.array([], dtype=np.int64)
    ps, pe = prev_block
    return np.arange(ps, pe, dtype=np.int64)


def _section_energies_from_source(
    *,
    section_energy_source: str,
    nominal_energies: Optional[np.ndarray],
    open_blocks,
    train_scalars: dict,
    is_open_raw: np.ndarray,
    valid_mask: np.ndarray,
    energy_round_decimals: Optional[int],
) -> tuple[np.ndarray, dict]:
    """
    Build one photon-energy value per detected open section.

    ``set_wavelength_*`` values are read as wavelength in nm and converted with
    E[eV] = hc / wavelength[nm]. When a wavelength source is selected, the
    converted value is treated as the section photon-energy axis.
    """
    source = str(section_energy_source).strip().lower()
    if source in ("config", "nominal", "nominal_energies"):
        if nominal_energies is None or nominal_energies.size == 0:
            raise ValueError("NOMINAL_ENERGIES is required when SECTION_ENERGY_SOURCE='config'.")
        energies = np.asarray(nominal_energies, dtype=np.float64).ravel()
        return energies, {
            "source": "config",
            "source_unit": "eV",
            "raw_eV": energies.copy(),
            "source_values": energies.copy(),
            "round_decimals": -1,
        }

    source_to_unit = {
        "set_wavelength_1": "nm",
        "set_wavelength_2": "nm",
        "opis_mean_wavelength": "nm",
        "opis_mean_photon_energy": "eV",
    }
    if source not in source_to_unit:
        allowed = ", ".join(["config", *source_to_unit])
        raise ValueError(f"unknown SECTION_ENERGY_SOURCE={section_energy_source!r}; allowed: {allowed}")
    if source not in train_scalars:
        raise KeyError(f"{source!r} is not available in train_scalars.")

    values_by_train = np.asarray(train_scalars[source], dtype=np.float64)
    section_values = np.full(len(open_blocks), np.nan, dtype=np.float64)
    raw_e = np.full(len(open_blocks), np.nan, dtype=np.float64)
    for i, (s, e) in enumerate(open_blocks):
        mask = np.zeros(valid_mask.shape[0], dtype=bool)
        mask[s:e] = True
        mask &= is_open_raw & valid_mask
        vals = values_by_train[mask]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            raise RuntimeError(f"section {i}: no finite positive {source} values.")
        section_values[i] = float(np.nanmedian(vals))
        if source_to_unit[source] == "nm":
            raw_e[i] = _HC_EV_NM / section_values[i]
        else:
            raw_e[i] = section_values[i]

    energies = raw_e.copy()
    if energy_round_decimals is not None:
        energies = np.round(energies, int(energy_round_decimals))

    return energies.astype(np.float64, copy=False), {
        "source": source,
        "source_unit": source_to_unit[source],
        "raw_eV": raw_e,
        "source_values": section_values,
        "round_decimals": (
            int(energy_round_decimals) if energy_round_decimals is not None else -1
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_static_xas(
    output_h5,
    *,
    run_no,
    nominal_energies=None,
    crop_roi: Tuple[int, int],
    signal_bunch_range: Tuple[int, int],
    bg_bunch_range: Tuple[int, int],
    vls_bunch_roll: int = 0,
    gmd_bunch_start: int = 0,
    config: int = 2,
    raw_dir=None,
    max_files: Optional[int] = None,
    train_length: Optional[int] = None,
    shutter_index_path: str = _DEFAULT_SHUTTER_INDEX_PATH,
    shutter_value_path: str = _DEFAULT_SHUTTER_VALUE_PATH,
    train_rate_hz: float = 10.0,
    transition_trim_seconds: float = 3.0,
    first_section_state: str = "open",
    ml_feature_names: Optional[list[str]] = None,
    section_energy_source: str = "config",
    section_energy_round_decimals: Optional[int] = None,
    config_path=None,
    verbose: bool = True,
) -> None:
    """
    Run the static-XAS shot-collection pipeline and write to ``output_h5``.

    Same processing as ``compute_xas_aggregates`` (pixel crop → per-train
    bg subtraction → shutter detection → per-section bg subtraction →
    signal-bunch selection → flatten to shots) but the result is stored
    as raw per-shot arrays rather than pre-binned aggregates.

    Parameters
    ----------
    output_h5 : str or Path
    run_no : int or sequence of int
    nominal_energies : array-like, optional
        Energies (eV) assigned to detected sections by index when
        ``section_energy_source='config'``.
    crop_roi : (int, int)
        Half-open VLS pixel ROI.
    signal_bunch_range : (int, int)
        Half-open bunch range for signal shots.
    bg_bunch_range : (int, int)
        Half-open bunch range for per-train baseline subtraction.
    vls_bunch_roll : int
        Cyclic shift (``np.roll``) applied to the VLS bunch axis right
        after the pixel crop. ``signal_bunch_range`` and
        ``bg_bunch_range`` apply in the rolled VLS frame.
    gmd_bunch_start : int
        Raw GMD bunch index corresponding to ``signal_bunch_range[0]``.
        The ML feature group stores this explicit VLS→GMD mapping.
    config : int
        Must be 2 (xas_static is config-2 only).
    raw_dir : Path, optional
        Overrides ``config.RAW_H5_DIR``.
    max_files, train_length :
        Forwarded to ``data_loading.load_raw_h5``.
    shutter_index_path, shutter_value_path : str
    train_rate_hz, transition_trim_seconds : float
    first_section_state : "open" | "closed"
    ml_feature_names : list[str], optional
        Optional subset/order of ``ML_FEATURE_SPECS`` names to write under
        ``/ml/features``. ``None`` writes all default compact features.
    section_energy_source : str
        ``"config"`` uses ``nominal_energies``. ``"set_wavelength_1"`` derives
        one section energy from the median undulator set wavelength in each
        open section and treats the converted value as the photon-energy axis.
    config_path : Path, optional  (recorded as a provenance attribute)
    verbose : bool
    """
    output_h5 = Path(output_h5)
    if int(config) != 2:
        raise ValueError("xas_static mode is config=2 only.")

    log = print if verbose else (lambda *a, **k: None)

    run_numbers = _normalize_run_numbers(run_no)
    if not run_numbers:
        raise ValueError("run_no must contain at least one run number.")

    if nominal_energies is None:
        nominal_energies_arr = None
    else:
        nominal_energies_arr = np.asarray(nominal_energies, dtype=np.float64).ravel()
    roi_min, roi_max = int(crop_roi[0]), int(crop_roi[1])
    n_pixels = roi_max - roi_min
    sig_b0, sig_b1 = int(signal_bunch_range[0]), int(signal_bunch_range[1])
    bg_b0,  bg_b1  = int(bg_bunch_range[0]),     int(bg_bunch_range[1])
    vls_bunch_roll = int(vls_bunch_roll)
    gmd_b0 = int(gmd_bunch_start)
    gmd_b1 = gmd_b0 + (sig_b1 - sig_b0)

    first_section_state = str(first_section_state).strip().lower()
    if first_section_state not in ("open", "closed"):
        raise ValueError("first_section_state must be 'open' or 'closed'.")

    if raw_dir is None:
        import config as path_config
        raw_dir = Path(path_config.RAW_H5_DIR)
    else:
        raw_dir = Path(raw_dir)

    log(f"run                : {_run_label(run_numbers)}")
    log(f"raw dir            : {raw_dir}")
    if str(section_energy_source).strip().lower() in ("config", "nominal", "nominal_energies"):
        if nominal_energies_arr is None or nominal_energies_arr.size == 0:
            raise ValueError("NOMINAL_ENERGIES is required when SECTION_ENERGY_SOURCE='config'.")
        log(f"nominal energies   : {nominal_energies_arr.size}  "
            f"({nominal_energies_arr[0]:.2f} .. {nominal_energies_arr[-1]:.2f} eV)")
    else:
        log(f"section energies   : from {section_energy_source}")
    log(f"VLS ROI            : [{roi_min}, {roi_max}) = {n_pixels} pixels")
    log(f"VLS bunch roll     : {vls_bunch_roll}")
    log(f"signal bunches     : [{sig_b0}, {sig_b1})")
    log(f"GMD signal bunches : [{gmd_b0}, {gmd_b1})")
    log(f"bg bunches         : [{bg_b0}, {bg_b1})")

    # ------------------------------------------------------------------
    # Load full run (gmd + vls), crop, per-train baseline subtraction
    # ------------------------------------------------------------------
    data_parts = [
        data_loading.load_raw_h5(
            run, config=2, raw_dir=raw_dir,
            train_length=train_length, max_files=max_files,
        )
        for run in run_numbers
    ]
    train_run_id = np.concatenate(
        [
            np.full(part.tID.shape, int(run), dtype=np.int64)
            for part, run in zip(data_parts, run_numbers)
        ]
    )
    data = _concat_experiment_data(data_parts)
    if data.vls is None:
        raise RuntimeError("load_raw_h5 returned no VLS data for this run.")

    data = data.crop_vls(roi_min, roi_max)
    data = data.roll_vls_bunches(vls_bunch_roll)
    m = data.vls.shape[1]
    if not (0 <= sig_b0 < sig_b1 <= m):
        raise ValueError(f"signal_bunch_range {signal_bunch_range} not within [0, {m}].")
    if not (0 <= bg_b0 < bg_b1 <= m):
        raise ValueError(f"bg_bunch_range {bg_bunch_range} not within [0, {m}].")
    vls_bg_noise = _global_vls_background_noise(data.vls, bg_b0, bg_b1)
    finite_sigma = vls_bg_noise["sigma"][np.isfinite(vls_bg_noise["sigma"])]
    sigma_msg = f"{np.nanmedian(finite_sigma):.3g}" if finite_sigma.size else "nan"
    log(f"raw VLS bg sigma   : global MAD, median pixel sigma={sigma_msg}")
    data = data.auto_subtract_background_trainwise((bg_b0, bg_b1))

    vls = np.asarray(data.vls, dtype=np.float64)   # (n_trains, m, n_pixels)
    gmd = np.asarray(data.gmd, dtype=np.float64)   # (n_trains, m), hall GMD
    n_trains = vls.shape[0]

    h5_paths = _list_raw_h5_files_for_runs(run_numbers, raw_dir, max_files)
    gmd_tunnel, gmd_tunnel_index_path, gmd_tunnel_value_path = _read_aligned_pulse_gmd(
        h5_paths,
        _GMD_TUNNEL_INDEX_CANDIDATES,
        _GMD_TUNNEL_VALUE_CANDIDATES,
        data.tID,
        m,
        channel=0,
    )
    gmd_tunnel_present = gmd_tunnel is not None
    if gmd_tunnel is None:
        gmd_tunnel = np.full_like(gmd, np.nan, dtype=np.float64)
        log("GMD tunnel         : not found; /gmd_tunnel will be NaN-filled")
    else:
        log(f"GMD tunnel         : {gmd_tunnel_value_path}")

    gmd_feature_length = max(m, gmd_b1)
    gmd_hall_channels, gmd_hall_index_path, gmd_hall_value_path = _read_aligned_pulse_gmd_channels(
        h5_paths,
        _GMD_HALL_INDEX_CANDIDATES,
        _GMD_HALL_VALUE_CANDIDATES,
        data.tID,
        gmd_feature_length,
        n_channels=8,
    )
    if gmd_hall_channels is None:
        gmd_hall_channels = np.full((n_trains, gmd_feature_length, 8), np.nan, dtype=np.float32)
        gmd_hall_channels[:, :m, 0] = gmd.astype(np.float32)
        log("GMD hall channels  : not found; channel 0 filled from compatibility GMD")
    else:
        log(f"GMD hall channels  : {gmd_hall_value_path}")

    gmd_tunnel_channels, gmd_tunnel_ch_index_path, gmd_tunnel_ch_value_path = _read_aligned_pulse_gmd_channels(
        h5_paths,
        _GMD_TUNNEL_INDEX_CANDIDATES,
        _GMD_TUNNEL_VALUE_CANDIDATES,
        data.tID,
        gmd_feature_length,
        n_channels=8,
    )
    if gmd_tunnel_channels is None:
        gmd_tunnel_channels = np.full((n_trains, gmd_feature_length, 8), np.nan, dtype=np.float32)
        gmd_tunnel_channels[:, :m, 0] = gmd_tunnel.astype(np.float32)
        log("GMD tunnel channels: not found; channel 0 filled from compatibility tunnel GMD")
    else:
        log(f"GMD tunnel channels: {gmd_tunnel_ch_value_path}")

    train_scalars = {
        f"sase_k_{i:02d}": _read_previous_value_scalar(
            h5_paths, f"/Electron Diagnostic/Undulator setting/SASE{i:02d} k value", data.tID,
        )
        for i in range(2, 14)
    }
    train_scalars.update(
        {
            f"sase_gap_{i:02d}": _read_previous_value_scalar_candidates(
                h5_paths, _UNDULATOR_GAP_PATH_CANDIDATES[i], data.tID,
            )
            for i in range(2, 14)
        }
    )
    train_scalars.update(
        {
            "set_wavelength_1": _read_previous_value_scalar(h5_paths, _SET_WAVELENGTH_1, data.tID),
            "set_wavelength_2": _read_previous_value_scalar(h5_paths, _SET_WAVELENGTH_2, data.tID),
            "gap_error": _read_previous_value_scalar(h5_paths, _GAP_ERROR, data.tID),
            "attenuator_pressure": _read_previous_value_scalar(h5_paths, _ATTENUATOR_PRESSURE, data.tID),
            "opis_mean_photon_energy": np.asarray(data.mpe, dtype=np.float64),
            "opis_mean_wavelength": _read_previous_value_scalar(
                h5_paths, _OPIS_MEAN_WAVELENGTH, data.tID,
            ),
            "opis_number_analysed_bunch": _read_previous_value_scalar(
                h5_paths, _OPIS_NUMBER_ANALYSED_BUNCH, data.tID,
            ),
        }
    )
    train_scalars.update(
        {
            name: _read_previous_value_scalar(h5_paths, path, data.tID)
            for name, path in _SLOW_SCALAR_FEATURE_PATHS.items()
        }
    )
    ml_feature_specs = _resolve_ml_feature_specs(ml_feature_names)

    # ------------------------------------------------------------------
    # Shutter section detection
    # ------------------------------------------------------------------
    shutter  = _read_aligned_shutter(
        h5_paths, shutter_index_path, shutter_value_path, data.tID,
    )
    train_score = np.nanmean(
        np.nansum(vls[:, sig_b0:sig_b1, :], axis=2), axis=1,
    )
    transition_trim_trains = int(round(transition_trim_seconds * train_rate_hz))
    open_blocks, closed_blocks, is_open_raw, is_closed_raw, valid_mask = (
        _detect_sections(shutter, train_score, transition_trim_trains)
    )
    if not open_blocks:
        raise RuntimeError("No open-shutter sections detected.")
    if not closed_blocks:
        raise RuntimeError("No closed-shutter sections detected.")

    section_energies, section_energy_meta = _section_energies_from_source(
        section_energy_source=section_energy_source,
        nominal_energies=nominal_energies_arr,
        open_blocks=open_blocks,
        train_scalars=train_scalars,
        is_open_raw=is_open_raw,
        valid_mask=valid_mask,
        energy_round_decimals=section_energy_round_decimals,
    )

    n_detected  = len(open_blocks)
    n_e = min(n_detected, section_energies.size)
    if n_detected != section_energies.size:
        log(f"warning: detected {n_detected} open sections vs "
            f"{section_energies.size} section energies; using leading {n_e}.")
    open_blocks      = open_blocks[:n_e]
    energies         = section_energies[:n_e]
    log(f"section energies   : {energies.size}  "
        f"({energies[0]:.2f} .. {energies[-1]:.2f} eV), "
        f"source={section_energy_meta['source']}")
    first_closed_idx = np.arange(
        closed_blocks[0][0], closed_blocks[0][1], dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Collect shots per section
    # ------------------------------------------------------------------
    section_bg  = np.full((n_e, m, n_pixels), np.nan, dtype=np.float64)
    vls_shots: list[np.ndarray] = []
    gmd_shots: list[np.ndarray] = []
    gmd_tunnel_shots: list[np.ndarray] = []
    train_id_shots: list[np.ndarray] = []
    run_id_shots: list[np.ndarray] = []
    section_index_shots: list[np.ndarray] = []
    vls_bunch_index_shots: list[np.ndarray] = []
    gmd_bunch_index_shots: list[np.ndarray] = []
    nominal_energy_shots: list[np.ndarray] = []
    gmd_hall_channel_shots: list[np.ndarray] = []
    gmd_tunnel_channel_shots: list[np.ndarray] = []
    feature_shots: list[np.ndarray] = []
    qc_ok_shots: list[np.ndarray] = []
    feature_names = [spec["name"] for spec in ml_feature_specs]
    feature_units = [spec.get("unit", "") for spec in ml_feature_specs]
    feature_roles = [spec.get("role", "predictor") for spec in ml_feature_specs]
    n_shots_arr = np.zeros(n_e, dtype=np.int64)

    log(f"collecting shots over {n_e} sections...")
    for i, (s, e) in enumerate(open_blocks):
        sec_open_mask = np.zeros(n_trains, dtype=bool)
        sec_open_mask[s:e] = True
        sec_open_mask &= is_open_raw & valid_mask
        sec_open_idx = np.where(sec_open_mask)[0]

        if sec_open_idx.size == 0:
            log(f"  section {i:>3d} (E={energies[i]:.2f} eV): no open trains; skipping")
            vls_shots.append(np.empty((0, n_pixels), dtype=np.float64))
            gmd_shots.append(np.empty(0, dtype=np.float64))
            gmd_tunnel_shots.append(np.empty(0, dtype=np.float64))
            train_id_shots.append(np.empty(0, dtype=np.int64))
            run_id_shots.append(np.empty(0, dtype=np.int64))
            section_index_shots.append(np.empty(0, dtype=np.int32))
            vls_bunch_index_shots.append(np.empty(0, dtype=np.int16))
            gmd_bunch_index_shots.append(np.empty(0, dtype=np.int16))
            nominal_energy_shots.append(np.empty(0, dtype=np.float64))
            gmd_hall_channel_shots.append(np.empty((0, 8), dtype=np.float32))
            gmd_tunnel_channel_shots.append(np.empty((0, 8), dtype=np.float32))
            n_features = len(feature_names)
            feature_shots.append(np.empty((0, n_features), dtype=np.float32))
            qc_ok_shots.append(np.empty(0, dtype=bool))
            continue

        sec_closed_idx = _preceding_closed_indices(
            s, closed_blocks, first_closed_idx, first_section_state, i,
        )
        if sec_closed_idx.size == 0:
            sec_closed_idx = np.where(is_closed_raw & valid_mask)[0]
        if sec_closed_idx.size == 0:
            raise RuntimeError(f"section {i}: no closed trains available for background.")

        bg2d = np.nanmean(vls[sec_closed_idx, :, :], axis=0)   # (m, n_pixels)
        section_bg[i] = bg2d

        A_block = (                                             # (n_sec, n_sig, n_pixels)
            vls[sec_open_idx][:, sig_b0:sig_b1, :]
            - bg2d[None, sig_b0:sig_b1, :]
        )
        # Root /gmd is written from this explicit raw-GMD bunch window, not
        # from the VLS signal-bunch indices.
        H_channel_block = gmd_hall_channels[sec_open_idx, gmd_b0:gmd_b1, :]
        T_channel_block = gmd_tunnel_channels[sec_open_idx, gmd_b0:gmd_b1, :]
        if H_channel_block.shape[1] != (sig_b1 - sig_b0):
            raise RuntimeError(
                f"section {i}: GMD slice [{gmd_b0}, {gmd_b1}) has "
                f"{H_channel_block.shape[1]} bunches, expected {sig_b1 - sig_b0}."
            )
        G_block = H_channel_block[:, :, 0].astype(np.float64, copy=False)
        G_tunnel_block = T_channel_block[:, :, 0].astype(np.float64, copy=False)

        A_flat = A_block.reshape(-1, n_pixels)
        G_flat = G_block.reshape(-1)
        G_tunnel_flat = G_tunnel_block.reshape(-1)
        H_channel_flat = H_channel_block.reshape(-1, 8)
        T_channel_flat = T_channel_block.reshape(-1, 8)

        n_sig = sig_b1 - sig_b0
        train_id_flat = np.repeat(data.tID[sec_open_idx].astype(np.int64), n_sig)
        run_id_flat = np.repeat(train_run_id[sec_open_idx].astype(np.int64), n_sig)
        section_index_flat = np.full(A_flat.shape[0], i, dtype=np.int32)
        vls_bunch_flat = np.tile(
            np.arange(sig_b0, sig_b1, dtype=np.int16), sec_open_idx.size,
        )
        gmd_bunch_flat = np.tile(
            np.arange(gmd_b0, gmd_b1, dtype=np.int16), sec_open_idx.size,
        )
        nominal_energy_flat = np.full(A_flat.shape[0], energies[i], dtype=np.float64)
        finite_vls_flat = np.all(np.isfinite(A_flat), axis=1)
        train_scalar_flat = {
            name: np.repeat(np.asarray(values)[sec_open_idx], n_sig)
            for name, values in train_scalars.items()
        }
        shutter_flat = np.repeat(shutter[sec_open_idx], n_sig)
        features, names, units, roles = _build_feature_block(
            nominal_energy=float(energies[i]),
            shutter_value=shutter_flat,
            gmd_hall_block=H_channel_flat,
            gmd_tunnel_block=T_channel_flat,
            train_scalar=train_scalar_flat,
            finite_vls=finite_vls_flat,
            feature_specs=ml_feature_specs,
        )
        if feature_names != names or feature_units != units or feature_roles != roles:
            raise RuntimeError("internal error: ML feature columns changed between sections")
        hall_tss_ok = (~np.isfinite(H_channel_flat[:, 7])) | (H_channel_flat[:, 7] == 0)
        tunnel_tss_ok = (~np.isfinite(T_channel_flat[:, 7])) | (T_channel_flat[:, 7] == 0)
        qc_ok = (
            finite_vls_flat
            & np.isfinite(G_flat)
            & np.isfinite(H_channel_flat[:, 0])
            & hall_tss_ok
            & tunnel_tss_ok
        )

        n_shots_arr[i] = A_flat.shape[0]
        vls_shots.append(A_flat)
        gmd_shots.append(G_flat)
        gmd_tunnel_shots.append(G_tunnel_flat)
        train_id_shots.append(train_id_flat)
        run_id_shots.append(run_id_flat)
        section_index_shots.append(section_index_flat)
        vls_bunch_index_shots.append(vls_bunch_flat)
        gmd_bunch_index_shots.append(gmd_bunch_flat)
        nominal_energy_shots.append(nominal_energy_flat)
        gmd_hall_channel_shots.append(H_channel_flat.astype(np.float32, copy=False))
        gmd_tunnel_channel_shots.append(T_channel_flat.astype(np.float32, copy=False))
        feature_shots.append(features)
        qc_ok_shots.append(qc_ok)
        log(f"  section {i:>3d} (E={energies[i]:.2f} eV): "
            f"open trains={sec_open_idx.size:>4d}  "
            f"bg trains={sec_closed_idx.size:>4d}  "
            f"shots={A_flat.shape[0]:>6d}")

    # ------------------------------------------------------------------
    # NaN-pad to rectangular arrays
    # ------------------------------------------------------------------
    n_shots_max = int(n_shots_arr.max()) if n_e > 0 else 0
    vls_out = np.full((n_e, n_shots_max, n_pixels), np.nan, dtype=np.float64)
    gmd_out = np.full((n_e, n_shots_max),            np.nan, dtype=np.float64)
    gmd_tunnel_out = np.full((n_e, n_shots_max),     np.nan, dtype=np.float64)
    n_features = len(feature_names)
    ml_train_id_out = np.full((n_e, n_shots_max), -1, dtype=np.int64)
    ml_run_id_out = np.full((n_e, n_shots_max), -1, dtype=np.int64)
    ml_section_index_out = np.full((n_e, n_shots_max), -1, dtype=np.int32)
    ml_vls_bunch_index_out = np.full((n_e, n_shots_max), -1, dtype=np.int16)
    ml_gmd_bunch_index_out = np.full((n_e, n_shots_max), -1, dtype=np.int16)
    ml_nominal_energy_out = np.full((n_e, n_shots_max), np.nan, dtype=np.float64)
    ml_gmd_hall_channels_out = np.full(
        (n_e, n_shots_max, 8), np.nan, dtype=np.float32,
    )
    ml_gmd_tunnel_channels_out = np.full(
        (n_e, n_shots_max, 8), np.nan, dtype=np.float32,
    )
    ml_features_out = np.full(
        (n_e, n_shots_max, n_features), np.nan, dtype=np.float32,
    )
    ml_qc_ok_out = np.zeros((n_e, n_shots_max), dtype=bool)

    for i, (A, G, Gt) in enumerate(zip(vls_shots, gmd_shots, gmd_tunnel_shots)):
        n = A.shape[0]
        if n > 0:
            vls_out[i, :n] = A
            gmd_out[i, :n] = G
            gmd_tunnel_out[i, :n] = Gt
            ml_train_id_out[i, :n] = train_id_shots[i]
            ml_run_id_out[i, :n] = run_id_shots[i]
            ml_section_index_out[i, :n] = section_index_shots[i]
            ml_vls_bunch_index_out[i, :n] = vls_bunch_index_shots[i]
            ml_gmd_bunch_index_out[i, :n] = gmd_bunch_index_shots[i]
            ml_nominal_energy_out[i, :n] = nominal_energy_shots[i]
            ml_gmd_hall_channels_out[i, :n, :] = gmd_hall_channel_shots[i]
            ml_gmd_tunnel_channels_out[i, :n, :] = gmd_tunnel_channel_shots[i]
            ml_features_out[i, :n, :] = feature_shots[i]
            ml_qc_ok_out[i, :n] = qc_ok_shots[i]

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}  (vls: {vls_out.shape}, gmd: {gmd_out.shape})")
    with h5py.File(output_h5, "w") as fout:
        fout.create_dataset("vls",              data=vls_out,     compression="gzip")
        fout.create_dataset("gmd",              data=gmd_out,     compression="gzip")
        fout.create_dataset("gmd_tunnel",       data=gmd_tunnel_out, compression="gzip")
        fout.create_dataset("n_shots",          data=n_shots_arr)
        fout.create_dataset("nominal_energies", data=energies)
        fout.create_dataset(
            "section_energy_source_values",
            data=np.asarray(section_energy_meta["source_values"][:n_e], dtype=np.float64),
        )
        fout.create_dataset(
            "section_energy_raw_eV",
            data=np.asarray(section_energy_meta["raw_eV"][:n_e], dtype=np.float64),
        )
        fout.create_dataset("vls_pixels",
                            data=np.arange(roi_min, roi_max, dtype=np.int64))
        fout.create_dataset("section_bg",       data=section_bg,  compression="gzip")

        noise = fout.create_group("noise")
        noise.create_dataset("vls_bg_median_global", data=vls_bg_noise["median"])
        noise.create_dataset("vls_bg_mad_global", data=vls_bg_noise["mad"])
        noise.create_dataset("vls_bg_sigma_global", data=vls_bg_noise["sigma"])
        noise.create_dataset("vls_bg_n_finite", data=vls_bg_noise["n_finite"])
        noise.attrs["schema_version"] = "1.0"
        noise.attrs["source"] = (
            "raw VLS after crop and optional bunch roll, before train-wise "
            "and per-section background subtraction"
        )
        noise.attrs["method"] = "sigma = 1.4826 * MAD over all train x background-bunch samples"
        noise.attrs["bg_bunch_range"] = np.asarray([bg_b0, bg_b1], dtype=np.int64)
        noise.attrs["n_background_bunches"] = int(vls_bg_noise["n_background_bunches"])
        noise.attrs["n_trains"] = int(vls_bg_noise["n_trains"])
        noise.attrs["mad_scale"] = float(vls_bg_noise["mad_scale"])

        ml = fout.create_group("ml")
        ml.create_dataset("train_id", data=ml_train_id_out, compression="gzip")
        ml.create_dataset("run_id", data=ml_run_id_out, compression="gzip")
        ml.create_dataset("section_index", data=ml_section_index_out, compression="gzip")
        ml.create_dataset("vls_bunch_index", data=ml_vls_bunch_index_out, compression="gzip")
        ml.create_dataset("gmd_bunch_index", data=ml_gmd_bunch_index_out, compression="gzip")
        ml.create_dataset("nominal_energy", data=ml_nominal_energy_out, compression="gzip")
        ml.create_dataset(
            "gmd_hall_channels", data=ml_gmd_hall_channels_out, compression="gzip",
        )
        ml.create_dataset(
            "gmd_tunnel_channels", data=ml_gmd_tunnel_channels_out, compression="gzip",
        )
        ml.create_dataset("features", data=ml_features_out, compression="gzip")
        ml.create_dataset("qc_ok", data=ml_qc_ok_out, compression="gzip")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        ml.create_dataset(
            "feature_names",
            data=np.asarray(feature_names or [], dtype=object),
            dtype=str_dtype,
        )
        ml.create_dataset(
            "feature_units",
            data=np.asarray(feature_units or [], dtype=object),
            dtype=str_dtype,
        )
        ml.create_dataset(
            "feature_roles",
            data=np.asarray(feature_roles or [], dtype=object),
            dtype=str_dtype,
        )
        ml.attrs["schema_version"] = "1.0"
        ml.attrs["description"] = (
            "Compact per-shot ML contract aligned to root /vls flatten order."
        )
        ml.attrs["gmd_bunch_start"] = int(gmd_b0)
        ml.attrs["gmd_bunch_range"] = np.asarray([gmd_b0, gmd_b1], dtype=np.int64)
        ml.attrs["feature_source"] = (
            "GMD hall/tunnel channels, reduced undulator K, attenuator pressure, "
            "OPIS mean photon energy, setpoints, and QC flags."
        )
        if gmd_hall_index_path is not None:
            ml.attrs["gmd_hall_index_path"] = gmd_hall_index_path
        if gmd_hall_value_path is not None:
            ml.attrs["gmd_hall_value_path"] = gmd_hall_value_path
        if gmd_tunnel_ch_index_path is not None:
            ml.attrs["gmd_tunnel_index_path"] = gmd_tunnel_ch_index_path
        if gmd_tunnel_ch_value_path is not None:
            ml.attrs["gmd_tunnel_value_path"] = gmd_tunnel_ch_value_path

        fout.attrs["mode"]                    = "xas_static"
        fout.attrs["ml_schema_version"]       = "1.0"
        fout.attrs["config"]                  = int(config)
        fout.attrs["run_no"]                  = (
            int(run_numbers[0]) if len(run_numbers) == 1
            else np.asarray(run_numbers, dtype=np.int64)
        )
        fout.attrs["raw_dir"]                 = str(raw_dir)
        fout.attrs["n_sections_detected"]     = int(n_detected)
        fout.attrs["n_sections_used"]         = int(n_e)
        fout.attrs["section_energy_source"]   = str(section_energy_meta["source"])
        fout.attrs["section_energy_source_unit"] = str(section_energy_meta["source_unit"])
        fout.attrs["section_energy_round_decimals"] = int(section_energy_meta["round_decimals"])
        fout.attrs["section_energy_conversion"] = (
            f"E_eV = {_HC_EV_NM:.13g} / wavelength_nm"
            if section_energy_meta["source_unit"] == "nm"
            else "source values already in eV"
        )
        fout.attrs["vls_crop_roi"]            = np.asarray([roi_min, roi_max], dtype=np.int64)
        fout.attrs["vls_bunch_roll"]          = int(vls_bunch_roll)
        fout.attrs["signal_bunch_range"]      = np.asarray([sig_b0, sig_b1], dtype=np.int64)
        fout.attrs["gmd_bunch_start"]         = int(gmd_b0)
        fout.attrs["gmd_signal_bunch_range"]  = np.asarray([gmd_b0, gmd_b1], dtype=np.int64)
        fout.attrs["bg_bunch_range"]          = np.asarray([bg_b0, bg_b1],   dtype=np.int64)
        fout.attrs["train_rate_hz"]           = float(train_rate_hz)
        fout.attrs["transition_trim_seconds"] = float(transition_trim_seconds)
        fout.attrs["transition_trim_trains"]  = int(transition_trim_trains)
        fout.attrs["first_section_state"]     = first_section_state
        fout.attrs["shutter_index_path"]      = shutter_index_path
        fout.attrs["shutter_value_path"]      = shutter_value_path
        fout.attrs["gmd_tunnel_present"]      = bool(gmd_tunnel_present)
        if gmd_tunnel_index_path is not None:
            fout.attrs["gmd_tunnel_index_path"] = gmd_tunnel_index_path
        if gmd_tunnel_value_path is not None:
            fout.attrs["gmd_tunnel_value_path"] = gmd_tunnel_value_path
        fout.attrs["gmd_source"]                = "ml/gmd_hall_channels[..., 0]"
        fout.attrs["gmd_tunnel_source"]         = "ml/gmd_tunnel_channels[..., 0]"
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)

    log(f"done: {int(n_shots_arr.sum())} shots across {n_e} energies.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-shot VLS/GMD for a static XAS scan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Config module exposing RUN_NO, CROP_ROI, SIGNAL_BUNCH_RANGE, "
             "BG_BUNCH_RANGE, CONFIG=2, plus either NOMINAL_ENERGIES or "
             "SECTION_ENERGY_SOURCE.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output H5 path (default: processed/xas_static/run<N>_static_xas.h5)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config_path)

    if int(getattr(cfg, "CONFIG", 2)) != 2:
        raise ValueError("compute_static_xas requires CONFIG = 2.")
    mode = getattr(cfg, "MODE", "xas")
    if mode not in ACCEPTED_XAS_MODES:
        raise ValueError(
            f"this entry point expects MODE in {ACCEPTED_XAS_MODES}, "
            f"got {mode!r}."
        )

    run_no = _normalize_run_numbers(cfg.RUN_NO)
    if args.output is None:
        import config as path_config
        out_dir = Path(path_config.COMBINED_DIR).parent / "xas_static"
        out = out_dir / f"{_run_label(run_no)}_static_xas.h5"
    else:
        out = args.output

    compute_static_xas(
        output_h5=out,
        run_no=run_no,
        nominal_energies=getattr(cfg, "NOMINAL_ENERGIES", None),
        crop_roi=tuple(cfg.CROP_ROI),
        signal_bunch_range=tuple(cfg.SIGNAL_BUNCH_RANGE),
        bg_bunch_range=tuple(cfg.BG_BUNCH_RANGE),
        vls_bunch_roll=int(getattr(cfg, "VLS_BUNCH_ROLL", 0)),
        gmd_bunch_start=int(getattr(cfg, "GMD_BUNCH_START", 0)),
        config=int(getattr(cfg, "CONFIG", 2)),
        raw_dir=getattr(cfg, "RAW_DIR", None),
        max_files=getattr(cfg, "MAX_FILES", None),
        train_length=getattr(cfg, "TRAIN_LENGTH", None),
        shutter_index_path=getattr(cfg, "SHUTTER_INDEX_PATH",
                                   _DEFAULT_SHUTTER_INDEX_PATH),
        shutter_value_path=getattr(cfg, "SHUTTER_VALUE_PATH",
                                   _DEFAULT_SHUTTER_VALUE_PATH),
        train_rate_hz=float(getattr(cfg, "TRAIN_RATE_HZ", 10.0)),
        transition_trim_seconds=float(getattr(cfg, "TRANSITION_TRIM_SECONDS", 3.0)),
        first_section_state=str(getattr(cfg, "FIRST_SECTION_STATE", "open")),
        ml_feature_names=getattr(cfg, "ML_FEATURE_NAMES", None),
        section_energy_source=str(getattr(cfg, "SECTION_ENERGY_SOURCE", "config")),
        section_energy_round_decimals=getattr(cfg, "SECTION_ENERGY_ROUND_DECIMALS", None),
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
