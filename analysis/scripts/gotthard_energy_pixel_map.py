#!/usr/bin/env python3
"""
Map photon energy (eV) to GOTTHARD pixel index for a grating spectrometer.

Physics model (matching this repo sign convention):
  beta = -asin(sin(alpha) - m * lambda / sigma0)

Dispersion coordinate at the screen plane:
  y_screen = d * cos(beta)

After optical magnification M onto GOTTHARD:
  y_sensor = M * y_screen

Pixel mapping using one calibration anchor (E_cal, pixel_cal):
  pixel(E) = pixel_cal + axis_sign * (y_sensor(E) - y_sensor(E_cal)) / pixel_pitch

Notes
-----
- axis_sign = +1 means larger y_sensor gives larger pixel index.
- axis_sign = -1 flips detector orientation.
- With your statement "higher pixel number -> lower energy", choose axis_sign so
  that trend is satisfied for your setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


# ==============================
# User Configuration (edit here)
# ==============================
USER_CONFIG = {
    "source_to_grating_m": 0.85,
    "grating_to_screen_m": 0.5429,
    "incident_angle_deg": 86.78,
    "groove_density_lines_per_mm": 1200.0,
    "diffraction_order": 1,
    "magnification": 3,
    "pixel_pitch_m": 50e-6,
    "axis_sign": 1,
}

USER_CALIBRATION = {
    "pixel": 590.0,
    "energy_eV": 270.0,
}

USER_QUERY = {
    # Input pixels you want to convert.
    "pixels": [530, 560, 590],
    # Energy range used for inverse interpolation.
    "inverse_range_eV": (260.0, 290.0),
    "inverse_samples": 10000,
}


HC_EV_NM = 1240.0


@dataclass
class SpectrometerConfig:
    # Geometry
    source_to_grating_m: float = 0.85
    grating_to_screen_m: float = 0.5429

    # Grating / beam
    incident_angle_deg: float = 86.78
    groove_density_lines_per_mm: float = 1200.0
    diffraction_order: int = 1

    # Detector chain
    magnification: float = 8.0
    pixel_pitch_m: float = 50e-6
    axis_sign: int = 1


@dataclass
class CalibrationPoint:
    pixel: float
    energy_eV: float


def _sigma0_from_lines_per_mm(lines_per_mm: float) -> float:
    if lines_per_mm <= 0:
        raise ValueError("Groove density must be positive.")
    return 1.0 / (lines_per_mm * 1000.0)


def _wavelength_m_from_eV(energy_eV: np.ndarray) -> np.ndarray:
    if np.any(energy_eV <= 0):
        raise ValueError("Photon energy must be positive.")
    return HC_EV_NM / energy_eV * 1e-9


def beta_from_energy(energy_eV: np.ndarray, cfg: SpectrometerConfig) -> np.ndarray:
    alpha = np.deg2rad(cfg.incident_angle_deg)
    sigma0 = _sigma0_from_lines_per_mm(cfg.groove_density_lines_per_mm)
    lam = _wavelength_m_from_eV(np.asarray(energy_eV, dtype=float))

    arg = np.sin(alpha) - cfg.diffraction_order * (lam / sigma0)
    if np.any(np.abs(arg) > 1.0):
        raise ValueError(
            "No real diffraction angle for at least one energy. "
            "Adjust energy/angle/order/groove-density configuration."
        )
    return -np.arcsin(arg)


def dispersion_coordinate_screen_m(
    energy_eV: np.ndarray, cfg: SpectrometerConfig
) -> np.ndarray:
    beta = beta_from_energy(energy_eV, cfg)
    return cfg.grating_to_screen_m * np.cos(beta)


def energy_to_pixel(
    energy_eV: np.ndarray, cfg: SpectrometerConfig, cal: CalibrationPoint
) -> np.ndarray:
    energies = np.asarray(energy_eV, dtype=float)
    y = dispersion_coordinate_screen_m(energies, cfg)
    y_cal = dispersion_coordinate_screen_m(np.array([cal.energy_eV]), cfg)[0]

    y_sensor = cfg.magnification * y
    y_sensor_cal = cfg.magnification * y_cal

    delta_pix = cfg.axis_sign * (y_sensor - y_sensor_cal) / cfg.pixel_pitch_m
    return cal.pixel + delta_pix


def pixel_to_energy(
    pixel: np.ndarray,
    cfg: SpectrometerConfig,
    cal: CalibrationPoint,
    e_min: float,
    e_max: float,
    n_samples: int = 10000,
) -> np.ndarray:
    pixels = np.asarray(pixel, dtype=float)
    e_grid = np.linspace(e_min, e_max, n_samples)
    p_grid = energy_to_pixel(e_grid, cfg, cal)

    order = np.argsort(p_grid)
    p_sorted = p_grid[order]
    e_sorted = e_grid[order]

    if np.any(pixels < p_sorted[0]) or np.any(pixels > p_sorted[-1]):
        raise ValueError(
            "Requested pixel is outside interpolation range. "
            "Expand e_min/e_max."
        )

    return np.interp(pixels, p_sorted, e_sorted)


def _float_list(values: Iterable[float]) -> str:
    return ", ".join(f"{v:.6f}" for v in values)


def _print_summary(cfg: SpectrometerConfig, cal: CalibrationPoint) -> None:
    print("=== Spectrometer Configuration ===")
    print(f"source_to_grating_m          : {cfg.source_to_grating_m}")
    print(f"grating_to_screen_m          : {cfg.grating_to_screen_m}")
    print(f"incident_angle_deg           : {cfg.incident_angle_deg}")
    print(f"groove_density_lines_per_mm  : {cfg.groove_density_lines_per_mm}")
    print(f"diffraction_order            : {cfg.diffraction_order}")
    print(f"magnification                : {cfg.magnification}")
    print(f"pixel_pitch_m                : {cfg.pixel_pitch_m}")
    print(f"axis_sign                    : {cfg.axis_sign}")
    print("=== Calibration ===")
    print(f"pixel at calibration energy  : {cal.pixel}")
    print(f"calibration energy (eV)      : {cal.energy_eV}")
    print()


def _default_user_config() -> Tuple[SpectrometerConfig, CalibrationPoint, dict]:
    cfg = SpectrometerConfig(
        source_to_grating_m=USER_CONFIG["source_to_grating_m"],
        grating_to_screen_m=USER_CONFIG["grating_to_screen_m"],
        incident_angle_deg=USER_CONFIG["incident_angle_deg"],
        groove_density_lines_per_mm=USER_CONFIG["groove_density_lines_per_mm"],
        diffraction_order=USER_CONFIG["diffraction_order"],
        magnification=USER_CONFIG["magnification"],
        pixel_pitch_m=USER_CONFIG["pixel_pitch_m"],
        axis_sign=USER_CONFIG["axis_sign"],
    )

    cal = CalibrationPoint(
        pixel=USER_CALIBRATION["pixel"],
        energy_eV=USER_CALIBRATION["energy_eV"],
    )

    query = dict(USER_QUERY)
    return cfg, cal, query


def main() -> None:
    cfg, cal, query = _default_user_config()

    _print_summary(cfg, cal)

    pixels_input = np.asarray(query.get("pixels", []), dtype=float)
    if pixels_input.size == 0:
        raise ValueError("query['pixels'] is empty.")

    pixels = pixels_input
    e_min, e_max = query.get("inverse_range_eV", (270, 290))
    n_samples = int(query.get("inverse_samples", 10000))
    energies = pixel_to_energy(
        pixels,
        cfg,
        cal,
        e_min=float(e_min),
        e_max=float(e_max),
        n_samples=n_samples,
    )

    print("Pixel -> Energy")
    for p, e in zip(pixels, energies):
        print(f"  pixel = {p:10.6f} -> E = {e:12.6f} eV")
    print()
    print(f"P list: {_float_list(pixels)}")
    print(f"E list: {_float_list(energies)}")


if __name__ == "__main__":
    main()