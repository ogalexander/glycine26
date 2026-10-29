import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "analysis" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import static_xas_process_good as pg
import static_xas_analysis as sxa
import static_xas_good_xas as gx
import static_xas_good_reference as gr
from gotthard_energy_pixel_map import energy_to_pixel, SpectrometerConfig, CalibrationPoint


def write_processed_h5(path, *, include_tunnel=True):
    hall = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    tunnel = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    trans = 5.0 + 2.0 * tunnel
    template = np.array([0.0, 0.0, 1.0, 3.0, 10.0, 3.0, 1.0, 0.0, 0.0], dtype=np.float64)
    spectra = np.vstack([template * (value / np.sum(template)) for value in trans])

    with h5py.File(path, "w") as h5:
        h5.attrs["run_no"] = 12345
        h5.create_dataset("vls", data=spectra.reshape(1, 3, template.size))
        h5.create_dataset("gmd", data=hall.reshape(1, 3))
        if include_tunnel:
            h5.create_dataset("gmd_tunnel", data=tunnel.reshape(1, 3))
        h5.create_dataset("n_shots", data=np.array([3], dtype=np.int64))
        h5.create_dataset("nominal_energies", data=np.array([271.0], dtype=np.float64))
        h5.create_dataset("vls_pixels", data=520 + np.arange(template.size, dtype=np.float64))

        noise = h5.create_group("noise")
        noise.create_dataset("vls_bg_sigma_global", data=np.full(template.size, 0.01, dtype=np.float64))

        ml = h5.create_group("ml")
        hall_channels = np.full((1, 3, 8), np.nan, dtype=np.float32)
        tunnel_channels = np.full((1, 3, 8), np.nan, dtype=np.float32)
        hall_channels[0, :, 0] = hall
        tunnel_channels[0, :, 0] = tunnel
        ml.create_dataset("gmd_hall_channels", data=hall_channels)
        if include_tunnel:
            ml.create_dataset("gmd_tunnel_channels", data=tunnel_channels)
        ml.create_dataset("qc_ok", data=np.ones((1, 3), dtype=bool))


def config_for(path, **kwargs):
    defaults = dict(
        input_h5=path,
        output_h5=None,
        chunk_size=2,
        peak_snr_threshold=0.1,
        prominence_inner_px=1,
        prominence_outer_px=4,
        prominence_baseline_percentile=20.0,
        sigma_half_window_px=1,
        center_half_window_px=1,
        min_side_pixels=1,
    )
    defaults.update(kwargs)
    return pg.ProcessGoodConfig(**defaults)


class StaticXASProcessGoodTests(unittest.TestCase):
    def test_fixed_map_anchor_and_inverse_match_legacy(self):
        cfg = pg.ProcessGoodConfig("unused.h5")
        self.assertEqual(cfg.center_method, "smooth_peak")
        self.assertAlmostEqual(pg.map_pixels_to_energy(np.array([524.0]), cfg)[0], 284.2, places=7)
        pixels = np.array([524.0, 540.25, 570.5, 610.0])
        expected, _, _ = sxa.convert_pixels_to_energy(
            pixels, cfg.pixel_map_config, cfg.pixel_map_calibration,
            cfg.inverse_map_range_ev, cfg.inverse_map_samples,
        )
        np.testing.assert_allclose(pg.map_pixels_to_energy(pixels, cfg), expected, atol=1e-12)
        energies = np.array([265.0, 275.0, 284.2])
        physical_pixels = energy_to_pixel(
            energies, SpectrometerConfig(**cfg.pixel_map_config),
            CalibrationPoint(**cfg.pixel_map_calibration),
        )
        np.testing.assert_allclose(pg.map_pixels_to_energy(physical_pixels, cfg), energies, atol=1e-7)

    def test_map_rejects_out_of_range_and_preserves_missing_centers(self):
        cfg = pg.ProcessGoodConfig("unused.h5")
        self.assertTrue(np.isnan(pg.map_pixels_to_energy(np.array([np.nan]), cfg)[0]))
        with self.assertRaisesRegex(ValueError, "adjust inverse_map_range_ev"):
            pg.map_pixels_to_energy(np.array([0.0]), cfg)

    def test_single_section_energy_does_not_fit_or_shift_physical_map(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path)
            first = pg.compute_processed_qc(config_for(path))
            with h5py.File(path, "r+") as h5:
                h5["nominal_energies"][0] = 280.0
            second = pg.compute_processed_qc(config_for(path))
        np.testing.assert_allclose(first.proxy_energy_eV, second.proxy_energy_eV)
        np.testing.assert_allclose(
            first.energy_calibration_table["mapped_energy_eV"],
            second.energy_calibration_table["mapped_energy_eV"],
        )
        np.testing.assert_allclose(
            second.energy_calibration_table["section_minus_mapped_eV"]
            - first.energy_calibration_table["section_minus_mapped_eV"], 9.0,
        )

    def test_quadratic_recovers_subpixel_vertex_with_roi_offset(self):
        pixels = np.arange(510.0, 541.0)
        spectrum = np.maximum(100.0 - (pixels - 524.3)**2, 0.0)
        cfg = replace(config_for("unused.h5"), center_half_window_px=3)
        peak = pg.peak_metrics(spectrum[None, :], np.ones(pixels.size), pixels, cfg)
        self.assertAlmostEqual(peak["center_pixel"][0], 524.3, places=10)

    def test_legacy_methods_match_and_preserve_qc(self):
        pixels = np.arange(510.0, 541.0)
        spectra = (3 + 50 * np.exp(-0.5 * ((pixels - 524.3) / 2)**2))[None, :]
        cfg = replace(config_for("unused.h5"), center_half_window_px=3)
        baseline = pg.peak_metrics(spectra, np.ones(pixels.size), pixels, cfg)
        for method in ("com", "peak", "smooth_peak", "gaussian_logfit", "local_com"):
            with self.subTest(method=method):
                result = pg.peak_metrics(spectra, np.ones(pixels.size), pixels, replace(cfg, center_method=method))
                if method != "local_com":
                    expected, _ = sxa.estimate_center_pixels(spectra, method=method, half_window=3)
                    np.testing.assert_allclose(result["center_pixel"], expected + pixels[0])
                else:
                    i = np.argmax(spectra[0])
                    y = spectra[0, i-3:i+4]
                    w = np.clip(y - np.percentile(y, 20), 0, None)
                    expected = np.sum(pixels[i-3:i+4] * w) / np.sum(w)
                    self.assertAlmostEqual(result["center_pixel"][0], expected)
                for key in ("peak_height", "prominence", "peak_snr", "local_bg"):
                    np.testing.assert_allclose(result[key], baseline[key])

    def test_quadratic_invalid_window_falls_back_to_com(self):
        pixels = np.arange(520.0, 529.0)
        spectra = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
        cfg = replace(config_for("unused.h5"), center_half_window_px=3, prominence_outer_px=2)
        quadratic = pg.peak_metrics(spectra, np.ones(9), pixels, cfg)
        com = pg.peak_metrics(spectra, np.ones(9), pixels, replace(cfg, center_method="com"))
        self.assertEqual(quadratic["center_pixel"][0], 522.0)
        np.testing.assert_allclose(quadratic["center_pixel"], com["center_pixel"])

    def test_nonfinite_and_edge_peaks_are_rejected(self):
        pixels = np.arange(520.0, 529.0)
        spectra = np.array([[np.nan] * 9, [10, 2, 0, 0, 0, 0, 0, 0, 0]])
        for method in ("smooth_peak", "local_com"):
            result = pg.peak_metrics(spectra, np.ones(9), pixels, config_for("unused.h5", center_method=method))
            self.assertTrue(np.all(np.isnan(result["center_pixel"])))

    def test_local_baseline_option_is_independent_of_side_background_qc(self):
        pixels = np.arange(520.0, 529.0)
        spectra = np.array([[0, 0, 1, 3, 10, 6, 2, 0, 0]], dtype=float)
        cfg = config_for("unused.h5", center_method="local_com", center_half_window_px=2)
        low = pg.peak_metrics(spectra, np.ones(9), pixels, replace(cfg, center_baseline_percentile=0))
        high = pg.peak_metrics(spectra, np.ones(9), pixels, replace(cfg, center_baseline_percentile=50))
        self.assertNotAlmostEqual(low["center_pixel"][0], high["center_pixel"][0])
        np.testing.assert_equal(low["peak_snr"], high["peak_snr"])

    def test_invalid_mapping_and_center_settings_fail_before_file_access(self):
        cases = [
            ({"center_method": "unknown"}, "center_method"),
            ({"center_half_window_px": -1}, "center_half_window_px"),
            ({"center_baseline_percentile": 101}, "center_baseline_percentile"),
            ({"inverse_map_range_ev": (290, 260)}, "inverse_map_range_ev"),
            ({"inverse_map_samples": 1}, "inverse_map_samples"),
            ({"pixel_map_calibration": {"pixel": 524, "energy_eV": -1}}, "pixel_map_calibration"),
        ]
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                pg.compute_processed_qc(pg.ProcessGoodConfig("does_not_exist.h5", **options))

    def test_saved_mapping_and_downstream_readers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path)
            for source in ("hall", "tunnel"):
                with self.subTest(source=source):
                    cfg = config_for(path, gmd_source=source)
                    result = pg.compute_processed_qc(cfg)
                    saved = pg.save_good_static_xas(result)
                    with h5py.File(saved, "r") as h5:
                        centers = h5["qc/peak_center_pixel"][0, :3]
                        expected = pg.map_pixels_to_energy(centers, cfg)
                        np.testing.assert_allclose(h5["energy_vls_calibrated"][0, :3], expected)
                        np.testing.assert_allclose(h5["ml/converted_energy"][0, :3], expected)
                        groups = h5["ml/converted_proxy_bin_index"][0, :3]
                        np.testing.assert_allclose(h5["ml/converted_energy_group"][0, :3], result.proxy_energy_eV[groups])
                        np.testing.assert_allclose(result.proxy_energy_eV[groups], expected)
                        energy = h5["energy"]
                        self.assertEqual(energy.attrs["vls_calibration_method"], "physical_grating")
                        self.assertEqual(energy.attrs["center_method"], "smooth_peak")
                        self.assertEqual(dict(energy["pixel_map_config"].attrs), cfg.pixel_map_config)
                        self.assertEqual(dict(energy["pixel_map_calibration"].attrs), cfg.pixel_map_calibration)
                        self.assertNotIn("vls_pixel_to_energy_coefficients", energy)
                        self.assertNotIn("vls_calibration_order", energy.attrs)
                        np.testing.assert_allclose(energy["section_center_mapped_energy_eV"][:],
                                                   pg.map_pixels_to_energy(result.section_center_pixel, cfg))
                    np.testing.assert_allclose(gx.read_good_shots(saved, gmd_source=source).converted_energy, expected)
                    np.testing.assert_allclose(gr.read_reference_shots(saved, gmd_source=source).converted_energy, expected)
                    xas = gx.compute_good_xas(gx.GoodXASConfig(
                        saved, gmd_source=source, gmd_edges=(0.0, 40.0), gmd_max_threshold=None,
                        unit_reference_window_ev=(280.0, 286.0),
                    ))
                    self.assertEqual(int(np.sum(xas.converted.counts)), 3)
                    self.assertTrue(np.all(np.isfinite(xas.converted.od)))
                    reference = gr.compute_reference_xas(gr.ReferenceXASConfig(
                        saved, saved, gmd_source=source, gmd_edges=(0.0, 40.0), gmd_max_threshold=None,
                    ))
                    np.testing.assert_allclose(reference.nominal.absorbance, 0.0, atol=1e-12)
                    np.testing.assert_allclose(reference.converted.absorbance, 0.0, atol=1e-12)

    def test_diagnostic_curve_uses_physical_map(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path)
            result = pg.compute_processed_qc(config_for(path))
            fig, axes = pg.plot_energy_calibration_summary(result)
            try:
                line = axes[0].lines[1]
                np.testing.assert_allclose(line.get_ydata(), pg.map_pixels_to_energy(line.get_xdata(), result.config))
                self.assertEqual(line.get_label(), "fixed physical map")
                self.assertNotIn("fit", axes[1].get_ylabel())
            finally:
                plt.close(fig)

    def test_notebook_exposes_all_config_fields(self):
        import ast
        import json
        path = REPO_ROOT / "analysis/notebooks/xas_static/process_processed_data_xas_static.ipynb"
        nb = json.loads(path.read_text())
        source = "".join(nb["cells"][1]["source"])
        tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "ProcessGoodConfig"]
        self.assertEqual({kw.arg for kw in calls[0].keywords}, {f.name for f in fields(pg.ProcessGoodConfig)})

    def test_default_hall_behavior_uses_hall_gmd_for_fit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path)

            result = pg.compute_processed_qc(config_for(path))

        self.assertEqual(result.config.gmd_source, "hall")
        self.assertAlmostEqual(result.fit_nominal_after.intercept[0], 15.0, places=10)

    def test_tunnel_gmd_source_uses_tunnel_for_fit_and_correction(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path)

            result = pg.compute_processed_qc(config_for(path, gmd_source="tunnel"))
            saved = pg.save_good_static_xas(result)
            with h5py.File(saved, "r") as h5:
                corrected = h5["trans_intercept_corrected_nominal"][0, :3]
                hall_out = h5["gmd"][0, :3]
                tunnel_out = h5["gmd_tunnel"][0, :3]
                attrs = dict(h5.attrs)
                fit_attrs = dict(h5["fits"].attrs)

        np.testing.assert_allclose(corrected, np.array([20.0, 40.0, 60.0]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hall_out, np.array([1.0, 2.0, 4.0]))
        np.testing.assert_allclose(tunnel_out, np.array([10.0, 20.0, 30.0]))
        self.assertAlmostEqual(result.fit_nominal_after.intercept[0], 5.0, places=10)
        self.assertEqual(attrs["processing_gmd_source"], "tunnel")
        self.assertEqual(attrs["processing_gmd_dataset"], "ml/gmd_tunnel_channels[..., 0]")
        self.assertEqual(fit_attrs["gmd_source"], "tunnel")

    def test_missing_tunnel_source_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run12345_static_xas.h5"
            write_processed_h5(path, include_tunnel=False)

            with self.assertRaisesRegex(KeyError, "gmd_tunnel_channels.*processed-to-good"):
                pg.compute_processed_qc(config_for(path, gmd_source="tunnel"))

    def test_tunnel_default_output_path_gets_suffix(self):
        cfg = pg.ProcessGoodConfig("run12345_static_xas.h5", gmd_source="tunnel")
        self.assertEqual(pg._default_output_path(cfg).name, "run12345_good_static_xas_tunnel.h5")

    def test_hall_default_output_path_is_unchanged(self):
        cfg = pg.ProcessGoodConfig("run12345_static_xas.h5")
        self.assertEqual(pg._default_output_path(cfg).name, "run12345_good_static_xas.h5")


if __name__ == "__main__":
    unittest.main()
