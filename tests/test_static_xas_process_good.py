import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "analysis" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import static_xas_process_good as pg


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
        h5.create_dataset("vls_pixels", data=np.arange(template.size, dtype=np.float64))

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
        energy_calibration_poly_order=0,
        energy_calibration_min_sections=1,
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
