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

import static_xas_analysis as sxa


def gaussian(center, sigma=3.0, amp=100.0, n=80):
    x = np.arange(n, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class StaticXASAnalysisTests(unittest.TestCase):
    def make_run(self):
        vls = np.full((1, 5, 20), np.nan, dtype=np.float64)
        gmd = np.full((1, 5), np.nan, dtype=np.float64)
        vls[0, 0] = np.ones(20) * 2.0
        vls[0, 1] = np.ones(20) * 4.0
        vls[0, 2] = np.ones(20) * 6.0
        vls[0, 3] = np.ones(20) * 8.0
        vls[0, 4] = np.ones(20) * 0.1
        gmd[0, :5] = [1.0, 2.0, 3.0, 5.0, 2.0]
        return sxa.StaticXASRun(
            path=None,
            vls=vls,
            gmd=gmd,
            n_shots=np.array([5]),
            nominal_energies=np.array([284.0]),
            vls_pixels=np.arange(20),
            attrs={},
        )

    def test_load_static_xas_run(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mini.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("vls", data=np.zeros((1, 2, 3)))
                f.create_dataset("gmd", data=np.ones((1, 2)))
                f.create_dataset("n_shots", data=np.array([2]))
                f.create_dataset("nominal_energies", data=np.array([284.0]))
                f.create_dataset("vls_pixels", data=np.array([10, 11, 12]))
                f.attrs["mode"] = "xas_static"
            run = sxa.load_static_xas_run(path)
        self.assertEqual(run.vls.shape, (1, 2, 3))
        self.assertEqual(run.attrs["mode"], "xas_static")

    def test_center_methods_recover_single_gaussian(self):
        spectra = np.vstack([gaussian(35.4, sigma=4.0, amp=200.0)])
        com = sxa.center_com(spectra)[0]
        peak = sxa.center_peak(spectra)[0]
        smooth = sxa.center_smooth_peak(spectra, half_window=6)[0]
        logfit = sxa.center_gaussian_logfit(spectra, half_window=6)[0]
        self.assertAlmostEqual(com, 35.4, delta=0.2)
        self.assertAlmostEqual(peak, 35.0, delta=1.0)
        self.assertAlmostEqual(smooth, 35.4, delta=0.3)
        self.assertAlmostEqual(logfit, 35.4, delta=0.4)

    def test_windowed_com_uses_local_peak_window(self):
        spectrum = gaussian(20.0, sigma=2.0, amp=120.0) + gaussian(58.0, sigma=6.0, amp=70.0)
        spectra = spectrum[None, :]
        full_com = sxa.center_com(spectra)[0]
        local_com = sxa.center_com(spectra, half_window=5)[0]
        estimated, method = sxa.estimate_center_pixels(spectra, method="com", half_window=5)
        self.assertEqual(method, "com")
        self.assertGreater(full_com, 30.0)
        self.assertAlmostEqual(local_com, 20.0, delta=0.2)
        self.assertAlmostEqual(estimated[0], local_com, delta=1e-12)

    def test_peak_ignores_center_window(self):
        spectrum = np.zeros(80, dtype=np.float64)
        spectrum[12] = 100.0
        spectrum[60] = 95.0
        spectra = spectrum[None, :]
        center, method = sxa.estimate_center_pixels(spectra, method="peak", half_window=1)
        self.assertEqual(method, "peak")
        self.assertEqual(float(center[0]), 12.0)

    def test_low_intensity_rejection(self):
        run = self.make_run()
        prepared = sxa.prepare_proxy_shots(run, vls_intensity_threshold=1.0, center_method="peak")
        self.assertEqual(prepared.n_shots, 4)
        self.assertEqual(prepared.summary["after_peak_threshold"], 4)

    def test_low_gmd_filter_rejects_small_gmd_shots(self):
        run = self.make_run()
        prepared = sxa.prepare_proxy_shots(
            run,
            vls_intensity_threshold=1.0,
            gmd_min_threshold=2.5,
            center_method="peak",
        )
        self.assertEqual(prepared.n_shots, 2)
        self.assertEqual(prepared.summary["after_gmd_filter"], 2)
        self.assertEqual(prepared.summary["low_gmd_rejected"], 2)

    def test_high_gmd_filter_rejects_large_gmd_shots(self):
        run = self.make_run()
        prepared = sxa.prepare_proxy_shots(
            run,
            vls_intensity_threshold=1.0,
            gmd_max_threshold=2.0,
            center_method="peak",
        )
        self.assertEqual(prepared.n_shots, 2)
        self.assertEqual(prepared.summary["after_gmd_filter"], 2)
        self.assertEqual(prepared.summary["high_gmd_rejected"], 2)

    def test_gmd_range_filter_keeps_only_in_range_shots(self):
        run = self.make_run()
        prepared = sxa.prepare_proxy_shots(
            run,
            vls_intensity_threshold=1.0,
            gmd_min_threshold=1.5,
            gmd_max_threshold=3.5,
            center_method="peak",
        )
        self.assertEqual(prepared.n_shots, 2)
        self.assertTrue(np.all((prepared.gmd >= 1.5) & (prepared.gmd <= 3.5)))
        self.assertEqual(prepared.summary["low_gmd_rejected"], 1)
        self.assertEqual(prepared.summary["high_gmd_rejected"], 1)

    def test_double_peak_detection(self):
        single = gaussian(35.0, sigma=3.0, amp=100.0)
        double = gaussian(30.0, sigma=2.0, amp=100.0) + gaussian(40.0, sigma=2.0, amp=70.0)
        xcorr, xsm, noise = sxa.preprocess_shots_for_peakfinding(
            np.vstack([single, double]), sg_window=9, sg_poly=2
        )
        diag = sxa.detect_double_peaks_pairscan(
            xcorr,
            xsm,
            noise,
            min_distance_px=3,
            max_distance_px=15,
            min_rel_peak_height=0.25,
        )
        self.assertFalse(bool(diag["is_candidate"][0]))
        self.assertTrue(bool(diag["is_candidate"][1]))

    def test_gmd_bin_ratio_uses_sum_gmd_over_sum_vls(self):
        run = self.make_run()
        binned = sxa.bin_nominal_energy_run(
            run,
            np.array([0.0, 2.0, 4.0]),
            vls_intensity_threshold=1.0,
            gmd_min_threshold=0.0,
            gmd_max_threshold=4.0,
        )
        # First bin: one shot, GMD=1, VLS sum=40 -> 0.025.
        self.assertAlmostEqual(binned.xas[0, 0], 1.0 / 40.0)
        # Second bin: two shots, GMD=2+3, VLS sum=80+120 -> 0.025.
        self.assertAlmostEqual(binned.xas[0, 1], 5.0 / 200.0)

    def test_gmd_final_edge_is_included_in_last_bin(self):
        run = self.make_run()
        binned = sxa.bin_nominal_energy_run(
            run,
            np.array([0.0, 2.0, 4.0, 5.0]),
            vls_intensity_threshold=1.0,
            gmd_min_threshold=0.0,
            gmd_max_threshold=5.0,
        )
        self.assertEqual(binned.n_per_bin[0, 2], 1)
        self.assertAlmostEqual(binned.xas[0, 2], 5.0 / 160.0)



    def make_absorbance_run(self, energies, intensities, *, gmd_values=None, vls_pixels=None):
        energies = np.asarray(energies, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        if vls_pixels is None:
            vls_pixels = np.arange(20)
        vls_pixels = np.asarray(vls_pixels, dtype=np.int64)
        n_e = energies.size
        vls = np.full((n_e, 1, vls_pixels.size), np.nan, dtype=np.float64)
        gmd = np.full((n_e, 1), np.nan, dtype=np.float64)
        for i, intensity in enumerate(intensities):
            vls[i, 0, :] = float(intensity)
            gmd[i, 0] = 1.0 if gmd_values is None else float(gmd_values[i])
        return sxa.StaticXASRun(
            path=None,
            vls=vls,
            gmd=gmd,
            n_shots=np.ones(n_e, dtype=np.int64),
            nominal_energies=energies,
            vls_pixels=vls_pixels,
            attrs={},
        )

    def test_integrated_absorbance_formula(self):
        trans = self.make_absorbance_run([284.0], [1.0])
        ref = self.make_absorbance_run([284.0], [10.0])
        curve = sxa.build_nominal_absorbance_curve(
            trans,
            ref,
            vls_intensity_threshold=None,
        )
        self.assertEqual(curve.x.size, 1)
        self.assertAlmostEqual(curve.absorbance[0], 1.0, delta=1e-12)

    def test_nominal_absorbance_uses_common_bins_only(self):
        trans = self.make_absorbance_run([281.0, 282.0], [1.0, 2.0])
        ref = self.make_absorbance_run([282.0, 283.0], [20.0, 30.0])
        curve = sxa.build_nominal_absorbance_curve(
            trans,
            ref,
            vls_intensity_threshold=None,
        )
        np.testing.assert_allclose(curve.x, np.array([282.0]))
        self.assertEqual(curve.n_trans[0], 1)
        self.assertEqual(curve.n_incident[0], 1)

    def test_duplicate_reference_nominal_energies_are_combined(self):
        trans = self.make_absorbance_run([282.0], [2.0])
        ref = self.make_absorbance_run([282.0, 282.0], [10.0, 30.0])
        curve = sxa.build_nominal_absorbance_curve(
            trans,
            ref,
            vls_intensity_threshold=None,
        )
        self.assertEqual(curve.n_incident[0], 2)
        self.assertAlmostEqual(curve.incident_intensity[0], 20.0 * 20.0)
        self.assertAlmostEqual(curve.absorbance[0], 1.0, delta=1e-12)

    def test_actual_absorbance_uses_fixed_absolute_pixel_bins(self):
        pixels = np.arange(20)
        trans = self.make_absorbance_run([284.0], [0.0], vls_pixels=pixels)
        ref = self.make_absorbance_run([284.0], [0.0], vls_pixels=pixels)
        trans.vls[0, 0, 10] = 10.0
        ref.vls[0, 0, 11] = 100.0
        config = {
            "source_to_grating_m": 0.85,
            "grating_to_screen_m": 0.5429,
            "incident_angle_deg": 86.78,
            "groove_density_lines_per_mm": 1200.0,
            "diffraction_order": 1,
            "magnification": 3.0,
            "pixel_pitch_m": 50e-6,
            "axis_sign": 1,
        }
        calibration = {"pixel": 10.5, "energy_eV": 284.2}
        curve = sxa.build_actual_absorbance_curve(
            trans,
            ref,
            config,
            calibration,
            (280.0, 288.0),
            n_samples=2000,
            pixel_bin_width=2,
            vls_intensity_threshold=None,
            center_method="peak",
        )
        self.assertEqual(curve.x.size, 1)
        self.assertAlmostEqual(curve.metadata["pixel_bin_centers"][0], 10.5)
        self.assertAlmostEqual(curve.absorbance[0], 1.0, delta=1e-12)

    def test_absorbance_does_not_use_gmd_when_filter_keeps_shots(self):
        ref = self.make_absorbance_run([284.0], [10.0], gmd_values=[1.0])
        trans_low_gmd = self.make_absorbance_run([284.0], [1.0], gmd_values=[1.0])
        trans_high_gmd = self.make_absorbance_run([284.0], [1.0], gmd_values=[9.0])
        kwargs = dict(vls_intensity_threshold=None, gmd_min_threshold=0.0, gmd_max_threshold=10.0)
        low = sxa.build_nominal_absorbance_curve(trans_low_gmd, ref, **kwargs)
        high = sxa.build_nominal_absorbance_curve(trans_high_gmd, ref, **kwargs)
        np.testing.assert_allclose(low.absorbance, high.absorbance)


    def test_nominal_gmd_binned_absorbance_formula(self):
        trans = self.make_absorbance_run([284.0, 284.0], [1.0, 2.0], gmd_values=[0.5, 1.5])
        ref = self.make_absorbance_run([284.0, 284.0], [10.0, 20.0], gmd_values=[0.5, 1.5])
        curve = sxa.build_nominal_gmd_binned_absorbance_curve(
            trans,
            ref,
            np.array([0.0, 1.0, 2.0]),
            vls_intensity_threshold=None,
        )
        self.assertEqual(curve.absorbance.shape, (1, 2))
        np.testing.assert_allclose(curve.absorbance[0], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(curve.n_trans[0], np.array([1, 1]))
        np.testing.assert_array_equal(curve.n_incident[0], np.array([1, 1]))

    def test_nominal_gmd_binned_absorbance_includes_final_edge(self):
        trans = self.make_absorbance_run([284.0], [1.0], gmd_values=[2.0])
        ref = self.make_absorbance_run([284.0], [10.0], gmd_values=[2.0])
        curve = sxa.build_nominal_gmd_binned_absorbance_curve(
            trans,
            ref,
            np.array([0.0, 1.0, 2.0]),
            vls_intensity_threshold=None,
        )
        self.assertEqual(curve.n_trans[0, 1], 1)
        self.assertEqual(curve.n_incident[0, 1], 1)
        self.assertAlmostEqual(curve.absorbance[0, 1], 1.0, delta=1e-12)

    def test_actual_gmd_binned_absorbance_formula(self):
        pixels = np.arange(20)
        trans = self.make_absorbance_run([284.0, 284.0], [0.0, 0.0], gmd_values=[0.5, 1.5], vls_pixels=pixels)
        ref = self.make_absorbance_run([284.0, 284.0], [0.0, 0.0], gmd_values=[0.5, 1.5], vls_pixels=pixels)
        trans.vls[0, 0, 10] = 10.0
        trans.vls[1, 0, 12] = 20.0
        ref.vls[0, 0, 10] = 100.0
        ref.vls[1, 0, 12] = 200.0
        config = {
            "source_to_grating_m": 0.85,
            "grating_to_screen_m": 0.5429,
            "incident_angle_deg": 86.78,
            "groove_density_lines_per_mm": 1200.0,
            "diffraction_order": 1,
            "magnification": 3.0,
            "pixel_pitch_m": 50e-6,
            "axis_sign": 1,
        }
        calibration = {"pixel": 10.0, "energy_eV": 284.2}
        curve = sxa.build_actual_gmd_binned_absorbance_curve(
            trans,
            ref,
            np.array([0.0, 1.0, 2.0]),
            config,
            calibration,
            (280.0, 288.0),
            n_samples=2000,
            pixel_bin_width=1,
            vls_intensity_threshold=None,
            center_method="peak",
        )
        self.assertEqual(curve.absorbance.shape, (2, 2))
        pixel_centers = np.asarray(curve.metadata["pixel_bin_centers"])
        row10 = int(np.where(np.isclose(pixel_centers, 10.0))[0][0])
        row12 = int(np.where(np.isclose(pixel_centers, 12.0))[0][0])
        self.assertAlmostEqual(curve.absorbance[row10, 0], 1.0, delta=1e-12)
        self.assertTrue(np.isnan(curve.absorbance[row10, 1]))
        self.assertTrue(np.isnan(curve.absorbance[row12, 0]))
        self.assertAlmostEqual(curve.absorbance[row12, 1], 1.0, delta=1e-12)

    def test_pixel_to_energy_wrapper_hits_calibration_anchor(self):
        config = {
            "source_to_grating_m": 0.85,
            "grating_to_screen_m": 0.5429,
            "incident_angle_deg": 86.78,
            "groove_density_lines_per_mm": 1200.0,
            "diffraction_order": 1,
            "magnification": 3.0,
            "pixel_pitch_m": 50e-6,
            "axis_sign": 1,
        }
        calibration = {"pixel": 524.0, "energy_eV": 284.2}
        energy, _, _ = sxa.convert_pixels_to_energy(
            np.array([524.0]), config, calibration, (280.0, 288.0), n_samples=2000
        )
        self.assertAlmostEqual(float(energy[0]), 284.2, delta=0.01)


if __name__ == "__main__":
    unittest.main()
