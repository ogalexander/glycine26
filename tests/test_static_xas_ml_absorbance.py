import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML" / "src"))

from static_xas_ml.absorbance import (
    baseline_slope_rows,
    bin_integrated_absorbance_by_energy,
    bin_integrated_absorbance_by_nominal,
)


class StaticXASMLAbsorbanceTests(unittest.TestCase):
    def test_nominal_absorbance_integrates_mean_spectra_then_logs(self):
        transmitted = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        incident = np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)

        df = bin_integrated_absorbance_by_nominal(
            transmitted,
            incident,
            nominal_energy=np.asarray([271.0, 271.0]),
            gmd=np.asarray([2.0, 2.5]),
            pixel_axis=np.asarray([0.0, 1.0]),
            gmd_edges=np.asarray([1.0, 3.0]),
            clip_negative=False,
        )

        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.loc[0, "trans_intensity"], 3.0)
        self.assertAlmostEqual(df.loc[0, "incident_intensity"], 30.0)
        self.assertAlmostEqual(df.loc[0, "absorbance_mean"], 1.0)

    def test_gmd_right_edge_is_included(self):
        df = bin_integrated_absorbance_by_nominal(
            np.asarray([[1.0, 1.0]], dtype=np.float64),
            np.asarray([[10.0, 10.0]], dtype=np.float64),
            nominal_energy=np.asarray([271.0]),
            gmd=np.asarray([3.0]),
            pixel_axis=np.asarray([0.0, 1.0]),
            gmd_edges=np.asarray([1.0, 2.0, 3.0]),
            clip_negative=False,
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "gmd_bin"], "2-3")

    def test_converted_energy_binner_uses_existing_energy_values(self):
        df = bin_integrated_absorbance_by_energy(
            np.asarray([[5.0, 5.0]], dtype=np.float64),
            np.asarray([[50.0, 50.0]], dtype=np.float64),
            energy=np.asarray([271.125]),
            gmd=np.asarray([2.0]),
            pixel_axis=np.asarray([0.0, 1.0]),
            gmd_edges=np.asarray([1.0, 3.0]),
            binning="converted_energy",
            x_label="Converted photon energy (eV)",
            energy_column="converted_energy_eV",
            clip_negative=False,
        )

        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.loc[0, "converted_energy_eV"], 271.125)
        self.assertAlmostEqual(df.loc[0, "absorbance_mean"], 1.0)

    def test_baseline_slope_excludes_feature_windows(self):
        df = bin_integrated_absorbance_by_energy(
            np.ones((6, 2), dtype=np.float64),
            np.asarray(
                [[10.0, 10.0], [11.0, 11.0], [50.0, 50.0], [12.0, 12.0], [60.0, 60.0], [13.0, 13.0]],
                dtype=np.float64,
            ),
            energy=np.asarray([271.0, 272.0, 273.0, 274.0, 275.0, 276.0]),
            gmd=np.full(6, 2.0),
            pixel_axis=np.asarray([0.0, 1.0]),
            gmd_edges=np.asarray([1.0, 3.0]),
            binning="converted_energy",
            x_label="Converted photon energy (eV)",
            energy_column="converted_energy_eV",
            clip_negative=False,
        )

        rows = baseline_slope_rows(df, exclude_windows=((272.7, 273.4), (274.6, 275.4)))

        self.assertEqual(rows[0]["n_points"], 4)
        self.assertTrue(np.isfinite(rows[0]["slope_absorbance_per_eV"]))


if __name__ == "__main__":
    unittest.main()
