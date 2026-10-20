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

import static_xas_good_reference as gr


def write_good_h5(
    path,
    *,
    nominal_energies,
    gmd_rows,
    tunnel_gmd_rows=None,
    trans_nominal_rows,
    trans_converted_rows=None,
    converted_rows=None,
    center_rows=None,
):
    nominal_energies = np.asarray(nominal_energies, dtype=np.float64)
    n_sections = nominal_energies.size
    n_shots = np.asarray([len(row) for row in gmd_rows], dtype=np.int64)
    max_shots = int(n_shots.max()) if n_shots.size else 0
    if tunnel_gmd_rows is None:
        tunnel_gmd_rows = gmd_rows
    if trans_converted_rows is None:
        trans_converted_rows = trans_nominal_rows
    if converted_rows is None:
        converted_rows = [
            np.full(len(row), nominal_energies[i], dtype=np.float64)
            for i, row in enumerate(gmd_rows)
        ]
    if center_rows is None:
        center_rows = [np.zeros(len(row), dtype=np.float64) for row in gmd_rows]

    def filled(rows, fill=np.nan):
        out = np.full((n_sections, max_shots), fill, dtype=np.float64)
        for i, row in enumerate(rows):
            row = np.asarray(row, dtype=np.float64)
            out[i, : row.size] = row
        return out

    gmd = filled(gmd_rows)
    gmd_tunnel = filled(tunnel_gmd_rows)
    trans_nominal = filled(trans_nominal_rows)
    trans_converted = filled(trans_converted_rows)
    converted = filled(converted_rows)
    centers = filled(center_rows)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("n_shots", data=n_shots)
        h5.create_dataset("nominal_energies", data=nominal_energies)
        h5.create_dataset("gmd", data=gmd)
        h5.create_dataset("gmd_tunnel", data=gmd_tunnel)
        h5.create_dataset("energy_vls_calibrated", data=converted)
        h5.create_dataset("vls_pixels", data=np.arange(3, dtype=np.float64))
        h5.create_dataset("trans_intercept_corrected_nominal", data=trans_nominal)
        h5.create_dataset("trans_intercept_corrected_converted", data=trans_converted)

        # Misleading raw datasets prove the helper does not use them.
        h5.create_dataset("trans_raw", data=np.full_like(trans_nominal, 9999.0))
        h5.create_dataset("vls", data=np.full((n_sections, max_shots, 3), 123.0))

        qc = h5.create_group("qc")
        qc.create_dataset("peak_center_pixel", data=centers)


class StaticXASGoodReferenceTests(unittest.TestCase):
    def test_nominal_absorbance_uses_mean_reference_over_mean_sample(self):
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / "sample.h5"
            reference = Path(td) / "reference.h5"
            write_good_h5(
                sample,
                nominal_energies=[271.0],
                gmd_rows=[[1.2, 1.8]],
                trans_nominal_rows=[[10.0, 20.0]],
            )
            write_good_h5(
                reference,
                nominal_energies=[271.0],
                gmd_rows=[[1.3, 1.7]],
                trans_nominal_rows=[[100.0, 200.0]],
            )

            result = gr.compute_reference_xas(
                gr.ReferenceXASConfig(sample, reference, gmd_edges=[1.0, 2.0])
            )

        self.assertEqual(result.nominal.absorbance.shape, (1, 1))
        self.assertEqual(result.nominal.sample_counts[0, 0], 2)
        self.assertEqual(result.nominal.reference_counts[0, 0], 2)
        self.assertAlmostEqual(result.nominal.absorbance[0, 0], 1.0)

    def test_gmd_final_edge_is_included_in_last_bin(self):
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / "sample.h5"
            reference = Path(td) / "reference.h5"
            write_good_h5(
                sample,
                nominal_energies=[271.0],
                gmd_rows=[[2.0]],
                trans_nominal_rows=[[10.0]],
            )
            write_good_h5(
                reference,
                nominal_energies=[271.0],
                gmd_rows=[[2.0]],
                trans_nominal_rows=[[100.0]],
            )

            result = gr.compute_reference_xas(
                gr.ReferenceXASConfig(sample, reference, gmd_edges=[1.0, 2.0])
            )

        self.assertEqual(result.nominal.sample_counts[0, 0], 1)
        self.assertEqual(result.nominal.reference_counts[0, 0], 1)
        self.assertAlmostEqual(result.nominal.absorbance[0, 0], 1.0)

    def test_converted_reference_interpolation_is_nan_outside_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / "sample.h5"
            reference = Path(td) / "reference.h5"
            write_good_h5(
                sample,
                nominal_energies=[271.0],
                gmd_rows=[[1.5, 1.5, 1.5]],
                trans_nominal_rows=[[10.0, 10.0, 10.0]],
                trans_converted_rows=[[10.0, 10.0, 10.0]],
                converted_rows=[[270.0, 271.0, 272.0]],
                center_rows=[[0.0, 1.0, 2.0]],
            )
            write_good_h5(
                reference,
                nominal_energies=[271.0],
                gmd_rows=[[1.5]],
                trans_nominal_rows=[[100.0]],
                trans_converted_rows=[[100.0]],
                converted_rows=[[271.0]],
                center_rows=[[1.0]],
            )

            result = gr.compute_reference_xas(
                gr.ReferenceXASConfig(sample, reference, gmd_edges=[1.0, 2.0])
            )

        np.testing.assert_allclose(result.converted.x, np.array([270.0, 271.0, 272.0]))
        self.assertTrue(np.isnan(result.converted.absorbance[0, 0]))
        self.assertAlmostEqual(result.converted.absorbance[1, 0], 1.0)
        self.assertTrue(np.isnan(result.converted.absorbance[2, 0]))

    def test_corrected_transmission_datasets_are_used_without_unit_scale(self):
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / "sample.h5"
            reference = Path(td) / "reference.h5"
            write_good_h5(
                sample,
                nominal_energies=[271.0],
                gmd_rows=[[1.5]],
                trans_nominal_rows=[[7.0]],
                trans_converted_rows=[[5.0]],
                converted_rows=[[271.0]],
                center_rows=[[1.0]],
            )
            write_good_h5(
                reference,
                nominal_energies=[271.0],
                gmd_rows=[[1.5]],
                trans_nominal_rows=[[70.0]],
                trans_converted_rows=[[50.0]],
                converted_rows=[[271.0]],
                center_rows=[[1.0]],
            )

            result = gr.compute_reference_xas(
                gr.ReferenceXASConfig(sample, reference, gmd_edges=[1.0, 2.0])
            )

        self.assertAlmostEqual(result.nominal.absorbance[0, 0], 1.0)
        self.assertAlmostEqual(result.converted.absorbance[0, 0], 1.0)
        self.assertNotIn("vls_per_uJ", result.nominal.metadata)
        self.assertNotIn("vls_per_uJ", result.converted.metadata)

    def test_tunnel_gmd_source_bins_by_gmd_tunnel(self):
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / "sample.h5"
            reference = Path(td) / "reference.h5"
            write_good_h5(
                sample,
                nominal_energies=[271.0],
                gmd_rows=[[9.5]],
                tunnel_gmd_rows=[[1.5]],
                trans_nominal_rows=[[10.0]],
            )
            write_good_h5(
                reference,
                nominal_energies=[271.0],
                gmd_rows=[[9.5]],
                tunnel_gmd_rows=[[1.5]],
                trans_nominal_rows=[[100.0]],
            )

            result = gr.compute_reference_xas(
                gr.ReferenceXASConfig(sample, reference, gmd_edges=[1.0, 2.0], gmd_source="tunnel")
            )

        self.assertEqual(result.nominal.sample_counts[0, 0], 1)
        self.assertEqual(result.nominal.reference_counts[0, 0], 1)
        self.assertAlmostEqual(result.nominal.absorbance[0, 0], 1.0)
        self.assertEqual(result.nominal.metadata["gmd_source"], "tunnel")
        self.assertEqual(result.nominal.metadata["gmd_dataset"], "gmd_tunnel")


if __name__ == "__main__":
    unittest.main()
