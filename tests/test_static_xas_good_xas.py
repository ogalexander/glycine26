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

import static_xas_good_xas as gx


def write_good_h5(
    path,
    *,
    hall_gmd,
    tunnel_gmd=None,
    trans_raw=None,
    nominal_energy=271.0,
):
    hall_gmd = np.asarray(hall_gmd, dtype=np.float64)
    n = hall_gmd.size
    if tunnel_gmd is not None:
        tunnel_gmd = np.asarray(tunnel_gmd, dtype=np.float64)
        if tunnel_gmd.size != n:
            raise ValueError("tunnel_gmd must have the same length as hall_gmd")
    if trans_raw is None:
        trans_raw = np.ones(n, dtype=np.float64)
    trans_raw = np.asarray(trans_raw, dtype=np.float64)
    if trans_raw.size != n:
        raise ValueError("trans_raw must have the same length as hall_gmd")

    shape = (1, n)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("n_shots", data=np.array([n], dtype=np.int64))
        h5.create_dataset("nominal_energies", data=np.array([nominal_energy], dtype=np.float64))
        h5.create_dataset("gmd", data=hall_gmd.reshape(shape))
        if tunnel_gmd is not None:
            h5.create_dataset("gmd_tunnel", data=tunnel_gmd.reshape(shape))
        h5.create_dataset("energy_vls_calibrated", data=np.full(shape, nominal_energy, dtype=np.float64))
        h5.create_dataset("vls_pixels", data=np.arange(max(n, 1), dtype=np.float64))
        h5.create_dataset("trans_raw", data=trans_raw.reshape(shape))

        qc = h5.create_group("qc")
        qc.create_dataset("peak_center_pixel", data=np.arange(n, dtype=np.float64).reshape(shape))


class StaticXASGoodXASTests(unittest.TestCase):
    def test_default_gmd_source_reads_hall_gmd(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "good.h5"
            write_good_h5(path, hall_gmd=[1.0, 2.0], tunnel_gmd=[10.0, 20.0])

            shots = gx.read_good_shots(path)

        np.testing.assert_allclose(shots.gmd, np.array([1.0, 2.0]))
        self.assertEqual(shots.attrs["selected_gmd_source"], "hall")
        self.assertEqual(shots.attrs["selected_gmd_dataset"], "/gmd")

    def test_tunnel_gmd_source_reads_gmd_tunnel(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "good.h5"
            write_good_h5(path, hall_gmd=[1.0, 2.0], tunnel_gmd=[10.0, 20.0])

            shots = gx.read_good_shots(path, gmd_source="tunnel")

        np.testing.assert_allclose(shots.gmd, np.array([10.0, 20.0]))
        self.assertEqual(shots.attrs["selected_gmd_source"], "tunnel")
        self.assertEqual(shots.attrs["selected_gmd_dataset"], "/gmd_tunnel")

    def test_tunnel_source_missing_dataset_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "good.h5"
            write_good_h5(path, hall_gmd=[1.0, 2.0])

            with self.assertRaisesRegex(KeyError, "gmd_tunnel.*processed-to-good"):
                gx.read_good_shots(path, gmd_source="tunnel")

    def test_invalid_gmd_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "good.h5"
            write_good_h5(path, hall_gmd=[1.0, 2.0], tunnel_gmd=[10.0, 20.0])

            with self.assertRaisesRegex(ValueError, "gmd_source"):
                gx.compute_good_xas(gx.GoodXASConfig(path, gmd_source="foo"))

    def test_od_formula_uses_selected_gmd_over_vls_uj(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "good.h5"
            tunnel = np.array([2.0, 3.0, 5.0])
            trans = np.array([10.0, 10.0, 10.0])
            write_good_h5(path, hall_gmd=[1.0, 1.0, 1.0], tunnel_gmd=tunnel, trans_raw=trans)

            result = gx.compute_good_xas(
                gx.GoodXASConfig(
                    path,
                    energy_window_ev=(270.5, 271.5),
                    unit_scale_axis="nominal",
                    unit_reference_window_ev=(270.5, 271.5),
                    gmd_edges=(0.0, 10.0),
                    gmd_min_threshold=None,
                    gmd_max_threshold=None,
                    correction_mode="none",
                    gmd_source="tunnel",
                )
            )

        expected = np.log10(np.sum(tunnel) / np.sum(trans / result.unit_scale.vls_per_uJ))
        self.assertEqual(result.nominal.counts[0, 0], 3)
        self.assertAlmostEqual(result.nominal.gmd_sum[0, 0], float(np.sum(tunnel)))
        self.assertAlmostEqual(result.nominal.od[0, 0], expected)
        self.assertEqual(result.nominal.metadata["gmd_source"], "tunnel")
        self.assertEqual(result.nominal.metadata["gmd_dataset"], "gmd_tunnel")


if __name__ == "__main__":
    unittest.main()
