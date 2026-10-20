import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML" / "src"))

from static_xas_ml.data import load_good_ml_h5


def _write_string_dataset(group, name, values):
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _write_processed(path: Path, *, train_ids=None) -> None:
    if train_ids is None:
        train_ids = np.asarray([[10, 11, 12, 13]], dtype=np.int64)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("n_shots", data=np.asarray([4], dtype=np.int64))
        ml = h5.create_group("ml")
        features = np.asarray(
            [
                [
                    [1.0, 10.0, 100.0],
                    [2.0, 20.0, 200.0],
                    [3.0, 30.0, 300.0],
                    [4.0, 40.0, 400.0],
                ]
            ],
            dtype=np.float32,
        )
        ml.create_dataset("features", data=features)
        _write_string_dataset(ml, "feature_names", ["f0", "f1", "f2"])
        _write_string_dataset(ml, "feature_units", ["au", "mm", "eV"])
        _write_string_dataset(ml, "feature_roles", ["predictor", "predictor", "candidate"])
        ml.create_dataset("train_id", data=train_ids)
        ml.create_dataset("vls_bunch_index", data=np.asarray([[20, 21, 22, 23]], dtype=np.int16))
        ml.create_dataset("gmd_bunch_index", data=np.asarray([[0, 1, 2, 3]], dtype=np.int16))


def _write_good(path: Path, *, train_ids=None) -> None:
    if train_ids is None:
        train_ids = np.asarray([[11, 13]], dtype=np.int64)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("n_shots", data=np.asarray([2], dtype=np.int64))
        h5.create_dataset("vls_pixels", data=np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
        h5.create_dataset(
            "vls_intercept_corrected_converted",
            data=np.asarray([[[10.0, 11.0, 12.0], [30.0, 31.0, 32.0]]], dtype=np.float64),
        )
        h5.create_dataset(
            "trans_intercept_corrected_converted",
            data=np.asarray([[33.0, 93.0]], dtype=np.float64),
        )
        h5.create_dataset("gmd", data=np.asarray([[2.5, 4.5]], dtype=np.float64))
        h5.create_dataset("gmd_tunnel", data=np.asarray([[5.5, 7.5]], dtype=np.float64))
        qc = h5.create_group("qc")
        qc.create_dataset("source_shot_index", data=np.asarray([[1, 3]], dtype=np.int64))
        ml = h5.create_group("ml")
        ml.create_dataset("qc_ok", data=np.asarray([[True, True]], dtype=bool))
        ml.create_dataset("train_id", data=train_ids)
        ml.create_dataset("run_id", data=np.asarray([[58794, 58794]], dtype=np.int64))
        ml.create_dataset("section_index", data=np.asarray([[0, 0]], dtype=np.int32))
        ml.create_dataset("vls_bunch_index", data=np.asarray([[21, 23]], dtype=np.int16))
        ml.create_dataset("gmd_bunch_index", data=np.asarray([[1, 3]], dtype=np.int16))
        ml.create_dataset("nominal_energy", data=np.asarray([[271.0, 272.0]], dtype=np.float64))
        ml.create_dataset("converted_energy", data=np.asarray([[271.1, 272.1]], dtype=np.float64))
        ml.create_dataset(
            "converted_energy_group",
            data=np.asarray([[271.125, 272.125]], dtype=np.float64),
        )
        ml.create_dataset("converted_proxy_bin_index", data=np.asarray([[101, 102]], dtype=np.int32))


class StaticXASMLGoodTableTests(unittest.TestCase):
    def test_good_loader_joins_processed_features_by_source_shot_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed.h5"
            good = tmp_path / "good.h5"
            _write_processed(processed)
            _write_good(good)

            table = load_good_ml_h5(good, processed)

        np.testing.assert_allclose(table.features[:, 0], [2.0, 4.0])
        np.testing.assert_allclose(table.features[:, 1], [20.0, 40.0])
        np.testing.assert_allclose(table.spectra, [[10.0, 11.0, 12.0], [30.0, 31.0, 32.0]])
        np.testing.assert_allclose(table.trans, [33.0, 93.0])
        np.testing.assert_allclose(table.gmd, [2.5, 4.5])
        np.testing.assert_array_equal(table.source_shot_index, [1, 3])

    def test_good_loader_can_select_tunnel_gmd(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed.h5"
            good = tmp_path / "good.h5"
            _write_processed(processed)
            _write_good(good)

            table = load_good_ml_h5(good, processed, gmd_source="tunnel")

        np.testing.assert_allclose(table.gmd, [5.5, 7.5])

    def test_good_loader_rejects_train_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed.h5"
            good = tmp_path / "good.h5"
            _write_processed(processed)
            _write_good(good, train_ids=np.asarray([[999, 13]], dtype=np.int64))

            with self.assertRaisesRegex(ValueError, "train_id mismatch"):
                load_good_ml_h5(good, processed)


if __name__ == "__main__":
    unittest.main()
