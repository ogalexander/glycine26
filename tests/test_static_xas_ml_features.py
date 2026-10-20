import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML" / "src"))

from static_xas_ml.data import GoodMLTable
from static_xas_ml.features import (
    DEFAULT_HALL_CORE_FEATURES,
    build_feature_matrix,
    filter_table_rows,
    sanitize_selected_feature_columns,
    select_feature_columns,
)
from static_xas_ml.splits import make_reference_energy_block_split


def _table(features: np.ndarray, *, train_ids=None, gmd=None, nominal_energy=None) -> GoodMLTable:
    n = int(features.shape[0])
    if train_ids is None:
        train_ids = np.arange(n, dtype=np.int64)
    if gmd is None:
        gmd = np.linspace(1.0, 3.0, n, dtype=np.float32)
    if nominal_energy is None:
        nominal_energy = np.full(n, 271.0, dtype=np.float64)
    return GoodMLTable(
        good_h5=Path("good.h5"),
        processed_h5=Path("processed.h5"),
        correction_axis="converted",
        gmd_source="hall",
        spectra=np.ones((n, 4), dtype=np.float32),
        trans=np.ones(n, dtype=np.float32) * 4,
        gmd=np.asarray(gmd, dtype=np.float32),
        gmd_tunnel=np.asarray(gmd, dtype=np.float32) + 1,
        features=np.asarray(features, dtype=np.float32),
        feature_names=["a", "b", "qc_flag"] + DEFAULT_HALL_CORE_FEATURES,
        feature_units=["", "", ""] + ["au"] * len(DEFAULT_HALL_CORE_FEATURES),
        feature_roles=["predictor", "predictor", "qc"] + ["predictor"] * len(DEFAULT_HALL_CORE_FEATURES),
        qc_ok=np.ones(n, dtype=bool),
        train_id=np.asarray(train_ids, dtype=np.int64),
        run_id=np.full(n, 58780, dtype=np.int64),
        section_index=np.zeros(n, dtype=np.int32),
        source_shot_index=np.arange(n, dtype=np.int64),
        vls_bunch_index=np.arange(n, dtype=np.int16),
        gmd_bunch_index=np.arange(n, dtype=np.int16),
        nominal_energy=np.asarray(nominal_energy, dtype=np.float64),
        converted_energy=np.asarray(nominal_energy, dtype=np.float64) + 0.1,
        converted_energy_group=np.asarray(nominal_energy, dtype=np.float64) + 0.125,
        converted_proxy_bin_index=np.arange(n, dtype=np.int32),
        vls_pixels=np.arange(4, dtype=np.float64),
    )


class StaticXASMLFeatureTests(unittest.TestCase):
    def test_default_hall_core_features_are_selected(self):
        n_features = 3 + len(DEFAULT_HALL_CORE_FEATURES)
        table = _table(np.arange(5 * n_features, dtype=np.float32).reshape(5, n_features))

        columns = select_feature_columns(table.feature_names, table.feature_roles)

        self.assertEqual([table.feature_names[i] for i in columns], DEFAULT_HALL_CORE_FEATURES)

    def test_qc_feature_selection_is_rejected(self):
        table = _table(np.ones((5, 3 + len(DEFAULT_HALL_CORE_FEATURES)), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "QC/status"):
            select_feature_columns(
                table.feature_names,
                table.feature_roles,
                selected_feature_names=["qc_flag"],
            )

    def test_sanitize_drops_constant_reference_feature(self):
        ref = _table(
            np.asarray(
                [
                    [1.0, 5.0, 0.0],
                    [2.0, 5.0, 0.0],
                    [3.0, 5.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        sam = _table(
            np.asarray(
                [
                    [1.5, 6.0, 0.0],
                    [2.5, 7.0, 0.0],
                    [3.5, 8.0, 0.0],
                ],
                dtype=np.float32,
            )
        )

        columns, dropped = sanitize_selected_feature_columns(ref, sam, np.asarray([0, 1]))

        np.testing.assert_array_equal(columns, [0])
        self.assertEqual(dropped[0]["feature_name"], "b")

    def test_filter_uses_hall_gmd_and_selected_feature_finiteness(self):
        n_features = 3 + len(DEFAULT_HALL_CORE_FEATURES)
        features = np.ones((4, n_features), dtype=np.float32)
        features[1, 3] = np.nan
        table = _table(features, gmd=np.asarray([1.0, 2.0, 11.0, 4.0]))

        filtered, counts = filter_table_rows(
            table,
            selected_columns=np.asarray([3]),
            gmd_max_threshold=10.0,
        )

        self.assertEqual(counts["kept_rows"], 2)
        np.testing.assert_allclose(filtered.gmd, [1.0, 4.0])
        np.testing.assert_allclose(build_feature_matrix(filtered, np.asarray([3])), [[1.0], [1.0]])

    def test_reference_split_has_no_train_id_leakage(self):
        n_features = 3 + len(DEFAULT_HALL_CORE_FEATURES)
        train_ids = np.concatenate([np.arange(100, 106), np.arange(200, 206)])
        energy = np.concatenate([np.full(6, 271.0), np.full(6, 272.0)])
        table = _table(
            np.ones((12, n_features), dtype=np.float32),
            train_ids=train_ids,
            nominal_energy=energy,
        )

        train, val, test, policy = make_reference_energy_block_split(table)

        self.assertEqual(policy, "per_nominal_energy_contiguous_train_id_block_split")
        self.assertTrue(np.any(train))
        self.assertTrue(np.any(val))
        self.assertTrue(np.any(test))
        train_ids_set = set(table.train_id[train])
        val_ids_set = set(table.train_id[val])
        test_ids_set = set(table.train_id[test])
        self.assertFalse(train_ids_set & val_ids_set)
        self.assertFalse(train_ids_set & test_ids_set)
        self.assertFalse(val_ids_set & test_ids_set)


if __name__ == "__main__":
    unittest.main()
