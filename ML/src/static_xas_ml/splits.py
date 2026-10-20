"""Grouped reference splits for static-XAS ML."""

from __future__ import annotations

import numpy as np

from .data import GoodMLTable


def _validate_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=float)
    if np.any(fractions <= 0):
        raise ValueError("train/validation/test fractions must all be positive.")
    if not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("train/validation/test fractions must sum to 1.")


def _split_sorted_ids(
    ids: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(np.unique(ids)), dtype=np.int64)
    n = int(ids.size)
    if n == 0:
        return ids, ids, ids
    if n < 3:
        return ids, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    n_train = int(np.floor(n * float(train_fraction)))
    n_val = int(np.floor(n * float(validation_fraction)))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def _fill_masks_from_id_sets(
    train_id: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.isin(train_id, train_ids)
    validation_mask = np.isin(train_id, validation_ids)
    test_mask = np.isin(train_id, test_ids)
    return train_mask, validation_mask, test_mask


def _assert_disjoint(train_mask: np.ndarray, validation_mask: np.ndarray, test_mask: np.ndarray) -> None:
    if np.any(train_mask & validation_mask) or np.any(train_mask & test_mask) or np.any(validation_mask & test_mask):
        raise ValueError("Split masks overlap.")


def make_reference_energy_block_split(
    table: GoodMLTable,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    energy_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Split reference rows by contiguous train-ID blocks inside each nominal energy.

    If a pathological run reuses train IDs across nominal energies and that
    creates split conflicts, the function falls back to one global train-ID
    block split to preserve the stronger no-train-leakage invariant.
    """
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if table.n_rows < 3:
        raise ValueError("Need at least three reference rows for a split.")

    energy_key = np.round(np.asarray(table.nominal_energy, dtype=np.float64), int(energy_decimals))
    train_mask = np.zeros(table.n_rows, dtype=bool)
    validation_mask = np.zeros(table.n_rows, dtype=bool)
    test_mask = np.zeros(table.n_rows, dtype=bool)

    assigned: dict[int, str] = {}
    conflict = False
    for energy in np.unique(energy_key[np.isfinite(energy_key)]):
        rows = np.where(energy_key == energy)[0]
        tr_ids, val_ids, te_ids = _split_sorted_ids(
            table.train_id[rows],
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        for split_name, ids in (
            ("train", tr_ids),
            ("validation", val_ids),
            ("test", te_ids),
        ):
            for train_id in ids:
                previous = assigned.get(int(train_id))
                if previous is not None and previous != split_name:
                    conflict = True
                assigned[int(train_id)] = split_name
        train_mask[rows] |= np.isin(table.train_id[rows], tr_ids)
        validation_mask[rows] |= np.isin(table.train_id[rows], val_ids)
        test_mask[rows] |= np.isin(table.train_id[rows], te_ids)

    _assert_disjoint(train_mask, validation_mask, test_mask)
    if conflict or not np.any(validation_mask) or not np.any(test_mask):
        train_ids, validation_ids, test_ids = _split_sorted_ids(
            table.train_id,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        train_mask, validation_mask, test_mask = _fill_masks_from_id_sets(
            table.train_id,
            train_ids,
            validation_ids,
            test_ids,
        )
        policy = "global_contiguous_train_id_block_split"
    else:
        policy = "per_nominal_energy_contiguous_train_id_block_split"

    _assert_disjoint(train_mask, validation_mask, test_mask)
    if not np.any(train_mask) or not np.any(validation_mask) or not np.any(test_mask):
        raise ValueError("Could not create non-empty train/validation/test splits.")
    return train_mask, validation_mask, test_mask, policy
