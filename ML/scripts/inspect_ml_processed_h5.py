#!/usr/bin/env python3
"""Inspect the enriched /ml group in a processed static-XAS H5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _decode(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        print(f"file: {path}")
        print("root datasets:")
        for key in ("vls", "gmd", "gmd_tunnel", "n_shots", "nominal_energies", "vls_pixels"):
            if key in h5:
                print(f"  /{key}: shape={h5[key].shape} dtype={h5[key].dtype}")
        if "ml" not in h5:
            print("\nERROR: no /ml group. Rerun compute_static_xas.py on the server.")
            return

        ml = h5["ml"]
        print("\n/ml datasets:")
        for key in ml:
            obj = ml[key]
            if isinstance(obj, h5py.Dataset):
                print(f"  /ml/{key}: shape={obj.shape} dtype={obj.dtype}")
        print("\n/ml attrs:")
        for key, value in ml.attrs.items():
            print(f"  {key}: {value}")

        n_shots = np.asarray(h5["n_shots"][...], dtype=int)
        finite_slots = int(n_shots.sum())
        valid_train = int(np.sum(ml["train_id"][...] >= 0))
        print("\nshot identity:")
        print(f"  sum(n_shots): {finite_slots}")
        print(f"  nonnegative /ml/train_id slots: {valid_train}")
        print(f"  feature matrix leading shape: {ml['features'].shape[:2]}")

        names = _decode(ml["feature_names"][...])
        roles = _decode(ml["feature_roles"][...])
        units = _decode(ml["feature_units"][...])
        print("\nfeatures:")
        for i, (name, unit, role) in enumerate(zip(names, units, roles)):
            print(f"  {i:02d}  {name:32s}  role={role:13s} unit={unit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="Processed static-XAS H5 file")
    args = parser.parse_args()
    inspect(args.h5)


if __name__ == "__main__":
    main()
