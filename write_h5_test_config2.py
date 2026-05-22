"""
Build a *synthetic* config 2 test H5 file by stitching together two
unrelated runs from the 11022188 beamtime.

Config 2 differs from config 1 in that:

- the electron spectrometer is a different detector (``liq_tofs_e``);
- the ion TOF is replaced by an x-ray spectrometer (Gotthard / ``vls``);
- there is no ion-side TOF data at all.

Because we don't yet have a real config 2 acquisition, this script fakes
one by:

1.  Loading every field from the already-produced ``test_config1.h5``
    (which is itself written from run48346 + the matching local DAQ).
2.  Renaming ``tofs_e`` to ``liq_tofs_e`` (same zero-padded structure;
    physically a different spectrometer but structurally identical, so
    fine for code-development purposes).
3.  Loading the Gotthard image stack from a *different* run (run54609)
    and aligning it by **sequential index**, not by train ID — the two
    runs share no train IDs.

Both stacks are truncated to the shorter of the two before stitching,
so every shot in the output has a matching VLS frame.

Run this from the project root::

    python write_h5_test_config2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure analysis/scripts is importable so we can pull paths from config.py
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Inputs / output
# ---------------------------------------------------------------------------

CONFIG1_H5: Path = config.COMBINED_DIR / "test_config1.h5"
VLS_H5:     Path = config.RAW_H5_DIR / "FLASH2_USER1_main_run54609_file20_20251104T015112.1.h5"
OUTPUT_H5:  Path = config.COMBINED_DIR / "test_config2.h5"

# Path inside the VLS run's H5 file
VLS_DATASET = "/FL2/Support Infrastructure/Gotthard/images/value"

# Per-bunch fields that need to be truncated from config 1's bunch count
# (typically 400) down to the VLS bunch count (typically 100).
PER_BUNCH_1D_FIELDS = ("gmd", "z", "z_std")     # shape (n, m)
PER_BUNCH_3D_FIELDS = ("tofs_e",)                # shape (n, m, max_hits) — renamed below

# Map: source key in test_config1.h5  ->  destination key in test_config2.h5
RENAME_ON_COPY = {"tofs_e": "liq_tofs_e"}

# Keys we drop entirely from the config 2 output (config 2 has no ion TOF).
DROP_FROM_CONFIG1 = ("tofs_i",)

# Per-train fields (no bunch axis) that copy across as-is.
PER_TRAIN_FIELDS = ("tID", "mpe", "hor_pos", "ver_pos",
                    "between_tdc_files", "local_DAQ_running")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_per_bunch(arr: np.ndarray, m_target: int) -> np.ndarray:
    """
    Truncate a per-bunch array down to ``m_target`` bunches along axis 1.

    Parameters
    ----------
    arr : np.ndarray
        Array with shape ``(n, m, ...)`` where ``m`` is the bunch axis.
    m_target : int
        Number of bunches to retain.

    Returns
    -------
    np.ndarray
        ``arr[:, :m_target, ...]``.
    """
    if arr.shape[1] < m_target:
        raise ValueError(
            f"Cannot truncate per-bunch array of shape {arr.shape} "
            f"down to {m_target} bunches — source has fewer."
        )
    return arr[:, :m_target, ...]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build ``test_config2.h5`` from config 1 + VLS sources and report."""
    print(f"Config 1 source : {CONFIG1_H5}")
    print(f"VLS source      : {VLS_H5}")
    print(f"Output          : {OUTPUT_H5}")
    print()

    if not CONFIG1_H5.exists():
        raise FileNotFoundError(
            f"{CONFIG1_H5} not found — run write_h5_test_config1.py first."
        )
    if not VLS_H5.exists():
        raise FileNotFoundError(f"{VLS_H5} not found.")

    # --- Pull all fields from test_config1.h5 ---------------------------
    with h5py.File(CONFIG1_H5, "r") as f1:
        src_keys = list(f1.keys())
        print(f"test_config1.h5 keys ({len(src_keys)}): {src_keys}")
        n1 = f1["tID"].shape[0]
        print(f"  trains in config 1 source : {n1}")
        # Load everything into memory — these test files are small enough.
        cfg1: dict[str, np.ndarray] = {k: f1[k][...] for k in src_keys
                                       if k not in DROP_FROM_CONFIG1}

    # --- Pull VLS from the run54609 H5 ----------------------------------
    with h5py.File(VLS_H5, "r") as f2:
        vls_dset = f2[VLS_DATASET]
        n2, m_vls, n_px = vls_dset.shape
        print(f"VLS dataset shape         : {vls_dset.shape} (n, m, pixels)")
        # We load the whole stack; (4936, 100, 1280) uint16 ≈ 1.2 GB which
        # is still comfortable on a workstation. Cast to float32 to match
        # the project's general numerical dtype.
        vls = vls_dset[...].astype(np.float32)

    # --- Align by sequential index, truncating to the shorter ----------
    # The two runs share no train IDs (run48346 ~ 1.8e9, run54609 ~ 2.5e9),
    # so train-ID-based merging is impossible. Per the project spec, we
    # truncate both to min(n1, n2) and assume shot i ↔ shot i.
    n = min(n1, n2)
    print(f"\nAligning by index, truncating both to n = {n} trains.")

    for k, v in cfg1.items():
        cfg1[k] = v[:n]
    vls = vls[:n]

    # --- Match the bunch axis between the two stacks --------------------
    # Config 1 has m=400 bunches/train; VLS has m=100. We pick m=100 as
    # the canonical bunch count and trim the per-bunch config 1 fields
    # accordingly. Per-train fields are unaffected.
    m_out = m_vls
    print(f"Truncating per-bunch axis to m = {m_out} bunches.\n")

    for k in PER_BUNCH_1D_FIELDS + PER_BUNCH_3D_FIELDS:
        if k in cfg1:
            before = cfg1[k].shape
            cfg1[k] = _truncate_per_bunch(cfg1[k], m_out)
            print(f"  truncated {k}: {before} -> {cfg1[k].shape}")

    # --- Write the output -----------------------------------------------
    print(f"\nWriting {OUTPUT_H5} ...")
    if OUTPUT_H5.exists():
        OUTPUT_H5.unlink()  # h5py 'w' overwrites, but be explicit

    with h5py.File(OUTPUT_H5, "w") as f_out:
        # All cfg1 fields, optionally renamed
        for src_key, arr in cfg1.items():
            dst_key = RENAME_ON_COPY.get(src_key, src_key)
            f_out.create_dataset(dst_key, data=arr,
                                 compression="gzip" if arr.ndim >= 2 else None)
        # And the VLS stack
        f_out.create_dataset("vls", data=vls, compression="gzip")

    # --- Report ---------------------------------------------------------
    print(f"\n=== test_config2.h5 written successfully ===")
    with h5py.File(OUTPUT_H5, "r") as f_out:
        print(f"Trains written  : {f_out['tID'].shape[0]}")
        print(f"Bunches/train   : {f_out['gmd'].shape[1]}")
        print(f"Datasets:")
        for k in sorted(f_out.keys()):
            d = f_out[k]
            print(f"  /{k:<22} shape={d.shape!s:<24} dtype={d.dtype}")


if __name__ == "__main__":
    main()
