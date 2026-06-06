"""
Run compute_aggregates.py for every photon energy in the 5 June WL scan.

Usage:
    python run_aggregates_5June_energy_scan.py [--dry-run]

Walks the energy axis 270.0 .. 283.0 eV in 0.5 eV steps and calls
``compute_aggregates.py`` once per energy with the
``aggregates_config1_5June_energy_scan.py`` config against the
matching ``glycine_WL_scan_<energy>eV.h5`` combined H5 file.

Stops on the first failure unless ``--keep-going`` is passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
CONFIG    = REPO_ROOT / "analysis" / "configs" / "aggregates_config1_5June_energy_scan.py"
H5_DIR    = REPO_ROOT / "11022188" / "processed" / "combined"
SCRIPT    = REPO_ROOT / "analysis" / "scripts" / "compute_aggregates.py"

E_START, E_STOP, E_STEP = 270.0, 283.0, 0.5


def _fmt(e: float) -> str:
    """270.0 -> '270', 270.5 -> '270.5'."""
    return f"{e:g}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    p.add_argument("--keep-going", action="store_true",
                   help="continue past failing energies instead of aborting")
    args = p.parse_args()

    energies = np.arange(E_START, E_STOP + 0.5 * E_STEP, E_STEP)
    print(f"{len(energies)} energies: {energies[0]:g} .. {energies[-1]:g} eV "
          f"(step {E_STEP} eV)")

    failures: list[tuple[float, int]] = []
    for k, e in enumerate(energies, 1):
        h5 = H5_DIR / f"glycine_WL_scan_{_fmt(float(e))}eV.h5"
        cmd = [sys.executable, str(SCRIPT), str(CONFIG), str(h5)]
        print(f"\n[{k}/{len(energies)}] {e:g} eV  ->  {h5.name}")
        print("  $", " ".join(cmd))
        if args.dry_run:
            continue
        if not h5.exists():
            print(f"  SKIP: {h5} does not exist")
            failures.append((float(e), -1))
            if not args.keep_going:
                break
            continue
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            print(f"  FAIL: exit {rc}")
            failures.append((float(e), rc))
            if not args.keep_going:
                break

    if failures:
        print("\nfailed energies:")
        for e, rc in failures:
            print(f"  {e:g} eV  (rc={rc})")
        return 1
    print("\nall energies processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
