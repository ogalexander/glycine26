import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "analysis" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compute_static_xas as csx


class ComputeStaticXASTests(unittest.TestCase):
    def test_global_vls_background_noise_uses_pooled_background_bunches(self):
        vls = np.array(
            [
                [[0.0, 10.0], [2.0, 14.0], [100.0, 100.0]],
                [[4.0, 18.0], [6.0, 22.0], [100.0, 100.0]],
            ],
            dtype=np.float64,
        )
        stats = csx._global_vls_background_noise(vls, 0, 2)
        bg_flat = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0], [6.0, 22.0]])
        med = np.median(bg_flat, axis=0)
        mad = np.median(np.abs(bg_flat - med[None, :]), axis=0)
        np.testing.assert_allclose(stats["median"], med)
        np.testing.assert_allclose(stats["mad"], mad)
        np.testing.assert_allclose(stats["sigma"], 1.4826 * mad)
        np.testing.assert_array_equal(stats["n_finite"], np.array([4, 4]))

    def test_section_energies_from_set_wavelength_preserves_repeated_sections(self):
        wavelengths_nm = np.array([
            1239.8419843320026 / 284.0,
            1239.8419843320026 / 283.5,
            1239.8419843320026 / 283.5,
            1239.8419843320026 / 283.0,
        ])
        energies, meta = csx._section_energies_from_source(
            section_energy_source="set_wavelength_1",
            nominal_energies=None,
            open_blocks=[(0, 1), (1, 2), (2, 3), (3, 4)],
            train_scalars={"set_wavelength_1": wavelengths_nm},
            is_open_raw=np.ones(4, dtype=bool),
            valid_mask=np.ones(4, dtype=bool),
            energy_round_decimals=3,
        )

        np.testing.assert_allclose(energies, np.array([284.0, 283.5, 283.5, 283.0]))
        np.testing.assert_allclose(meta["source_values"], wavelengths_nm)
        self.assertEqual(meta["source"], "set_wavelength_1")
        self.assertEqual(meta["source_unit"], "nm")


if __name__ == "__main__":
    unittest.main()
