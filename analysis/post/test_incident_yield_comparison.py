"""Small synthetic checks: no experiment data or full scans required.

Run: .venv/bin/python -m unittest discover -s analysis/post -p 'test_incident_yield_comparison.py'
"""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from incident_yield_comparison import read_run, json_default


class YieldComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.h5'
        rng = np.random.default_rng(22)
        self.g = rng.uniform(1, 7, (24, 4))
        self.g[3, 1] = np.nan
        self.tof = rng.choice([0., 10., 11., 12., 13., 14.], (24, 4, 5))
        self.ion = rng.choice([0., 1., 5., 11.], (24, 4, 3))
        self.bad = np.zeros(24, bool)
        self.bad[[0, 9]] = True
        self.z = np.tile(np.arange(4), (24, 1)).astype(float)
        self.z[2, 0] = np.nan
        with h5py.File(self.path, 'w') as f:
            for key, value in dict(gmd=self.g, tofs_e=self.tof, tofs_i=self.ion,
                                   between_tdc_files=self.bad, z=self.z,
                                   tID=np.arange(24) + 100).items():
                f[key] = value
        self.settings = dict(
            calibration=dict(t0_tof_units=0., slope_eV_tof_unit2=30000., E0_eV=0.),
            gmd_ranges=[[1, 12], [3, 4]],
            ke_windows=[[150, 310], [220, 250], [260, 310], [150, 220]],
            reference_energy=274., bunch_range=[1, 4], delay_stage_range=[1, 3],
            ion_tof_range=[0, 10], delay_zero_stage=0., fs_per_stage_unit=1.,
            delay_edges_fs=[0, 1, 2, 3], n_blocks=4, chunk_trains=3, max_trains=None,
        )
        self.ds = dict(source_h5=str(self.path), family='150C', photon_energy_ev=274.,
                       trim_start=1, trim_end=2)

    def test_counts_alignment_and_uncertainty_against_direct_reference(self):
        result = read_run(self.ds, self.settings)
        keep = np.flatnonzero(~self.bad)[1:-2]
        mask = (np.isfinite(self.g) & (self.g >= 3) & (self.g < 4)
                & (self.z >= 1) & (self.z < 3))
        mask[:, :1] = False
        mask[np.setdiff1d(np.arange(24), keep)] = False
        ke = np.divide(30000, self.tof ** 2, out=np.full(self.tof.shape, np.nan),
                       where=self.tof > 0)
        direct = [np.sum((ke[mask] >= lo) & (ke[mask] < hi))
                  for lo, hi in self.settings['ke_windows']]
        np.testing.assert_allclose(result['total_counts'][1, :4], direct)
        self.assertEqual(result['shots'][:, 1].sum(), mask.sum())
        np.testing.assert_allclose(result['gmd'][:, 1].sum(), self.g[mask].sum())
        self.assertEqual(result['total_counts'][1, 8],
                         np.sum((self.ion[mask] > 0) & (self.ion[mask] < 10)))
        np.testing.assert_allclose(result['bunch_counts'][1].sum(axis=0),
                                   result['total_counts'][1, :8])
        self.assertEqual(result['delay_e'][1].sum(), direct[0])
        per_shot = ((ke >= 150) & (ke < 310)).sum(axis=-1)
        ct = np.where(mask, per_shot, 0).sum(axis=1)
        gt = np.where(mask, self.g, 0).sum(axis=1)
        good = mask.any(axis=1)
        n = good.sum()
        rate = ct.sum() / gt.sum()
        se = np.sqrt(n / (n - 1) * np.sum((ct[good] - rate * gt[good]) ** 2)) / gt.sum()
        np.testing.assert_allclose(result['train_se'][1, 0], se)

    def test_chunk_invariance_and_json_roundtrip(self):
        a = read_run(self.ds, self.settings)
        b = read_run(self.ds, dict(self.settings, chunk_trains=7))
        for key in ['counts', 'gmd', 'shots', 'train_se', 'bunch_counts', 'delay_e']:
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)
        restored = json.loads(json.dumps(a, default=json_default))
        np.testing.assert_allclose(restored['yield'], a['yield'], equal_nan=True)
        self.assertEqual(json.loads(json.dumps(self.settings)), self.settings)

    def test_preview_missing_ions_and_empty_selection(self):
        with h5py.File(self.path, 'a') as f:
            del f['tofs_i']
        result = read_run(self.ds, dict(self.settings, max_trains=5))
        self.assertEqual(result['read_trains'], 5)
        self.assertTrue(np.isnan(result['yield'][:, 8]).all())
        self.assertTrue(np.isnan(result['monitors']['shutter']).all())
        empty = read_run(self.ds, dict(self.settings, gmd_ranges=[[50, 60]]))
        self.assertEqual(empty['shots'].sum(), 0)
        self.assertTrue(np.isnan(empty['yield']).all())


if __name__ == '__main__':
    unittest.main()
