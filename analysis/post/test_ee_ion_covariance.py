"""Small numerical checks; no experiment-scale jobs or writable raw inputs."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compute_aggregates import AggregatesData, compute_aggregates
from data_loading import load_data
from ee_ion_covariance import (iter_shots, accumulate_pairs, coincidence_projection,
                               pool_ee, read_ee_aggregates, tof_to_ke)


class CoincidenceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'raw.h5'
        self.cal = dict(t0_tof_units=0., slope_eV_tof_unit2=100., E0_eV=0.)
        e = np.array([[[1,2,4,0], [2,2,0,0]]] * 8, float)
        e[4, 0] = [0,0,0,0]
        g = np.array([[2.,3.]] * 8); g[3, 1] = np.nan; g[5, 0] = 12.
        with h5py.File(self.path, 'w') as f:
            for key, a in dict(gmd=g, tofs_e=e, tofs_i=np.array([[[10.,0], [20.,0]]] * 8),
                               z=np.zeros((8,2)), mpe=np.full(8,275.), tID=np.arange(8),
                               between_tdc_files=np.arange(8) == 2).items(): f[key] = a
        self.ds = dict(source=self.path, shape=(8,2), trim_start=1, trim_end=1)
        self.settings = dict(bunch_range=(0,2), gmd_range=(1.,12.), chunk_trains=2, max_trains=None,
                             delay_window_fs=None, delay_zero_stage=0., fs_per_stage_unit=1.)

    def test_stream_matches_loader_and_unique_pairs(self):
        loaded = load_data(self.path, config=1, trim_start=1, trim_end=1)
        chunks = list(iter_shots(self.ds, self.settings))
        np.testing.assert_array_equal(np.concatenate([c['e'] for c in chunks]), loaded.tofs_e)
        np.testing.assert_array_equal(np.concatenate([c['tID'] for c in chunks]), loaded.tID)
        edges = np.arange(0.,102.)
        actual = accumulate_pairs(self.ds, self.settings, self.cal, edges, {'signal':(9.,11.), 'control':(19.,21.)})
        valid = np.isfinite(loaded.gmd) & (loaded.gmd >= 1) & (loaded.gmd < 12)
        expected = np.zeros((101,101), int); single = np.zeros(101, int)
        pairs = one = two = 0
        for hits in loaded.tofs_e[valid]:
            ke = tof_to_ke(hits[hits > 0], self.cal)
            one += len(ke) >= 1; two += len(ke) >= 2
            single += np.histogram(ke, edges)[0]
            for i in range(len(ke)):
                for j in range(i+1, len(ke)):
                    slow, fast = sorted([ke[i],ke[j]])
                    expected[int(slow),int(fast)] += 1; pairs += 1
        r = actual['ungated']
        np.testing.assert_array_equal(r['single'], single)
        np.testing.assert_array_equal(r['pairs'], expected)
        self.assertEqual(r['shots'], valid.sum()); self.assertEqual(r['one'], one); self.assertEqual(r['two'], two)
        self.assertEqual(r['raw_pairs'], pairs); self.assertEqual(r['map_pairs'], pairs)
        self.assertEqual(actual['signal']['shots'] + actual['control']['shots'], r['shots'])
        np.testing.assert_array_equal(actual['signal']['pairs'] + actual['control']['pairs'], expected)
        cb = np.array([2.,3.])
        corrected = accumulate_pairs(self.ds, self.settings, self.cal, edges, cb=cb)['ungated']
        np.testing.assert_array_equal(corrected['pairs'], r['pairs'])
        self.assertAlmostEqual(corrected['sum_norm'], (loaded.gmd * cb)[valid].sum())
        projection, _, _ = coincidence_projection(expected, edges, 275., (249.,250.))
        tags = np.zeros(101)
        for hits in loaded.tofs_e[valid]:
            ke = tof_to_ke(hits[hits > 0], self.cal)
            for i, pe in enumerate(ke):
                if 25 <= pe < 26:
                    for j, partner in enumerate(ke):
                        if i != j: tags[int(partner)] += 1
        np.testing.assert_array_equal(projection, tags)
        # Same-bin real pairs produce two tags; no pair with itself is invented.
        self.assertGreater(projection[25], 0)

    def test_pool_matches_direct_moments(self):
        rng = np.random.default_rng(7)
        d = rng.poisson(1.5, (9,4)).astype(float); g = rng.uniform(1,10,9)
        groups = [slice(0,3), slice(3,9)]
        agg = AggregatesData(G=np.array([g[s].mean() for s in groups]),
                            GtG=np.array([(g[s]**2).mean() for s in groups]), n_per_bin=np.array([3,6]),
                            gmd_edges=np.array([0,3,12]), D=np.array([d[s].mean(0) for s in groups]),
                            DtD=np.array([d[s].T @ d[s] / len(d[s]) for s in groups]),
                            DtG=np.array([d[s].T @ g[s] / len(d[s]) for s in groups]))
        actual = pool_ee(agg)
        cov = d.T @ d / len(d) - np.outer(d.mean(0),d.mean(0))
        dg = d.T @ g / len(d) - d.mean(0) * g.mean()
        pcov = cov - np.outer(dg,dg) / g.var()
        np.testing.assert_allclose(actual['cov'],cov,atol=1e-14)
        np.testing.assert_allclose(actual['pcov'],pcov,atol=1e-14)
        agg.G[:] = 2; agg.GtG[:] = 4
        self.assertTrue(np.isnan(pool_ee(agg)['pcov']).all())

    def test_selective_aggregate_reader_matches_writer_and_coarsening(self):
        output = Path(self.tmp.name) / 'aggregate.h5'
        edges = np.array([1.,2.,3.,4.,5.])
        compute_aggregates(self.path, output, config=1, gmd_edges=[0.,3.,12.], tof_edges=edges,
                           ion_tof_edges=[0.,15.,30.], trim_start=1, trim_end=1, chunk_size=2, verbose=False)
        ds = dict(self.ds, aggregate=output)
        actual = read_ee_aggregates(ds, self.cal, np.array([4.,10.,30.,100.]), (0.,12.), max_bins=None, row_chunk=1)
        data = load_data(self.path, 1, trim_start=1, trim_end=1)
        valid = np.isfinite(data.gmd) & (data.gmd >= 0) & (data.gmd < 12)
        hits = data.tofs_e[valid]
        native = np.array([np.histogram(row[row>0],edges)[0] for row in hits])
        boundaries = np.rint(np.sqrt(100. / actual['edges'][::-1])).astype(int) - 1
        coarse = np.stack([native[:,a:b].sum(1) for a,b in zip(boundaries[:-1],boundaries[1:])],axis=1)[:,::-1]
        g = data.gmd[valid]
        cov = coarse.T @ coarse / len(g) - np.outer(coarse.mean(0),coarse.mean(0))
        dg = coarse.T @ g / len(g) - coarse.mean(0) * g.mean()
        np.testing.assert_allclose(actual['cov'],cov,atol=1e-14)
        np.testing.assert_allclose(actual['pcov'],cov-np.outer(dg,dg)/g.var(),atol=1e-14)
        with self.assertRaisesRegex(ValueError,'existing bin edges'):
            read_ee_aggregates(ds,self.cal,[4.,100.],(1.,12.))


if __name__ == '__main__': unittest.main()
