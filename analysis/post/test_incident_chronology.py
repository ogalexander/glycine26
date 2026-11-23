"""Synthetic full-run tests; no experimental-data scan required."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from incident_chronology import read_full_run, inspect_chronology, ratio, clean_json
from incident_yield_comparison import read_run


class ChronologyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic.h5'
        rng=np.random.default_rng(9)
        self.g=rng.uniform(2,5,(29,4))
        self.g[3,0]=np.nan
        self.tof=rng.choice([0.,10.,11.,12.],(29,4,3))
        self.ion=rng.choice([0.,1.,9.,15.],(29,4,4))
        self.bad=np.zeros(29,bool); self.bad[[0,5,7,21]]=True
        # Deliberately unsorted chronological IDs; source alignment must survive.
        self.tid=np.arange(29,dtype=np.int64)[::-1]+100
        self.pos=np.arange(29,dtype=float)-14
        with h5py.File(self.path,'w') as f:
            for k,v in dict(gmd=self.g,tofs_e=self.tof,tofs_i=self.ion,
                            tID=self.tid,between_tdc_files=self.bad,
                            hor_pos=self.pos,ver_pos=-self.pos,mpe=self.pos+274,
                            z=np.tile(np.arange(4),(29,1))).items(): f[k]=v
            f.attrs['detector_gain']=2.
            f['machine_state']=np.array(3)
        self.ds=dict(source_h5=str(self.path),family='150C',photon_energy_ev=274.,trim_start=1,trim_end=2)
        self.settings=dict(calibration=dict(t0_tof_units=0.,slope_eV_tof_unit2=30000.,E0_eV=0.),
                           gmd_range=[3.,4.],ke_windows=[[180.,310.],[220.,260.],[270.,310.]],
                           bunch_range=[1,4],delay_stage_range=None,ion_tof_range=[0,10],
                           n_blocks=8,chunk_trains=3,max_trains=None)

    def test_equal_valid_blocks_order_alignment_and_direct_counts(self):
        r=read_full_run(self.ds,self.settings)
        ids=np.flatnonzero(~self.bad)[1:-2]
        ids=ids[np.argsort(self.tid[ids])]
        expected_parts=np.array_split(ids,8)
        self.assertFalse(r['source_tid_monotonic'])
        self.assertEqual(r['start_tid'],int(self.tid[ids[0]]))
        for b,ii in zip(r['blocks'],expected_parts):
            self.assertEqual(b['valid_trains'],len(ii))
            self.assertEqual(b['start_tid'],int(self.tid[ii[0]]))
            g=self.g[ii,1:]
            mask=np.isfinite(g)&(g>=3)&(g<4)
            tof=self.tof[ii,1:]
            ke=np.divide(30000,tof**2,out=np.full_like(tof,np.nan),where=tof>0)
            expected=[np.sum((ke>=lo)&(ke<hi)&mask[...,None]) for lo,hi in self.settings['ke_windows']]
            ions=self.ion[ii,1:]
            expected.append(np.sum((ions>0)&(ions<10)&mask[...,None]))
            np.testing.assert_allclose(b['counts'],expected)
            np.testing.assert_allclose(b['sum_gmd'],np.sum(g[mask]))
            self.assertEqual(b['shots'],int(mask.sum()))
            pos=self.pos[ii][mask.any(axis=1)]
            if len(pos): np.testing.assert_allclose(b['monitors']['hor_pos']['mean'],pos.mean())
        sizes=[b['valid_trains'] for b in r['blocks']]
        self.assertLessEqual(max(sizes)-min(sizes),1)
        np.testing.assert_allclose(np.sum([b['counts'] for b in r['blocks']],axis=0),r['whole']['counts'])

    def test_unchanged_whole_run_yields_and_chunk_invariance(self):
        a=read_full_run(self.ds,self.settings)
        b=read_full_run(self.ds,dict(self.settings,chunk_trains=8))
        self.assertEqual(clean_json(a),clean_json(b))
        old=dict(self.settings,gmd_ranges=[[3,4]],reference_energy=274.,delay_zero_stage=0.,
                 fs_per_stage_unit=1.,delay_edges_fs=[0,1,2,3])
        ref=read_run(self.ds,old)
        np.testing.assert_allclose(a['whole']['counts'],ref['total_counts'][0,[0,1,2,6]])
        for j,key in enumerate(['E/G','A/G','B/G','I/G']):
            ix=[0,1,2,6][j]
            np.testing.assert_allclose(a['whole']['metrics'][key]['value'],ref['yield'][0,ix])
            np.testing.assert_allclose(a['whole']['metrics'][key]['se'],ref['train_se'][0,ix],rtol=1e-12)

    def test_zero_ion_trains_must_not_be_dropped_from_ratio(self):
        r=ratio([2.,4.,6.],[0.,2.,2.])
        self.assertEqual(r['value'],3.)
        self.assertAlmostEqual(r['se'],np.sqrt(3/2*np.sum(np.array([2.,-2.,0.])**2))/4)

    def test_missing_fields_no_shots_duplicate_tid_and_json(self):
        with h5py.File(self.path,'a') as f:
            del f['tofs_i']; del f['hor_pos']
        r=read_full_run(self.ds,self.settings)
        self.assertTrue(np.isnan(r['whole']['metrics']['I/G']['value']))
        self.assertTrue(np.isnan(r['whole']['monitors']['hor_pos']['mean']))
        json.dumps(clean_json(r),allow_nan=False)
        empty=read_full_run(self.ds,dict(self.settings,gmd_range=[99,100]))
        self.assertEqual(empty['whole']['shots'],0)
        self.assertTrue(np.isnan(empty['whole']['metrics']['E/G']['value']))
        with h5py.File(self.path,'a') as f: f['tID'][3]=f['tID'][2]
        invalid = inspect_chronology(self.ds)
        self.assertFalse(invalid['chronology_valid'])
        self.assertIsNone(invalid['start_tid'])
        flagged = read_full_run(self.ds,self.settings)
        self.assertEqual(flagged['block_order_basis'],'source rows; chronology unavailable')
        self.assertTrue(np.isnan(flagged['whole']['metrics']['E/G']['se']))
        with self.assertRaises(ValueError): read_full_run(self.ds,dict(self.settings,max_trains=1000))


if __name__=='__main__': unittest.main()
