"""Full-run chronology and common-factor diagnostics for the incident-energy scans.

No correction, smoothing, preview sampling, or raw-data writes. Only per-train
scalars are retained; hit arrays are read in chunks. Use Slurm for full scans.
"""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from incident_yield_comparison import divide

METRICS = ('E/G', 'A/G', 'B/G', 'I/G', 'E/I', 'A/B', 'A/I', 'B/I', 'mean_GMD')


def clean_json(value):
    """Strict JSON: represent missing/nonfinite values as null."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def discover(combined_dir):
    """Use aggregate provenance for source paths and the existing trim settings."""
    out = []
    for family, stem, lo, hi in [('150C', 'glycine_delay_scan_150C', 271.5, 275.),
                                ('147C (7 June)', 'glycine_delay_scan_7June_147C', 274., 276.5)]:
        for path in sorted(Path(combined_dir).glob(stem + '_*eV_aggregates_tr_etof1.h5')):
            match = re.fullmatch(re.escape(stem) + r'_(\d+(?:\.\d+)?)eV_aggregates_tr_etof1.h5', path.name)
            if not match or not lo <= float(match[1]) <= hi:
                continue
            with h5py.File(path, 'r') as f:
                src = f.attrs.get('input_h5', '')
                if isinstance(src, bytes):
                    src = src.decode()
                source = Path(src)
                if not src or not source.is_file():
                    raise FileNotFoundError(f'{path.name}: missing source from input_h5={src!r}')
                out.append(dict(family=family, photon_energy_ev=float(match[1]),
                                source_h5=str(source), agg_path=str(path),
                                trim_start=int(f.attrs.get('trim_start', 0) or 0),
                                trim_end=int(f.attrs.get('trim_end', 0) or 0)))
    if not out:
        raise FileNotFoundError('No matching scan aggregates')
    keys = [(d['family'], d['photon_energy_ev']) for d in out]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate energies within a family; resolve provenance before comparing')
    return out


def valid_indices(f, ds):
    # Exactly the established missing-flag fallback and good-train trimming.
    if 'between_tdc_files' not in f:
        return np.arange(f['gmd'].shape[0])
    idx = np.flatnonzero(~f['between_tdc_files'][:].astype(bool))
    a, b = ds['trim_start'], ds['trim_end']
    return idx[a:len(idx) - b if b else None]


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, mad=np.nan)
    med = np.median(x)
    return dict(n=len(x), mean=np.mean(x), std=np.std(x, ddof=1) if len(x)>1 else np.nan,
                median=med, mad=np.median(np.abs(x-med)))


def ratio(x, y):
    """Ratio of sums and train-cluster SE; keep zero-denominator trains."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if not len(x) or y.sum() <= 0:
        return dict(value=np.nan, se=np.nan)
    r = x.sum()/y.sum()
    se = np.sqrt(len(x)/(len(x)-1)*np.sum((x-r*y)**2))/y.sum() if len(x)>1 else np.nan
    return dict(value=r, se=se)


def summarise(tid, shots, gmd, counts, monitors):
    active = shots > 0
    c, g, n = counts[active], gmd[active], shots[active]
    e, a, b, i = c.T
    pairs = [(e,g), (a,g), (b,g), (i,g), (e,i), (a,b), (a,i), (b,i), (g,n)]
    return dict(valid_trains=len(tid), selected_trains=int(active.sum()), shots=int(shots.sum()),
                sum_gmd=float(gmd.sum()), counts=np.sum(counts, axis=0),
                tid=stats(tid), start_tid=int(tid[0]) if len(tid) and np.isfinite(tid[0]) else None,
                end_tid=int(tid[-1]) if len(tid) and np.isfinite(tid[-1]) else None,
                selected_start_tid=int(tid[active][0]) if active.any() and np.isfinite(tid[active][0]) else None,
                selected_end_tid=int(tid[active][-1]) if active.any() and np.isfinite(tid[active][-1]) else None,
                metrics={name: ratio(*pair) for name,pair in zip(METRICS,pairs)},
                # Equal train weights; compare robust median/MAD with mean/std.
                monitors={k:stats(v[active]) for k,v in monitors.items()},
                monitors_all_valid={k:stats(v) for k,v in monitors.items()})


def metadata_inventory(f):
    """Bounded root inventory; never load unknown large arrays or use them as cuts."""
    attrs = {}
    for key in f.attrs:
        attr = f.attrs.get_id(key)
        if attr.get_storage_size() > 4096:
            attrs[key] = '<attribute larger than 4096 bytes; not read>'
            continue
        v = f.attrs[key]
        arr = np.asarray(v)
        if arr.size <= 16:
            attrs[key] = str(v) if arr.dtype.kind not in 'biuf' else clean_json(arr.tolist())
        else:
            attrs[key] = f'<shape={arr.shape}, dtype={arr.dtype}>'
    datasets = {}
    terms = re.compile(r'atten|transmi|energy|gmd|mpe|pos|point|state|tim|delay|shutter|volt|gain|detect|temp|pressure|daq', re.I)
    for key, item in f.items():
        if not isinstance(item, h5py.Dataset):
            datasets[key] = dict(kind='group')
            continue
        desc = dict(shape=item.shape, dtype=str(item.dtype), candidate=bool(terms.search(key)))
        if item.size <= 16 and item.dtype.kind in 'biuf':
            desc['small_value'] = clean_json(item[()].tolist())
        datasets[key] = desc
    return dict(root_attributes=attrs, root_datasets=datasets)


def inspect_chronology(ds):
    """Read only the small train-ID vector for chronology and provenance."""
    with h5py.File(ds['source_h5'], 'r') as f:
        idx = valid_indices(f, ds)
        tid = np.asarray(f['tID'][idx]).reshape(-1)
        if tid.shape != idx.shape:
            raise ValueError('tID must be one scalar per valid train')
        finite = np.isfinite(tid)
        unique = np.unique(tid[finite])
        duplicate_rows = int(finite.sum()-len(unique))
        valid = bool(len(tid) and finite.all() and np.all(tid>0) and duplicate_rows==0)
        issue = None if valid else 'Missing, nonpositive or duplicate train IDs; chronology unavailable'
        order = np.argsort(tid, kind='stable') if valid else np.arange(len(tid))
        return dict(dataset=ds, valid_trains=len(idx),
                    chronology_valid=valid, chronology_issue=issue,
                    duplicate_tid_rows=duplicate_rows, nonpositive_tid_rows=int(np.sum(tid<=0)),
                    tid=stats(tid if valid else np.array([])),
                    start_tid=int(tid[order[0]]) if valid else None,
                    end_tid=int(tid[order[-1]]) if valid else None,
                    source_tid_monotonic=bool(valid and np.all(np.diff(tid.astype(np.int64))>0)),
                    metadata=metadata_inventory(f))


def read_full_run(ds, settings):
    """All valid trains, eight equal consecutive blocks in tID order."""
    if settings.get('max_trains') is not None:
        raise ValueError('Chronology analysis requires all valid trains, not a preview')
    if settings['n_blocks'] != 8:
        raise ValueError('This diagnostic defines exactly eight chronological blocks')
    if settings['chunk_trains'] < 1:
        raise ValueError('chunk_trains must be positive')
    if len(settings['ke_windows']) != 3:
        raise ValueError('Expected existing broad, feature A and feature B windows')
    header = inspect_chronology(ds)
    with h5py.File(ds['source_h5'], 'r') as f:
        idx = valid_indices(f, ds)
        tid = np.asarray(f['tID'][idx]).reshape(-1)
        n = len(idx)
        shots, gmd = np.zeros(n), np.zeros(n)
        counts = np.zeros((n,4))
        mon = {k:np.full(n,np.nan) for k in ['hor_pos','ver_pos','mpe']}
        b0,b1 = settings['bunch_range']
        b1 = f['gmd'].shape[1] if b1 is None else b1
        if not 0 <= b0 < b1 <= f['gmd'].shape[1]:
            raise ValueError('Bunch selection is outside the source bunch axis')
        cal = settings['calibration']
        for off in range(0,n,settings['chunk_trains']):
            inds = idx[off:off+settings['chunk_trains']]
            sl = slice(off,off+len(inds))
            # Read monotonically in source order; reorder only the small summaries.
            g = f['gmd'][inds,b0:b1].astype(float)
            tof = f['tofs_e'][inds,b0:b1,:]
            selected = np.isfinite(g)&(g>=settings['gmd_range'][0])&(g<settings['gmd_range'][1])
            if settings['delay_stage_range'] is not None:
                z = f['z'][inds,b0:b1]
                lo,hi = settings['delay_stage_range']
                selected &= np.isfinite(z)&(z>=lo)&(z<hi)
            shots[sl] = selected.sum(axis=1)
            gmd[sl] = np.where(selected,g,0).sum(axis=1)
            valid = np.isfinite(tof)&(tof>0)&(tof>cal['t0_tof_units'])
            ke = np.full(tof.shape,np.nan)
            ke[valid] = cal['slope_eV_tof_unit2']/(tof[valid]-cal['t0_tof_units'])**2+cal['E0_eV']
            for j,(lo,hi) in enumerate(settings['ke_windows']):
                counts[sl,j] = ((ke>=lo)&(ke<hi)&selected[...,None]).sum(axis=(1,2))
            if 'tofs_i' in f:
                ions = f['tofs_i'][inds,b0:b1,:]
                lo,hi = settings['ion_tof_range']
                counts[sl,3] = (np.isfinite(ions)&(ions>0)&(ions>=lo)&(ions<hi)&selected[...,None]).sum(axis=(1,2))
            else:
                counts[sl,3] = np.nan
            for key in mon:
                if key in f:
                    values = np.asarray(f[key][inds],float)
                    if values.shape != (len(inds),):
                        raise ValueError(f'{key}: expected one value per train, got {values.shape}')
                    mon[key][sl] = values
        order = np.argsort(tid,kind='stable') if header['chronology_valid'] else np.arange(n)
        tid,shots,gmd,counts = tid[order],shots[order],gmd[order],counts[order]
        mon = {k:v[order] for k,v in mon.items()}
        if not header['chronology_valid']:
            tid = np.full(n,np.nan)  # Do not label source-row order as acquisition time.
        whole = summarise(tid,shots,gmd,counts,mon)
        blocks = []
        for ii in np.array_split(np.arange(n),8):
            blocks.append(summarise(tid[ii],shots[ii],gmd[ii],counts[ii],{k:v[ii] for k,v in mon.items()}))
        if not header['chronology_valid']:
            # We cannot verify independent train clusters without usable IDs.
            for summary in [whole,*blocks]:
                for metric in summary['metrics'].values():
                    metric['se'] = np.nan
        return dict(**header, whole=whole, blocks=blocks, full_data=True,
                    block_order_basis='tID' if header['chronology_valid'] else 'source rows; chronology unavailable')


def run(manifest):
    rows=[]
    for ds in manifest['datasets']:
        print(f"Full chronology: {ds['family']} {ds['photon_energy_ev']:g} eV",flush=True)
        rows.append(read_full_run(ds,manifest['settings']))
    rows.sort(key=lambda r: (r['start_tid'] is None,r['start_tid'] or 0))
    return dict(schema='incident_chronology_v1',manifest=manifest,rows=rows)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new result path')
    result=run(json.loads(args.manifest.read_text()))
    with args.output.open('x') as f:
        json.dump(clean_json(result),f,allow_nan=False)
