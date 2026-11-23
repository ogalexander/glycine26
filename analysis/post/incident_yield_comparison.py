"""Stream matched 150C/147C intensity diagnostics; inputs are read-only.

Run full scans on a compute node:
    python incident_yield_comparison.py manifest.json results.json
The notebook writes the manifest and supplies calibration and all selections.
"""
from pathlib import Path
import argparse
import json

import h5py
import numpy as np


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def read_run(ds, settings):
    """Accumulate yields, train-cluster uncertainty moments and stability bins.

    Keep/trim follows the notebook convention. No new DAQ, shutter or delay
    rejection is applied unless explicitly requested in settings. Preview
    selects evenly spaced kept trains and is not a full-run measurement.
    """
    cal = settings['calibration']
    groups = np.asarray(settings['gmd_ranges'], float)
    windows = np.asarray(settings['ke_windows'], float)
    moving = windows + float(ds['photon_energy_ev']) - settings['reference_energy']
    windows = np.concatenate((windows, moving))
    ng, nw = len(groups), len(windows)
    # Electron windows followed by broad ion yield and filled electron slots.
    nm = nw + 2
    nb = int(settings['n_blocks'])
    if nb < 2 or settings['chunk_trains'] < 1:
        raise ValueError('Need >=2 acquisition blocks and a positive chunk size')
    with h5py.File(ds['source_h5'], 'r') as f:
        nt, nbu = f['gmd'].shape
        if 'between_tdc_files' in f:
            kept = np.flatnonzero(~f['between_tdc_files'][:].astype(bool))
            start, end = int(ds['trim_start']), int(ds['trim_end'])
            kept = kept[start:len(kept) - end if end else None]
        else:
            kept = np.arange(nt)  # Matches the notebook's missing-flag fallback.
        n_kept = len(kept)
        limit = settings.get('max_trains')
        if limit is not None:
            if int(limit) < 1:
                raise ValueError('max_trains must be positive or null')
            kept = kept[np.unique(np.linspace(0, n_kept - 1, min(int(limit), n_kept)).astype(int))] if n_kept else kept
        b0, b1 = settings['bunch_range']
        b1 = nbu if b1 is None else b1
        if not (0 <= b0 < b1 <= nbu):
            raise ValueError(f'Invalid bunch range for {ds["source_h5"]}: {nbu} bunches')
        stage_range = settings.get('delay_stage_range')
        if stage_range is not None and 'z' not in f:
            raise KeyError('Delay selection requested but /z is missing')
        shots = np.zeros((nb, ng))
        exposure = np.zeros_like(shots)
        counts = np.zeros((nb, ng, nm))
        # Sufficient statistics across independent train clusters, not shots.
        ntrain = np.zeros(ng)
        g2 = np.zeros(ng)
        c2 = np.zeros((ng, nm))
        cg = np.zeros_like(c2)
        bunch_shots = np.zeros((ng, nbu))
        bunch_gmd = np.zeros_like(bunch_shots)
        bunch_counts = np.zeros((ng, nbu, nw))
        delay_edges = np.asarray(settings['delay_edges_fs'], float)
        delay_shots = np.zeros((ng, len(delay_edges) - 1))
        delay_gmd = np.zeros_like(delay_shots)
        delay_e = np.zeros_like(delay_shots)
        delay_missing = np.zeros(ng)
        delay_outside = np.zeros(ng)
        monitor_names = ['hor_pos', 'ver_pos', 'mpe', 'shutter', 'local_DAQ_running']
        monitor_sum = {k: np.zeros((nb, ng)) for k in monitor_names}
        monitor_n = {k: np.zeros((nb, ng)) for k in monitor_names}
        has_ions = 'tofs_i' in f
        first_tid = last_tid = None
        for offset in range(0, len(kept), settings['chunk_trains']):
            idx = kept[offset:offset + settings['chunk_trains']]
            g = f['gmd'][idx].astype(float)
            tof = f['tofs_e'][idx]
            hit = np.isfinite(tof) & (tof > 0) & (tof > cal['t0_tof_units'])
            ke = np.full(tof.shape, np.nan)
            ke[hit] = cal['slope_eV_tof_unit2'] / (tof[hit] - cal['t0_tof_units']) ** 2 + cal['E0_eV']
            ec = np.stack([np.sum(hit & (ke >= lo) & (ke < hi), axis=-1) for lo, hi in windows], axis=-1)
            filled = np.all(np.isfinite(tof) & (tof > 0), axis=-1)
            if has_ions:
                ions = f['tofs_i'][idx]
                ilo, ihi = settings['ion_tof_range']
                ic = np.sum(np.isfinite(ions) & (ions > 0) & (ions >= ilo) & (ions < ihi), axis=-1)
            else:
                ic = np.zeros(g.shape)
            obs = np.concatenate((ec, ic[..., None], filled[..., None]), axis=-1)
            base = np.zeros(g.shape, bool)
            base[:, b0:b1] = True
            z = f['z'][idx] if 'z' in f else np.full(g.shape, np.nan)
            if z.shape != g.shape:
                raise ValueError('/z must have the same train/bunch shape as /gmd')
            if stage_range is not None:
                base &= np.isfinite(z) & (z >= stage_range[0]) & (z < stage_range[1])
            delay = (z - settings['delay_zero_stage']) * settings['fs_per_stage_unit']
            blocks = np.minimum(idx * nb // max(nt, 1), nb - 1)
            monitors = {}
            for key in monitor_names:
                if key in f:
                    v = np.asarray(f[key][idx], float)
                    if v.shape != (len(idx),):
                        raise ValueError(f'Expected per-train /{key}, got {v.shape}')
                    monitors[key] = v
            if 'tID' in f:
                tids = f['tID'][idx]
                if first_tid is None:
                    first_tid = int(tids[0])
                last_tid = int(tids[-1])
            for j, (lo, hi) in enumerate(groups):
                mask = base & np.isfinite(g) & (g >= lo) & (g < hi)
                n = mask.sum(axis=1)
                gt = np.where(mask, g, 0).sum(axis=1)
                ct = np.where(mask[..., None], obs, 0).sum(axis=1)
                np.add.at(shots[:, j], blocks, n)
                np.add.at(exposure[:, j], blocks, gt)
                np.add.at(counts[:, j], blocks, ct)
                ntrain[j] += np.count_nonzero(n)
                g2[j] += np.sum(gt ** 2)
                c2[j] += np.sum(ct ** 2, axis=0)
                cg[j] += np.sum(ct * gt[:, None], axis=0)
                bunch_shots[j] += mask.sum(axis=0)
                bunch_gmd[j] += np.where(mask, g, 0).sum(axis=0)
                bunch_counts[j] += np.where(mask[..., None], ec, 0).sum(axis=0)
                d = delay[mask]
                finite = np.isfinite(d)
                delay_missing[j] += np.sum(~finite)
                delay_outside[j] += np.sum(finite & ((d < delay_edges[0]) | (d > delay_edges[-1])))
                delay_shots[j] += np.histogram(d[finite], delay_edges)[0]
                delay_gmd[j] += np.histogram(d[finite], delay_edges, weights=g[mask][finite])[0]
                delay_e[j] += np.histogram(d[finite], delay_edges, weights=ec[..., 0][mask][finite])[0]
                for key, v in monitors.items():
                    good = np.isfinite(v)
                    np.add.at(monitor_sum[key][:, j], blocks, np.where(good, v, 0) * n)
                    np.add.at(monitor_n[key][:, j], blocks, good * n)
        total_c, total_g = counts.sum(axis=0), exposure.sum(axis=0)
        yield_g = divide(total_c, total_g[:, None])
        residual_ss = np.maximum(c2 - 2 * yield_g * cg + yield_g ** 2 * g2[:, None], 0)
        se = divide(np.sqrt(residual_ss * divide(ntrain, ntrain - 1)[:, None]), total_g[:, None])
        if not has_ions:
            counts[..., nw] = np.nan
            total_c[:, nw] = yield_g[:, nw] = se[:, nw] = np.nan
        return {
            'dataset': dict(ds), 'preview': limit is not None, 'kept_trains': n_kept,
            'read_trains': len(kept), 'train_id_range': [first_tid, last_tid],
            'run_no': str(f.attrs.get('run_no', 'not recorded')),
            'shots': shots, 'gmd': exposure, 'counts': counts,
            'total_counts': total_c, 'yield': yield_g, 'train_se': se,
            'bunch_shots': bunch_shots, 'bunch_gmd': bunch_gmd, 'bunch_counts': bunch_counts,
            'delay_shots': delay_shots, 'delay_gmd': delay_gmd, 'delay_e': delay_e,
            'delay_missing': delay_missing, 'delay_outside': delay_outside,
            'monitors': {k: divide(monitor_sum[k], monitor_n[k]) for k in monitor_names},
            'available_monitors': [k for k in monitor_names if k in f], 'has_ions': has_ions,
        }


def run_manifest(manifest):
    rows = []
    for ds in manifest['datasets']:
        print(f"Reading {ds['family']} {ds['photon_energy_ev']:g} eV", flush=True)
        rows.append(read_run(ds, manifest['settings']))
    return {'manifest': manifest, 'rows': rows}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        parser.error('Output must differ from manifest')
    manifest = json.loads(args.manifest.read_text())
    result = run_manifest(manifest)
    # Exclusive creation protects any previous results.
    with args.output.open('x') as out:
        json.dump(result, out, default=json_default)
