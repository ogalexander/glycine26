"""Read-only, streamed helpers for the config-1 Auger-search notebook.

Train trimming reuses compute_aggregates; calibration is the saved post-analysis
model. No detector-efficiency correction, accidental subtraction or smoothing.
"""
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from compute_aggregates import AggregatesData, _build_keep_mask


def tof_to_ke(tof, calibration):
    """Same saved model/domain as Bunch investigations (TOF in 100 ps)."""
    tof = np.asarray(tof, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ke = calibration['slope_eV_tof_unit2'] / (tof - calibration['t0_tof_units']) ** 2 + calibration['E0_eV']
    return np.where(tof > calibration['t0_tof_units'], ke, np.nan)


def discover_datasets(combined_dir, energies, stem, aggregate_suffix):
    """Verify exact source names, aggregate provenance and train/bunch shapes."""
    root = Path(combined_dir)
    found = {p.name: p for p in root.glob(f'{stem}_*eV*.h5')}
    rows = []
    for hv in energies:
        base = f'{stem}_{hv:.1f}eV'
        raw, agg = found.get(base + '.h5'), found.get(base + aggregate_suffix)
        if raw is None or agg is None:
            raise FileNotFoundError(f'Missing source or reference aggregate for {base} in {root}')
        with h5py.File(agg, 'r') as f:
            meta = dict(f.attrs)
            source = meta.get('input_h5', '')
            if isinstance(source, bytes): source = source.decode()
            if Path(source).name != raw.name or int(meta.get('config', -1)) != 1:
                raise ValueError(f'Aggregate source/config mismatch: {agg}')
            shapes = {k: f[k].shape for k in ('D', 'DtD', 'G', 'DtG', 'GtG', 'n_per_bin')}
        with h5py.File(raw, 'r') as f:
            nt, nb = f['gmd'].shape
            for key in ('tofs_e', 'tofs_i', 'z'):
                if f[key].shape[:2] != (nt, nb): raise ValueError(f'Misaligned /{key}: {raw}')
            if f['tofs_e'].ndim != 3 or f['tofs_i'].ndim != 3 or f['z'].ndim != 2:
                raise ValueError(f'Unexpected hit/stage axes: {raw}')
            if f['tID'].shape != (nt,) or f['between_tdc_files'].shape != (nt,):
                raise ValueError(f'Misaligned train metadata: {raw}')
            run = f.attrs.get('run_no', 'not recorded')
        rows.append(dict(hv=float(hv), source=raw, aggregate=agg, trim_start=int(meta['trim_start']),
                         trim_end=int(meta['trim_end']), shape=(nt, nb), run_no=run, aggregate_shapes=shapes))
    return rows


def iter_shots(ds, settings, load_ions=False):
    """Yield aligned train chunks; selections match the 7 June raw diagnostics.

    Half-open bunch/GMD/delay cuts. No new DAQ/shutter/tID rejection. Preview
    takes the first retained trains and must not be interpreted as full-run data.
    Only the small train keep-mask is loaded for the full file. Ion hits are
    read only when load_ions=True; otherwise the returned ions field is None.
    """
    b0, b1 = settings['bunch_range']
    nb = ds['shape'][1]
    b1 = nb if b1 is None else b1
    if not 0 <= b0 < b1 <= nb: raise ValueError('Invalid bunch range')
    size, limit = settings['chunk_trains'], settings['max_trains']
    if size < 1 or (limit is not None and limit < 1): raise ValueError('Invalid chunk/preview size')
    with h5py.File(ds['source'], 'r') as f:
        idx = np.flatnonzero(_build_keep_mask(f, ds['trim_start'], ds['trim_end']))
        if limit is not None: idx = idx[:limit]
        for start in range(0, len(idx), size):
            ii = idx[start:start + size]
            # Contiguous slices are fast; sparse retained indices keep the read
            # bounded even if a large rejected-train gap lies inside this chunk.
            rows = slice(int(ii[0]), int(ii[-1]) + 1) if np.all(np.diff(ii) == 1) else ii
            g = f['gmd'][rows, b0:b1].astype(float)
            z = f['z'][rows, b0:b1]
            e = f['tofs_e'][rows, b0:b1]
            ions = f['tofs_i'][rows, b0:b1] if load_ions else None
            if z.shape != g.shape or e.shape[:2] != g.shape or (ions is not None and ions.shape[:2] != g.shape):
                raise ValueError('Train/bunch alignment failed')
            lo, hi = settings['gmd_range']
            valid = np.isfinite(g)
            if lo is not None: valid &= g >= lo
            if hi is not None: valid &= g < hi
            delay = settings['delay_window_fs']
            if delay is not None:
                fs = (z - settings['delay_zero_stage']) * settings['fs_per_stage_unit']
                valid &= np.isfinite(fs) & (fs >= delay[0]) & (fs < delay[1])
            yield dict(gmd=g, e=e, ions=ions, valid=valid, tID=f['tID'][rows], bunch_start=b0)


def inspect_coverage(ds, settings, calibration):
    """Electron coverage with no ROI; ion_hist stays empty for schema compatibility."""
    if settings['tof_bin_width'] <= 0 or settings['ion_bin_width'] <= 0:
        raise ValueError('Positive TOF bin widths required')
    hist_e, hist_i = Counter(), Counter()
    result = dict(shots=0, electrons=0, calibrated=0, below180=0, negative_ke=0,
                  tof_min=np.inf, tof_max=-np.inf, ke_min=np.inf, ke_max=-np.inf)
    for block in iter_shots(ds, settings, load_ions=False):
        result['shots'] += int(block['valid'].sum())
        for key, hist, width in [('e', hist_e, settings['tof_bin_width'])]:
            hits = block[key][block['valid']]
            hits = hits[np.isfinite(hits) & (hits > 0)]
            bins, counts = np.unique(np.floor(hits / width).astype(np.int64), return_counts=True)
            hist.update(dict(zip(bins.tolist(), counts.tolist())))
            if key == 'e' and hits.size:
                ke = tof_to_ke(hits, calibration)
                good = ke[np.isfinite(ke)]
                result['electrons'] += hits.size
                result['calibrated'] += good.size
                result['below180'] += int(np.sum((good >= 0) & (good < 180)))
                result['negative_ke'] += int(np.sum(good < 0))
                result['tof_min'] = min(result['tof_min'], float(hits.min()))
                result['tof_max'] = max(result['tof_max'], float(hits.max()))
                if good.size:
                    result['ke_min'] = min(result['ke_min'], float(good.min()))
                    result['ke_max'] = max(result['ke_max'], float(good.max()))
    result.update(tof_hist=hist_e, ion_hist=hist_i)
    return result


def histogram_from_counter(hist, width, limits=None):
    if not hist: return np.array([0., width]), np.zeros(1)
    lo, hi = min(hist), max(hist) + 1
    if limits is not None:
        lo, hi = int(np.floor(limits[0] / width)), int(np.ceil(limits[1] / width))
    edges = np.arange(lo, hi + 1, dtype=float) * width
    counts = np.array([hist.get(i, 0) for i in range(lo, hi)], dtype=float)
    return edges, counts


def make_ke_edges(coverage, width, max_bins):
    if width <= 0 or max_bins < 1: raise ValueError('Positive KE width and map budget required')
    good = [r for r in coverage if r['calibrated']]
    if not good: raise ValueError('No calibratable electrons in this selection')
    lo = np.floor(min(r['ke_min'] for r in good) / width) * width
    hi = (np.floor(max(r['ke_max'] for r in good) / width) + 1) * width
    if (hi - lo) / width > max_bins:
        raise ValueError('Full KE range exceeds map budget: increase KE bin width explicitly after inspecting coverage')
    return np.arange(lo, hi + width * .5, width)


def accumulate_pairs(ds, settings, calibration, ke_edges, gates=None, cb=None):
    """Unique hit pairs i<j; map rows=slower KE, columns=faster KE.

    All raw multiplicities/pairs are counted before calibration/bin-range cuts.
    Gates are Boolean Q=any positive ion hit in [lo,hi); no 3D tensor.
    Cb optionally changes only the summed exposure, as in the existing notebook.
    """
    gates = {} if gates is None else gates
    n = len(ke_edges) - 1
    out = {name: dict(single=np.zeros(n, np.int64), pairs=np.zeros((n, n), np.int64),
                     shots=0, one=0, two=0, raw_pairs=0, map_pairs=0, sum_gmd=0., sum_norm=0., filled=0)
           for name in ['ungated', *gates]}
    for name, roi in gates.items():
        if name == 'ungated' or len(roi) != 2 or roi[0] >= roi[1]: raise ValueError('Invalid ion gate')
    for block in iter_shots(ds, settings, load_ions=bool(gates)):
        valid = block['valid']
        e = block['e'][valid]
        ions = block['ions'][valid] if gates else None
        g = block['gmd'][valid]
        norm = g
        if cb is not None:
            cb = np.asarray(cb, float)
            if cb.shape != (ds['shape'][1],) or not np.all(np.isfinite(cb) & (cb > 0)):
                raise ValueError('Cb must contain one positive finite factor per original bunch')
            b0 = block['bunch_start']
            factors = np.broadcast_to(cb[b0:b0 + valid.shape[1]], valid.shape)[valid]
            norm = g * factors
        masks = {'ungated': np.ones(len(g), bool)}
        for name, (lo, hi) in gates.items():
            masks[name] = np.any(np.isfinite(ions) & (ions > 0) & (ions >= lo) & (ions < hi), axis=-1)
        hit = np.isfinite(e) & (e > 0)
        multiplicity = hit.sum(axis=-1)
        ke_block = tof_to_ke(np.where(hit, e, np.nan), calibration)
        for name, q in masks.items():
            r = out[name]
            r['shots'] += int(q.sum())
            r['sum_gmd'] += float(g[q].sum())
            r['sum_norm'] += float(norm[q].sum())
            r['one'] += int(np.sum(q & (multiplicity >= 1)))
            r['two'] += int(np.sum(q & (multiplicity >= 2)))
            r['raw_pairs'] += int(np.sum(multiplicity[q] * (multiplicity[q] - 1) // 2))
            r['filled'] += int(np.sum(q & (multiplicity == e.shape[-1])))
            values = ke_block[q]
            r['single'] += np.histogram(values[np.isfinite(values)], ke_edges)[0]
        # One shot at a time bounds pair memory by the detector hit-slot count.
        for s in np.flatnonzero(np.sum(np.isfinite(ke_block), axis=-1) >= 2):
            ke = ke_block[s, np.isfinite(ke_block[s])]
            i, j = np.triu_indices(len(ke), 1)
            slow, fast = np.minimum(ke[i], ke[j]), np.maximum(ke[i], ke[j])
            bi, bj = np.searchsorted(ke_edges, slow, side='right') - 1, np.searchsorted(ke_edges, fast, side='right') - 1
            # Match numpy histogram's inclusive final edge.
            bi[slow == ke_edges[-1]] = n - 1
            bj[fast == ke_edges[-1]] = n - 1
            inside = (bi >= 0) & (bi < n) & (bj >= 0) & (bj < n)
            for name, q in masks.items():
                if not q[s]: continue
                r = out[name]
                np.add.at(r['pairs'], (bi[inside], bj[inside]), 1)
                r['map_pairs'] += int(inside.sum())
    for r in out.values():
        if int(r['pairs'].sum()) != r['map_pairs'] or np.any(np.tril(r['pairs'], -1)):
            raise AssertionError('Pair count/axis ordering failed')
    return out


def coincidence_projection(pairs, edges, hv, be_window):
    """Electron-tagged partner counts: either member may be the PE.

    Whole KE bins selected by centre, no fractional-bin interpolation. A pair
    with both members in the PE window contributes two partner tags, including
    two real distinct hits in one bin. This does not add self-pairs.
    """
    centres = (edges[:-1] + edges[1:]) / 2
    lo, hi = hv - be_window[1], hv - be_window[0]
    w = ((centres >= lo) & (centres < hi)).astype(float)
    if not w.any(): raise ValueError(f'PE window [{lo},{hi}) has no bins')
    return pairs @ w + pairs.T @ w, (lo, hi), np.flatnonzero(w)


def pool_ee(agg):
    """Pool per-bin MEANS with n before covariance (population 1/N convention)."""
    n = np.asarray(agg.n_per_bin, float)
    if n.sum() <= 0: raise ValueError('No aggregate shots selected')
    k = agg.D.shape[-1]
    for field, shape in [('D', n.shape + (k,)), ('DtG', n.shape + (k,)),
                         ('DtD', n.shape + (k, k)), ('G', n.shape), ('GtG', n.shape)]:
        if np.shape(getattr(agg, field)) != shape: raise ValueError(f'Unexpected {field} shape')
    use = n > 0
    def mean(field):
        a = np.asarray(getattr(agg, field))[use]
        if not np.all(np.isfinite(a)): raise ValueError(f'Non-finite occupied {field}')
        w = n[use].reshape((-1,) + (1,) * (a.ndim - 1))
        return np.sum(a * w, axis=0) / n[use].sum()
    d, dd, dg, g, gg = (mean(key) for key in ['D', 'DtD', 'DtG', 'G', 'GtG'])
    cov = dd - np.outer(d, d)
    cov_dg, var = dg - d * g, float(gg - g * g)
    tolerance = np.finfo(float).eps * max(abs(float(gg)), float(g * g), 1.) * 32
    pcov = cov - np.outer(cov_dg, cov_dg) / var if var > tolerance else np.full_like(cov, np.nan)
    if not np.allclose(cov, cov.T, atol=1e-10): raise ValueError('Asymmetric DtD/covariance')
    return dict(cov=cov, pcov=pcov, mean=d, var_gmd=var, shots=int(n.sum()))


def read_ee_aggregates(ds, calibration, target_edges, gmd_range, stage_range=None, max_bins=1, row_chunk=64):
    """Select exact GMD/z cells and pool rebinned e-e moments while streaming DtD rows."""
    if row_chunk < 1 or (max_bins is not None and max_bins < 1):
        raise ValueError("Positive aggregate row/cell limits required")

    with h5py.File(ds["aggregate"], "r") as f:
        if int(f.attrs["config"]) != 1: raise ValueError("Expected config 1")

        n = f["n_per_bin"][:]
        tof = f["tof_edges"][:]
        native = len(tof) - 1
        prefix = n.shape

        for key, shape in [("D", prefix + (native,)), ("DtG", prefix + (native,)),
                           ("DtD", prefix + (native, native)), ("G", prefix), ("GtG", prefix)]:
            if f[key].shape != shape: raise ValueError(f"Invalid {key} shape: {f[key].shape}")

        ge = f["gmd_edges"][:]
        if not all(np.any(np.isclose(ge, x)) for x in gmd_range):
            raise ValueError("Aggregate GMD cut must use existing bin edges; cannot recover sub-bin cuts")

        keep_g = (ge[:-1] >= gmd_range[0]) & (ge[1:] <= gmd_range[1])
        select = np.broadcast_to(keep_g.reshape((-1,) + (1,) * (n.ndim - 1)), n.shape).copy()

        if n.ndim == 2:
            ze = f["z_edges"][:]
            if stage_range is not None and not all(np.any(np.isclose(ze, x)) for x in stage_range):
                raise ValueError("Aggregate delay cut must use existing z bin edges")
            keep_z = np.ones(len(ze) - 1, bool) if stage_range is None else (ze[:-1] >= stage_range[0]) & (ze[1:] <= stage_range[1])
            select &= keep_z[None, :]
        elif n.ndim != 1 or stage_range is not None:
            raise ValueError("Unsupported aggregate axes/delay selection")

        cells = [tuple(i) for i in np.argwhere(select & (n > 0))]
        total_cells = len(cells)
        if max_bins is not None: cells = cells[:max_bins]
        if not cells: raise ValueError("No occupied aggregate cells in this selection")

        ke = tof_to_ke(tof, calibration)
        good_edges = np.flatnonzero(np.isfinite(ke) & (ke >= target_edges[0]) & (ke <= target_edges[-1]))
        if len(good_edges) < 2: raise ValueError("No overlapping aggregate KE coverage")

        a, b = good_edges[0], good_edges[-1]
        if not np.all(np.diff(ke[a:b + 1]) < 0): raise ValueError("Non-monotonic KE calibration")

        boundaries = np.unique([a, b, *[a + int(np.argmin(abs(ke[a:b + 1] - x))) for x in target_edges]])
        group = np.repeat(np.arange(len(boundaries) - 1), np.diff(boundaries))
        group = len(boundaries) - 2 - group
        k = len(boundaries) - 1
        r = csr_matrix((np.ones(b - a), (group, np.arange(b - a))), shape=(k, b - a))

        # Pool weighted moments immediately; never retain one k x k matrix per cell.
        total_n = 0.0
        sum_d = np.zeros(k)
        sum_dg = np.zeros(k)
        sum_dd = np.zeros((k, k))
        sum_g = 0.0
        sum_gg = 0.0

        for cell in cells:
            nc = float(n[cell])
            if nc <= 0: continue

            d = np.asarray(r @ f["D"][cell + (slice(a, b),)])
            dg = np.asarray(r @ f["DtG"][cell + (slice(a, b),)])
            g = float(f["G"][cell])
            gg = float(f["GtG"][cell])

            dd = np.zeros((k, k))
            for start in range(a, b, row_chunk):
                stop = min(start + row_chunk, b)
                block = f["DtD"][cell + (slice(start, stop), slice(a, b))]
                dd += r[:, start - a:stop - a] @ (block @ r.T)

            total_n += nc
            sum_d += nc * d
            sum_dg += nc * dg
            sum_g += nc * g
            sum_gg += nc * gg
            dd *= nc
            sum_dd += dd

        if total_n <= 0: raise ValueError("No aggregate shots selected")

        d = sum_d / total_n
        dg = sum_dg / total_n
        dd = sum_dd / total_n
        g = sum_g / total_n
        gg = sum_gg / total_n

        cov = dd - np.outer(d, d)
        cov_dg = dg - d * g
        var = float(gg - g * g)
        tolerance = np.finfo(float).eps * max(abs(float(gg)), float(g * g), 1.0) * 32
        pcov = cov - np.outer(cov_dg, cov_dg) / var if var > tolerance else np.full_like(cov, np.nan)

        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Asymmetric DtD/covariance")

        return {
            "cov": cov,
            "pcov": pcov,
            "mean": d,
            "var_gmd": var,
            "shots": int(total_n),
            "edges": ke[boundaries][::-1],
            "cells": cells,
            "total_cells": total_cells,
            "shapes": {key: f[key].shape for key in ["D", "DtD", "G", "DtG", "GtG"]},
            "gmd_intervals": [(ge[c[0]], ge[c[0] + 1]) for c in cells],
            "stage_intervals": [(ze[c[1]], ze[c[1] + 1]) for c in cells] if n.ndim == 2 else None,
        }