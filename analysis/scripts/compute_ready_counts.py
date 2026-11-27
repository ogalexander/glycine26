"""
Dead-time-corrected electron spectra for an existing aggregates file.

Separate from ``compute_aggregates.py``: it reads an aggregates file, goes back
to the combined H5 that file was made from and writes a companion file
``<aggregates>_ready.h5``. Notebooks that work with aggregates load it if they
want the correction; the aggregates themselves are not changed.

The electron detector / TDC records no second hit within its dead time
(none below 1.5 ns, partly up to ~6 ns) after a recorded hit. The loss grows with the hit density, so with the
pulse energy, and is largest on the fast photolines. Every shot is put into
the same cell as in ``compute_aggregates.py`` (same keep mask, bunch range,
GMD / photon-energy / z bins, hit caps, VLS frame check) and, per cell and per
TDC tick (0.1 ns), the script counts

    R  ready shots: no hit recorded in the preceding tau
    H  hits recorded at that tick in ready shots

For a detector that is dead for at most tau after each recorded hit and
Poisson arrivals, H / R is the arrival probability at that tick, whatever
happened earlier in the shot. The mean number of electrons per shot in an
aggregates TOF bin is then

    D_live = sum over the bin's ticks of -ln(1 - H / R)

(the Coates correction for pile-up in TOF spectra). It is exact for any tau at
or above the real dead time (and too small below it) when every shot has the
same pulse energy. Inside a wide GMD bin the shots that are ready are on
average the weaker ones, which under-corrects; so R and H are counted in
``gmd_sub`` sub-bins of every GMD bin (quantiles of the GMD) and the corrected
means are combined weighted by shots (synthetic test: 7 % low with one
sub-bin, within 2 % with 3-6). Ticks with fewer than ``MIN_READY`` ready shots
in a sub-bin fall back to the correction of the whole cell.

The file holds D_live for several tau and, per kinetic-energy window, the
zero-hit fraction p0 and mu0 = -ln p0 (combined over the same sub-bins). Dead
time never removes the first hit of a shot, so mu0 is a window's mean number
of electrons without losses (Poisson at fixed pulse energy). Use a tau on the
plateau where sum(D_live over the window) no longer changes and agrees with
mu0. Run 60611 (fast lines up to 1.3 recorded hits per shot): the plateau starts at
tau = 6 ns, where the fast-window totals equal mu0 to 0.1 % in every GMD bin
(x 1.56 on the recorded hits in the top bin); the hit spacing shows no pairs
below 1.5 ns, so the detector is partly dead up to ~6 ns.

Only the means are corrected. The second moments of the aggregates (DtD, DtC,
DtG) stay as recorded; covariances need the recorded D, not D_live.

Output ``<aggregates stem>_ready.h5`` (cells = the aggregates' leading axes,
e.g. (N_E, N_GMD) in energy mode):

    taus_ns        (n_tau,)
    n_per_bin      (cells)                    shots per cell (= the aggregates')
    D              (cells, n_tof)             recorded hits per shot per bin (= the aggregates')
    D_live         (n_tau, cells, n_tof)      dead-time-corrected electrons per shot per bin
    ready_frac     (n_tau, cells, n_tof)      fraction of shots ready, averaged over the bin's ticks
    p0_windows_ke  (n_w, 2)                   eV; a window holds the bins whose centre lies inside
    p0             (cells, n_w)               fraction of shots without a recorded hit in the window
    mu0            (cells, n_w)               -ln p0 per GMD sub-bin, combined weighted by shots
    tof_edges      (n_tof + 1,)               TDC units, as the aggregates
    attrs: aggregates_h5, input_h5, mode, gmd_sub, min_ready, check_n_equal, check_D_max_abs_diff, ...

Usage:
    python analysis/scripts/compute_ready_counts.py AGG.h5 [--taus 1 2 3 4 5 6 8 10]
           [--windows 210:285 10:40] [--gmd-sub 4] [--workers 16] [-o OUT.h5] [--no-vls-check]

AGG.h5 is resolved against config.COMBINED_DIR when it is not a path that exists.
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import calibration
import config
from compute_aggregates import _build_keep_mask, _iter_chunks, _pool_map

REPO_ROOT = Path(__file__).resolve().parents[2]
BIG = np.int64(1) << 40          # sorts zero-padding after every real hit
MIN_READY = 5                    # sub-bin ticks with fewer ready shots use the whole cell's correction


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _accumulate(task):
    """Worker: per GMD sub-cell (cell x gmd_sub), shot counts, recorded hits per bin, zero-hit counts and,
    per tau, H and the difference array of the not-ready intervals, over the chunks ``task["chunks"]``."""
    p = task
    n_tof, L, T0, n_sub = p["n_tof"], p["L"], p["T0"], p["n_sub"]
    n_sc = p["n_cells"] * n_sub
    taus, wins = p["tau_ticks"], p["win_bins"]
    n_w = max(len(wins), 1)
    edges = p["tof_edges"]
    b0, b1 = p["b0"], p["b1"]
    out = {"n": np.zeros(n_sc, np.int64), "D": np.zeros(n_sc * n_tof, np.int64),
           "zero": np.zeros(n_sc * n_w, np.int64),
           "H": np.zeros((len(taus), n_sc * L), np.int32),
           "dR": np.zeros((len(taus), n_sc * (L + 1)), np.int32)}
    with h5py.File(p["input_h5"], "r") as f:
        for s, e in p["chunks"]:
            sub = p["keep_mask"][s:e]
            if not sub.any():
                continue
            # cell of every shot, with the rules of compute_aggregates._accumulate_cfg1
            g = f["gmd"][s:e, b0:b1][sub].ravel()
            te = f["tofs_e"][s:e, b0:b1][sub]
            ti = f["tofs_i"][s:e, b0:b1][sub]
            gbin = np.digitize(g, p["gmd_edges"]) - 1
            valid = np.isfinite(g) & (gbin >= 0) & (gbin < p["n_gmd"])
            if p["max_hits_e"] is not None:
                valid &= (te != 0).sum(axis=-1).ravel() <= int(p["max_hits_e"])
            if p["max_hits_i"] is not None:
                valid &= (ti != 0).sum(axis=-1).ravel() <= int(p["max_hits_i"])
            del ti
            if p["vls_col"] is not None:
                valid &= np.isfinite(f["vls"][s:e, b0:b1, p["vls_col"]][sub]).ravel()
            if p["mode"] == "spectral":
                cell = gbin
            elif p["mode"] == "energy":
                ev = f[p["energy_key"]][s:e, b0:b1][sub].ravel()
                ebin = np.digitize(ev, p["e_edges"]) - 1
                valid &= np.isfinite(ev) & (ebin >= 0) & (ebin < p["n_e"])
                cell = ebin * p["n_gmd"] + gbin
            else:
                zv = f["z"][s:e, b0:b1][sub].ravel()
                zbin = np.digitize(zv, p["z_edges"]) - 1
                valid &= np.isfinite(zv) & (zbin >= 0) & (zbin < p["n_z"])
                cell = gbin * p["n_z"] + zbin
            vq = np.flatnonzero(valid)
            if not len(vq):
                continue
            # GMD sub-bin inside the cell's GMD bin (quantile edges of that bin)
            local = (g[vq, None] >= p["sub_interior"][gbin[vq]]).sum(axis=1) if n_sub > 1 else 0
            cv = cell[vq] * n_sub + local
            out["n"] += np.bincount(cv, minlength=n_sc)

            # hits of the valid shots, sorted in time; padding (0) -> BIG at the end of each row
            t = te.reshape(-1, te.shape[-1])[vq].astype(np.int64)
            del te
            t = np.where(t > 0, t, BIG)
            t.sort(axis=1)
            hit = t < BIG
            cb = np.broadcast_to(cv[:, None], t.shape)

            # recorded hits per aggregates bin ([e_i, e_i+1), last bin closed, as np.histogram2d)
            j = np.searchsorted(edges, t, side="right") - 1
            j[t == edges[-1]] = n_tof - 1
            inb = hit & (j >= 0) & (j < n_tof)
            out["D"] += np.bincount(cb[inb] * n_tof + j[inb], minlength=n_sc * n_tof)
            for w, (j0, j1) in enumerate(wins):
                has = (inb & (j >= j0) & (j <= j1)).any(axis=1)
                out["zero"] += np.bincount(cv[~has] * n_w + w, minlength=n_sc * n_w)

            # ready hits and not-ready intervals per tau
            gap = np.full(t.shape, BIG)
            gap[:, 1:] = t[:, 1:] - t[:, :-1]               # time since the previous recorded hit
            nxt = np.full(t.shape, BIG)
            nxt[:, :-1] = t[:, 1:]                          # next recorded hit (BIG: none)
            tick = t - T0
            on_grid = hit & (tick >= 0) & (tick < L)
            for k, tau in enumerate(taus):
                rdy = on_grid & (gap > tau)
                out["H"][k] += np.bincount(cb[rdy] * L + tick[rdy], minlength=n_sc * L).astype(np.int32)
                # a hit at h makes the shot not ready at ticks h+1 .. h+tau; clipping at the next hit keeps the
                # intervals of one shot disjoint (the next hit opens its own)
                si = np.clip(t + 1 - T0, 0, L)
                ei = np.clip(np.minimum(t + tau, nxt) - T0 + 1, 0, L)
                use = hit & (ei > si)
                idx = cb[use] * (L + 1)
                out["dR"][k] += (np.bincount(idx + si[use], minlength=n_sc * (L + 1))
                                 - np.bincount(idx + ei[use], minlength=n_sc * (L + 1))).astype(np.int32)
    return out


def _mu(H, R):
    """-ln(1 - H/R) and whether it is usable (enough ready shots, not every ready shot hit)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (R >= MIN_READY) & (H < R)
        mu = np.where(ok, -np.log1p(-H / np.where(R > 0, R, 1)), np.nan)
    return mu, ok


def compute_ready_counts(agg_h5, output_h5, taus_ns=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
                         windows_ke=((210.0, 285.0), (10.0, 40.0)), gmd_sub=4, workers=1, vls_check=True,
                         log=print):
    t_start = time.time()
    agg_h5 = Path(agg_h5)
    with h5py.File(agg_h5, "r") as a:
        at = dict(a.attrs)
        mode = str(at["mode"])
        if int(at["config"]) != 1:
            raise ValueError("compute_ready_counts: config-1 aggregates (electron TOF in tofs_e) only")
        tof_edges = a["tof_edges"][...].astype(float)
        gmd_edges = a["gmd_edges"][...]
        n_ref = a["n_per_bin"][...]
        D_ref = a["D"][...]
        e_edges = a["energy_edges"][...] if "energy_edges" in a else None
        z_edges = a["z_edges"][...] if "z_edges" in a else None
        has_vls = "A" in a
    input_h5 = _resolve(at["input_h5"])
    b0, b1 = (int(x) for x in at["bunch_range"])
    n_gmd = len(gmd_edges) - 1
    n_tof = len(tof_edges) - 1
    cell_shape = n_ref.shape
    n_cells = int(np.prod(cell_shape))

    vls_col = None
    if vls_check and has_vls:
        if not str(at.get("vls_source", "")).startswith("combined"):
            raise RuntimeError(f"VLS of the aggregates from {at.get('vls_source')!r}: frame check not supported; "
                               "use --no-vls-check (n_per_bin is still compared)")
        vls_col = int(np.asarray(at["vls_crop_roi"])[0])

    T0 = int(np.floor(tof_edges[0]))
    T1 = int(np.ceil(tof_edges[-1]))
    L = T1 - T0 + 1
    tau_ticks = [int(round(10 * x)) for x in taus_ns]
    kc = calibration.ke_of_tof(0.5 * (tof_edges[1:] + tof_edges[:-1]) / 10.0)   # bin centres, eV (TOF order)
    win_bins = []
    for lo, hi in windows_ke:
        jj = np.flatnonzero((kc >= lo) & (kc < hi))
        if not len(jj):
            raise ValueError(f"window {lo}-{hi} eV holds no electron bin")
        win_bins.append((int(jj.min()), int(jj.max())))

    with h5py.File(input_h5, "r") as f:
        keep_mask = _build_keep_mask(f, int(at["trim_start"]), int(at["trim_end"]))
        n_trains = f["gmd"].shape[0]
        g_all = f["gmd"][:, b0:b1][keep_mask].ravel()
    # GMD sub-bins: quantiles of the GMD inside each GMD bin of the aggregates
    n_sub = max(1, int(gmd_sub))
    gb_all = np.digitize(g_all, gmd_edges) - 1
    sub_interior = np.zeros((n_gmd, n_sub - 1))
    for k in range(n_gmd):
        v = g_all[gb_all == k]
        sub_interior[k] = (np.quantile(v, np.linspace(0, 1, n_sub + 1)[1:-1]) if len(v)
                           else np.full(n_sub - 1, gmd_edges[k + 1]))
    del g_all, gb_all
    chunk = int(at.get("chunk_size", 500))
    chunks = list(_iter_chunks(n_trains, chunk))
    log(f"aggregates : {agg_h5.name} ({mode}, cells {cell_shape}, {n_tof} electron bins)")
    log(f"input      : {input_h5}")
    log(f"dead times : {list(taus_ns)} ns; ticks {T0}..{T1} ({L}); windows {list(windows_ke)} eV; "
        f"{n_sub} GMD sub-bins per GMD bin; VLS frame check {'on' if vls_col is not None else 'off'}")

    task = dict(input_h5=str(input_h5), keep_mask=keep_mask, b0=b0, b1=b1, mode=mode, gmd_edges=gmd_edges,
                n_gmd=n_gmd, max_hits_e=at.get("max_hits_e"), max_hits_i=at.get("max_hits_i"), vls_col=vls_col,
                energy_key=str(at.get("energy_key", "photon_energy/measured")), e_edges=e_edges,
                n_e=None if e_edges is None else len(e_edges) - 1, z_edges=z_edges,
                n_z=None if z_edges is None else len(z_edges) - 1, n_cells=n_cells, n_tof=n_tof, L=L, T0=T0,
                tau_ticks=tau_ticks, win_bins=win_bins, tof_edges=tof_edges, n_sub=n_sub,
                sub_interior=sub_interior)
    workers = max(1, int(workers))
    if workers == 1:
        tot = _accumulate(dict(task, chunks=chunks))
    else:
        groups = [g for g in np.array_split(np.arange(len(chunks)), workers) if len(g)]
        tot = None
        for k_done, res in enumerate(_pool_map(_accumulate, [dict(task, chunks=[chunks[i] for i in g])
                                                             for g in groups], log), 1):
            if tot is None:
                tot = res
            else:
                for k in tot:
                    tot[k] += res[k]
            log(f"  worker {k_done}/{len(groups)} done")

    n_s = tot["n"].reshape(n_cells, n_sub)                  # shots per GMD sub-bin
    n = n_s.sum(axis=1)
    nn = np.where(n > 0, n, np.nan)
    D = tot["D"].reshape(n_cells, n_sub, n_tof).sum(axis=1) / nn[:, None]
    n_w = max(len(win_bins), 1)
    zero_s = tot["zero"].reshape(n_cells, n_sub, n_w)
    p0 = zero_s.sum(axis=1) / nn[:, None]
    has = n_s[..., None] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0_s = -np.log(zero_s / n_s[..., None])
        sub_ok = (np.isfinite(mu0_s) | ~has).all(axis=1)
        mu0 = np.where(sub_ok, (np.where(has, mu0_s, 0.0) * n_s[..., None]).sum(axis=1) / nn[:, None],
                       -np.log(p0))
    ticks = T0 + np.arange(L)
    tick_bin = np.searchsorted(tof_edges, ticks, side="right") - 1
    tick_bin[ticks == tof_edges[-1]] = n_tof - 1
    D_live = np.full((len(tau_ticks), n_cells, n_tof), np.nan)
    ready = np.full((len(tau_ticks), n_cells, n_tof), np.nan)
    n_fallback = []
    for k in range(len(tau_ticks)):
        R_s = n_s[..., None] - np.cumsum(tot["dR"][k].reshape(n_cells, n_sub, L + 1).astype(np.int64),
                                         axis=2)[..., :L]
        H_s = tot["H"][k].reshape(n_cells, n_sub, L).astype(np.int64)
        mu_s, ok_s = _mu(H_s, R_s)
        mu_c, _ = _mu(H_s.sum(axis=1), R_s.sum(axis=1))
        # sub-bins combined weighted by shots; ticks where a populated sub-bin is unusable take the whole cell
        bad = (~ok_s & has).any(axis=1)
        comb = (np.where(has, np.nan_to_num(mu_s), 0.0) * n_s[..., None]).sum(axis=1) / nn[:, None]
        mu = np.where(bad, mu_c, comb)
        n_fallback.append(int(bad[n > 0].sum()))
        rf = R_s.sum(axis=1) / nn[:, None]
        for jb in range(n_tof):
            m = tick_bin == jb
            if m.any():
                D_live[k, :, jb] = mu[:, m].sum(axis=1)
                ready[k, :, jb] = rf[:, m].mean(axis=1)
    D_live[:, n == 0] = np.nan
    log(f"fallback   : ticks using the whole cell (per tau) {n_fallback} of {int((n > 0).sum()) * L}")

    n_equal = bool(np.array_equal(n.reshape(cell_shape), n_ref))
    d_diff = float(np.nanmax(np.abs(D.reshape(cell_shape + (n_tof,)) - D_ref))) if n_equal else np.nan
    log(f"check      : shots per cell {'equal to' if n_equal else 'DIFFERENT from'} the aggregates "
        f"({n.sum()} vs {n_ref.sum()}); max |D - D_aggregates| = {d_diff:.3g}")

    output_h5 = Path(output_h5)
    tmp = output_h5.with_name(output_h5.name + ".part")
    with h5py.File(tmp, "w") as o:
        o["taus_ns"] = np.asarray(taus_ns, float)
        o["n_per_bin"] = n.reshape(cell_shape)
        o["D"] = D.reshape(cell_shape + (n_tof,))
        o["D_live"] = D_live.reshape((len(tau_ticks),) + cell_shape + (n_tof,))
        o["ready_frac"] = ready.reshape((len(tau_ticks),) + cell_shape + (n_tof,))
        o["p0_windows_ke"] = np.asarray(windows_ke, float).reshape(-1, 2)
        o["p0"] = p0.reshape(cell_shape + (n_w,))
        o["mu0"] = mu0.reshape(cell_shape + (n_w,))
        o["tof_edges"] = tof_edges
        o["gmd_sub_interior"] = sub_interior
        o.attrs.update(aggregates_h5=str(agg_h5), input_h5=str(at["input_h5"]), mode=mode,
                       check_n_equal=n_equal, check_D_max_abs_diff=d_diff, gmd_sub=n_sub, min_ready=MIN_READY,
                       method="Coates live-time correction per TDC tick in GMD sub-bins: D_live = sum -ln(1 - H/R)",
                       vls_frame_check=vls_col is not None)
    tmp.replace(output_h5)
    log(f"wrote      : {output_h5} ({time.time() - t_start:.0f} s)")
    if not n_equal:
        log("WARNING: the cells do not reproduce the aggregates; do not use this file")
    return output_h5


def load_ready_counts(path):
    """The datasets and attrs of a ``*_ready.h5`` file as a dict of arrays (attrs under ``"attrs"``)."""
    with h5py.File(path, "r") as f:
        out = {k: f[k][...] for k in f}
        out["attrs"] = dict(f.attrs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("aggregates", help="aggregates H5 (path, or name in config.COMBINED_DIR)")
    ap.add_argument("-o", "--output", default=None, help="default: <aggregates stem>_ready.h5 next to it")
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
                    help="dead times, ns")
    ap.add_argument("--windows", nargs="+", default=["210:285", "10:40"], help="KE windows lo:hi (eV) for p0")
    ap.add_argument("--gmd-sub", type=int, default=4, help="GMD sub-bins per GMD bin for the correction")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-vls-check", action="store_true", help="skip the VLS-frame condition of the aggregates")
    args = ap.parse_args(argv)
    agg = Path(args.aggregates)
    if not agg.exists():
        agg = config.COMBINED_DIR / agg
    out = Path(args.output) if args.output else agg.with_name(agg.stem + "_ready.h5")
    if out.exists():
        sys.exit(f"{out} exists; not overwriting")
    windows = [tuple(float(x) for x in w.split(":")) for w in args.windows]
    compute_ready_counts(agg, out, taus_ns=args.taus, windows_ke=windows, gmd_sub=args.gmd_sub,
                         workers=args.workers, vls_check=not args.no_vls_check, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
