"""
Per-shot photon energy measured with the VLS (Gotthard), for undulator scans.

The `vls` dataset of a combined H5 is streamed in blocks of trains. Per shot,
after subtracting a pedestal (mean of the dark bunch slot ``dark_slot``, a
Gotthard line without a pulse) and a common mode (median over
``bg_pixels``), the peak is found on a boxcar-smoothed spectrum and its
centre of mass taken within ``half_width`` pixels of it. The signal in
that window is compared with what the same estimator gives on the dark
slot; shots above ``min_snr`` robust standard deviations of that null get
a centre.

Pixel -> eV: the centre of each nominal step's mean spectrum (trains at
least ``settle_trains`` after a change of ``nominal_energy``) is fitted
against the nominal energy with a polynomial of order ``cal_order``. The
measured energies are therefore on the undulator (nominal) energy scale;
they add the shot-to-shot spread and the real timing of the steps.

The result goes into a ``photon_energy/`` group of the combined H5:

    measured        (n, m) float32  eV; NaN = no pulse or VLS signal too weak
    vls_centre      (n, m) float32  peak centre (Gotthard pixel)
    vls_window_sum  (n, m) float32  ADU in the centre window
    cal_coef        (cal_order+1,)  np.polyval coefficients, E(pixel)
    cal_energies, cal_centres, cal_residuals   (n_steps,)
    step_spectra    (n_steps, n_px) mean spectrum per nominal step
    pixels          (n_px,)         Gotthard pixels of step_spectra
    attrs: the parameters, threshold, dark median / sigma

The VLS frames come from the combined file's ``vls`` dataset or, with
``raw_run``, straight from that run's raw FLASH files (Gotthard line
``b + vls_offset`` -> bunch ``b``, as ``write_h5.py`` stores them). The
raw route serves combined files written without VLS (``--no-vls``, e.g.
live writes of a long run) and gives the same numbers.

CLI:
    python photon_energy.py COMBINED.h5 [--raw-run N] [--workers N] [--overwrite] [--roi LO HI] ...
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

import config
from data_loading import _VLS_INDEX, _VLS_VALUE, _crop_vls_block

__all__ = ["GROUP", "DEFAULTS", "measure_photon_energy", "save_photon_energy", "load_photon_energy"]

GROUP = "photon_energy"
DEFAULTS = dict(
    roi=(200, 650),            # 200-650 holds the 265-282 eV peak of run 60559
    bg_pixels=(650, 800),
    bad_pixels=(0, 1, 291, 292, 679, 680, 681, *range(832, 864)),
    dark_slot=100,             # Gotthard line 102 with VLS_OFFSET = 2
    smooth=9,
    half_width=30,
    min_snr=5.0,
    settle_trains=30,
    cal_order=1,
)


def _centre(x, px, smooth, half_width):
    """Centre of mass within half_width of the smoothed maximum, along the last axis."""
    sm = uniform_filter1d(np.nan_to_num(x), smooth, axis=-1, mode="nearest")
    peak = px[np.argmax(sm, axis=-1)]
    w = np.abs(px - np.asarray(peak)[..., None]) <= half_width
    xc = np.where(w, np.clip(x, 0, None), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = (xc * px).sum(-1) / xc.sum(-1)
    return centre, np.where(w, x, 0).sum(-1)


def _raw_paths(run_no, min_age=120):
    """Raw FLASH files of a run, in file order; files modified in the last min_age s are skipped."""
    paths = glob.glob(str(Path(config.RAW_H5_DIR) / f"*run{run_no}_*.h5"))
    paths = [q for q in paths if time.time() - os.path.getmtime(q) >= min_age]
    return sorted(paths, key=lambda q: int(Path(q).name.split("_file")[1].split("_")[0]))


def _iter_vls(f, tid, m, block, raw_paths, vls_offset, rows=None):
    """
    Yield ``(rows, frames)``: combined-file row indices (ascending) and their
    VLS frames ``(len(rows), m, n_pixels)`` float32 lined up with the bunch
    axis. ``rows`` restricts to a subset of rows. From the combined ``vls``
    dataset, or (``raw_paths``) from raw files matched by train ID; trains
    in no raw file are not yielded, a train in two files only once.
    """
    rows = np.arange(len(tid)) if rows is None else np.asarray(rows)
    if raw_paths is None:
        d = f["vls"]
        for i in range(0, len(rows), block):
            r = rows[i:i + block]
            yield r, (d[r[0]:r[-1] + 1] if r[-1] - r[0] + 1 == len(r) else d[r])
        return
    done = np.zeros(len(tid), bool)
    for path in raw_paths:
        with h5py.File(path, "r") as raw:
            idx = raw[_VLS_INDEX][...].astype(np.int64)
            order = np.argsort(idx, kind="stable")
            pos = np.clip(np.searchsorted(idx[order], tid[rows]), 0, len(idx) - 1)
            hit = (idx[order][pos] == tid[rows]) & ~done[rows]
            r_hit, src = rows[hit], order[pos[hit]]
            dset = raw[_VLS_VALUE]
            for i in range(0, len(r_hit), block):
                s_blk = src[i:i + block]
                o = np.argsort(s_blk)                      # h5py needs increasing indices
                frames = np.empty((len(s_blk), m, dset.shape[2]), np.float32)
                frames[o] = _crop_vls_block(dset[s_blk[o]], m, dset.shape[2], vls_offset)
                done[r_hit[i:i + block]] = True
                yield r_hit[i:i + block], frames


def _pe_dark(task):
    """Worker: sum and finite count of the dark-slot frames of ``task["rows"]`` (pedestal pass)."""
    tot = cnt = None
    with h5py.File(task["h5_path"], "r") as f:
        for _, x in _iter_vls(f, task["tid"], task["m"], task["block"], task["raw_paths"], task["vls_offset"],
                              rows=task["rows"]):
            d = x[:, task["dark_slot"], :].astype(np.float64)
            if tot is None:
                tot, cnt = np.zeros(d.shape[1]), np.zeros(d.shape[1])
            tot += np.nansum(d, axis=0)
            cnt += np.isfinite(d).sum(axis=0)
    return tot, cnt


def _pe_part(task):
    """
    Worker: per-shot peak centres and window sums, and per-step sums of the
    spectra, for the rows ``task["rows"]`` (ascending) of the combined file.
    """
    rows, m, px, bg, ped = task["rows"], task["m"], task["px"], task["bg"], task["ped"]
    centre = np.full((len(rows), m), np.nan, np.float32)
    win = np.full((len(rows), m), np.nan, np.float32)
    spec_sum = np.zeros((task["n_steps"], len(px)))
    spec_n = np.zeros(task["n_steps"])
    with h5py.File(task["h5_path"], "r") as f:
        for r, x in _iter_vls(f, task["tid"], m, task["block"], task["raw_paths"], task["vls_offset"], rows=rows):
            loc = np.searchsorted(rows, r)
            frame = np.isfinite(x[:, :, px[0]])
            cm = np.median(x[:, :, bg] - ped[bg], axis=-1, keepdims=True)
            x = x[:, :, px] - ped[px] - cm
            c, sm = _centre(x, px, task["smooth"], task["half_width"])
            centre[loc] = np.where(frame, c, np.nan)
            win[loc] = np.where(frame, sm, np.nan)
            lit = frame & np.isfinite(task["gmd"][loc])
            st = task["step_of"][loc]
            for k in np.unique(st):
                if k < 0:
                    continue
                mk = lit & (st == k)[:, None]
                spec_sum[k] += x[mk].sum(axis=0)
                spec_n[k] += mk.sum()
    return dict(rows=rows, centre=centre, win=win, spec_sum=spec_sum, spec_n=spec_n)


def _map(fn, tasks, workers, log):
    """Results of fn over tasks: in-process for one worker, else spawned processes (compute_aggregates._pool_map)."""
    if workers <= 1 or len(tasks) == 1:
        return [fn(t) for t in tasks]
    from compute_aggregates import _pool_map
    return list(_pool_map(fn, tasks, log))


def measure_photon_energy(h5_path, *, block=100, verbose=True, raw_run=None, vls_offset=None, workers=1,
                          **params) -> dict:
    """
    Stream the VLS of a combined H5 (or, with ``raw_run``, of that run's raw
    files) and return per-shot centres, energies and the calibration.
    ``workers`` > 1 splits the trains over that many processes.
    """
    p = {**DEFAULTS, **params}
    log = print if verbose else (lambda *a, **k: None)
    bad = set(p["bad_pixels"])
    px = np.array([q for q in range(*p["roi"]) if q not in bad])
    bg = np.array([q for q in range(*p["bg_pixels"]) if q not in bad])

    with h5py.File(h5_path, "r") as f:
        if "nominal_energy" not in f:
            raise RuntimeError("no nominal_energy in the file: rewrite it with the current write_h5.py")
        tid = f["tID"][...].astype(np.int64)
        gmd = f["gmd"][...]
        E_nom = np.round(f["nominal_energy"][...].astype(np.float64), 3)
        n, m = gmd.shape
        if raw_run is None:
            if "vls" not in f:
                raise RuntimeError("no vls dataset in the file: pass raw_run to read the raw files")
            raw_paths = None
            source = "combined vls"
        else:
            vls_offset = config.VLS_OFFSET if vls_offset is None else int(vls_offset)
            raw_paths = _raw_paths(raw_run)
            source = f"raw files of run {raw_run} ({len(raw_paths)}), line b+{vls_offset} -> bunch b"
        if np.isfinite(gmd[:, p["dark_slot"]]).any():
            raise ValueError(f"dark_slot {p['dark_slot']} has GMD pulses")

        # Calibration steps: trains at least settle_trains after a change of the nominal energy
        is_change = np.r_[False, E_nom[1:] != E_nom[:-1]]
        change_t = tid[is_change]
        k_last = np.searchsorted(change_t, tid, side="right") - 1
        since = np.where(k_last >= 0, tid - change_t[np.clip(k_last, 0, None)], np.iinfo(np.int64).max)
        settled = np.isfinite(E_nom) & (since >= p["settle_trains"])
        steps = np.unique(E_nom[settled])
        step_of = np.full(n, -1)
        step_of[settled] = np.searchsorted(steps, E_nom[settled])

    # Pedestal (dark slot, every 10th train), then the per-shot pass; each split over the workers
    workers = max(1, int(workers))
    common = dict(h5_path=str(h5_path), tid=tid, m=m, block=block, raw_paths=raw_paths, vls_offset=vls_offset)
    ped_rows = np.arange(0, n, 10)
    parts = _map(_pe_dark, [dict(common, rows=r, dark_slot=p["dark_slot"])
                            for r in np.array_split(ped_rows, workers) if len(r)], workers, log)
    tot = sum(t for t, _ in parts if t is not None)
    cnt = sum(c for _, c in parts if c is not None)
    with np.errstate(invalid="ignore", divide="ignore"):
        ped = np.where(cnt > 0, tot / cnt, np.nan)
    log(f"{Path(h5_path).name}: {n} trains x {m} slots, {len(steps)} nominal steps; "
        f"streaming the VLS from {source}" + (f" with {workers} workers" if workers > 1 else "") + "...")
    tasks = [dict(common, rows=r, px=px, bg=bg, ped=ped, smooth=p["smooth"], half_width=p["half_width"],
                  gmd=gmd[r], step_of=step_of[r], n_steps=len(steps))
             for r in np.array_split(np.arange(n), workers) if len(r)]
    centre = np.full((n, m), np.nan, np.float32)
    win = np.full((n, m), np.nan, np.float32)
    spec_sum = np.zeros((len(steps), len(px)))
    spec_n = np.zeros(len(steps))
    for res in _map(_pe_part, tasks, workers, log):
        centre[res["rows"]], win[res["rows"]] = res["centre"], res["win"]
        spec_sum += res["spec_sum"]
        spec_n += res["spec_n"]

    dark = win[:, p["dark_slot"]][np.isfinite(win[:, p["dark_slot"]])].astype(np.float64)
    dark_med = float(np.median(dark))
    dark_sd = float(1.4826 * np.median(np.abs(dark - dark_med)))
    thr = dark_med + p["min_snr"] * dark_sd
    ok = (win > thr) & np.isfinite(centre) & np.isfinite(gmd)

    spectra = spec_sum / spec_n[:, None]
    cal_centres = np.array([_centre(s, px, p["smooth"], p["half_width"])[0] for s in spectra])
    coef = np.polyfit(cal_centres, steps, p["cal_order"])
    resid = steps - np.polyval(coef, cal_centres)
    measured = np.where(ok, np.polyval(coef, centre), np.nan).astype(np.float32)
    lit_all = np.isfinite(gmd)
    log(f"threshold {thr:.0f} ADU (dark {dark_med:.0f} +- {dark_sd:.0f}); "
        f"measured energy for {ok[lit_all].mean():.1%} of the {lit_all.sum()} shots with a pulse")
    log(f"calibration E(px) = {' + '.join(f'{c:.6g} px^{len(coef) - 1 - i}' for i, c in enumerate(coef))}; "
        f"rms residual {np.sqrt(np.mean(resid ** 2)):.3f} eV over {len(steps)} steps")
    return dict(tID=tid, measured=measured, vls_centre=centre, vls_window_sum=win, cal_coef=coef,
                cal_energies=steps, cal_centres=cal_centres, cal_residuals=resid, step_spectra=spectra,
                pixels=px, threshold=thr, dark_median=dark_med, dark_sigma=dark_sd,
                params={**p, "source": source})


def save_photon_energy(h5_path, res: dict, overwrite: bool = False) -> None:
    """Write the result of :func:`measure_photon_energy` into the ``photon_energy/`` group of the same file."""
    with h5py.File(h5_path, "a") as f:
        if not np.array_equal(f["tID"][...].astype(np.int64), res["tID"]):
            raise ValueError("train IDs differ from the file the energies were measured on")
        if GROUP in f:
            if not overwrite:
                raise FileExistsError(f"{GROUP}/ already in {h5_path}; pass overwrite=True")
            del f[GROUP]
        g = f.create_group(GROUP)
        for k in ("measured", "vls_centre", "vls_window_sum"):
            g.create_dataset(k, data=res[k], compression="gzip")
        for k in ("cal_coef", "cal_energies", "cal_centres", "cal_residuals", "step_spectra", "pixels"):
            g.create_dataset(k, data=res[k])
        for k in ("threshold", "dark_median", "dark_sigma"):
            g.attrs[k] = res[k]
        for k, v in res["params"].items():
            g.attrs[k] = v if isinstance(v, str) else np.asarray(v)


def load_photon_energy(h5_path) -> dict:
    """Read the ``photon_energy/`` group back (arrays + attrs)."""
    with h5py.File(h5_path, "r") as f:
        g = f[GROUP]
        out = {k: g[k][...] for k in g}
        out.update({k: g.attrs[k] for k in g.attrs})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure the per-shot photon energy with the VLS and store it "
                                             "in the combined H5 (photon_energy/ group).")
    ap.add_argument("combined_h5", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing photon_energy/ group")
    ap.add_argument("--raw-run", type=int, default=None,
                    help="read the VLS from this run's raw files (for combined files written without VLS)")
    ap.add_argument("--vls-offset", type=int, default=None,
                    help="with --raw-run: Gotthard line stored as bunch 0 (default config.VLS_OFFSET)")
    ap.add_argument("--roi", type=int, nargs=2, default=DEFAULTS["roi"])
    ap.add_argument("--bg-pixels", type=int, nargs=2, default=DEFAULTS["bg_pixels"])
    ap.add_argument("--dark-slot", type=int, default=DEFAULTS["dark_slot"])
    ap.add_argument("--min-snr", type=float, default=DEFAULTS["min_snr"])
    ap.add_argument("--settle-trains", type=int, default=DEFAULTS["settle_trains"])
    ap.add_argument("--cal-order", type=int, default=DEFAULTS["cal_order"])
    ap.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    a = ap.parse_args(argv)
    res = measure_photon_energy(a.combined_h5, roi=tuple(a.roi), bg_pixels=tuple(a.bg_pixels),
                                dark_slot=a.dark_slot, min_snr=a.min_snr, settle_trains=a.settle_trains,
                                cal_order=a.cal_order, raw_run=a.raw_run, vls_offset=a.vls_offset,
                                workers=a.workers)
    save_photon_energy(a.combined_h5, res, overwrite=a.overwrite)
    print(f"wrote {GROUP}/ into {a.combined_h5}")


if __name__ == "__main__":
    main()
