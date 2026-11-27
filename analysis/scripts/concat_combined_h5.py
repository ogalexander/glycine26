"""
Concatenate combined H5 files (``write_h5.py`` output) along the train axis, e.g. the parts of one scan that
were acquired as separate runs.

Every top-level dataset is concatenated in train order. The inputs must hold the same datasets, agree in the
trailing shapes, dtypes and dataset attributes, and must not overlap in train ID. Datasets stored one train per
chunk (the VLS) are copied chunk by chunk without recompression; the others are copied in blocks.

A ``photon_energy/`` group (``photon_energy.py``) is carried over when every input has one. Each input keeps its
own pixel -> eV calibration, i.e. its run's nominal energy scale, which is how ``photon_energy.py`` calibrates a
run (a single fit through the steps of several runs would mix their VLS offsets):

    measured, vls_centre, vls_window_sum     (n, m)    concatenated
    cal_energies, cal_centres, cal_residuals (n_steps) concatenated over the inputs; step_spectra likewise
    cal_source                               (n_steps) input index of each calibration step
    cal_coef                                 (n_inputs, order + 1), row i = input i
    pixels                                   (n_px)    the same for every input (checked)
    attrs: equal in every input -> kept as they are; different (dark_median, threshold, source, ...) -> one
           entry per input

Other groups are not copied (listed in the log). The root attributes record the inputs: ``sources`` (file
names), ``source_first_index`` and ``source_n_trains``. The output is written as ``<out>.partial`` and renamed
when complete; an existing output is never overwritten. Inputs are opened read-only.

CLI:
    python concat_combined_h5.py OUT.h5 IN1.h5 IN2.h5 [...] [--check N]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

__all__ = ["concat_combined"]

PE = "photon_energy"
PE_SHOT = ("measured", "vls_centre", "vls_window_sum")
PE_STEP = ("cal_energies", "cal_centres", "cal_residuals", "step_spectra")


def _create_like(parent, name, src, n):
    """Dataset of n trains with the creation properties (chunks, filters, fill value) of ``src``."""
    chunks = None if src.chunks is None else (min(src.chunks[0], n),) + src.chunks[1:]
    return parent.create_dataset(name, shape=(n,) + src.shape[1:], dtype=src.dtype, chunks=chunks,
                                 compression=src.compression, compression_opts=src.compression_opts,
                                 shuffle=src.shuffle, fletcher32=src.fletcher32, fillvalue=src.fillvalue)


def _copy_rows(src, dst, off):
    """Copy all trains of ``src`` into ``dst`` starting at train ``off``."""
    n = src.shape[0]
    rest = (0,) * (src.ndim - 1)
    direct = (src.chunks is not None and src.chunks == dst.chunks and src.chunks[0] == 1
              and src.chunks[1:] == src.shape[1:] and src.compression == dst.compression
              and src.compression_opts == dst.compression_opts and src.shuffle == dst.shuffle)
    if direct:                                      # one train per chunk: copy the stored bytes
        for i in range(n):
            try:
                mask, buf = src.id.read_direct_chunk((i,) + rest)
            except (KeyError, OSError, RuntimeError, ValueError):   # chunk never written: copy the fill values
                dst[off + i] = src[i]
                continue
            dst.id.write_direct_chunk((off + i,) + rest, buf, mask)
        return
    block = max(src.chunks[0] if src.chunks else 0, 2000)
    for s in range(0, n, block):
        e = min(s + block, n)
        dst[off + s:off + e] = src[s:e]


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype.kind == b.dtype.kind and bool(
        np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def _attr_per_input(values):
    """One attribute value per input, in a form h5py stores."""
    if all(isinstance(v, (str, bytes)) for v in values):
        return [v.decode() if isinstance(v, bytes) else v for v in values]
    try:
        return np.stack([np.asarray(v) for v in values])
    except ValueError:                              # different shapes: keep them as text
        return [repr(np.asarray(v).tolist()) for v in values]


def concat_combined(out_path, in_paths, *, check: int = 5, log=print) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; not overwriting")
    tmp = out_path.with_name(out_path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists (an interrupted run?); remove it first")
    srcs = [h5py.File(p, "r") for p in in_paths]
    try:
        first_tid = [float(f["tID"][0]) for f in srcs]
        order = np.argsort(first_tid)
        srcs = [srcs[i] for i in order]
        in_paths = [Path(in_paths[i]) for i in order]
        tids = [f["tID"][...] for f in srcs]
        for p, t in zip(in_paths, tids):
            if not np.all(np.diff(t) > 0):
                raise ValueError(f"{p.name}: tID not strictly increasing")
        for (pa, a), (pb, b) in zip(zip(in_paths, tids), zip(in_paths[1:], tids[1:])):
            if b[0] <= a[-1]:
                raise ValueError(f"{pb.name} starts at train {b[0]:.0f}, before the end of {pa.name} ({a[-1]:.0f})")
        ns = [len(t) for t in tids]
        first = np.r_[0, np.cumsum(ns)[:-1]].astype(np.int64)
        n = int(sum(ns))

        names = sorted(k for k, v in srcs[0].items() if isinstance(v, h5py.Dataset))
        groups = sorted(k for k, v in srcs[0].items() if isinstance(v, h5py.Group))
        for p, f, ni in zip(in_paths, srcs, ns):
            have = sorted(k for k, v in f.items() if isinstance(v, h5py.Dataset))
            if have != names:
                raise ValueError(f"{p.name}: datasets {have} differ from {names}")
            for k in names:
                d, d0 = f[k], srcs[0][k]
                if d.shape[0] != ni or d.shape[1:] != d0.shape[1:] or d.dtype != d0.dtype:
                    raise ValueError(f"{p.name}:{k} shape {d.shape} {d.dtype} vs {d0.shape} {d0.dtype}")
                if sorted(d.attrs) != sorted(d0.attrs) or not all(_same(d.attrs[a], d0.attrs[a]) for a in d0.attrs):
                    raise ValueError(f"{p.name}:{k} attributes differ from {in_paths[0].name}")
        with_pe = PE in groups and all(PE in f for f in srcs)
        skipped = [g for g in groups if g != PE or not with_pe]
        log(f"inputs ({n} trains):")
        for p, ni, o, t in zip(in_paths, ns, first, tids):
            log(f"  {p.name}: {ni} trains {t[0]:.0f}..{t[-1]:.0f} -> rows {o}..{o + ni - 1}")
        if skipped:
            log(f"groups not copied: {skipped}" + ("  (photon_energy/ is missing in an input: run photon_energy.py "
                                                  "on the output)" if PE in skipped else ""))

        t0 = time.time()
        with h5py.File(tmp, "w", rdcc_nbytes=512 * 1024 ** 2) as fo:
            for k in names:
                d0 = srcs[0][k]
                dst = _create_like(fo, k, d0, n)
                for a in d0.attrs:
                    dst.attrs[a] = d0.attrs[a]
                for f, o in zip(srcs, first):
                    _copy_rows(f[k], dst, int(o))
                log(f"  {k:20s} {str(dst.shape):22s} {dst.dtype}  ({time.time() - t0:.0f} s)")

            if with_pe:
                gs = [f[PE] for f in srcs]
                go = fo.create_group(PE)
                for k in PE_SHOT:
                    dst = _create_like(go, k, gs[0][k], n)
                    for g, o in zip(gs, first):
                        if g[k].shape[0] != len(g.file["tID"]):
                            raise ValueError(f"{PE}/{k} does not have one row per train")
                        _copy_rows(g[k], dst, int(o))
                for k in PE_STEP:
                    go.create_dataset(k, data=np.concatenate([g[k][...] for g in gs]))
                go.create_dataset("cal_source", data=np.concatenate(
                    [np.full(len(g["cal_energies"]), i, np.int64) for i, g in enumerate(gs)]))
                go.create_dataset("cal_coef", data=np.stack([g["cal_coef"][...] for g in gs]))
                if not all(_same(g["pixels"][...], gs[0]["pixels"][...]) for g in gs):
                    raise ValueError(f"{PE}/pixels differ between the inputs")
                go.create_dataset("pixels", data=gs[0]["pixels"][...])
                keys = sorted(set().union(*(g.attrs.keys() for g in gs)))
                per_input = []
                for a in keys:
                    vals = [g.attrs.get(a) for g in gs]
                    if all(v is not None for v in vals) and all(_same(v, vals[0]) for v in vals):
                        go.attrs[a] = vals[0]
                    else:
                        go.attrs[a] = _attr_per_input(["" if v is None else v for v in vals])
                        per_input.append(a)
                go.attrs["concatenated"] = (
                    f"per-shot datasets concatenated from {len(gs)} inputs (root attrs sources / source_first_index); "
                    "each keeps its own calibration: cal_coef[i] and the steps with cal_source == i belong to input i; "
                    f"one entry per input in the attributes {per_input}")
                log(f"  {PE}/ carried over; per-input attributes: {per_input}")

            fo.attrs["sources"] = [p.name for p in in_paths]
            fo.attrs["source_first_index"] = first
            fo.attrs["source_n_trains"] = np.asarray(ns, np.int64)
            fo.attrs["concatenated_by"] = "analysis/scripts/concat_combined_h5.py"

        # spot check: random trains of every input against the output
        if check:
            rng = np.random.default_rng(0)
            with h5py.File(tmp, "r") as fo:
                keys = names + ([f"{PE}/{k}" for k in PE_SHOT] if with_pe else [])
                for f, o, ni in zip(srcs, first, ns):
                    for i in np.unique(np.r_[0, ni - 1, rng.integers(0, ni, check)]):
                        for k in keys:
                            if not _same(f[k][i], fo[k][o + i]):
                                raise RuntimeError(f"check failed: {k} train {i} of {Path(f.filename).name}")
            log(f"check: {check}+2 trains per input equal in every dataset")
        os.replace(tmp, out_path)
        log(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f} s)")
        return out_path
    finally:
        for f in srcs:
            f.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Concatenate combined H5 files along the train axis.")
    ap.add_argument("output", type=Path)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--check", type=int, default=5, help="random trains per input compared after writing (0: none)")
    a = ap.parse_args(argv)
    if len(a.inputs) < 2:
        ap.error("give at least two inputs")
    concat_combined(a.output, a.inputs, check=a.check)


if __name__ == "__main__":
    main()
