"""
Inspect raw MCS6A .lst eTOF spectra before combined-H5 writing.

This script uses the same decoder and bunch-folding helper as write_h5.py,
then writes compact diagnostic spectra for each input .lst file:

  * per-channel event counts
  * folded eTOF spectrum for channel 1
  * folded iTOF spectrum for channel 2
  * folded liquid-jet eTOF spectrum for channel 3
  * channel-1 modulo-folded eTOF spectrum, using timedata % folding_parameter
    without relying on train-ID/eventcount splitting
  * optional per-bunch channel-1 eTOF heatmap

Run from the repo root on the FLASH cluster, for example:

  FLASH_ENV=remote python analysis/scripts/inspect_lst_tof.py \
    /asap3/flash/gpfs/fl24/2026/data/11022188/processed/local_DAQ/TDC/glycine_WL_scan_270.0eV_2659932961.lst \
    /asap3/flash/gpfs/fl24/2026/data/11022188/processed/local_DAQ/TDC/glycine_WL_scan_270.5eV_2659936861.lst \
    --tof-min 700 --tof-max 1200 --tof-bin 1 --plot

To scan the folding parameter around the nominal value:

  FLASH_ENV=remote python analysis/scripts/inspect_lst_tof.py \
    /path/to/file.lst --scan-folding 39876.9 50 0.5 \
    --tof-min 700 --tof-max 1200 --tof-bin 1 --plot
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_writer_helpers():
    root = _repo_root()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "analysis" / "scripts"))
    sys.path.insert(
        0,
        str(root / "11022188" / "processed" / "analysis_tools" / "decoding_script"),
    )
    sys.path.append(
        "/asap3/flash/gpfs/fl24/2026/data/11022188/processed/analysis_tools/decoding_script"
    )

    import write_h5  # noqa: PLC0415

    return write_h5


def _split_by_counts(values: np.ndarray, counts: np.ndarray):
    offset = 0
    for n in counts:
        n = int(n)
        yield values[offset : offset + n]
        offset += n


def _fold_all_trains(
    flat_tofs: np.ndarray,
    counts_per_train: np.ndarray,
    *,
    train_length: int,
    folding_parameter: float,
    fold_into_bunches,
):
    bunch_edges = np.arange(train_length + 1, dtype=np.float64) * folding_parameter
    per_bunch = [[] for _ in range(train_length)]

    for train_tofs in _split_by_counts(flat_tofs, counts_per_train):
        folded = fold_into_bunches(
            train_tofs,
            train_length,
            folding_parameter,
            bunch_edges,
        )
        for b, vals in enumerate(folded):
            if vals.size:
                per_bunch[b].append(np.asarray(vals, dtype=np.float64))

    out = []
    for chunks in per_bunch:
        if chunks:
            out.append(np.concatenate(chunks))
        else:
            out.append(np.empty(0, dtype=np.float64))
    return out


def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(edges.size - 1, dtype=np.int64)
    return np.histogram(values, bins=edges)[0].astype(np.int64)


def _modulo_fold(values: np.ndarray, folding_parameter: float) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    folded = np.mod(np.asarray(values, dtype=np.float64), folding_parameter)
    return folded[folded > 0]


def _write_spectrum_csv(path: Path, centres: np.ndarray, counts: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tof", "counts"])
        writer.writerows(zip(centres, counts))


def _write_bunch_csv(path: Path, centres: np.ndarray, heatmap: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bunch", *[f"{x:g}" for x in centres]])
        for b, row in enumerate(heatmap):
            writer.writerow([b, *row.astype(int)])


def _maybe_plot(
    out_png: Path,
    centres: np.ndarray,
    counts: np.ndarray,
    modulo_counts: np.ndarray,
    heatmap: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].step(centres, counts, where="mid", lw=0.8, label="split + fold")
    axes[0].step(
        centres,
        modulo_counts,
        where="mid",
        lw=0.8,
        alpha=0.7,
        label="timedata modulo fp",
    )
    axes[0].set_xlabel("folded TOF")
    axes[0].set_ylabel("channel-1 counts")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    im = axes[1].imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[centres[0], centres[-1], -0.5, heatmap.shape[0] - 0.5],
    )
    axes[1].set_xlabel("folded TOF")
    axes[1].set_ylabel("bunch")
    fig.colorbar(im, ax=axes[1], label="channel-1 counts")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _sharpness_metric(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return np.nan

    # A narrow, well-folded spectrum concentrates counts into fewer bins.
    # This is scale-normalised so candidates with slightly different totals
    # are still comparable.
    return float(np.sum(counts * counts) / (total * total))


def _scan_folding_parameter(
    args,
    lst_path: Path,
    tofs_e: np.ndarray,
    edges: np.ndarray,
) -> None:
    centre, span, step = args.scan_folding
    centre = float(centre)
    span = float(span)
    step = float(step)
    if span <= 0 or step <= 0:
        raise ValueError("--scan-folding requires positive span and step")

    fps = np.arange(centre - span, centre + span + 0.5 * step, step)
    rows = []
    for fp in fps:
        counts = _hist(_modulo_fold(tofs_e, fp), edges)
        rows.append(
            (
                float(fp),
                _sharpness_metric(counts),
                int(np.sum(counts)),
                int(np.max(counts)) if counts.size else 0,
            )
        )

    metrics = np.array([r[1] for r in rows], dtype=float)
    best_idx = int(np.nanargmax(metrics))
    best_fp, best_metric, best_hits, best_peak = rows[best_idx]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = lst_path.stem
    out_csv = out_dir / f"{stem}_folding_parameter_scan.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folding_parameter", "sharpness", "hits_in_window", "max_bin"])
        writer.writerows(rows)

    print("  folding-parameter scan:")
    print(f"    centre/span/step       : {centre:g} +/- {span:g} / {step:g}")
    print(f"    best folding parameter : {best_fp:.6f}")
    print(f"    nominal metric         : {rows[int(np.argmin(np.abs(fps - centre)))][1]:.8g}")
    print(f"    best metric            : {best_metric:.8g}")
    print(f"    best hits/max bin      : {best_hits} / {best_peak}")
    print(f"    wrote scan CSV         : {out_csv}")

    if args.plot:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        axes[0].plot(fps, metrics, "o-", ms=3)
        axes[0].axvline(centre, color="k", lw=0.8, ls="--", label="nominal")
        axes[0].axvline(best_fp, color="r", lw=0.8, ls=":", label="best")
        axes[0].set_xlabel("folding parameter")
        axes[0].set_ylabel("sharpness")
        axes[0].set_title(f"{lst_path.name}: folding-parameter scan")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        centres = 0.5 * (edges[:-1] + edges[1:])
        candidates = [centre, best_fp]
        if abs(best_fp - centre) < step * 0.5:
            candidates.extend([centre - span, centre + span])
        for fp in candidates:
            counts = _hist(_modulo_fold(tofs_e, fp), edges)
            label = f"fp={fp:.3f}"
            axes[1].step(centres, counts, where="mid", lw=0.8, label=label)
        axes[1].set_xlabel("folded TOF")
        axes[1].set_ylabel("channel-1 counts")
        axes[1].grid(alpha=0.2)
        axes[1].legend()

        out_png = out_dir / f"{stem}_folding_parameter_scan.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"    wrote scan PNG         : {out_png}")


def inspect_file(args, lst_path: Path, write_h5) -> None:
    decoder = write_h5.decoding.Decoder(
        wanted_data=["channel", "timedata", "sweep", "tagbits"]
    )
    sweeps_per_file = write_h5.metadata_decoding.from_file(
        str(lst_path),
        keywords="swpreset=",
        paragraph_kw="MPA4A",
    )[0]

    (
        train_ids,
        eventcounts_e,
        tofs_e,
        eventcounts_i,
        tofs_i,
        eventcounts_le,
        tofs_le,
    ) = write_h5.extract_data_from_single_file(
        str(lst_path),
        decoder,
        int(sweeps_per_file),
    )

    edges = np.arange(args.tof_min, args.tof_max + args.tof_bin, args.tof_bin)
    centres = 0.5 * (edges[:-1] + edges[1:])

    folded_e = _fold_all_trains(
        tofs_e,
        eventcounts_e,
        train_length=args.train_length,
        folding_parameter=args.folding_parameter,
        fold_into_bunches=write_h5._fold_into_bunches,
    )
    folded_i = _fold_all_trains(
        tofs_i,
        eventcounts_i,
        train_length=args.train_length,
        folding_parameter=args.folding_parameter,
        fold_into_bunches=write_h5._fold_into_bunches,
    )
    folded_le = _fold_all_trains(
        tofs_le,
        eventcounts_le,
        train_length=args.train_length,
        folding_parameter=args.folding_parameter,
        fold_into_bunches=write_h5._fold_into_bunches,
    )

    e_all = np.concatenate(folded_e) if any(x.size for x in folded_e) else np.empty(0)
    i_all = np.concatenate(folded_i) if any(x.size for x in folded_i) else np.empty(0)
    le_all = np.concatenate(folded_le) if any(x.size for x in folded_le) else np.empty(0)

    counts_e = _hist(e_all, edges)
    counts_i = _hist(i_all, edges)
    counts_le = _hist(le_all, edges)
    counts_e_mod = _hist(_modulo_fold(tofs_e, args.folding_parameter), edges)
    heatmap_e = np.vstack([_hist(vals, edges) for vals in folded_e])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = lst_path.stem

    _write_spectrum_csv(out_dir / f"{stem}_ch1_folded_etof.csv", centres, counts_e)
    _write_spectrum_csv(out_dir / f"{stem}_ch1_modulo_etof.csv", centres, counts_e_mod)
    _write_spectrum_csv(out_dir / f"{stem}_ch2_folded_itof.csv", centres, counts_i)
    _write_spectrum_csv(out_dir / f"{stem}_ch3_folded_liq_etof.csv", centres, counts_le)
    _write_bunch_csv(out_dir / f"{stem}_ch1_bunch_heatmap.csv", centres, heatmap_e)

    if args.plot:
        _maybe_plot(
            out_dir / f"{stem}_ch1_folded_diagnostic.png",
            centres,
            counts_e,
            counts_e_mod,
            heatmap_e,
            f"{lst_path.name}: channel 1 folded eTOF",
        )

    print(f"\n{lst_path}")
    print(f"  sweeps_per_file metadata : {int(sweeps_per_file)}")
    print(f"  train IDs decoded        : {len(train_ids)}")
    print(f"  channel 1 events         : {len(tofs_e)}")
    print(f"  channel 2 events         : {len(tofs_i)}")
    print(f"  channel 3 events         : {len(tofs_le)}")
    print(f"  ch1 hits in TOF window   : {int(counts_e.sum())}")
    print(f"  ch1 modulo hits in window: {int(counts_e_mod.sum())}")
    print(f"  ch2 hits in TOF window   : {int(counts_i.sum())}")
    print(f"  ch3 hits in TOF window   : {int(counts_le.sum())}")
    print(f"  ch1 split-vs-mod max diff: {int(np.max(np.abs(counts_e - counts_e_mod)))}")

    if args.scan_folding is not None:
        _scan_folding_parameter(args, lst_path, tofs_e, edges)

    print(f"  ch1 top bunches in window:")
    bunch_totals = heatmap_e.sum(axis=1)
    top = np.argsort(bunch_totals)[-10:][::-1]
    for b in top:
        if bunch_totals[b] <= 0:
            continue
        print(f"    bunch {b:3d}: {int(bunch_totals[b])}")
    print(f"  wrote diagnostics to     : {out_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lst_paths", nargs="+", type=Path)
    parser.add_argument("--tof-min", type=float, default=700.0)
    parser.add_argument("--tof-max", type=float, default=1200.0)
    parser.add_argument("--tof-bin", type=float, default=1.0)
    parser.add_argument("--train-length", type=int, default=101)
    parser.add_argument("--folding-parameter", type=float, default=39876.9)
    parser.add_argument(
        "--scan-folding",
        nargs=3,
        metavar=("CENTRE", "SPAN", "STEP"),
        type=float,
        default=None,
        help=(
            "Scan channel-1 modulo-folded spectra over folding parameters "
            "CENTRE +/- SPAN with spacing STEP, using a peak sharpness metric."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/post/lst_tof_diagnostics"),
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    write_h5 = _import_writer_helpers()

    for lst_path in args.lst_paths:
        inspect_file(args, lst_path, write_h5)


if __name__ == "__main__":
    main()
