from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import h5py
import numpy as np


GOTTHARD_INDEX = "/FL2/Support Infrastructure/Gotthard/images/index"


def safe_name(path: str) -> str:
    name = path.strip("/").replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name[:180]


def parse_shape(text: str) -> tuple[int, ...]:
    stripped = text.strip().strip("()")
    if not stripped:
        return ()
    return tuple(int(part.strip()) for part in stripped.split(",") if part.strip())


def robust_limits(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def downsample_2d(arr: np.ndarray, max_rows: int = 700, max_cols: int = 700) -> np.ndarray:
    row_step = max(1, int(np.ceil(arr.shape[0] / max_rows)))
    col_step = max(1, int(np.ceil(arr.shape[1] / max_cols)))
    return arr[::row_step, ::col_step]


def align_positions(source_index: np.ndarray, master_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = {int(t): i for i, t in enumerate(source_index)}
    src = np.array([pos.get(int(t), -1) for t in master_index], dtype=int)
    valid = src >= 0
    return src, valid


def load_aligned_2d(h5: h5py.File, path: str, master_index: np.ndarray) -> np.ndarray:
    idx = np.asarray(h5[path + "/index"][...]).reshape(-1)
    ds = h5[path + "/value"]
    src, valid = align_positions(idx, master_index)
    out = np.full((master_index.size, ds.shape[1]), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = np.asarray(ds[src[valid], :], dtype=np.float32)
    return out


def plot_2d_parameter(path: str, arr: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    finite = np.isfinite(arr)
    row_mean = np.nanmean(np.where(finite, arr, np.nan), axis=1)
    col_mean = np.nanmean(np.where(finite, arr, np.nan), axis=0)
    valid_rows = np.flatnonzero(np.isfinite(row_mean))
    if valid_rows.size >= 5:
        example_rows = valid_rows[np.linspace(0, valid_rows.size - 1, 5, dtype=int)]
    else:
        example_rows = np.arange(min(arr.shape[0], 5), dtype=int)
    heat = downsample_2d(arr)
    vmin, vmax = robust_limits(heat)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [2.0, 1.0]})
    im = axs[0, 0].imshow(heat.T, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("value heatmap, aligned to Gotthard trains")
    axs[0, 0].set_xlabel("train index, downsampled")
    axs[0, 0].set_ylabel("second-axis index")
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

    axs[0, 1].plot(col_mean, lw=1.2)
    axs[0, 1].set_title("mean over trains by second-axis index")
    axs[0, 1].set_xlabel("second-axis index")
    axs[0, 1].set_ylabel("mean value")

    axs[1, 0].plot(row_mean, lw=0.8)
    axs[1, 0].set_title("mean over second axis by train")
    axs[1, 0].set_xlabel("Gotthard train row")
    axs[1, 0].set_ylabel("mean value")

    x = np.arange(arr.shape[1])
    for row in example_rows:
        axs[1, 1].plot(x, arr[row], lw=1.0, label=f"train row {row}")
    axs[1, 1].set_title("five example trains along second axis")
    axs[1, 1].set_xlabel("second-axis index")
    axs[1, 1].set_ylabel("value")
    axs[1, 1].legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_h5", type=Path)
    parser.add_argument("--train-resolved-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures" / "train_resolved_2d_params"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with args.train_resolved_csv.open() as f:
        rows = list(csv.DictReader(f))
    array_rows = [r for r in rows if len(parse_shape(r["value_shape"])) == 2]

    index_lines = [
        "# Train-Resolved 2D Parameter Plots",
        "",
        f"Raw file: `{args.raw_h5}`",
        f"Source table: `{args.train_resolved_csv}`",
        "",
        "The plots use exact train-ID alignment to the Gotthard master axis.",
        "Only 2D parameters with shape `(train, second_axis)` are included; VLS/Gotthard and GMD 3D arrays are excluded.",
        "",
        "Each plot shows: value heatmap, mean vs second-axis index, mean vs train, and five example train traces along the second axis.",
        "",
        "| parameter | shape | plot |",
        "|---|---:|---|",
    ]

    with h5py.File(args.raw_h5, "r") as h5:
        master_index = np.asarray(h5[GOTTHARD_INDEX][...]).reshape(-1)
        for row in array_rows:
            path = row["path"]
            shape = parse_shape(row["value_shape"])
            out_png = fig_dir / f"{safe_name(path)}.png"
            title = f"{path}  shape={row['value_shape']}"
            arr = load_aligned_2d(h5, path, master_index)
            plot_2d_parameter(path, arr, out_png, title)
            rel = out_png.relative_to(args.out_dir)
            index_lines.append(f"| `{path}` | `{row['value_shape']}` | [{rel}]({rel}) |")
            print(out_png)

    n_plotted = sum(1 for line in index_lines if line.startswith("| `"))
    (args.out_dir / "train_resolved_2d_param_plot_index.md").write_text("\n".join(index_lines) + "\n")
    print(f"wrote {args.out_dir / 'train_resolved_2d_param_plot_index.md'}")
    print(f"plotted {n_plotted} 2D train-resolved parameters")


if __name__ == "__main__":
    main()
