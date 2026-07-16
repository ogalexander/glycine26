from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


GOTTHARD_INDEX = "/FL2/Support Infrastructure/Gotthard/images/index"
GOTTHARD_VALUE = "/FL2/Support Infrastructure/Gotthard/images/value"
GMD_HALL = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall"
GMD_TUNNEL = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel"
DEFAULT_VLS_BUNCH_RANGE = (10, 40)
DEFAULT_GMD_BUNCH_START = 0


def dataset_shape(ds) -> str:
    return "(" + ", ".join(str(x) for x in ds.shape) + ("," if len(ds.shape) == 1 else "") + ")"


def iter_index_value_groups(h5: h5py.File):
    groups = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "index" in obj and "value" in obj:
            if isinstance(obj["index"], h5py.Dataset) and isinstance(obj["value"], h5py.Dataset):
                groups.append("/" + name if not name.startswith("/") else name)

    h5.visititems(visit)
    return sorted(groups)


def sample_numeric(ds: h5py.Dataset, max_values: int = 200_000) -> np.ndarray:
    if ds.dtype.kind not in "biufc":
        return np.array([], dtype=np.float64)
    if ds.shape == ():
        arr = ds[()]
        return np.asarray([arr], dtype=np.float64)
    total = int(np.prod(ds.shape))
    if total == 0:
        return np.array([], dtype=np.float64)
    if total <= max_values:
        arr = ds[...]
    else:
        trailing = int(np.prod(ds.shape[1:])) if len(ds.shape) > 1 else 1
        n0 = ds.shape[0]
        take0 = max(1, min(n0, max_values // max(trailing, 1)))
        idx = np.unique(np.linspace(0, n0 - 1, take0, dtype=int))
        arr = ds[idx, ...]
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    flat = arr.reshape(-1)
    if flat.size > max_values:
        step = int(math.ceil(flat.size / max_values))
        flat = flat[::step]
    return flat.astype(np.float64, copy=False)


def robust_stats(ds: h5py.Dataset) -> dict[str, object]:
    vals = sample_numeric(ds)
    out: dict[str, object] = {
        "sample_n": int(vals.size),
        "finite_n": 0,
        "finite_fraction": np.nan,
        "min": np.nan,
        "p01": np.nan,
        "median": np.nan,
        "p99": np.nan,
        "max": np.nan,
        "unique_n_sample": np.nan,
    }
    if vals.size == 0:
        return out
    finite = vals[np.isfinite(vals)]
    out["finite_n"] = int(finite.size)
    out["finite_fraction"] = float(finite.size / vals.size)
    if finite.size == 0:
        return out
    out["min"] = float(np.min(finite))
    out["p01"] = float(np.percentile(finite, 1))
    out["median"] = float(np.percentile(finite, 50))
    out["p99"] = float(np.percentile(finite, 99))
    out["max"] = float(np.max(finite))
    if finite.size <= 200_000:
        out["unique_n_sample"] = int(np.unique(finite).size)
    return out


def read_1d_index(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds[...])
    return arr.reshape(-1)


def match_counts(index: np.ndarray, master: np.ndarray) -> tuple[int, int]:
    if index.size == 0 or master.size == 0:
        return 0, 0
    common = np.intersect1d(index, master, assume_unique=False)
    exact_order = int(index.size == master.size and np.array_equal(index, master))
    return int(common.size), exact_order


def classify_time_resolution(path: str, index_len: int, match_master: int, exact_order: int, value_shape: tuple[int, ...], master_n: int) -> str:
    if path == "/FL2/Support Infrastructure/Gotthard/images":
        return "gotthard_master_image"
    if path.startswith(GMD_HALL) or path.startswith(GMD_TUNNEL):
        return "pulse_resolved_bunch_multi_channel"
    if not value_shape or value_shape[0] != index_len:
        return "index_value_shape_mismatch"
    master_frac = match_master / master_n if master_n else 0.0
    if master_frac >= 0.98:
        if len(value_shape) == 1:
            return "train_resolved_scalar"
        if len(value_shape) == 2 and value_shape[1] <= 200:
            return "train_resolved_matrix_small_axis"
        if len(value_shape) >= 2:
            return "train_resolved_array"
    if match_master > 0:
        return "sparse_train_indexed_monitor"
    if exact_order:
        return "train_resolved_scalar"
    return "unmatched_or_other"


def mapped_gmd_range(vls_b0: int, vls_b1: int, gmd_b0: int, gmd_bunch_count: int) -> tuple[int, int]:
    n_vls = max(0, vls_b1 - vls_b0)
    return gmd_b0, min(gmd_b0 + n_vls, gmd_bunch_count)


def channel_stats(
    h5: h5py.File,
    base: str,
    master_index: np.ndarray,
    vls_b0: int,
    vls_b1: int,
    gmd_b0: int,
) -> list[dict[str, object]]:
    idx_path = base + "/index"
    val_path = base + "/value"
    if idx_path not in h5 or val_path not in h5:
        return []
    idx = read_1d_index(h5[idx_path])
    val = h5[val_path]
    if len(val.shape) != 3:
        return []
    gmd_b1 = mapped_gmd_range(vls_b0, vls_b1, gmd_b0, val.shape[2])[1]
    if gmd_b0 >= gmd_b1:
        return []
    positions = {int(t): i for i, t in enumerate(idx)}
    src = np.array([positions.get(int(t), -1) for t in master_index], dtype=int)
    valid_train = src >= 0
    rows: list[dict[str, object]] = []
    label = "hall" if "energy hall" in base else "tunnel"
    for ch in range(val.shape[1]):
        arr = val[src[valid_train], ch, gmd_b0:gmd_b1]
        flat = np.asarray(arr).reshape(-1).astype(np.float64)
        finite = flat[np.isfinite(flat)]
        row: dict[str, object] = {
            "gmd": label,
            "channel": ch,
            "raw_shape": str(tuple(val.shape)),
            "aligned_trains": int(valid_train.sum()),
            "vls_bunch_start": vls_b0,
            "vls_bunch_stop": vls_b1,
            "gmd_bunch_start": gmd_b0,
            "gmd_bunch_stop": gmd_b1,
            "bunch_count": gmd_b1 - gmd_b0,
            "sample_n": int(flat.size),
            "finite_n": int(finite.size),
            "finite_fraction": float(finite.size / flat.size) if flat.size else np.nan,
            "min": np.nan,
            "p01": np.nan,
            "median": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
        if finite.size:
            row.update(
                {
                    "min": float(np.min(finite)),
                    "p01": float(np.percentile(finite, 1)),
                    "median": float(np.percentile(finite, 50)),
                    "p99": float(np.percentile(finite, 99)),
                    "max": float(np.max(finite)),
                }
            )
        rows.append(row)
    return rows


def gmd_bunch_stats(h5: h5py.File, base: str, master_index: np.ndarray) -> list[dict[str, object]]:
    idx_path = base + "/index"
    val_path = base + "/value"
    if idx_path not in h5 or val_path not in h5:
        return []
    idx = read_1d_index(h5[idx_path])
    val = h5[val_path]
    if len(val.shape) != 3:
        return []
    positions = {int(t): i for i, t in enumerate(idx)}
    src = np.array([positions.get(int(t), -1) for t in master_index], dtype=int)
    valid_train = src >= 0
    label = "hall" if "energy hall" in base else "tunnel"
    rows: list[dict[str, object]] = []
    for ch in range(val.shape[1]):
        arr = np.asarray(val[src[valid_train], ch, :], dtype=np.float64)
        for bunch in range(val.shape[2]):
            flat = arr[:, bunch]
            finite = flat[np.isfinite(flat)]
            row: dict[str, object] = {
                "gmd": label,
                "channel": ch,
                "bunch": bunch,
                "aligned_trains": int(valid_train.sum()),
                "sample_n": int(flat.size),
                "finite_n": int(finite.size),
                "finite_fraction": float(finite.size / flat.size) if flat.size else np.nan,
                "min": np.nan,
                "p01": np.nan,
                "median": np.nan,
                "p99": np.nan,
                "max": np.nan,
            }
            if finite.size:
                row.update(
                    {
                        "min": float(np.min(finite)),
                        "p01": float(np.percentile(finite, 1)),
                        "median": float(np.percentile(finite, 50)),
                        "p99": float(np.percentile(finite, 99)),
                        "max": float(np.max(finite)),
                    }
                )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir: Path, inventory: list[dict[str, object]], gmd_rows: list[dict[str, object]], gmd_bunch_rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(row["time_resolution"] for row in inventory)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax.bar(range(len(labels)), vals, color="#577590")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("index/value parameter count")
    ax.set_title("Raw parameter time-resolution classes")
    fig.tight_layout()
    fig.savefig(fig_dir / "time_resolution_counts.png", dpi=160)
    plt.close(fig)

    if gmd_rows:
        for label in sorted(set(row["gmd"] for row in gmd_rows)):
            rows = [row for row in gmd_rows if row["gmd"] == label]
            ch = [int(row["channel"]) for row in rows]
            med = [float(row["median"]) for row in rows]
            p01 = [float(row["p01"]) for row in rows]
            p99 = [float(row["p99"]) for row in rows]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ch, med, marker="o", label="median")
            ax.fill_between(ch, p01, p99, alpha=0.2, label="p1-p99")
            ax.set_xlabel("GMD channel")
            ax.set_ylabel("pulse energy value")
            ax.set_title(f"GMD {label}: mapped raw bunch range")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"gmd_{label}_channel_summary.png", dpi=160)
            plt.close(fig)

    if gmd_bunch_rows:
        for label in sorted(set(row["gmd"] for row in gmd_bunch_rows)):
            rows = [
                row for row in gmd_bunch_rows
                if row["gmd"] == label and int(row["channel"]) == 0
            ]
            bunch = [int(row["bunch"]) for row in rows]
            med = [float(row["median"]) for row in rows]
            p01 = [float(row["p01"]) for row in rows]
            p99 = [float(row["p99"]) for row in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(bunch, med, marker="o", label="median")
            ax.fill_between(bunch, p01, p99, alpha=0.2, label="p1-p99")
            ax.set_xlabel("GMD bunch index")
            ax.set_ylabel("channel 0 pulse energy value")
            ax.set_title(f"GMD {label}: channel 0 by bunch")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"gmd_{label}_channel0_by_bunch.png", dpi=160)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_h5", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vls-bunch-start", type=int, default=DEFAULT_VLS_BUNCH_RANGE[0])
    parser.add_argument("--vls-bunch-stop", type=int, default=DEFAULT_VLS_BUNCH_RANGE[1])
    parser.add_argument("--gmd-bunch-start", type=int, default=DEFAULT_GMD_BUNCH_START)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.raw_h5, "r") as h5:
        master_index = read_1d_index(h5[GOTTHARD_INDEX])
        gotthard_value = h5[GOTTHARD_VALUE]
        master_n = int(master_index.size)
        groups = iter_index_value_groups(h5)
        rows: list[dict[str, object]] = []
        for group in groups:
            idx_ds = h5[group + "/index"]
            val_ds = h5[group + "/value"]
            idx = read_1d_index(idx_ds)
            match_master, exact_order = match_counts(idx, master_index)
            stats = robust_stats(val_ds)
            row: dict[str, object] = {
                "path": group,
                "index_path": group + "/index",
                "value_path": group + "/value",
                "index_shape": dataset_shape(idx_ds),
                "value_shape": dataset_shape(val_ds),
                "index_dtype": str(idx_ds.dtype),
                "value_dtype": str(val_ds.dtype),
                "index_len": int(idx.size),
                "gotthard_master_len": master_n,
                "sample_ratio_vs_gotthard": float(idx.size / master_n) if master_n else np.nan,
                "matched_gotthard_trains": match_master,
                "match_fraction_of_gotthard": float(match_master / master_n) if master_n else np.nan,
                "exact_gotthard_order": exact_order,
                "time_resolution": classify_time_resolution(group, int(idx.size), match_master, exact_order, tuple(val_ds.shape), master_n),
            }
            row.update(stats)
            rows.append(row)

        gmd_rows = []
        gmd_rows.extend(channel_stats(h5, GMD_HALL, master_index, args.vls_bunch_start, args.vls_bunch_stop, args.gmd_bunch_start))
        gmd_rows.extend(channel_stats(h5, GMD_TUNNEL, master_index, args.vls_bunch_start, args.vls_bunch_stop, args.gmd_bunch_start))
        gmd_bunch_rows = []
        gmd_bunch_rows.extend(gmd_bunch_stats(h5, GMD_HALL, master_index))
        gmd_bunch_rows.extend(gmd_bunch_stats(h5, GMD_TUNNEL, master_index))

        summary_lines = [
            "# Raw H5 Feature EDA Summary",
            "",
            f"Raw file: `{args.raw_h5}`",
            "",
            "## Master Axis",
            "",
            f"- Gotthard index path: `{GOTTHARD_INDEX}`",
            f"- Gotthard value path: `{GOTTHARD_VALUE}`",
            f"- Gotthard index shape: `{tuple(master_index.shape)}`",
            f"- Gotthard value shape: `{tuple(gotthard_value.shape)}`",
            f"- Gotthard train IDs: {int(master_index[0])} .. {int(master_index[-1])}",
            "",
            "## GMD",
            "",
        ]
        for base in (GMD_HALL, GMD_TUNNEL):
            if base + "/value" in h5:
                ds = h5[base + "/value"]
                if len(ds.shape) == 3:
                    gmd_b0, gmd_b1 = mapped_gmd_range(args.vls_bunch_start, args.vls_bunch_stop, args.gmd_bunch_start, ds.shape[2])
                    summary_lines.append(
                        f"- `{base}/value`: shape `{tuple(ds.shape)}`; VLS selected bunch range "
                        f"`{args.vls_bunch_start}:{args.vls_bunch_stop}` maps to GMD raw bunch range `{gmd_b0}:{gmd_b1}`."
                    )
                else:
                    summary_lines.append(f"- `{base}/value`: shape `{tuple(ds.shape)}`.")
        summary_lines.extend(
            [
                "",
                "## Time-Resolution Classes",
                "",
            ]
        )
        for klass, count in Counter(row["time_resolution"] for row in rows).most_common():
            summary_lines.append(f"- {klass}: {count}")
        summary_lines.extend(
            [
                "",
                "## Outputs",
                "",
                "- `raw_param_inventory.csv`: all index/value parameter shapes, alignment coverage, and robust sampled ranges.",
                "- `train_resolved_params.csv`: full-coverage Gotthard/GMD/train-resolved rows.",
                "- `sparse_train_indexed_params.csv`: partial train-indexed monitor rows.",
                "- `gmd_channel_stats.csv`: hall/tunnel GMD channel ranges after exact train alignment to Gotthard.",
                "- `gmd_bunch_stats.csv`: hall/tunnel GMD channel ranges by bunch index after exact train alignment to Gotthard.",
                "- `figures/`: quicklook plots.",
            ]
        )
        (args.out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

    write_csv(args.out_dir / "raw_param_inventory.csv", rows)
    write_csv(
        args.out_dir / "train_resolved_params.csv",
        [
            row for row in rows
            if row["time_resolution"].startswith("train_resolved")
            or row["time_resolution"] == "pulse_resolved_bunch_multi_channel"
            or row["time_resolution"] == "gotthard_master_image"
        ],
    )
    write_csv(
        args.out_dir / "sparse_train_indexed_params.csv",
        [row for row in rows if row["time_resolution"] == "sparse_train_indexed_monitor"],
    )
    write_csv(args.out_dir / "gmd_channel_stats.csv", gmd_rows)
    write_csv(args.out_dir / "gmd_bunch_stats.csv", gmd_bunch_rows)
    make_plots(args.out_dir, rows, gmd_rows, gmd_bunch_rows)

    print(f"wrote {args.out_dir}")
    print(f"parameters: {len(rows)}")
    print("classes:", dict(Counter(row["time_resolution"] for row in rows)))
    if gmd_rows:
        print("gmd rows:", len(gmd_rows))
    if gmd_bunch_rows:
        print("gmd bunch rows:", len(gmd_bunch_rows))


if __name__ == "__main__":
    main()
