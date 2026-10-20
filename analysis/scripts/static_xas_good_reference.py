"""Reference-normalized downstream helpers for good static XAS.

This module starts from two ``run*_good_static_xas.h5`` files:

* a transmitted/sample run, e.g. 5 mmol glycine run 58794,
* a no-sample reference run, e.g. run 58780.

The absorbance convention is intentionally different from
``static_xas_good_xas.py``.  GMD is used only for fluence-bin matching, not as
the numerator of the XAS signal:

    A(E, GMD bin) = log10(mean(reference VLS) / mean(sample VLS))

Both VLS means are read from the intercept-corrected good-H5 transmission
datasets.  No VLS-to-uJ scale is fitted or applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np

from static_xas_good_xas import (
    GMD_SOURCE_TO_DATASET,
    finite_x_order,
    format_gmd_labels,
    gmd_bin_indices,
    resolve_bin_index,
    safe_divide,
)


@dataclass(frozen=True)
class ReferenceXASConfig:
    """Configuration for reference-normalized good-H5 static XAS."""

    sample_h5: str | Path
    reference_h5: str | Path
    gmd_edges: Sequence[float] = tuple(np.arange(1.0, 11.0, 1.0))
    gmd_min_threshold: Optional[float] = None
    gmd_max_threshold: Optional[float] = 10.0
    nominal_group_decimals: int = 3
    proxy_pixel_bin_width: int = 1
    baseline_window_ev: Sequence[float] = (271.0, 280.0)
    exclude_feature_windows_ev: Sequence[Sequence[float]] = (
        (272.8, 273.2),
        (274.8, 275.2),
    )
    gmd_source: str = "hall"  # "hall" or "tunnel"


@dataclass(frozen=True)
class ReferenceShots:
    """Flattened good-shot table with corrected VLS transmissions."""

    nominal_energy: np.ndarray
    converted_energy: np.ndarray
    center_pixel: np.ndarray
    gmd: np.ndarray
    trans_nominal: np.ndarray
    trans_converted: np.ndarray
    section_index: np.ndarray
    good_shot_index: np.ndarray
    vls_pixels: np.ndarray
    attrs: dict[str, Any]

    @property
    def n_shots(self) -> int:
        return int(self.gmd.size)


@dataclass(frozen=True)
class TransmissionGrid:
    """Mean corrected-VLS transmission grouped by x-axis and GMD bin."""

    x: np.ndarray
    counts: np.ndarray
    trans_sum: np.ndarray
    trans_sumsq: np.ndarray
    gmd_sum: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    x_label: str
    metadata: dict[str, Any]

    @property
    def mean_trans(self) -> np.ndarray:
        return safe_divide(self.trans_sum, self.counts)

    @property
    def mean_gmd(self) -> np.ndarray:
        return safe_divide(self.gmd_sum, self.counts)

    @property
    def sem_trans(self) -> np.ndarray:
        counts = np.asarray(self.counts, dtype=np.float64)
        mean = self.mean_trans
        mean2 = safe_divide(self.trans_sumsq, counts)
        var = np.maximum(mean2 - mean * mean, 0.0)
        sem = np.full(mean.shape, np.nan, dtype=np.float64)
        np.divide(np.sqrt(var), np.sqrt(counts), out=sem, where=counts > 0)
        sem[~np.isfinite(sem)] = np.nan
        return sem


@dataclass(frozen=True)
class ReferenceBinnedXAS:
    """Reference-normalized absorbance on one energy axis."""

    x: np.ndarray
    absorbance: np.ndarray
    absorbance_sem: np.ndarray
    sample_counts: np.ndarray
    reference_counts: np.ndarray
    sample_mean_vls: np.ndarray
    reference_mean_vls: np.ndarray
    sample_mean_gmd: np.ndarray
    reference_mean_gmd: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    x_label: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReferenceXASResult:
    """Computed state for 58780-reference-normalized good static XAS."""

    config: ReferenceXASConfig
    sample_path: Path
    reference_path: Path
    sample_shots: ReferenceShots
    reference_shots: ReferenceShots
    sample_keep: np.ndarray
    reference_keep: np.ndarray
    gmd_edges: np.ndarray
    gmd_labels: list[str]
    nominal: ReferenceBinnedXAS
    converted: ReferenceBinnedXAS
    sample_nominal_grid: TransmissionGrid
    reference_nominal_grid: TransmissionGrid
    sample_converted_grid: TransmissionGrid
    reference_converted_grid: TransmissionGrid


def compute_reference_xas(config: ReferenceXASConfig) -> ReferenceXASResult:
    """Compute reference-normalized nominal and converted static XAS."""

    _validate_config(config)
    sample_path = Path(config.sample_h5)
    reference_path = Path(config.reference_h5)
    sample = read_reference_shots(sample_path, gmd_source=config.gmd_source)
    reference = read_reference_shots(reference_path, gmd_source=config.gmd_source)
    gmd_edges = _validate_edges(np.asarray(config.gmd_edges, dtype=np.float64))

    sample_keep = _base_keep_mask(sample, config)
    reference_keep = _base_keep_mask(reference, config)
    sample_gbin, sample_in_gmd = gmd_bin_indices(sample.gmd, gmd_edges)
    reference_gbin, reference_in_gmd = gmd_bin_indices(reference.gmd, gmd_edges)
    sample_keep &= sample_in_gmd
    reference_keep &= reference_in_gmd
    label_gmd = np.concatenate(
        [sample.gmd[sample_keep], reference.gmd[reference_keep]]
    )
    gmd_labels = format_gmd_labels(gmd_edges, label_gmd)

    nominal, sample_nominal, reference_nominal = compute_nominal_reference_xas(
        sample,
        reference,
        sample_keep,
        reference_keep,
        sample_gbin,
        reference_gbin,
        gmd_edges,
        gmd_labels,
        config,
    )
    converted, sample_converted, reference_converted = compute_converted_reference_xas(
        sample,
        reference,
        sample_keep,
        reference_keep,
        sample_gbin,
        reference_gbin,
        gmd_edges,
        gmd_labels,
        config,
    )

    return ReferenceXASResult(
        config=config,
        sample_path=sample_path,
        reference_path=reference_path,
        sample_shots=sample,
        reference_shots=reference,
        sample_keep=sample_keep,
        reference_keep=reference_keep,
        gmd_edges=gmd_edges,
        gmd_labels=gmd_labels,
        nominal=nominal,
        converted=converted,
        sample_nominal_grid=sample_nominal,
        reference_nominal_grid=reference_nominal,
        sample_converted_grid=sample_converted,
        reference_converted_grid=reference_converted,
    )


def read_reference_shots(path: str | Path, *, gmd_source: str = "hall") -> ReferenceShots:
    """Read flattened good-H5 shots and corrected transmission arrays."""

    gmd_source_norm = _normalize_gmd_source(gmd_source)
    gmd_dataset = _gmd_dataset_for_source(gmd_source_norm)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with h5py.File(p, "r") as h5:
        required = [
            "n_shots",
            "nominal_energies",
            gmd_dataset,
            "energy_vls_calibrated",
            "qc/peak_center_pixel",
            "vls_pixels",
            "trans_intercept_corrected_nominal",
            "trans_intercept_corrected_converted",
        ]
        missing = [name for name in required if name not in h5]
        if missing:
            message = f"{p} is missing required good-H5 datasets for gmd_source={gmd_source_norm!r}: {missing}"
            if gmd_source_norm == "tunnel" and gmd_dataset in missing:
                message += (
                    ". Use a tunnel-good file produced by process_processed_data_xas_static.py/"
                    "static_xas_process_good.py with PROCESS_GMD_SOURCE='tunnel'."
                )
            raise KeyError(message)

        n_shots = h5["n_shots"][...].astype(np.int64)
        nominal_energies = h5["nominal_energies"][...].astype(np.float64)
        parts: dict[str, list[np.ndarray]] = {
            "nominal_energy": [],
            "converted_energy": [],
            "center_pixel": [],
            "gmd": [],
            "trans_nominal": [],
            "trans_converted": [],
            "section_index": [],
            "good_shot_index": [],
        }
        for ie, n in enumerate(n_shots.astype(int)):
            if n <= 0:
                continue
            sl = slice(0, int(n))
            if "ml/nominal_energy" in h5:
                nominal = h5["ml/nominal_energy"][ie, sl].astype(np.float64)
            else:
                nominal = np.full(int(n), nominal_energies[ie], dtype=np.float64)
            parts["nominal_energy"].append(nominal)
            parts["converted_energy"].append(
                h5["energy_vls_calibrated"][ie, sl].astype(np.float64)
            )
            parts["center_pixel"].append(
                h5["qc/peak_center_pixel"][ie, sl].astype(np.float64)
            )
            parts["gmd"].append(h5[gmd_dataset][ie, sl].astype(np.float64))
            parts["trans_nominal"].append(
                h5["trans_intercept_corrected_nominal"][ie, sl].astype(np.float64)
            )
            parts["trans_converted"].append(
                h5["trans_intercept_corrected_converted"][ie, sl].astype(np.float64)
            )
            parts["section_index"].append(np.full(int(n), ie, dtype=np.int32))
            parts["good_shot_index"].append(np.arange(int(n), dtype=np.int64))

        flattened: dict[str, np.ndarray] = {}
        for name, chunks in parts.items():
            if chunks:
                flattened[name] = np.concatenate(chunks)
            else:
                dtype = np.int64 if name == "good_shot_index" else np.float64
                if name == "section_index":
                    dtype = np.int32
                flattened[name] = np.empty(0, dtype=dtype)

        attrs = dict(h5.attrs)
        attrs["selected_gmd_source"] = gmd_source_norm
        attrs["selected_gmd_dataset"] = f"/{gmd_dataset}"

        return ReferenceShots(
            vls_pixels=h5["vls_pixels"][...].astype(np.float64),
            attrs=attrs,
            **flattened,
        )


def compute_nominal_reference_xas(
    sample: ReferenceShots,
    reference: ReferenceShots,
    sample_keep: np.ndarray,
    reference_keep: np.ndarray,
    sample_gbin: np.ndarray,
    reference_gbin: np.ndarray,
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    config: ReferenceXASConfig,
) -> tuple[ReferenceBinnedXAS, TransmissionGrid, TransmissionGrid]:
    """Reference-normalized XAS on exact common rounded nominal energies."""

    decimals = int(config.nominal_group_decimals)
    sample_values = np.round(sample.nominal_energy, decimals)
    reference_values = np.round(reference.nominal_energy, decimals)
    sample_x = _unique_finite(sample_values[sample_keep])
    reference_x = _unique_finite(reference_values[reference_keep])
    x = np.intersect1d(sample_x, reference_x)

    sample_group = _assign_groups(sample_values, x)
    reference_group = _assign_groups(reference_values, x)
    sample_grid = _finish_grid(
        x,
        _accumulate_grid(
            x.size,
            gmd_edges.size - 1,
            sample_group,
            sample_gbin,
            sample_keep,
            sample.gmd,
            sample.trans_nominal,
        ),
        gmd_edges,
        gmd_labels,
        "nominal photon energy (eV)",
        {
            "axis": "nominal",
            "dataset": "trans_intercept_corrected_nominal",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )
    reference_grid = _finish_grid(
        x,
        _accumulate_grid(
            x.size,
            gmd_edges.size - 1,
            reference_group,
            reference_gbin,
            reference_keep,
            reference.gmd,
            reference.trans_nominal,
        ),
        gmd_edges,
        gmd_labels,
        "nominal photon energy (eV)",
        {
            "axis": "nominal",
            "dataset": "trans_intercept_corrected_nominal",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )
    binned = _make_reference_binned(
        x,
        sample_grid,
        reference_grid,
        "nominal photon energy (eV)",
        {
            "mode": "nominal",
            "nominal_group_decimals": decimals,
            "formula": "log10(mean(reference corrected VLS) / mean(sample corrected VLS))",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )
    return binned, sample_grid, reference_grid


def compute_converted_reference_xas(
    sample: ReferenceShots,
    reference: ReferenceShots,
    sample_keep: np.ndarray,
    reference_keep: np.ndarray,
    sample_gbin: np.ndarray,
    reference_gbin: np.ndarray,
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    config: ReferenceXASConfig,
) -> tuple[ReferenceBinnedXAS, TransmissionGrid, TransmissionGrid]:
    """Reference-normalized XAS on the sample converted/proxy energy grid."""

    width = int(config.proxy_pixel_bin_width)
    sample_proxy = _proxy_groups(
        _center_pixel_to_relative(sample.center_pixel, sample.vls_pixels),
        sample.vls_pixels.size,
        width,
    )
    reference_proxy = _proxy_groups(
        _center_pixel_to_relative(reference.center_pixel, reference.vls_pixels),
        reference.vls_pixels.size,
        width,
    )
    n_sample_proxy = int(np.ceil(sample.vls_pixels.size / width))
    n_reference_proxy = int(np.ceil(reference.vls_pixels.size / width))

    sample_proxy_keep = (
        sample_keep
        & (sample_proxy >= 0)
        & (sample_proxy < n_sample_proxy)
        & np.isfinite(sample.converted_energy)
    )
    reference_proxy_keep = (
        reference_keep
        & (reference_proxy >= 0)
        & (reference_proxy < n_reference_proxy)
        & np.isfinite(reference.converted_energy)
    )

    sample_proxy_energy = _mean_energy_per_proxy(
        n_sample_proxy, sample_proxy, sample_proxy_keep, sample.converted_energy
    )
    reference_proxy_energy = _mean_energy_per_proxy(
        n_reference_proxy,
        reference_proxy,
        reference_proxy_keep,
        reference.converted_energy,
    )
    sample_occupied = np.where(np.isfinite(sample_proxy_energy))[0]
    sample_occupied = sample_occupied[np.argsort(sample_proxy_energy[sample_occupied])]
    reference_occupied = np.where(np.isfinite(reference_proxy_energy))[0]
    reference_occupied = reference_occupied[
        np.argsort(reference_proxy_energy[reference_occupied])
    ]

    sample_grid_full = _finish_grid(
        sample_proxy_energy,
        _accumulate_grid(
            n_sample_proxy,
            gmd_edges.size - 1,
            sample_proxy,
            sample_gbin,
            sample_proxy_keep,
            sample.gmd,
            sample.trans_converted,
        ),
        gmd_edges,
        gmd_labels,
        "converted photon energy (eV)",
        {
            "axis": "converted",
            "dataset": "trans_intercept_corrected_converted",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )
    reference_grid_full = _finish_grid(
        reference_proxy_energy,
        _accumulate_grid(
            n_reference_proxy,
            gmd_edges.size - 1,
            reference_proxy,
            reference_gbin,
            reference_proxy_keep,
            reference.gmd,
            reference.trans_converted,
        ),
        gmd_edges,
        gmd_labels,
        "converted photon energy (eV)",
        {
            "axis": "converted",
            "dataset": "trans_intercept_corrected_converted",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )

    sample_grid = _subset_grid(sample_grid_full, sample_occupied)
    reference_grid = _subset_grid(reference_grid_full, reference_occupied)
    binned = _make_interpolated_reference_binned(
        sample_grid,
        reference_grid,
        "converted photon energy (eV)",
        {
            "mode": "converted_proxy_pixel",
            "proxy_pixel_bin_width": width,
            "formula": "log10(interpolated mean(reference corrected VLS) / mean(sample corrected VLS))",
            "reference_interpolation": "linear in converted photon energy, independently per GMD bin",
            "gmd_source": _normalize_gmd_source(config.gmd_source),
            "gmd_dataset": _gmd_dataset_for_source(config.gmd_source),
        },
    )
    return binned, sample_grid, reference_grid


def print_reference_xas_summary(result: ReferenceXASResult) -> None:
    """Print a compact summary for reference-normalized XAS."""

    print(f"sample H5           : {result.sample_path}")
    print(f"reference H5        : {result.reference_path}")
    print(
        "GMD source          : "
        f"{_normalize_gmd_source(result.config.gmd_source)} ({_gmd_dataset_for_source(result.config.gmd_source)})"
    )
    print(f"sample good shots   : {result.sample_shots.n_shots:,}")
    print(f"reference good shots: {result.reference_shots.n_shots:,}")
    print(f"sample accepted     : {int(np.sum(result.sample_keep)):,}")
    print(f"reference accepted  : {int(np.sum(result.reference_keep)):,}")
    print(f"nominal bins        : {result.nominal.x.size:,}")
    print(f"converted bins      : {result.converted.x.size:,}")
    print(f"proxy pixel width   : {result.config.proxy_pixel_bin_width}")
    print("formula             : log10(mean(reference VLS) / mean(sample VLS))")
    print("VLS datasets        : trans_intercept_corrected_nominal / converted")
    _print_count_summary("nominal", result.nominal)
    _print_count_summary("converted", result.converted)


def print_gmd_difference_summary(
    result: ReferenceXASResult, *, bin_a: int = 0, bin_b: int = -1
) -> None:
    """Print resolved GMD-bin labels for the plotted difference."""

    ia = resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = resolve_bin_index(bin_b, len(result.gmd_labels))
    print(f"Delta direction     : bin {ib} - bin {ia}")
    print(f"bin A               : {result.gmd_labels[ia]}")
    print(f"bin B               : {result.gmd_labels[ib]}")


def counts_table(result: ReferenceXASResult, *, axis: str = "nominal"):
    """Return a flat table of counts, mean GMD, and absorbance."""

    import pandas as pd

    binned = _axis_binned(result, axis)
    rows = []
    for ix, energy in enumerate(binned.x):
        for ig, label in enumerate(binned.gmd_labels):
            rows.append(
                {
                    "axis": axis,
                    "energy_eV": float(energy),
                    "gmd_bin": ig,
                    "gmd_label": label,
                    "sample_count": float(binned.sample_counts[ix, ig]),
                    "reference_count": float(binned.reference_counts[ix, ig]),
                    "sample_mean_gmd": binned.sample_mean_gmd[ix, ig],
                    "reference_mean_gmd": binned.reference_mean_gmd[ix, ig],
                    "sample_mean_vls": binned.sample_mean_vls[ix, ig],
                    "reference_mean_vls": binned.reference_mean_vls[ix, ig],
                    "absorbance": binned.absorbance[ix, ig],
                }
            )
    return pd.DataFrame(rows)


def baseline_slope_table(
    result: ReferenceXASResult,
    *,
    window_ev: Optional[Sequence[float]] = None,
    exclude_windows_ev: Optional[Sequence[Sequence[float]]] = None,
    min_points: int = 4,
):
    """Fit pre-edge baseline slopes per axis/GMD bin, excluding features."""

    import pandas as pd

    window = _normal_window(window_ev or result.config.baseline_window_ev)
    exclude = exclude_windows_ev
    if exclude is None:
        exclude = result.config.exclude_feature_windows_ev
    rows = []
    for axis, binned in (("nominal", result.nominal), ("converted", result.converted)):
        for ig, label in enumerate(binned.gmd_labels):
            x = np.asarray(binned.x, dtype=np.float64)
            y = np.asarray(binned.absorbance[:, ig], dtype=np.float64)
            mask = (
                np.isfinite(x)
                & np.isfinite(y)
                & (x >= window[0])
                & (x <= window[1])
            )
            for ex in exclude:
                lo, hi = _normal_window(ex)
                mask &= ~((x >= lo) & (x <= hi))
            xx = x[mask]
            yy = y[mask]
            slope = intercept = r2 = np.nan
            if xx.size >= int(min_points):
                slope, intercept = np.polyfit(xx, yy, 1)
                pred = slope * xx + intercept
                ss_res = float(np.sum((yy - pred) ** 2))
                ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
                r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
            rows.append(
                {
                    "axis": axis,
                    "gmd_bin": ig,
                    "gmd_label": label,
                    "n_points": int(xx.size),
                    "slope_per_eV": float(slope),
                    "intercept": float(intercept),
                    "r2": float(r2),
                    "window_eV": f"{window[0]:g}-{window[1]:g}",
                    "excluded_eV": ", ".join(
                        f"{_normal_window(ex)[0]:g}-{_normal_window(ex)[1]:g}"
                        for ex in exclude
                    ),
                }
            )
    return pd.DataFrame(rows)


def print_baseline_slope_summary(result: ReferenceXASResult) -> None:
    """Print median pre-edge slope diagnostics for nominal and converted axes."""

    table = baseline_slope_table(result)
    for axis in ("nominal", "converted"):
        vals = table.loc[table["axis"] == axis, "slope_per_eV"].to_numpy(
            dtype=np.float64
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(
                f"{axis} pre-edge slope: median {np.nanmedian(vals):.4g} "
                f"absorbance/eV over {vals.size} GMD bins"
            )
        else:
            print(f"{axis} pre-edge slope: no finite fits")


def plot_accepted_gmd_audit(
    result: ReferenceXASResult,
    *,
    bins: int = 80,
):
    """Plot accepted sample/reference GMD distributions and active bin edges."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True, constrained_layout=True)
    panels = [
        ("sample", result.sample_shots.gmd[result.sample_keep], "0.25"),
        ("reference", result.reference_shots.gmd[result.reference_keep], "tab:blue"),
    ]
    for ax, (label, gmd, color) in zip(axes, panels):
        vals = gmd[np.isfinite(gmd)]
        if vals.size:
            ax.hist(vals, bins=int(bins), color=color, alpha=0.82)
        else:
            ax.text(0.5, 0.5, "No accepted GMD values", ha="center", va="center", transform=ax.transAxes)
        for edge in result.gmd_edges:
            ax.axvline(edge, color="tab:red", lw=0.9, alpha=0.75)
        ax.set_ylabel("shots")
        ax.set_title(f"{label} accepted GMD")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("GMD / uJ")
    return fig, axes


def plot_reference_xas_matplotlib(
    result: ReferenceXASResult,
    *,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    absorbance_nominal_ylim: Optional[Sequence[float]] = None,
    absorbance_converted_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    share_xaxis: bool = True,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot nominal/converted reference-normalized XAS and GMD differences."""

    import matplotlib.pyplot as plt

    shared_xlim = _resolve_shared_xlim(nominal_xlim, converted_xlim) if share_xaxis else None
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.5), sharex=share_xaxis, constrained_layout=True)
    _plot_binned_absorbance(
        axes[0],
        result.nominal,
        title="Nominal energy: reference-normalized XAS",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=absorbance_nominal_ylim,
        show_errorbars=show_errorbars,
        show_legend=True,
    )
    _plot_delta_absorbance(
        axes[1],
        result.nominal,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Nominal energy: GMD-bin difference",
        xlim=shared_xlim if share_xaxis else nominal_xlim,
        ylim=delta_nominal_ylim,
        show_errorbars=show_errorbars,
    )
    _plot_binned_absorbance(
        axes[2],
        result.converted,
        title="Converted/proxy energy: reference-normalized XAS",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=absorbance_converted_ylim,
        show_errorbars=show_errorbars,
        show_legend=False,
    )
    _plot_delta_absorbance(
        axes[3],
        result.converted,
        bin_a=bin_a,
        bin_b=bin_b,
        title="Converted/proxy energy: GMD-bin difference",
        xlim=shared_xlim if share_xaxis else converted_xlim,
        ylim=delta_converted_ylim,
        show_errorbars=show_errorbars,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("photon energy (eV)")
    if title is None:
        title = "Reference-normalized good static XAS"
    fig.suptitle(title)
    return fig, axes


def plot_reference_xas_plotly(
    result: ReferenceXASResult,
    *,
    bin_a: int = 0,
    bin_b: int = -1,
    nominal_xlim: Optional[Sequence[float]] = None,
    converted_xlim: Optional[Sequence[float]] = None,
    absorbance_nominal_ylim: Optional[Sequence[float]] = None,
    absorbance_converted_ylim: Optional[Sequence[float]] = None,
    delta_nominal_ylim: Optional[Sequence[float]] = None,
    delta_converted_ylim: Optional[Sequence[float]] = None,
    show_errorbars: bool = True,
    title: Optional[str] = None,
):
    """Plot full absorbance and Delta absorbance in four Plotly rows."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on notebook env.
        raise ImportError("plotly is required for plot_reference_xas_plotly") from exc

    ia = resolve_bin_index(bin_a, len(result.gmd_labels))
    ib = resolve_bin_index(bin_b, len(result.gmd_labels))
    diff_label = f"{result.gmd_labels[ib]} - {result.gmd_labels[ia]}"
    if title is None:
        title = "Reference-normalized good static XAS"

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Nominal energy: reference-normalized XAS",
            "Nominal energy: GMD-bin difference",
            "Converted/proxy energy: reference-normalized XAS",
            "Converted/proxy energy: GMD-bin difference",
        ),
        shared_xaxes=False,
        vertical_spacing=0.075,
    )
    colors = [_rgba_to_hex(c) for c in _colors(len(result.gmd_labels))]

    for row, binned, showlegend in ((1, result.nominal, True), (3, result.converted, False)):
        order = finite_x_order(binned.x)
        x = binned.x[order]
        y = binned.absorbance[order]
        err = binned.absorbance_sem[order]
        for ig, label in enumerate(result.gmd_labels):
            error_y = {"type": "data", "array": err[:, ig], "visible": True} if show_errorbars else None
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[:, ig],
                    error_y=error_y,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=showlegend,
                    marker={"size": 5, "color": colors[ig]},
                    line={"width": 1.4, "color": colors[ig]},
                ),
                row=row,
                col=1,
            )

    for row, binned in ((2, result.nominal), (4, result.converted)):
        x, delta, sigma, _, _ = delta_absorbance_for_bins(binned, bin_a=bin_a, bin_b=bin_b)
        error_y = {"type": "data", "array": sigma, "visible": True} if show_errorbars else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                error_y=error_y,
                mode="lines+markers",
                name=diff_label,
                legendgroup="delta",
                showlegend=(row == 2),
                marker={"size": 6, "color": "#1f77b4"},
                line={"width": 1.8, "color": "#1f77b4"},
            ),
            row=row,
            col=1,
        )
        if x.size:
            fig.add_trace(
                go.Scatter(
                    x=[float(np.nanmin(x)), float(np.nanmax(x))],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="zero",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"width": 1.0, "color": "rgba(0,0,0,0.45)", "dash": "dot"},
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="nominal photon energy (eV)", range=nominal_xlim, row=2, col=1)
    fig.update_xaxes(title_text="converted photon energy (eV)", range=converted_xlim, row=4, col=1)
    fig.update_yaxes(title_text="absorbance", range=absorbance_nominal_ylim, row=1, col=1)
    fig.update_yaxes(title_text="Delta absorbance", range=delta_nominal_ylim, row=2, col=1)
    fig.update_yaxes(title_text="absorbance", range=absorbance_converted_ylim, row=3, col=1)
    fig.update_yaxes(title_text="Delta absorbance", range=delta_converted_ylim, row=4, col=1)
    fig.update_layout(
        title=title,
        height=980,
        width=980,
        template="plotly_white",
        legend_title_text="GMD bin",
        margin={"l": 75, "r": 25, "t": 100, "b": 60},
    )
    return fig


def delta_absorbance_for_bins(
    binned: ReferenceBinnedXAS,
    *,
    bin_a: int,
    bin_b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return x, Delta absorbance, and propagated sigma for two GMD bins."""

    ia = resolve_bin_index(bin_a, binned.absorbance.shape[1])
    ib = resolve_bin_index(bin_b, binned.absorbance.shape[1])
    if ia == ib:
        raise ValueError("bin_a and bin_b resolve to the same GMD bin")
    order = finite_x_order(binned.x)
    x = binned.x[order]
    y = binned.absorbance[order]
    err = binned.absorbance_sem[order]
    delta = y[:, ib] - y[:, ia]
    sigma = np.sqrt(err[:, ib] ** 2 + err[:, ia] ** 2)
    valid = np.isfinite(x) & np.isfinite(delta)
    return x[valid], delta[valid], sigma[valid], ia, ib


def _make_reference_binned(
    x: np.ndarray,
    sample_grid: TransmissionGrid,
    reference_grid: TransmissionGrid,
    x_label: str,
    metadata: dict[str, Any],
) -> ReferenceBinnedXAS:
    sample_mean = sample_grid.mean_trans
    reference_mean = reference_grid.mean_trans
    absorbance = _log10_ratio(reference_mean, sample_mean)
    absorbance_sem = _absorbance_sem(
        reference_mean,
        sample_mean,
        reference_grid.sem_trans,
        sample_grid.sem_trans,
    )
    return ReferenceBinnedXAS(
        x=np.asarray(x, dtype=np.float64),
        absorbance=absorbance,
        absorbance_sem=absorbance_sem,
        sample_counts=sample_grid.counts.astype(np.float64),
        reference_counts=reference_grid.counts.astype(np.float64),
        sample_mean_vls=sample_mean,
        reference_mean_vls=reference_mean,
        sample_mean_gmd=sample_grid.mean_gmd,
        reference_mean_gmd=reference_grid.mean_gmd,
        gmd_edges=sample_grid.gmd_edges,
        gmd_labels=list(sample_grid.gmd_labels),
        x_label=x_label,
        metadata=dict(metadata),
    )


def _make_interpolated_reference_binned(
    sample_grid: TransmissionGrid,
    reference_grid: TransmissionGrid,
    x_label: str,
    metadata: dict[str, Any],
) -> ReferenceBinnedXAS:
    sample_mean = sample_grid.mean_trans
    sample_sem = sample_grid.sem_trans
    reference_mean_raw = reference_grid.mean_trans
    reference_sem_raw = reference_grid.sem_trans
    reference_gmd_raw = reference_grid.mean_gmd

    reference_mean = np.full(sample_mean.shape, np.nan, dtype=np.float64)
    reference_sem = np.full(sample_mean.shape, np.nan, dtype=np.float64)
    reference_counts = np.full(sample_mean.shape, np.nan, dtype=np.float64)
    reference_gmd = np.full(sample_mean.shape, np.nan, dtype=np.float64)
    for ig in range(sample_mean.shape[1]):
        reference_mean[:, ig] = _interp_with_nan(
            sample_grid.x, reference_grid.x, reference_mean_raw[:, ig]
        )
        reference_sem[:, ig] = _interp_with_nan(
            sample_grid.x, reference_grid.x, reference_sem_raw[:, ig]
        )
        reference_counts[:, ig] = _interp_with_nan(
            sample_grid.x, reference_grid.x, reference_grid.counts[:, ig].astype(np.float64)
        )
        reference_gmd[:, ig] = _interp_with_nan(
            sample_grid.x, reference_grid.x, reference_gmd_raw[:, ig]
        )

    absorbance = _log10_ratio(reference_mean, sample_mean)
    absorbance_sem = _absorbance_sem(reference_mean, sample_mean, reference_sem, sample_sem)
    return ReferenceBinnedXAS(
        x=sample_grid.x.astype(np.float64, copy=True),
        absorbance=absorbance,
        absorbance_sem=absorbance_sem,
        sample_counts=sample_grid.counts.astype(np.float64),
        reference_counts=reference_counts,
        sample_mean_vls=sample_mean,
        reference_mean_vls=reference_mean,
        sample_mean_gmd=sample_grid.mean_gmd,
        reference_mean_gmd=reference_gmd,
        gmd_edges=sample_grid.gmd_edges,
        gmd_labels=list(sample_grid.gmd_labels),
        x_label=x_label,
        metadata=dict(metadata),
    )


def _accumulate_grid(
    n_x: int,
    n_gmd: int,
    xbin: np.ndarray,
    gbin: np.ndarray,
    keep: np.ndarray,
    gmd: np.ndarray,
    trans: np.ndarray,
) -> dict[str, np.ndarray]:
    grid = {
        "counts": np.zeros((int(n_x), int(n_gmd)), dtype=np.int64),
        "trans_sum": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "trans_sumsq": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
        "gmd_sum": np.zeros((int(n_x), int(n_gmd)), dtype=np.float64),
    }
    valid = (
        keep
        & (xbin >= 0)
        & (xbin < int(n_x))
        & (gbin >= 0)
        & (gbin < int(n_gmd))
        & np.isfinite(gmd)
        & np.isfinite(trans)
    )
    if np.any(valid):
        idx = (xbin[valid].astype(np.int64), gbin[valid].astype(np.int64))
        t = trans[valid].astype(np.float64, copy=False)
        g = gmd[valid].astype(np.float64, copy=False)
        np.add.at(grid["counts"], idx, 1)
        np.add.at(grid["trans_sum"], idx, t)
        np.add.at(grid["trans_sumsq"], idx, t * t)
        np.add.at(grid["gmd_sum"], idx, g)
    return grid


def _finish_grid(
    x: np.ndarray,
    grid: dict[str, np.ndarray],
    gmd_edges: np.ndarray,
    gmd_labels: list[str],
    x_label: str,
    metadata: dict[str, Any],
) -> TransmissionGrid:
    return TransmissionGrid(
        x=np.asarray(x, dtype=np.float64),
        gmd_edges=np.asarray(gmd_edges, dtype=np.float64),
        gmd_labels=list(gmd_labels),
        x_label=x_label,
        metadata=dict(metadata),
        **grid,
    )


def _subset_grid(grid: TransmissionGrid, rows: np.ndarray) -> TransmissionGrid:
    rows = np.asarray(rows, dtype=np.int64)
    return TransmissionGrid(
        x=grid.x[rows],
        counts=grid.counts[rows],
        trans_sum=grid.trans_sum[rows],
        trans_sumsq=grid.trans_sumsq[rows],
        gmd_sum=grid.gmd_sum[rows],
        gmd_edges=grid.gmd_edges,
        gmd_labels=list(grid.gmd_labels),
        x_label=grid.x_label,
        metadata=dict(grid.metadata),
    )


def _mean_energy_per_proxy(
    n_proxy: int,
    proxy: np.ndarray,
    keep: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    energy_sum = np.zeros(int(n_proxy), dtype=np.float64)
    energy_count = np.zeros(int(n_proxy), dtype=np.float64)
    valid = keep & (proxy >= 0) & (proxy < int(n_proxy)) & np.isfinite(energy)
    if np.any(valid):
        np.add.at(energy_sum, proxy[valid].astype(np.int64), energy[valid])
        np.add.at(energy_count, proxy[valid].astype(np.int64), 1.0)
    out = np.full(int(n_proxy), np.nan, dtype=np.float64)
    np.divide(energy_sum, energy_count, out=out, where=energy_count > 0)
    return out


def _log10_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    ratio = safe_divide(num, den)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.log10(ratio[valid])
    return out


def _absorbance_sem(
    reference_mean: np.ndarray,
    sample_mean: np.ndarray,
    reference_sem: np.ndarray,
    sample_sem: np.ndarray,
) -> np.ndarray:
    out = np.full(np.shape(reference_mean), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(reference_mean)
        & np.isfinite(sample_mean)
        & np.isfinite(reference_sem)
        & np.isfinite(sample_sem)
        & (reference_mean > 0.0)
        & (sample_mean > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.sqrt(
            (reference_sem[valid] / reference_mean[valid]) ** 2
            + (sample_sem[valid] / sample_mean[valid]) ** 2
        ) / np.log(10.0)
    out[~np.isfinite(out)] = np.nan
    return out


def _interp_with_nan(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_new = np.asarray(x_new, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    out = np.full(x_new.shape, np.nan, dtype=np.float64)
    if not np.any(finite):
        return out
    xx = x[finite]
    yy = y[finite]
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    unique_x, inv = np.unique(xx, return_inverse=True)
    if unique_x.size != xx.size:
        y_sum = np.zeros(unique_x.size, dtype=np.float64)
        y_count = np.zeros(unique_x.size, dtype=np.float64)
        np.add.at(y_sum, inv, yy)
        np.add.at(y_count, inv, 1.0)
        yy = y_sum / y_count
        xx = unique_x
    out[:] = np.interp(x_new, xx, yy, left=np.nan, right=np.nan)
    return out


def _base_keep_mask(shots: ReferenceShots, config: ReferenceXASConfig) -> np.ndarray:
    keep = (
        np.isfinite(shots.nominal_energy)
        & np.isfinite(shots.converted_energy)
        & np.isfinite(shots.center_pixel)
        & np.isfinite(shots.gmd)
        & np.isfinite(shots.trans_nominal)
        & np.isfinite(shots.trans_converted)
    )
    if config.gmd_min_threshold is not None:
        keep &= shots.gmd >= float(config.gmd_min_threshold)
    if config.gmd_max_threshold is not None:
        keep &= shots.gmd <= float(config.gmd_max_threshold)
    return keep


def _unique_finite(values: np.ndarray) -> np.ndarray:
    out = np.unique(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    out.sort()
    return out


def _assign_groups(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    group = np.full(values.shape, -1, dtype=np.int64)
    if centers.size == 0:
        return group
    idx = np.searchsorted(centers, values)
    valid = (idx >= 0) & (idx < centers.size)
    matched = valid & (centers[np.clip(idx, 0, centers.size - 1)] == values)
    group[matched] = idx[matched]
    return group


def _center_pixel_to_relative(center_pixel: np.ndarray, vls_pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(vls_pixels, dtype=np.float64)
    rel = np.arange(pixels.size, dtype=np.float64)
    if pixels.size == 0:
        return np.full(np.shape(center_pixel), np.nan, dtype=np.float64)
    if pixels[0] <= pixels[-1]:
        return np.interp(center_pixel, pixels, rel, left=np.nan, right=np.nan)
    return np.interp(center_pixel, pixels[::-1], rel[::-1], left=np.nan, right=np.nan)


def _proxy_groups(center_rel: np.ndarray, n_pixels: int, width: int) -> np.ndarray:
    group = np.full(center_rel.shape, -1, dtype=np.int64)
    valid = np.isfinite(center_rel)
    if np.any(valid):
        rel = np.clip(np.round(center_rel[valid]).astype(np.int64), 0, int(n_pixels) - 1)
        group[valid] = rel // int(width)
    return group


def _plot_binned_absorbance(
    ax,
    binned: ReferenceBinnedXAS,
    *,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
    show_legend: bool,
) -> None:
    order = finite_x_order(binned.x)
    x = binned.x[order]
    y = binned.absorbance[order]
    err = binned.absorbance_sem[order]
    colors = _colors(y.shape[1])
    for ig, color in enumerate(colors):
        valid = np.isfinite(x) & np.isfinite(y[:, ig])
        if not np.any(valid):
            continue
        if show_errorbars:
            ax.errorbar(x[valid], y[valid, ig], yerr=err[valid, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
        else:
            ax.plot(x[valid], y[valid, ig], marker="o", ms=3, lw=1.1, color=color, label=binned.gmd_labels[ig])
    ax.set_title(title)
    ax.set_ylabel("A = log10(mean ref VLS / mean sample VLS)")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(title="GMD bin", frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_delta_absorbance(
    ax,
    binned: ReferenceBinnedXAS,
    *,
    bin_a: int,
    bin_b: int,
    title: str,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    show_errorbars: bool,
) -> None:
    x, delta, sigma, ia, ib = delta_absorbance_for_bins(binned, bin_a=bin_a, bin_b=bin_b)
    label = f"{binned.gmd_labels[ib]} - {binned.gmd_labels[ia]}"
    if show_errorbars:
        ax.errorbar(x, delta, yerr=sigma, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    else:
        ax.plot(x, delta, marker="o", ms=3.5, lw=1.2, color="tab:blue", label=label)
    ax.axhline(0.0, color="0.25", lw=0.9, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Delta A = A[B] - A[A]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _axis_binned(result: ReferenceXASResult, axis: str) -> ReferenceBinnedXAS:
    if axis == "nominal":
        return result.nominal
    if axis == "converted":
        return result.converted
    raise ValueError("axis must be 'nominal' or 'converted'")


def _print_count_summary(name: str, binned: ReferenceBinnedXAS) -> None:
    valid = np.isfinite(binned.absorbance)
    sample_counts = binned.sample_counts[valid]
    reference_counts = binned.reference_counts[valid]
    if sample_counts.size:
        print(
            f"{name} sample/bin   : {np.nanmin(sample_counts):.0f} / "
            f"{np.nanmedian(sample_counts):.0f} / {np.nanmax(sample_counts):.0f} "
            "min/median/max"
        )
        print(
            f"{name} ref/bin      : {np.nanmin(reference_counts):.0f} / "
            f"{np.nanmedian(reference_counts):.0f} / {np.nanmax(reference_counts):.0f} "
            "min/median/max"
        )
    else:
        print(f"{name} counts/bin   : no finite absorbance bins")


def _normal_window(window: Sequence[float]) -> tuple[float, float]:
    if len(window) != 2:
        raise ValueError("energy window must contain exactly two values")
    lo, hi = float(window[0]), float(window[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("energy window values must be finite")
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        raise ValueError("energy window must have nonzero width")
    return lo, hi


def _resolve_shared_xlim(
    a: Optional[Sequence[float]], b: Optional[Sequence[float]]
) -> Optional[tuple[float, float]]:
    if a is None:
        return None if b is None else _normal_window(b)
    if b is None:
        return _normal_window(a)
    aa = _normal_window(a)
    bb = _normal_window(b)
    return min(aa[0], bb[0]), max(aa[1], bb[1])


def _colors(n: int):
    import matplotlib.pyplot as plt

    return plt.cm.inferno(np.linspace(0.08, 0.92, int(n)))


def _rgba_to_hex(rgba) -> str:
    vals = np.asarray(rgba[:3], dtype=np.float64)
    vals = np.clip(np.round(vals * 255), 0, 255).astype(int)
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _validate_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("GMD edges must be a 1D array with at least two values")
    if not np.all(np.isfinite(edges)):
        raise ValueError("GMD edges must be finite")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("GMD edges must be strictly increasing")
    return edges


def _validate_config(config: ReferenceXASConfig) -> None:
    _normalize_gmd_source(config.gmd_source)
    if int(config.proxy_pixel_bin_width) <= 0:
        raise ValueError("proxy_pixel_bin_width must be positive")
    _validate_edges(np.asarray(config.gmd_edges, dtype=np.float64))
    _normal_window(config.baseline_window_ev)
    for window in config.exclude_feature_windows_ev:
        _normal_window(window)


def _normalize_gmd_source(gmd_source: str) -> str:
    source = str(gmd_source).strip().lower()
    if source not in GMD_SOURCE_TO_DATASET:
        allowed = ", ".join(sorted(GMD_SOURCE_TO_DATASET))
        raise ValueError(f"gmd_source must be one of {{{allowed}}}; got {gmd_source!r}")
    return source


def _gmd_dataset_for_source(gmd_source: str) -> str:
    return GMD_SOURCE_TO_DATASET[_normalize_gmd_source(gmd_source)]
