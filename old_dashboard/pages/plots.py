"""Page 3 — Plot: load once, cache in session, generate multiple plot panels."""
from __future__ import annotations

import io
import sys
import uuid
import warnings
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))

import config  # noqa: E402
from compute_aggregates import load_aggregates  # noqa: E402
from data_loading import load_data  # noqa: E402
from binning import arb_bool_ar, bin_and_average  # noqa: E402
from plotting import (  # noqa: E402
    plot_delay_dependence,
    plot_diagnostics,
    plot_energy_binned_maps,
    plot_linearity,
)


def _init_state() -> None:
    st.session_state.setdefault("plot_loaded_data", None)
    st.session_state.setdefault("plot_loaded_meta", None)
    st.session_state.setdefault("plot_panels", [])
    st.session_state.setdefault("plot_load_status", None)


@st.cache_data(show_spinner=False)
def _detect_h5_type(filepath: Path) -> dict:
    with h5py.File(filepath, "r") as f:
        keys = set(f.keys())

    # Combined H5 files.
    if {"between_tdc_files", "gmd", "tID"}.issubset(keys):
        if {"vls", "liq_tofs_e"}.issubset(keys):
            return {"kind": "combined", "config": 2, "mode": "spectral"}
        if {"tofs_e", "tofs_i"}.issubset(keys):
            return {"kind": "combined", "config": 1, "mode": "spectral"}

    # Aggregates H5 files.
    if {"D", "DtD", "G", "GtG", "DtG", "n_per_bin", "gmd_edges", "tof_edges"}.issubset(keys):
        mode = "time_resolved" if "z_edges" in keys else "spectral"
        if {"C", "CtC", "DtC", "CtG", "ion_tof_edges"}.issubset(keys):
            return {"kind": "aggregates", "config": 1, "mode": mode}
        if {"A", "AtA", "AtD", "AtG", "vls_pixels"}.issubset(keys):
            return {"kind": "aggregates", "config": 2, "mode": mode}

    raise RuntimeError(f"Could not determine config from datasets in {filepath.name}")


@st.cache_data(show_spinner=False)
def _inspect_file(
    filepath: Path,
    config_no: int,
    trim_start: int,
    trim_end: int,
    downsample_n: int,
) -> dict:
    with h5py.File(filepath, "r") as f:
        between_tdc = f["between_tdc_files"][:].astype(bool)
        good_idx = np.where(~between_tdc)[0]
        n_total = int(between_tdc.shape[0])
        n_good = int(good_idx.shape[0])
        if trim_end > 0:
            final_idx = good_idx[trim_start:-trim_end:downsample_n]
        else:
            final_idx = good_idx[trim_start::downsample_n]

        meta = {
            "filepath": str(filepath),
            "config": int(config_no),
            "n_total": n_total,
            "n_good": n_good,
            "n_loaded": int(final_idx.shape[0]),
            "n_bunches": int(f["gmd"].shape[1]),
            "trim_start": int(trim_start),
            "trim_end": int(trim_end),
            "downsample": int(downsample_n),
        }
    return meta


@st.cache_data(show_spinner=False)
def _inspect_aggregates_file(filepath: Path, config_no: int, mode: str) -> dict:
    with h5py.File(filepath, "r") as f:
        n_per_bin = f["n_per_bin"][:]
        meta = {
            "filepath": str(filepath),
            "kind": "aggregates",
            "config": int(config_no),
            "mode": str(mode),
            "n_bins_total": int(n_per_bin.size),
            "n_bins_nonempty": int(np.count_nonzero(np.isfinite(n_per_bin) & (n_per_bin > 0))),
            "n_shots_total": int(np.nansum(n_per_bin)),
        }
        if "z_edges" in f:
            meta["n_z_bins"] = int(f["z_edges"].shape[0] - 1)
        if "vls_pixels" in f:
            meta["n_pixels"] = int(f["vls_pixels"].shape[0])
        if "ion_tof_edges" in f:
            meta["n_tof_i"] = int(f["ion_tof_edges"].shape[0] - 1)
        meta["n_tof"] = int(f["tof_edges"].shape[0] - 1)
    return meta


def _loaded_matches(filepath: Path, trim_start: int, trim_end: int, downsample_n: int) -> bool:
    meta = st.session_state.get("plot_loaded_meta")
    if not meta:
        return False
    return (
        meta["filepath"] == str(filepath)
        and meta.get("trim_start") == int(trim_start)
        and meta.get("trim_end") == int(trim_end)
        and meta.get("downsample") == int(downsample_n)
    )


def _safe_minmax(arr: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return fallback
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _add_panel(kind: str, params: dict) -> None:
    loaded_data = st.session_state.plot_loaded_data
    loaded_meta = st.session_state.get("plot_loaded_meta") or {}
    image_bytes = _build_panel_image(kind, params, loaded_data, loaded_meta)
    loaded_meta = st.session_state.get("plot_loaded_meta") or {}
    st.session_state.plot_panels.append(
        {
            "id": uuid.uuid4().hex[:8],
            "kind": kind,
            "params": params,
            "image": image_bytes,
            "source_filepath": loaded_meta.get("filepath", ""),
            "source_config": loaded_meta.get("config"),
            "source_kind": loaded_meta.get("kind", "combined"),
            "source_mode": loaded_meta.get("mode", "spectral"),
        }
    )


def _figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _safe_nanmean(arr: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _plot_agg_means(agg) -> tuple[plt.Figure, np.ndarray]:
    if agg.mode == "spectral":
        if agg.config == 2:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axes[0].pcolormesh(agg.gmd_centres, agg.vls_pixels, agg.A.T, shading="auto", cmap="viridis")
            fig.colorbar(im0, ax=axes[0], label="Mean intensity")
            axes[0].set_title("A (VLS) vs GMD bin")
            axes[0].set_xlabel("GMD bin centre (uJ)")
            axes[0].set_ylabel("VLS pixel")

            im1 = axes[1].pcolormesh(agg.gmd_centres, agg.tof_centres, agg.D.T, shading="auto", cmap="magma")
            fig.colorbar(im1, ax=axes[1], label="Mean counts")
            axes[1].set_title("D (eTOF) vs GMD bin")
            axes[1].set_xlabel("GMD bin centre (uJ)")
            axes[1].set_ylabel("eTOF (ns)")

            axes[2].plot(agg.gmd_centres, agg.n_per_bin, "o-", color="steelblue")
            axes[2].set_title("Shots per GMD bin")
            axes[2].set_xlabel("GMD bin centre (uJ)")
            axes[2].set_ylabel("n")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axes[0].pcolormesh(agg.gmd_centres, agg.tof_centres, agg.D.T, shading="auto", cmap="magma")
            fig.colorbar(im0, ax=axes[0], label="Mean counts")
            axes[0].set_title("D (electron TOF) vs GMD bin")
            axes[0].set_xlabel("GMD bin centre (uJ)")
            axes[0].set_ylabel("eTOF (ns)")

            im1 = axes[1].pcolormesh(
                agg.gmd_centres,
                agg.ion_tof_centres,
                agg.C.T,
                shading="auto",
                cmap="inferno",
            )
            fig.colorbar(im1, ax=axes[1], label="Mean counts")
            axes[1].set_title("C (ion TOF) vs GMD bin")
            axes[1].set_xlabel("GMD bin centre (uJ)")
            axes[1].set_ylabel("iTOF (ns)")

            axes[2].plot(agg.gmd_centres, agg.n_per_bin, "o-", color="steelblue")
            axes[2].set_title("Shots per GMD bin")
            axes[2].set_xlabel("GMD bin centre (uJ)")
            axes[2].set_ylabel("n")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        z_centres = agg.z_centres
        A_mean = _safe_nanmean(agg.A, axis=0)
        D_mean = _safe_nanmean(agg.D[..., 0], axis=0)
        n_mean = _safe_nanmean(agg.n_per_bin, axis=0)

        im0 = axes[0].pcolormesh(z_centres, agg.vls_pixels, A_mean.T, shading="auto", cmap="viridis")
        fig.colorbar(im0, ax=axes[0], label="Mean intensity")
        axes[0].set_title("A (VLS) vs z (avg over GMD bins)")
        axes[0].set_xlabel("z bin centre")
        axes[0].set_ylabel("VLS pixel")

        axes[1].plot(z_centres, D_mean, "o-", color="darkorange")
        axes[1].set_title("D count in TOF ROI vs z")
        axes[1].set_xlabel("z bin centre")
        axes[1].set_ylabel("Mean count")

        axes[2].plot(z_centres, n_mean, "o-", color="steelblue")
        axes[2].set_title("Shots per z bin (avg over GMD)")
        axes[2].set_xlabel("z bin centre")
        axes[2].set_ylabel("n")

    fig.suptitle(f"Aggregates means — Config {agg.config} ({agg.mode})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def _plot_agg_cross_cov(agg, g_idx: int) -> tuple[plt.Figure, np.ndarray]:
    if agg.mode != "spectral":
        raise RuntimeError("Cross-covariance view is only available for spectral aggregates.")

    g_idx = int(np.clip(g_idx, 0, agg.n_gmd_bins - 1))
    if agg.config == 2:
        cov = agg.AtD[g_idx] - np.outer(agg.A[g_idx], agg.D[g_idx])
        x = agg.tof_centres
        y = agg.vls_pixels
        title = f"Cov(A, D) at GMD bin {g_idx}"
        ylab = "VLS pixel"
        xlab = "eTOF (ns)"
    else:
        cov = agg.DtC[g_idx] - np.outer(agg.D[g_idx], agg.C[g_idx])
        x = agg.ion_tof_centres
        y = agg.tof_centres
        title = f"Cov(D, C) at GMD bin {g_idx}"
        ylab = "eTOF (ns)"
        xlab = "iTOF (ns)"

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.pcolormesh(x, y, cov, shading="auto", cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Covariance")
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.tight_layout()
    return fig, np.array([ax])


def _plot_agg_tr_slice(agg, g_idx: int) -> tuple[plt.Figure, np.ndarray]:
    if agg.mode != "time_resolved" or agg.config != 2:
        raise RuntimeError("TR slice is only available for config-2 time-resolved aggregates.")

    g_idx = int(np.clip(g_idx, 0, agg.n_gmd_bins - 1))
    A_slice = agg.A[g_idx]
    D_slice = agg.D[g_idx, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].pcolormesh(agg.z_centres, agg.vls_pixels, A_slice.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="Mean intensity")
    axes[0].set_title(f"A(z, pixel) at GMD bin {g_idx}")
    axes[0].set_xlabel("z bin centre")
    axes[0].set_ylabel("VLS pixel")

    axes[1].plot(agg.z_centres, D_slice, "o-", color="darkorange")
    axes[1].set_title(f"D(z) at GMD bin {g_idx}")
    axes[1].set_xlabel("z bin centre")
    axes[1].set_ylabel("Mean TOF-ROI count")

    fig.tight_layout()
    return fig, axes


def _require_vls(data) -> np.ndarray:
    vls = getattr(data, "vls", None)
    if vls is None:
        raise RuntimeError("This plot requires config-2 combined data with VLS.")
    return vls


def _vls_pixel_axis(data, n_pixels: int) -> np.ndarray:
    pix = getattr(data, "vls_pixels", None)
    return pix if pix is not None else np.arange(n_pixels)


def _safe_lognorm(arr: np.ndarray, floor: float = 1e-6):
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    vmin = max(float(np.nanmin(arr[finite])), floor)
    vmax = float(np.nanmax(arr[finite]))
    if vmax <= vmin:
        return None
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _plot_vls_mean_spectrum(data) -> tuple[plt.Figure, np.ndarray]:
    vls = _require_vls(data)
    n_pixels = vls.shape[-1]
    pixel_ax = _vls_pixel_axis(data, n_pixels)
    flat = vls.reshape(-1, n_pixels)
    mean_spec = _safe_nanmean(flat, axis=0)
    std_spec = np.nanstd(flat, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(pixel_ax, mean_spec, color="mediumseagreen", lw=1.2, label=f"mean ({flat.shape[0]} shots)")
    ax.fill_between(pixel_ax, mean_spec - std_spec, mean_spec + std_spec, color="mediumseagreen", alpha=0.25)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Intensity (arb.)")
    ax.set_title("Average VLS spectrum")
    ax.legend()
    fig.tight_layout()
    return fig, np.array([ax])


def _plot_vls_vs_bunch(data, log_map: bool) -> tuple[plt.Figure, np.ndarray]:
    vls = _require_vls(data)
    n_trains, n_bunches, n_pixels = vls.shape
    pixel_ax = _vls_pixel_axis(data, n_pixels)
    mean_by_bunch = _safe_nanmean(vls, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    norm = _safe_lognorm(mean_by_bunch, floor=1e-3) if log_map else None
    im = axes[0].pcolormesh(pixel_ax, np.arange(n_bunches), mean_by_bunch, cmap="inferno", shading="auto", norm=norm)
    fig.colorbar(im, ax=axes[0], label="Mean intensity (arb.)")
    axes[0].set_ylabel("Bunch index")
    axes[0].set_title("Mean VLS spectrum vs bunch index")

    total_per_bunch = np.nansum(mean_by_bunch, axis=1)
    axes[1].plot(total_per_bunch, np.arange(n_bunches), marker="o", ms=3, color="mediumseagreen")
    axes[1].set_xlabel("Integrated intensity")
    axes[1].set_title("Integrated VLS intensity per bunch")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _plot_vls_vs_train(data, log_map: bool) -> tuple[plt.Figure, np.ndarray]:
    vls = _require_vls(data)
    n_trains, _, n_pixels = vls.shape
    pixel_ax = _vls_pixel_axis(data, n_pixels)
    mean_by_train = _safe_nanmean(vls, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    norm = _safe_lognorm(mean_by_train, floor=1e-3) if log_map else None
    im = axes[0].pcolormesh(pixel_ax, data.tID, mean_by_train, cmap="inferno", shading="auto", norm=norm)
    fig.colorbar(im, ax=axes[0], label="Mean intensity (arb.)")
    axes[0].set_ylabel("Train ID")
    axes[0].set_title("Mean VLS spectrum vs train index")

    total_per_train = np.nansum(mean_by_train, axis=1)
    axes[1].plot(total_per_train, data.tID, marker="o", ms=3, color="mediumseagreen", ls="")
    axes[1].set_xlabel("Integrated intensity")
    axes[1].set_title("Integrated VLS intensity per train")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _plot_gmd_vs_vls_sum(data, n_gmd_bins: int) -> tuple[plt.Figure, np.ndarray]:
    vls = _require_vls(data)
    gmd_flat = data.gmd.ravel()
    vls_sum = np.nansum(vls, axis=-1).ravel()
    good = np.isfinite(gmd_flat) & np.isfinite(vls_sum)
    x = gmd_flat[good]
    y = vls_sum[good]
    if x.size < 2:
        raise RuntimeError("Not enough valid shots for GMD vs VLS plot.")

    gmd_edges = np.percentile(x, np.linspace(0, 100, int(n_gmd_bins) + 1))
    gmd_edges = np.unique(gmd_edges)
    if gmd_edges.size < 3:
        gmd_edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), int(n_gmd_bins) + 1)

    _, bool_ar = arb_bool_ar(gmd_edges, x)
    mean_vls, std_vls, _ = bin_and_average(bool_ar, y)
    mean_gmd, _, _ = bin_and_average(bool_ar, x)

    finite = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[finite], y[finite], 1)
    r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y_hi = float(np.nanpercentile(y, 95))
    h = axes[0].hist2d(
        x,
        y,
        bins=[gmd_edges, np.linspace(0.0, y_hi if y_hi > 0 else 1.0, 51)],
        cmap="viridis",
        cmin=1,
        norm=mcolors.LogNorm(),
    )
    fig.colorbar(h[3], ax=axes[0], label="Shots per bin")
    axes[0].plot(x_line, slope * x_line + intercept, color="white", lw=1.5)
    axes[0].set_xlabel("GMD (uJ)")
    axes[0].set_ylabel("Per-shot VLS sum (arb.)")
    axes[0].set_title(f"GMD vs VLS sum (r = {r:.3f})")

    axes[1].errorbar(mean_gmd, mean_vls, yerr=std_vls, fmt="o-", color="mediumseagreen", capsize=3, lw=1.2)
    axes[1].plot(x_line, slope * x_line + intercept, color="darkred", lw=1.5)
    axes[1].set_xlabel("Mean GMD per bin (uJ)")
    axes[1].set_ylabel("Mean VLS sum per shot (arb.)")
    axes[1].set_title("Binned mean GMD vs VLS sum")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _available_tof_keys(data) -> list[str]:
    keys: list[str] = []
    if getattr(data, "tofs_e", None) is not None:
        keys.append("tofs_e")
    if getattr(data, "tofs_i", None) is not None:
        keys.append("tofs_i")
    if getattr(data, "liq_tofs_e", None) is not None:
        keys.append("liq_tofs_e")
    return keys


def _require_tofs(data, tof_key: str) -> np.ndarray:
    tofs = getattr(data, tof_key, None)
    if tofs is None:
        raise RuntimeError(f"TOF source {tof_key!r} is not available in the loaded dataset.")
    return tofs


def _prepare_tof_hits(tofs: np.ndarray, tof_min: float, tof_max: float):
    n_trains, n_bunches, _ = tofs.shape
    valid = tofs != 0
    tof_of_hit = tofs[valid]
    train_of_hit = np.broadcast_to(np.arange(n_trains)[:, None, None], tofs.shape)[valid]
    bunch_of_hit = np.broadcast_to(np.arange(n_bunches)[None, :, None], tofs.shape)[valid]
    in_win = (tof_of_hit >= float(tof_min)) & (tof_of_hit < float(tof_max))
    return tof_of_hit, train_of_hit, bunch_of_hit, in_win, n_trains, n_bunches


def _plot_tof_mean_spectrum(data, tof_key: str, tof_min: float, tof_max: float, n_bins: int) -> tuple[plt.Figure, np.ndarray]:
    tofs = _require_tofs(data, tof_key)
    tof_edges = np.linspace(float(tof_min), float(tof_max), int(n_bins) + 1)
    tof_cents = 0.5 * (tof_edges[:-1] + tof_edges[1:])
    tof_of_hit, _, _, in_win, n_trains, n_bunches = _prepare_tof_hits(tofs, tof_min, tof_max)
    total_counts, _ = np.histogram(tof_of_hit[in_win], bins=tof_edges)
    mean_spec = total_counts / max(n_trains * n_bunches, 1)

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(tof_cents, mean_spec, color="steelblue", lw=1.2, label=f"mean ({n_trains * n_bunches} shots)")
    ax.set_xlabel("TOF (ns)")
    ax.set_ylabel("Counts per shot")
    ax.set_title(f"Average {tof_key} spectrum ({tof_min:.0f}-{tof_max:.0f} ns)")
    ax.legend()
    fig.tight_layout()
    return fig, np.array([ax])


def _plot_tof_vs_bunch(
    data,
    tof_key: str,
    tof_min: float,
    tof_max: float,
    n_bins: int,
    log_map: bool,
) -> tuple[plt.Figure, np.ndarray]:
    tofs = _require_tofs(data, tof_key)
    tof_edges = np.linspace(float(tof_min), float(tof_max), int(n_bins) + 1)
    tof_cents = 0.5 * (tof_edges[:-1] + tof_edges[1:])
    tof_of_hit, _, bunch_of_hit, in_win, n_trains, n_bunches = _prepare_tof_hits(tofs, tof_min, tof_max)
    bunch_edges = np.arange(n_bunches + 1)
    H_bunch, _, _ = np.histogram2d(bunch_of_hit[in_win], tof_of_hit[in_win], bins=[bunch_edges, tof_edges])
    mean_by_bunch = H_bunch / max(n_trains, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    norm = _safe_lognorm(mean_by_bunch) if log_map else None
    im = axes[0].pcolormesh(tof_cents, np.arange(n_bunches), mean_by_bunch, cmap="inferno", shading="auto", norm=norm)
    fig.colorbar(im, ax=axes[0], label="Mean counts per shot")
    axes[0].set_xlabel("TOF (ns)")
    axes[0].set_ylabel("Bunch index")
    axes[0].set_title(f"Mean {tof_key} spectrum vs bunch index")

    total_per_bunch = mean_by_bunch.sum(axis=1)
    axes[1].plot(total_per_bunch, np.arange(n_bunches), marker="o", ms=3, color="steelblue")
    axes[1].set_xlabel("Mean hits per shot")
    axes[1].set_title("Integrated hits per bunch")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _plot_tof_vs_train(
    data,
    tof_key: str,
    tof_min: float,
    tof_max: float,
    n_bins: int,
    log_map: bool,
) -> tuple[plt.Figure, np.ndarray]:
    tofs = _require_tofs(data, tof_key)
    tof_edges = np.linspace(float(tof_min), float(tof_max), int(n_bins) + 1)
    tof_cents = 0.5 * (tof_edges[:-1] + tof_edges[1:])
    tof_of_hit, train_of_hit, _, in_win, _, n_bunches = _prepare_tof_hits(tofs, tof_min, tof_max)
    n_trains = tofs.shape[0]
    train_edges = np.arange(n_trains + 1)
    H_train, _, _ = np.histogram2d(train_of_hit[in_win], tof_of_hit[in_win], bins=[train_edges, tof_edges])
    mean_by_train = H_train / max(n_bunches, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    norm = _safe_lognorm(mean_by_train) if log_map else None
    im = axes[0].pcolormesh(tof_cents, data.tID, mean_by_train, cmap="inferno", shading="auto", norm=norm)
    fig.colorbar(im, ax=axes[0], label="Mean counts per shot")
    axes[0].set_xlabel("TOF (ns)")
    axes[0].set_ylabel("Train ID")
    axes[0].set_title(f"Mean {tof_key} spectrum vs train index")

    total_per_train = mean_by_train.sum(axis=1)
    axes[1].plot(total_per_train, data.tID, marker="o", ms=3, color="steelblue", ls="")
    axes[1].set_xlabel("Mean hits per shot")
    axes[1].set_title("Integrated hits per train")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _plot_tof_per_gmd_bin(
    data,
    tof_key: str,
    tof_min: float,
    tof_max: float,
    n_bins: int,
    n_gmd_bins: int,
) -> tuple[plt.Figure, np.ndarray]:
    tofs = _require_tofs(data, tof_key)
    tof_edges = np.linspace(float(tof_min), float(tof_max), int(n_bins) + 1)
    tof_cents = 0.5 * (tof_edges[:-1] + tof_edges[1:])
    tof_of_hit, train_of_hit, bunch_of_hit, in_win, n_trains, n_bunches = _prepare_tof_hits(tofs, tof_min, tof_max)

    gmd_flat = data.gmd.ravel()
    good = np.isfinite(gmd_flat)
    x = gmd_flat[good]
    if x.size < 2:
        raise RuntimeError("Not enough valid GMD values for TOF-vs-GMD plot.")

    gmd_edges = np.percentile(x, np.linspace(0, 100, int(n_gmd_bins) + 1))
    gmd_edges = np.unique(gmd_edges)
    if gmd_edges.size < 3:
        gmd_edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), int(n_gmd_bins) + 1)

    shot_id = train_of_hit[in_win] * n_bunches + bunch_of_hit[in_win]
    shot_gmd = data.gmd.reshape(-1)[shot_id]
    gmd_bin_of_shot = np.digitize(shot_gmd, gmd_edges) - 1
    valid_bins = (gmd_bin_of_shot >= 0) & (gmd_bin_of_shot < len(gmd_edges) - 1) & np.isfinite(shot_gmd)
    H_gmd, _, _ = np.histogram2d(gmd_bin_of_shot[valid_bins], tof_of_hit[in_win][valid_bins], bins=[np.arange(len(gmd_edges)), tof_edges])

    _, bool_ar = arb_bool_ar(gmd_edges, x)
    mean_gmd, _, n_per_bin = bin_and_average(bool_ar, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_spec_by_gmd = H_gmd / n_per_bin[:, None]

    finite_bins = np.isfinite(mean_gmd)
    if not finite_bins.any():
        raise RuntimeError("No non-empty GMD bins for TOF-vs-GMD plot.")

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(np.nanmin(mean_gmd[finite_bins])), vmax=float(np.nanmax(mean_gmd[finite_bins])))
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for cent, spec, n_shots in zip(mean_gmd, mean_spec_by_gmd, n_per_bin):
        if not np.isfinite(cent) or n_shots <= 0:
            continue
        ax.plot(tof_cents, spec, color=cmap(norm(cent)), lw=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Mean GMD per bin (uJ)")
    ax.set_xlabel("TOF (ns)")
    ax.set_ylabel("Mean counts per shot")
    ax.set_title(f"Average {tof_key} spectrum per GMD bin")
    fig.tight_layout()
    return fig, np.array([ax])


def _build_panel_image(kind: str, params: dict, data, meta: dict) -> bytes:
    if meta.get("kind") == "aggregates":
        if kind == "agg_means":
            fig, _ = _plot_agg_means(data)
        elif kind == "agg_cross_cov":
            fig, _ = _plot_agg_cross_cov(data, params["gmd_bin_idx"])
        else:
            fig, _ = _plot_agg_tr_slice(data, params["gmd_bin_idx"])
        return _figure_to_png_bytes(fig)

    if kind == "diagnostics":
        fig, _ = plot_diagnostics(data)
    elif kind == "linearity":
        fig, _ = plot_linearity(data, n_gmd_bins=params["n_gmd_bins"])
    elif kind == "vls_mean":
        fig, _ = _plot_vls_mean_spectrum(data)
    elif kind == "vls_bunch":
        fig, _ = _plot_vls_vs_bunch(data, log_map=params["log_map"])
    elif kind == "vls_train":
        fig, _ = _plot_vls_vs_train(data, log_map=params["log_map"])
    elif kind == "vls_gmd":
        fig, _ = _plot_gmd_vs_vls_sum(data, n_gmd_bins=params["n_gmd_bins"])
    elif kind == "tof_mean":
        fig, _ = _plot_tof_mean_spectrum(data, params["tof_key"], params["tof_min"], params["tof_max"], params["n_bins"])
    elif kind == "tof_bunch":
        fig, _ = _plot_tof_vs_bunch(data, params["tof_key"], params["tof_min"], params["tof_max"], params["n_bins"], params["log_map"])
    elif kind == "tof_train":
        fig, _ = _plot_tof_vs_train(data, params["tof_key"], params["tof_min"], params["tof_max"], params["n_bins"], params["log_map"])
    elif kind == "tof_gmd":
        fig, _ = _plot_tof_per_gmd_bin(data, params["tof_key"], params["tof_min"], params["tof_max"], params["n_bins"], params["n_gmd_bins"])
    elif kind == "energy_maps":
        fig, _ = plot_energy_binned_maps(
            data,
            mpe_edges=params["mpe_edges"],
            e_tof_edges=params["e_tof_edges"],
            i_tof_edges=params["i_tof_edges"],
            bunch_start=params["bunch_start"],
            bunch_end=params["bunch_end"],
        )
    else:
        fig, _ = plot_delay_dependence(
            data,
            mpe_range=params["mpe_range"],
            z_edges=params["z_edges"],
            bunch_start=params["bunch_start"],
            bunch_end=params["bunch_end"],
            e_tof_range=params["e_tof_range"],
            i_tof_range=params["i_tof_range"],
        )
    return _figure_to_png_bytes(fig)


def _render_plot_controls(data, meta: dict) -> None:
    st.subheader("Generate Plots")

    if meta.get("kind") == "aggregates":
        _render_aggregate_plot_controls(data, meta)
        return

    with st.form("plot_add_form"):
        options = ["Diagnostics", "Linearity", "Energy-Binned Maps", "Delay Dependence"]
        options.extend(
            [
                "TOF Mean Spectrum (All Shots)",
                "TOF Spectrum vs Bunch Index",
                "TOF Spectrum vs Train Index",
                "TOF Spectrum per GMD Bin",
            ]
        )
        if int(meta.get("config", 0)) == 2:
            options.extend(
                [
                    "VLS Mean Spectrum (All Shots)",
                    "VLS Spectrum vs Bunch Index",
                    "VLS Spectrum vs Train Index",
                    "GMD vs VLS Intensity",
                ]
            )
        plot_type = st.selectbox(
            "Plot type",
            options=options,
            key="plot_type_selector",
        )

        if plot_type == "Diagnostics":
            add_clicked = st.form_submit_button("Add Diagnostics Plot")
            if add_clicked:
                _add_panel("diagnostics", {})

        elif plot_type == "Linearity":
            n_bins = st.number_input("Number of GMD bins", value=20, min_value=2, step=1, key="lin_n_bins")
            add_clicked = st.form_submit_button("Add Linearity Plot")
            if add_clicked:
                _add_panel("linearity", {"n_gmd_bins": int(n_bins)})

        elif plot_type == "VLS Mean Spectrum (All Shots)":
            add_clicked = st.form_submit_button("Add VLS Mean Spectrum")
            if add_clicked:
                _add_panel("vls_mean", {})

        elif plot_type == "VLS Spectrum vs Bunch Index":
            log_map = st.checkbox("Log color scale", value=False, key="vls_bunch_log")
            add_clicked = st.form_submit_button("Add VLS vs Bunch")
            if add_clicked:
                _add_panel("vls_bunch", {"log_map": bool(log_map)})

        elif plot_type == "VLS Spectrum vs Train Index":
            log_map = st.checkbox("Log color scale", value=False, key="vls_train_log")
            add_clicked = st.form_submit_button("Add VLS vs Train")
            if add_clicked:
                _add_panel("vls_train", {"log_map": bool(log_map)})

        elif plot_type == "GMD vs VLS Intensity":
            n_bins = st.number_input("Percentile GMD bins", value=10, min_value=4, step=1, key="vls_gmd_bins")
            add_clicked = st.form_submit_button("Add GMD vs VLS")
            if add_clicked:
                _add_panel("vls_gmd", {"n_gmd_bins": int(n_bins)})

        elif plot_type in {
            "TOF Mean Spectrum (All Shots)",
            "TOF Spectrum vs Bunch Index",
            "TOF Spectrum vs Train Index",
            "TOF Spectrum per GMD Bin",
        }:
            tof_options = _available_tof_keys(data)
            tof_key = st.selectbox("TOF source", options=tof_options, key=f"tof_key_{plot_type}")
            col1, col2, col3 = st.columns(3)
            with col1:
                tof_min = st.number_input("TOF min (ns)", value=0.0, step=10.0, key=f"tof_min_{plot_type}")
            with col2:
                tof_max = st.number_input("TOF max (ns)", value=2000.0, step=10.0, key=f"tof_max_{plot_type}")
            with col3:
                n_bins = st.number_input("TOF bins", value=500, min_value=10, step=10, key=f"tof_bins_{plot_type}")

            params = {
                "tof_key": str(tof_key),
                "tof_min": float(tof_min),
                "tof_max": float(tof_max),
                "n_bins": int(n_bins),
            }

            if plot_type in {"TOF Spectrum vs Bunch Index", "TOF Spectrum vs Train Index"}:
                params["log_map"] = bool(st.checkbox("Log color scale", value=False, key=f"tof_log_{plot_type}"))

            if plot_type == "TOF Spectrum per GMD Bin":
                params["n_gmd_bins"] = int(
                    st.number_input("Percentile GMD bins", value=10, min_value=4, step=1, key=f"tof_gmd_bins_{plot_type}")
                )

            button_label = {
                "TOF Mean Spectrum (All Shots)": "Add TOF Mean Spectrum",
                "TOF Spectrum vs Bunch Index": "Add TOF vs Bunch",
                "TOF Spectrum vs Train Index": "Add TOF vs Train",
                "TOF Spectrum per GMD Bin": "Add TOF per GMD Bin",
            }[plot_type]
            add_clicked = st.form_submit_button(button_label)
            if add_clicked:
                kind_map = {
                    "TOF Mean Spectrum (All Shots)": "tof_mean",
                    "TOF Spectrum vs Bunch Index": "tof_bunch",
                    "TOF Spectrum vs Train Index": "tof_train",
                    "TOF Spectrum per GMD Bin": "tof_gmd",
                }
                _add_panel(kind_map[plot_type], params)

        elif plot_type == "Energy-Binned Maps":
            mpe_min, mpe_max = _safe_minmax(data.mpe, (0.0, 1.0))
            col1, col2, col3 = st.columns(3)
            with col1:
                mpe_lo = st.number_input("Photon energy min (eV)", value=mpe_min, key="emap_mpe_lo")
            with col2:
                mpe_hi = st.number_input("Photon energy max (eV)", value=mpe_max, key="emap_mpe_hi")
            with col3:
                n_mpe = st.number_input("Photon energy bins", value=20, min_value=2, step=1, key="emap_n_mpe")

            col4, col5 = st.columns(2)
            with col4:
                bunch_start = st.number_input(
                    "Bunch start",
                    value=0,
                    min_value=0,
                    max_value=max(0, data.n_bunches - 1),
                    step=1,
                    key="emap_bunch_start",
                )
            with col5:
                bunch_end = st.number_input(
                    "Bunch end",
                    value=max(0, data.n_bunches - 1),
                    min_value=0,
                    max_value=max(0, data.n_bunches - 1),
                    step=1,
                    key="emap_bunch_end",
                )

            add_clicked = st.form_submit_button("Add Energy-Binned Map")
            if add_clicked:
                _add_panel(
                    "energy_maps",
                    {
                        "mpe_edges": np.linspace(float(mpe_lo), float(mpe_hi), int(n_mpe) + 1),
                        "e_tof_edges": np.linspace(500.0, 1750.0, 126),
                        "i_tof_edges": np.linspace(1000.0, 9000.0, 161) if meta["config"] == 1 else None,
                        "bunch_start": int(bunch_start),
                        "bunch_end": int(bunch_end),
                    },
                )

        else:
            mpe_min, mpe_max = _safe_minmax(data.mpe, (0.0, 1.0))
            z_min, z_max = _safe_minmax(data.z, (-1.0, 1.0))
            col1, col2, col3 = st.columns(3)
            with col1:
                mpe_lo = st.number_input("Photon energy min (eV)", value=mpe_min, key="delay_mpe_lo")
            with col2:
                mpe_hi = st.number_input("Photon energy max (eV)", value=mpe_max, key="delay_mpe_hi")
            with col3:
                n_z = st.number_input("Delay bins", value=20, min_value=2, step=1, key="delay_n_z")

            col4, col5, col6, col7 = st.columns(4)
            with col4:
                z_lo = st.number_input("Delay min", value=z_min, key="delay_z_lo")
            with col5:
                z_hi = st.number_input("Delay max", value=z_max, key="delay_z_hi")
            with col6:
                bunch_start = st.number_input(
                    "Bunch start",
                    value=0,
                    min_value=0,
                    max_value=max(0, data.n_bunches - 1),
                    step=1,
                    key="delay_bunch_start",
                )
            with col7:
                bunch_end = st.number_input(
                    "Bunch end",
                    value=max(0, data.n_bunches - 1),
                    min_value=0,
                    max_value=max(0, data.n_bunches - 1),
                    step=1,
                    key="delay_bunch_end",
                )

            add_clicked = st.form_submit_button("Add Delay Plot")
            if add_clicked:
                _add_panel(
                    "delay",
                    {
                        "mpe_range": (float(mpe_lo), float(mpe_hi)),
                        "z_edges": np.linspace(float(z_lo), float(z_hi), int(n_z) + 1),
                        "bunch_start": int(bunch_start),
                        "bunch_end": int(bunch_end),
                        "e_tof_range": (500.0, 1750.0),
                        "i_tof_range": (1000.0, 9000.0) if meta["config"] == 1 else None,
                    },
                )


def _render_aggregate_plot_controls(agg, meta: dict) -> None:
    opts = ["Aggregate Means"]
    if meta.get("mode") == "spectral":
        opts.append("Cross-Covariance (one GMD bin)")
    if meta.get("mode") == "time_resolved" and meta.get("config") == 2:
        opts.append("Time-Resolved Slice (one GMD bin)")

    with st.form("plot_add_form_agg"):
        plot_type = st.selectbox("Plot type", options=opts, key="plot_type_selector_agg")

        if plot_type == "Aggregate Means":
            add_clicked = st.form_submit_button("Add Aggregate Means")
            if add_clicked:
                _add_panel("agg_means", {})

        else:
            max_bin = max(0, int(agg.n_gmd_bins) - 1)
            gmd_bin_idx = st.number_input(
                "GMD bin index",
                value=0,
                min_value=0,
                max_value=max_bin,
                step=1,
                key="agg_gmd_bin_idx",
            )
            label = (
                "Add Cross-Covariance"
                if plot_type.startswith("Cross-Covariance")
                else "Add TR Slice"
            )
            add_clicked = st.form_submit_button(label)
            if add_clicked:
                _add_panel(
                    "agg_cross_cov" if plot_type.startswith("Cross-Covariance") else "agg_tr_slice",
                    {"gmd_bin_idx": int(gmd_bin_idx)},
                )


def _render_panel(panel: dict) -> bool:
    title_map = {
        "diagnostics": "Diagnostics",
        "linearity": "Linearity",
        "vls_mean": "VLS Mean Spectrum",
        "vls_bunch": "VLS Spectrum vs Bunch Index",
        "vls_train": "VLS Spectrum vs Train Index",
        "vls_gmd": "GMD vs VLS Intensity",
        "tof_mean": "TOF Mean Spectrum",
        "tof_bunch": "TOF Spectrum vs Bunch Index",
        "tof_train": "TOF Spectrum vs Train Index",
        "tof_gmd": "TOF Spectrum per GMD Bin",
        "energy_maps": "Energy-Binned Maps",
        "delay": "Delay Dependence",
        "agg_means": "Aggregate Means",
        "agg_cross_cov": "Aggregate Cross-Covariance",
        "agg_tr_slice": "Aggregate Time-Resolved Slice",
    }
    container = st.container(border=True)
    with container:
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(f"**{title_map[panel['kind']]}**")
            source = panel.get("source_filepath")
            if source:
                cfg = panel.get("source_config")
                cfg_txt = f" | Config {cfg}" if cfg is not None else ""
                kind = panel.get("source_kind", "combined")
                mode = panel.get("source_mode", "spectral")
                st.caption(f"{Path(source).name}{cfg_txt} | {kind} ({mode})")
            else:
                st.caption("Source file unknown")
        with c2:
            if st.button("Close", key=f"close_plot_{panel['id']}"):
                return True

        st.image(panel["image"], width="stretch")
    return False


def show() -> None:
    _init_state()
    st.header("Plot")
    st.caption("Load data once, then generate and compare plots from the loaded dataset.")

    if not config.COMBINED_DIR.exists():
        st.warning(f"Combined folder not found: `{config.COMBINED_DIR}`")
        return

    h5_files = sorted(config.COMBINED_DIR.glob("*.h5"))
    if not h5_files:
        st.info("No combined H5 files found. Run the Process step first.")
        return

    selected = st.selectbox("H5 file", options=[p.name for p in h5_files])
    selected_path = config.COMBINED_DIR / selected

    try:
        file_info = _detect_h5_type(selected_path)
    except Exception as exc:
        st.error(f"Failed to inspect file: {exc}")
        return

    st.caption(
        f"Detected: config {file_info['config']} | {file_info['kind']} ({file_info['mode']})."
    )

    trim_start = 0
    trim_end = 0
    downsample = 1
    if file_info["kind"] == "combined":
        with st.form("plot_load_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text_input("Detected config", value=f"Config {file_info['config']}", disabled=True)
            with col2:
                trim_start = st.number_input("Trim start (good trains)", value=2, min_value=0, step=1)
            with col3:
                trim_end = st.number_input("Trim end (good trains)", value=2, min_value=0, step=1)

            downsample = st.number_input(
                "Downsample N (keep 1 in N trains, 1 = no downsample)",
                value=10,
                min_value=1,
                step=1,
            )
            load_clicked = st.form_submit_button("Load Data")

        inspect = _inspect_file(
            selected_path,
            file_info["config"],
            int(trim_start),
            int(trim_end),
            int(downsample),
        )
        inspect["kind"] = "combined"
        inspect["mode"] = file_info["mode"]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total trains", inspect["n_total"])
        s2.metric("Good trains", inspect["n_good"])
        s3.metric("Loaded trains", inspect["n_loaded"])
        s4.metric("Bunches / train", inspect["n_bunches"])
        st.caption(
            f"Detected config {inspect['config']} from datasets. "
            "Loaded trains = good trains after trim/downsample."
        )
    else:
        with st.form("plot_load_form_agg"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text_input("Detected config", value=f"Config {file_info['config']}", disabled=True)
            with col2:
                st.text_input("Data type", value="Aggregates", disabled=True)
            with col3:
                st.text_input("Mode", value=file_info["mode"], disabled=True)
            load_clicked = st.form_submit_button("Load Data")

        inspect = _inspect_aggregates_file(selected_path, file_info["config"], file_info["mode"])
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Bins (total)", inspect["n_bins_total"])
        s2.metric("Bins (non-empty)", inspect["n_bins_nonempty"])
        s3.metric("Shots (sum)", inspect["n_shots_total"])
        s4.metric("TOF bins", inspect["n_tof"])
        extra = []
        if "n_pixels" in inspect:
            extra.append(f"pixels={inspect['n_pixels']}")
        if "n_tof_i" in inspect:
            extra.append(f"ion_tof_bins={inspect['n_tof_i']}")
        if "n_z_bins" in inspect:
            extra.append(f"z_bins={inspect['n_z_bins']}")
        if extra:
            st.caption("Extra: " + ", ".join(extra))

    status = st.session_state.get("plot_load_status")
    if status:
        level, message = status
        getattr(st, level)(message)

    if load_clicked:
        with st.spinner("Loading data…"):
            try:
                if file_info["kind"] == "combined":
                    data = load_data(
                        str(selected_path),
                        config=file_info["config"],
                        trim_start=int(trim_start),
                        trim_end=int(trim_end),
                        downsample_N=int(downsample),
                    )
                else:
                    data = load_aggregates(selected_path)
            except Exception as exc:
                st.session_state.plot_load_status = ("error", f"Failed to load data: {exc}")
                st.error(f"Failed to load data: {exc}")
                return

        st.session_state.plot_loaded_data = data
        st.session_state.plot_loaded_meta = inspect
        st.session_state.plot_load_status = (
            "success",
            (
                f"Loaded {data.n_trains} trains × {data.n_bunches} bunches from {selected}."
                if file_info["kind"] == "combined"
                else f"Loaded aggregates from {selected} (config {inspect['config']}, mode={inspect['mode']})."
            ),
        )
        if file_info["kind"] == "combined":
            st.success(f"Loaded {data.n_trains} trains × {data.n_bunches} bunches from {selected}.")
        else:
            st.success(f"Loaded aggregates from {selected} (config {inspect['config']}, mode={inspect['mode']}).")

    if file_info["kind"] == "combined" and not _loaded_matches(selected_path, int(trim_start), int(trim_end), int(downsample)):
        st.info("Current controls differ from the loaded dataset. Click Load Data to refresh the cache.")

    loaded_data = st.session_state.plot_loaded_data
    loaded_meta = st.session_state.plot_loaded_meta
    if loaded_data is None or loaded_meta is None:
        return

    st.divider()
    _render_plot_controls(loaded_data, loaded_meta)

    if not st.session_state.plot_panels:
        st.info("No plots yet. Add a plot above.")
        return

    st.divider()
    st.subheader("Open Plots")
    remove_ids: list[str] = []
    cols = st.columns(2)
    for idx, panel in enumerate(st.session_state.plot_panels):
        with cols[idx % 2]:
            should_remove = _render_panel(panel)
            if should_remove:
                remove_ids.append(panel["id"])

    if remove_ids:
        st.session_state.plot_panels = [
            panel for panel in st.session_state.plot_panels if panel["id"] not in remove_ids
        ]
        st.rerun()
