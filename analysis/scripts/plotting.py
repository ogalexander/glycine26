import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from typing import Optional, Tuple

from experiment_data import ExperimentData
from processing import count_hits, hits_to_spectrum, select_bunch_range
from binning import arb_bool_ar, bin_and_average, bin_and_sum_ratio


# ===========================================================================
# 1. DIAGNOSTIC PLOTS
# ===========================================================================

def plot_diagnostics(
    data: ExperimentData,
    e_tof_edges: Optional[np.ndarray] = None,
    i_tof_edges: Optional[np.ndarray] = None,
    e_tof_range: Optional[Tuple[float, float]] = None,
    i_tof_range: Optional[Tuple[float, float]] = None,
    gmd_bins: int = 100,
    count_bins: int = 50,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Produce a suite of diagnostic plots for the loaded dataset.

    Panels
    ------
    1. Histogram of electron (and ion) hit counts per shot.
    2. Histogram of per-shot pulse energies (GMD).
    3. False-colour 2D histogram: bunch index vs GMD.
    4. 2D histogram: GMD vs electron counts (Pearson r labelled).
    5. 2D histogram: GMD vs ion counts (config 1 only, Pearson r labelled).
    6. Average x-ray spectrum (config 2 only).
    7. Train ID vs photon energy (mpe).
    8. Train ID vs train-averaged delay stage position.

    Parameters
    ----------
    data : ExperimentData
        Loaded experiment data.
    e_tof_edges : np.ndarray, optional
        TOF bin edges (ns) for electron spectra. Used only for count_hits range.
    i_tof_edges : np.ndarray, optional
        TOF bin edges (ns) for ion spectra.
    e_tof_range : (float, float), optional
        (tof_min, tof_max) in ns for counting electron hits.
    i_tof_range : (float, float), optional
        (tof_min, tof_max) in ns for counting ion hits.
    gmd_bins : int
        Number of bins for GMD histograms.
    count_bins : int
        Number of bins for count histograms.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    config = data.config
    n_panels = 8 if config == 1 else 8  # same count; panel 5 swaps ion↔vls

    ncols = 3
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
    axes = axes.flatten()

    gmd_flat  = data.gmd.reshape(-1)                    # (n*m,)
    bunch_idx = np.tile(np.arange(data.n_bunches), data.n_trains)  # (n*m,)

    # ---- electron counts ------------------------------------------------
    tofs_e_src = data.tofs_e if config == 1 else data.liq_tofs_e
    e_min = e_tof_range[0] if e_tof_range else None
    e_max = e_tof_range[1] if e_tof_range else None
    e_counts = count_hits(tofs_e_src, e_min, e_max).reshape(-1)  # (n*m,)

    # ---- ion counts (config 1 only) ------------------------------------
    if config == 1:
        i_min = i_tof_range[0] if i_tof_range else None
        i_max = i_tof_range[1] if i_tof_range else None
        i_counts = count_hits(data.tofs_i, i_min, i_max).reshape(-1)

    # -------------------------------------------------------------------
    # Panel 1: Histogram of hit counts
    # -------------------------------------------------------------------
    ax = axes[0]
    ax.hist(e_counts, bins=count_bins, alpha=0.7, label="Electrons", color="steelblue")
    if config == 1:
        ax.hist(i_counts, bins=count_bins, alpha=0.7, label="Ions", color="tomato")
    ax.set_xlabel("Hits per shot")
    ax.set_ylabel("Counts")
    ax.set_title("Hit count distribution")
    ax.legend()

    # -------------------------------------------------------------------
    # Panel 2: Histogram of GMD (pulse energy)
    # -------------------------------------------------------------------
    ax = axes[1]
    ax.hist(gmd_flat, bins=gmd_bins, color="goldenrod", alpha=0.85)
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Counts")
    ax.set_title("Pulse energy distribution")

    # -------------------------------------------------------------------
    # Panel 3: False-colour 2D histogram — bunch index vs GMD
    # -------------------------------------------------------------------
    ax = axes[2]
    h, xedges, yedges = np.histogram2d(
        bunch_idx, gmd_flat,
        bins=[data.n_bunches, gmd_bins],
    )
    im = ax.pcolormesh(
        xedges, yedges, h.T,
        cmap="viridis", norm=mcolors.LogNorm(vmin=1),
    )
    fig.colorbar(im, ax=ax, label="Counts")
    ax.set_xlabel("Bunch index")
    ax.set_ylabel("Pulse energy (µJ)")
    ax.set_title("Bunch index vs pulse energy")

    # -------------------------------------------------------------------
    # Panel 4: 2D histogram — GMD vs electron counts
    # -------------------------------------------------------------------
    ax = axes[3]
    valid = np.isfinite(gmd_flat) & np.isfinite(e_counts)
    r_e, _ = pearsonr(gmd_flat[valid], e_counts[valid])
    h, xe, ye = np.histogram2d(gmd_flat[valid], e_counts[valid],
                               bins=[gmd_bins, count_bins])
    ax.pcolormesh(xe, ye, h.T, cmap="plasma", norm=mcolors.LogNorm(vmin=1))
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Electron hits per shot")
    ax.set_title(f"GMD vs electrons  (r = {r_e:.3f})")

    # -------------------------------------------------------------------
    # Panel 5: 2D histogram — GMD vs ion counts (config 1)
    #          or average VLS spectrum (config 2)
    # -------------------------------------------------------------------
    ax = axes[4]
    if config == 1:
        valid_i = np.isfinite(gmd_flat) & np.isfinite(i_counts)
        r_i, _ = pearsonr(gmd_flat[valid_i], i_counts[valid_i])
        h, xe, ye = np.histogram2d(gmd_flat[valid_i], i_counts[valid_i],
                                   bins=[gmd_bins, count_bins])
        ax.pcolormesh(xe, ye, h.T, cmap="plasma", norm=mcolors.LogNorm(vmin=1))
        ax.set_xlabel("Pulse energy (µJ)")
        ax.set_ylabel("Ion hits per shot")
        ax.set_title(f"GMD vs ions  (r = {r_i:.3f})")
    else:
        # Average VLS spectrum
        vls_flat = data.vls.reshape(-1, data.vls.shape[-1])
        mean_vls = np.nanmean(vls_flat, axis=0)
        pixel_ax = np.arange(mean_vls.shape[0])
        ax.plot(pixel_ax, mean_vls, color="mediumseagreen")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Intensity (arb.)")
        ax.set_title("Average x-ray spectrum (VLS)")

    # -------------------------------------------------------------------
    # Panel 6: spare / average VLS for config 1 (blank if not applicable)
    # -------------------------------------------------------------------
    ax = axes[5]
    ax.set_visible(False)  # Reserved for future use

    # -------------------------------------------------------------------
    # Panel 7: Train ID vs photon energy
    # -------------------------------------------------------------------
    ax = axes[6]
    ax.plot(data.tID, data.mpe, ".", markersize=3, color="mediumpurple")
    ax.set_xlabel("Train ID")
    ax.set_ylabel("Photon energy (eV)")
    ax.set_title("Photon energy vs train ID")

    # -------------------------------------------------------------------
    # Panel 8: Train ID vs train-averaged delay
    # -------------------------------------------------------------------
    ax = axes[7]
    mean_z = np.nanmean(data.z, axis=1)  # average over bunches → (n,)
    ax.plot(data.tID, mean_z, ".", markersize=3, color="darkorange")
    ax.set_xlabel("Train ID")
    ax.set_ylabel("Delay stage position (arb.)")
    ax.set_title("Delay stage vs train ID")

    # Hide unused last panel
    axes[8].set_visible(False)

    fig.suptitle(f"Diagnostics — Config {config}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ===========================================================================
# 2. LINEARITY INVESTIGATION
# ===========================================================================

def plot_linearity(
    data: ExperimentData,
    n_gmd_bins: int = 20,
    gmd_edges: Optional[np.ndarray] = None,
    e_tof_range: Optional[Tuple[float, float]] = None,
    i_tof_range: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Investigate the linearity between pulse energy and hit counts.

    Panels
    ------
    1. Mean electron counts vs mean GMD per bin (linearity check).
    2. Mean ion counts vs mean GMD per bin (config 1 only).
    3. False-colour: bunch index vs mean electron counts per GMD bin
       (checks bunch-index independence of the linear relationship).
    4. False-colour: bunch index vs mean ion counts per GMD bin (config 1).

    Parameters
    ----------
    data : ExperimentData
    n_gmd_bins : int
        Number of GMD bins (used if gmd_edges is None).
    gmd_edges : np.ndarray, optional
        Explicit GMD bin edges (µJ). If None, uniform bins over data range.
    e_tof_range : (float, float), optional
        TOF range (ns) for counting electron hits.
    i_tof_range : (float, float), optional
        TOF range (ns) for counting ion hits.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    config = data.config

    # --- Compute per-shot quantities (preserve train × bunch shape) ------
    tofs_e_src = data.tofs_e if config == 1 else data.liq_tofs_e
    e_counts = count_hits(tofs_e_src,
                          *(e_tof_range or (None, None)))  # (n, m)

    if config == 1:
        i_counts = count_hits(data.tofs_i,
                              *(i_tof_range or (None, None)))  # (n, m)

    gmd = data.gmd  # (n, m)

    # --- Flatten to shots ------------------------------------------------
    gmd_flat     = gmd.reshape(-1)
    e_counts_flat = e_counts.reshape(-1)
    bunch_idx    = np.tile(np.arange(data.n_bunches), data.n_trains)

    if config == 1:
        i_counts_flat = i_counts.reshape(-1)

    # --- Build GMD bin edges ---------------------------------------------
    if gmd_edges is None:
        gmd_edges = np.linspace(np.nanmin(gmd_flat), np.nanmax(gmd_flat),
                                n_gmd_bins + 1)

    gmd_cents, bool_ar = arb_bool_ar(gmd_edges, gmd_flat)

    # --- Bin counts vs GMD -----------------------------------------------
    e_binned, e_std, e_n = bin_and_average(bool_ar, e_counts_flat)
    gmd_binned, gmd_std, _ = bin_and_average(bool_ar, gmd_flat)

    if config == 1:
        i_binned, i_std, _ = bin_and_average(bool_ar, i_counts_flat)

    # --- Bunch-index dependence ------------------------------------------
    # For each bunch index b, bin the shots at that bunch index by GMD
    n_b = data.n_bunches
    e_by_bunch = np.full((n_b, len(gmd_cents)), np.nan)
    if config == 1:
        i_by_bunch = np.full((n_b, len(gmd_cents)), np.nan)

    for b in range(n_b):
        b_mask = bunch_idx == b
        sub_gmd   = gmd_flat[b_mask]
        sub_e     = e_counts_flat[b_mask]
        _, b_bool = arb_bool_ar(gmd_edges, sub_gmd)
        e_avg, _, _ = bin_and_average(b_bool, sub_e)
        e_by_bunch[b] = e_avg
        if config == 1:
            sub_i = i_counts_flat[b_mask]
            i_avg, _, _ = bin_and_average(b_bool, sub_i)
            i_by_bunch[b] = i_avg

    # --- Plotting --------------------------------------------------------
    ncols = 2
    nrows = 2 if config == 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Panel (0,0): electron counts vs GMD
    ax = axes[0, 0]
    ax.errorbar(gmd_binned, e_binned, yerr=e_std, fmt="o-",
                color="steelblue", capsize=3, label="Mean ± std")
    # Overlay linear fit
    finite = np.isfinite(gmd_binned) & np.isfinite(e_binned)
    if finite.sum() > 1:
        coeffs = np.polyfit(gmd_binned[finite], e_binned[finite], 1)
        fit_x = np.linspace(gmd_binned[finite].min(), gmd_binned[finite].max(), 200)
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "--", color="navy",
                label=f"Linear fit (slope={coeffs[0]:.3f})")
    ax.set_xlabel("Pulse energy (µJ)")
    ax.set_ylabel("Mean electron hits")
    ax.set_title("Linearity: GMD vs electrons")
    ax.legend()

    # Panel (0,1): false-colour bunch index vs electron counts per GMD bin
    ax = axes[0, 1]
    im = ax.pcolormesh(
        gmd_cents, np.arange(n_b), e_by_bunch,
        cmap="viridis", shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Mean electron hits")
    ax.set_xlabel("Pulse energy bin centre (µJ)")
    ax.set_ylabel("Bunch index")
    ax.set_title("Bunch-index independence (electrons)")

    if config == 1:
        # Panel (1,0): ion counts vs GMD
        ax = axes[1, 0]
        ax.errorbar(gmd_binned, i_binned, yerr=i_std, fmt="o-",
                    color="tomato", capsize=3, label="Mean ± std")
        finite_i = np.isfinite(gmd_binned) & np.isfinite(i_binned)
        if finite_i.sum() > 1:
            coeffs_i = np.polyfit(gmd_binned[finite_i], i_binned[finite_i], 1)
            fit_x = np.linspace(gmd_binned[finite_i].min(), gmd_binned[finite_i].max(), 200)
            ax.plot(fit_x, np.polyval(coeffs_i, fit_x), "--", color="darkred",
                    label=f"Linear fit (slope={coeffs_i[0]:.3f})")
        ax.set_xlabel("Pulse energy (µJ)")
        ax.set_ylabel("Mean ion hits")
        ax.set_title("Linearity: GMD vs ions")
        ax.legend()

        # Panel (1,1): false-colour bunch index vs ion counts per GMD bin
        ax = axes[1, 1]
        im = ax.pcolormesh(
            gmd_cents, np.arange(n_b), i_by_bunch,
            cmap="viridis", shading="auto",
        )
        fig.colorbar(im, ax=ax, label="Mean ion hits")
        ax.set_xlabel("Pulse energy bin centre (µJ)")
        ax.set_ylabel("Bunch index")
        ax.set_title("Bunch-index independence (ions)")

    fig.suptitle(f"Linearity — Config {config}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ===========================================================================
# 3. ENERGY-BINNED 2D MAPS
# ===========================================================================

def plot_energy_binned_maps(
    data: ExperimentData,
    mpe_edges: np.ndarray,
    e_tof_edges: np.ndarray,
    i_tof_edges: Optional[np.ndarray] = None,
    bunch_start: int = 0,
    bunch_end: Optional[int] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Bin data by photon energy (mpe) and plot 2D maps of spectra.

    For each photon energy bin the following are shown (raw and normalised):
      - Photon energy vs x-ray spectrum (config 2 only)
      - Photon energy vs electron TOF spectrum
      - Photon energy vs ion TOF spectrum (config 1 only)

    Normalisation is sum(spectra) / sum(GMD) per bin.

    Parameters
    ----------
    data : ExperimentData
    mpe_edges : np.ndarray
        Bin edges for photon energy (eV).
    e_tof_edges : np.ndarray
        TOF bin edges (ns) for electron spectra.
    i_tof_edges : np.ndarray, optional
        TOF bin edges (ns) for ion spectra (config 1).
    bunch_start : int
        First bunch index to include.
    bunch_end : int, optional
        Last bunch index to include (inclusive). Defaults to last bunch.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    if bunch_end is None:
        bunch_end = data.n_bunches - 1

    flat = select_bunch_range(data, bunch_start, bunch_end)

    mpe_flat = flat["mpe"]   # (N_shots,)
    gmd_flat = flat["gmd"]   # (N_shots,)

    mpe_cents, bool_ar = arb_bool_ar(mpe_edges, mpe_flat)

    # --- Compute single-shot spectra -------------------------------------
    config = data.config

    # Electron spectra
    tofs_e_src = (data.tofs_e if config == 1 else data.liq_tofs_e)
    sl = slice(bunch_start, bunch_end + 1)
    tofs_e_sel = tofs_e_src[:, sl, :]
    e_spectra_3d = hits_to_spectrum(tofs_e_sel, e_tof_edges)       # (n, m_sel, n_e)
    e_spectra = e_spectra_3d.reshape(-1, e_spectra_3d.shape[-1])   # (N_shots, n_e)
    e_cents = (e_tof_edges[:-1] + e_tof_edges[1:]) / 2

    if config == 1 and i_tof_edges is not None:
        tofs_i_sel = data.tofs_i[:, sl, :]
        i_spectra_3d = hits_to_spectrum(tofs_i_sel, i_tof_edges)
        i_spectra = i_spectra_3d.reshape(-1, i_spectra_3d.shape[-1])
        i_cents = (i_tof_edges[:-1] + i_tof_edges[1:]) / 2

    if config == 2:
        vls_sel = data.vls[:, sl, :]
        vls_flat = vls_sel.reshape(-1, vls_sel.shape[-1])           # (N_shots, n_vls)
        vls_cents = np.arange(vls_flat.shape[-1])

    # --- Determine layout ------------------------------------------------
    # Rows: [raw, normalised]  Cols: [e, i/vls]
    has_ion_or_vls = (config == 1 and i_tof_edges is not None) or config == 2
    ncols = 2 if has_ion_or_vls else 1
    nrows = 2  # raw + normalised
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    def _plot_map(ax, x_cents, y_cents, img, title, xlabel, ylabel, cbar_label):
        im = ax.pcolormesh(x_cents, y_cents, img.T, cmap="inferno", shading="auto")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # --- Raw maps --------------------------------------------------------
    e_map, _, _ = bin_and_average(bool_ar, e_spectra)  # (n_mpe_bins, n_e)
    _plot_map(axes[0, 0], mpe_cents, e_cents, e_map,
              "Photon energy vs electron spectrum (raw)",
              "Photon energy (eV)", "Electron TOF (ns)", "Mean counts / shot")

    if has_ion_or_vls:
        if config == 1:
            i_map, _, _ = bin_and_average(bool_ar, i_spectra)
            _plot_map(axes[0, 1], mpe_cents, i_cents, i_map,
                      "Photon energy vs ion spectrum (raw)",
                      "Photon energy (eV)", "Ion TOF (ns)", "Mean counts / shot")
        else:
            vls_map, _, _ = bin_and_average(bool_ar, vls_flat)
            _plot_map(axes[0, 1], mpe_cents, vls_cents, vls_map,
                      "Photon energy vs x-ray spectrum (raw)",
                      "Photon energy (eV)", "Pixel", "Mean intensity / shot")

    # --- Normalised maps (sum(spectrum)/sum(GMD) per bin) ----------------
    e_map_norm, _ = bin_and_sum_ratio(bool_ar, e_spectra, gmd_flat)
    _plot_map(axes[1, 0], mpe_cents, e_cents, e_map_norm,
              "Photon energy vs electron spectrum (GMD-normalised)",
              "Photon energy (eV)", "Electron TOF (ns)", "Counts / µJ")

    if has_ion_or_vls:
        if config == 1:
            i_map_norm, _ = bin_and_sum_ratio(bool_ar, i_spectra, gmd_flat)
            _plot_map(axes[1, 1], mpe_cents, i_cents, i_map_norm,
                      "Photon energy vs ion spectrum (GMD-normalised)",
                      "Photon energy (eV)", "Ion TOF (ns)", "Counts / µJ")
        else:
            vls_map_norm, _ = bin_and_sum_ratio(bool_ar, vls_flat, gmd_flat)
            _plot_map(axes[1, 1], mpe_cents, vls_cents, vls_map_norm,
                      "Photon energy vs x-ray spectrum (GMD-normalised)",
                      "Photon energy (eV)", "Pixel", "Intensity / µJ")

    fig.suptitle(f"Energy-binned maps — Config {config}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ===========================================================================
# 4. DELAY-DEPENDENT PLOTS
# ===========================================================================

def plot_delay_dependence(
    data: ExperimentData,
    mpe_range: Tuple[float, float],
    z_edges: np.ndarray,
    bunch_start: int,
    bunch_end: int,
    e_tof_edges: Optional[np.ndarray] = None,
    i_tof_edges: Optional[np.ndarray] = None,
    e_tof_range: Optional[Tuple[float, float]] = None,
    i_tof_range: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    At a fixed photon energy range, bin by delay stage position and plot
    energy-normalised observables as a function of delay.

    Panels
    ------
    1. Delay vs GMD / electron counts  (electrons per µJ)
    2. Delay vs GMD / ion counts       (ions per µJ, config 1 only)
    3. Delay vs log10(GMD / integrated VLS)  (config 2 only)

    Error bars are ±1 std (shaded region).

    Parameters
    ----------
    data : ExperimentData
    mpe_range : (float, float)
        Photon energy selection window (eV): (mpe_min, mpe_max).
    z_edges : np.ndarray
        Bin edges for the delay stage position.
    bunch_start : int
        First bunch index to include.
    bunch_end : int
        Last bunch index to include (inclusive).
    e_tof_edges : np.ndarray, optional
        TOF bin edges for electron spectra (needed if e_tof_range is None
        and you want range-based counting).
    i_tof_edges : np.ndarray, optional
        TOF bin edges for ion spectra.
    e_tof_range : (float, float), optional
        TOF range (ns) for counting electrons.
    i_tof_range : (float, float), optional
        TOF range (ns) for counting ions.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    config = data.config

    # --- Flatten selected bunch range ------------------------------------
    flat = select_bunch_range(data, bunch_start, bunch_end)

    mpe_flat = flat["mpe"]
    gmd_flat = flat["gmd"]
    z_flat   = flat["z"]

    # --- Photon energy filter --------------------------------------------
    mpe_min, mpe_max = mpe_range
    mpe_mask = (mpe_flat >= mpe_min) & (mpe_flat <= mpe_max)

    gmd_sel = gmd_flat[mpe_mask]
    z_sel   = z_flat[mpe_mask]

    # --- Electron counts -------------------------------------------------
    tofs_e_src = data.tofs_e if config == 1 else data.liq_tofs_e
    sl = slice(bunch_start, bunch_end + 1)
    tofs_e_flat = tofs_e_src[:, sl, :].reshape(-1, tofs_e_src.shape[-1])
    tofs_e_sel  = tofs_e_flat[mpe_mask]

    e_min = e_tof_range[0] if e_tof_range else None
    e_max = e_tof_range[1] if e_tof_range else None
    # count_hits expects (n, m, hits) — add a dummy train axis
    e_counts_sel = count_hits(tofs_e_sel[:, np.newaxis, :], e_min, e_max).reshape(-1)

    # --- Ion counts (config 1) -------------------------------------------
    if config == 1:
        tofs_i_flat = data.tofs_i[:, sl, :].reshape(-1, data.tofs_i.shape[-1])
        tofs_i_sel  = tofs_i_flat[mpe_mask]
        i_min = i_tof_range[0] if i_tof_range else None
        i_max = i_tof_range[1] if i_tof_range else None
        i_counts_sel = count_hits(tofs_i_sel[:, np.newaxis, :], i_min, i_max).reshape(-1)

    # --- VLS integrated intensity (config 2) ----------------------------
    if config == 2:
        vls_flat_all = data.vls[:, sl, :].reshape(-1, data.vls.shape[-1])
        vls_sel      = vls_flat_all[mpe_mask]
        vls_int_sel  = np.nansum(vls_sel, axis=-1)  # integrated spectrum per shot

    # --- Bin by delay stage ----------------------------------------------
    z_cents, bool_ar = arb_bool_ar(z_edges, z_sel)

    # Ratio: GMD / e_counts  (shots per µJ → µJ per electron)
    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        e_ratio = np.where(e_counts_sel > 0, gmd_sel / e_counts_sel, np.nan)

    e_ratio_binned, e_ratio_std, e_n = bin_and_average(bool_ar, e_ratio)

    if config == 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            i_ratio = np.where(i_counts_sel > 0, gmd_sel / i_counts_sel, np.nan)
        i_ratio_binned, i_ratio_std, _ = bin_and_average(bool_ar, i_ratio)

    if config == 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ratio = np.where(
                vls_int_sel > 0,
                np.log10(gmd_sel / vls_int_sel),
                np.nan,
            )
        log_ratio_binned, log_ratio_std, _ = bin_and_average(bool_ar, log_ratio)

    # --- Plotting --------------------------------------------------------
    n_panels = 2 if config == 1 else 2  # e + (i or log_vls)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))

    def _delay_plot(ax, z_cents, y, y_std, n, ylabel, title, color):
        ax.plot(z_cents, y, "o-", color=color)
        ax.fill_between(z_cents, y - y_std, y + y_std, alpha=0.3, color=color)
        ax.set_xlabel("Delay stage position (arb.)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Annotate with shot counts
        for xc, yc, ni in zip(z_cents, y, n):
            if np.isfinite(yc):
                ax.annotate(f"n={ni}", (xc, yc), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7, color="grey")

    _delay_plot(
        axes[0], z_cents, e_ratio_binned, e_ratio_std, e_n,
        ylabel="µJ / electron hit",
        title=f"Delay vs GMD/electrons\nmpe ∈ [{mpe_min:.1f}, {mpe_max:.1f}] eV",
        color="steelblue",
    )

    if config == 1:
        _delay_plot(
            axes[1], z_cents, i_ratio_binned, i_ratio_std, e_n,
            ylabel="µJ / ion hit",
            title=f"Delay vs GMD/ions\nmpe ∈ [{mpe_min:.1f}, {mpe_max:.1f}] eV",
            color="tomato",
        )
    else:
        _delay_plot(
            axes[1], z_cents, log_ratio_binned, log_ratio_std, e_n,
            ylabel=r"$\log_{10}$(GMD / integrated VLS)",
            title=f"Delay vs log₁₀(GMD/VLS)\nmpe ∈ [{mpe_min:.1f}, {mpe_max:.1f}] eV",
            color="mediumseagreen",
        )

    fig.suptitle(
        f"Delay dependence — Config {config}  |  "
        f"Bunches {bunch_start}–{bunch_end}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes