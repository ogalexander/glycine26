"""
Static XAS Dashboard — load processed H5, add plot cards, compare.

Run from the repo root:
    streamlit run dashboard/xas_static/app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                    # _data, _plots
sys.path.insert(0, str(_HERE.parents[1] / "analysis" / "scripts"))  # config

import streamlit.components.v1 as _components   # noqa: E402 (unused, ensures plotly registered)
from _data import bin_by_gmd, load_h5          # noqa: E402
from _plots import (                            # noqa: E402
    plot_correlation,
    plot_mean_gmd,
    plot_shot_counts,
    plot_vls_spectra,
    plot_xas,
)
import config as path_config                    # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Static XAS", layout="wide")
st.title("Static XAS Dashboard")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "cards" not in st.session_state:
    st.session_state.cards = []
if "loaded_path" not in st.session_state:
    st.session_state.loaded_path = None        # path that was explicitly loaded
if "applied_edges" not in st.session_state:
    st.session_state.applied_edges = None      # edges that were explicitly applied

# ---------------------------------------------------------------------------
# Plot registry: label → (type_key, render_fn, has_params)
# ---------------------------------------------------------------------------
PLOTS = {
    "Correlation (GMD vs VLS)": "correlation",
    "Shot counts":              "shot_counts",
    "Mean GMD":                 "mean_gmd",
    "VLS spectra":              "vls_spectra",
    "XAS(E)":                   "xas",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_XAS_DIR = Path(path_config.COMBINED_DIR).parent / "xas_static"

with st.sidebar:
    st.header("Data")
    run_no = st.number_input("Run number", min_value=1, step=1, value=None,
                             placeholder="e.g. 58793")
    _h5 = str(_XAS_DIR / f"run{int(run_no)}_static_xas.h5") if run_no else ""
    if st.button("Load data", use_container_width=True, type="primary"):
        _p = Path(_h5) if _h5 else None
        if not _p:
            st.error("Enter a run number first.")
        elif not _p.exists():
            st.error(f"File not found: `{_h5}`")
        elif _p.is_dir():
            st.error(f"`{_h5}` is a directory.")
        else:
            st.session_state.loaded_path = _h5
            st.session_state.applied_edges = None
            st.session_state.cards = []
            st.rerun()

    st.divider()
    st.header("GMD bin edges (µJ)")
    gmd_str = st.text_input(
        "Edges",
        value="0.0, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8",
        placeholder="e.g. 0.0, 0.5, 1.0, 2.0",
        help="Comma-separated numbers, like a Python list.",
    )
    if st.button("Apply binning", use_container_width=True):
        try:
            _edges = tuple(float(x.strip()) for x in gmd_str.split(",") if x.strip())
            if len(_edges) < 2:
                raise ValueError
            st.session_state.applied_edges = _edges
            st.rerun()
        except ValueError:
            st.error("Enter at least 2 valid numbers separated by commas.")

    st.divider()
    st.header("Add card")
    plot_label = st.selectbox("Plot type", list(PLOTS.keys()))
    if st.button("＋ Add card", use_container_width=True):
        st.session_state.cards.append(
            {"id": uuid.uuid4().hex[:8], "type": PLOTS[plot_label], "label": plot_label}
        )
        st.rerun()
    if st.button("Clear all", use_container_width=True):
        st.session_state.cards = []
        st.rerun()

# ---------------------------------------------------------------------------
# Gate: wait for explicit Load + Apply
# ---------------------------------------------------------------------------
h5_path = st.session_state.loaded_path
if not h5_path:
    st.info("Enter a run number and click **Load data** to begin.")
    st.stop()

gmd_edges_tuple = st.session_state.applied_edges
if gmd_edges_tuple is None:
    with st.spinner("Loading data…"):
        vls, gmd, n_shots, nominal_energies, vls_pixels, attrs = load_h5(h5_path)
    N_E, _, n_pixels = vls.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run", str(attrs.get("run_no", "—")))
    c2.metric("Energies", N_E)
    c3.metric("Total shots", f"{int(n_shots.sum()):,}")
    c4.metric("VLS pixels", n_pixels)
    st.info("Set GMD bin edges in the sidebar and click **Apply binning**.")
    st.stop()

# ---------------------------------------------------------------------------
# Load + bin (both @st.cache_data — only recomputes on new path / new edges)
# ---------------------------------------------------------------------------
with st.spinner("Loading data…"):
    vls, gmd, n_shots, nominal_energies, vls_pixels, attrs = load_h5(h5_path)
with st.spinner("Binning by GMD…"):
    A_mean, G_mean, n_per_bin = bin_by_gmd(h5_path, gmd_edges_tuple)

N_E, _, n_pixels = vls.shape
N_GMD = len(gmd_edges_tuple) - 1
gmd_edges_arr = np.array(gmd_edges_tuple)

# ---------------------------------------------------------------------------
# Cached figure builders  (keyed on hashable args — avoid re-building on rerun)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _fig_correlation(path: str, n_bins: int) -> go.Figure:
    v, g, ns, *_ = load_h5(path)
    return plot_correlation(v, g, ns, n_bins=n_bins)

@st.cache_data(show_spinner=False)
def _fig_shot_counts(path: str, edges: tuple) -> go.Figure:
    _, _, _, energies, _, _ = load_h5(path)
    _, _, npb = bin_by_gmd(path, edges)
    return plot_shot_counts(npb, energies, np.array(edges))

@st.cache_data(show_spinner=False)
def _fig_mean_gmd(path: str, edges: tuple) -> go.Figure:
    _, _, _, energies, _, _ = load_h5(path)
    am, gm, npb = bin_by_gmd(path, edges)
    return plot_mean_gmd(gm, npb, energies, np.array(edges))

@st.cache_data(show_spinner=False)
def _fig_vls_spectra(path: str, edges: tuple, energy_idx: int) -> go.Figure:
    _, _, _, energies, pixels, _ = load_h5(path)
    am, _, _ = bin_by_gmd(path, edges)
    return plot_vls_spectra(am, energies, pixels, np.array(edges), energy_idx=energy_idx)

@st.cache_data(show_spinner=False)
def _fig_xas(path: str, edges: tuple) -> go.Figure:
    _, _, _, energies, _, _ = load_h5(path)
    am, gm, _ = bin_by_gmd(path, edges)
    return plot_xas(am, gm, energies, np.array(edges))

# ---------------------------------------------------------------------------
# Info bar
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Run", str(attrs.get("run_no", "—")))
c2.metric("Energies", N_E)
c3.metric("Total shots", f"{int(n_shots.sum()):,}")
c4.metric("GMD bins", N_GMD)
c5.metric("VLS pixels", n_pixels)
st.divider()

# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------
if not st.session_state.cards:
    st.info("Use the sidebar to add plot cards.")
    st.stop()


@st.fragment
def _render_card(card: dict) -> None:
    cid = card["id"]
    ptype = card["type"]

    with st.container(border=True):
        title_col, rm_col = st.columns([7, 1])
        title_col.markdown(f"**{card['label']}**")
        if rm_col.button("✕", key=f"rm_{cid}"):
            st.session_state.cards = [
                c for c in st.session_state.cards if c["id"] != cid
            ]
            st.rerun(scope="app")

        # --- per-type params and plot ---
        if ptype == "correlation":
            n_bins = st.slider("Histogram bins", 20, 100, 50, key=f"bins_{cid}")
            fig = _fig_correlation(h5_path, n_bins)

        elif ptype == "shot_counts":
            fig = _fig_shot_counts(h5_path, gmd_edges_tuple)

        elif ptype == "mean_gmd":
            fig = _fig_mean_gmd(h5_path, gmd_edges_tuple)

        elif ptype == "vls_spectra":
            e_idx = st.slider(
                "Energy index", 0, max(N_E - 1, 0), 0, key=f"eidx_{cid}",
                format=f"%d  ({nominal_energies[0]:.1f} eV)",
            )
            st.caption(f"Energy: {nominal_energies[e_idx]:.2f} eV")
            fig = _fig_vls_spectra(h5_path, gmd_edges_tuple, e_idx)

        elif ptype == "xas":
            fig = _fig_xas(h5_path, gmd_edges_tuple)

        else:
            st.warning(f"Unknown plot type: {ptype}")
            return

        st.plotly_chart(fig, width="stretch")


# 2-column grid
cards = st.session_state.cards
left, right = st.columns(2)
for i, card in enumerate(cards):
    with (left if i % 2 == 0 else right):
        _render_card(card)
