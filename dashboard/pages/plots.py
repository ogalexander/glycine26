"""Page 3 — Plot: load a combined H5 and show diagnostic plots."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))

import config  # noqa: E402
from data_loading import load_data  # noqa: E402
from plotting import plot_diagnostics  # noqa: E402


def show() -> None:
    st.header("Plot")
    st.caption("Load a combined H5 and generate diagnostic plots.")

    # ------------------------------------------------------------------ File picker
    if not config.COMBINED_DIR.exists():
        st.warning(f"Combined folder not found: `{config.COMBINED_DIR}`")
        return

    h5_files = sorted(config.COMBINED_DIR.glob("*.h5"))
    if not h5_files:
        st.info("No combined H5 files found. Run the **Process** step first.")
        return

    selected = st.selectbox(
        "Combined H5 file",
        options=[p.name for p in h5_files],
    )
    selected_path = config.COMBINED_DIR / selected

    # ------------------------------------------------------------------ Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        cfg = st.selectbox("Config", options=[1, 2], index=0)
    with col2:
        trim_start = st.number_input("Trim start (trains)", value=2, min_value=0, step=1)
    with col3:
        trim_end = st.number_input("Trim end (trains)", value=2, min_value=0, step=1)

    downsample = st.number_input(
        "Downsample N (keep 1 in N trains, 1 = no downsample)",
        value=1, min_value=1, step=1,
    )

    # ------------------------------------------------------------------ Plot
    if st.button("Load & plot diagnostics"):
        with st.spinner("Loading data…"):
            try:
                data = load_data(
                    str(selected_path),
                    config=cfg,
                    trim_start=int(trim_start),
                    trim_end=int(trim_end),
                    downsample_N=int(downsample),
                )
            except Exception as exc:
                st.error(f"Failed to load data: {exc}")
                return

        st.success(
            f"Loaded {data.n_trains} trains × {data.n_bunches} bunches."
        )

        with st.spinner("Generating plot…"):
            try:
                fig, _ = plot_diagnostics(data)
                st.pyplot(fig)
            except Exception as exc:
                st.error(f"plot_diagnostics failed: {exc}")
