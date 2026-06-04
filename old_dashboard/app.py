"""
FLASH beamtime analysis dashboard.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pages import monitor, process, plots  # noqa: E402

st.set_page_config(
    page_title="FLASH Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
)

PAGES = {
    "Monitor": monitor,
    "Process": process,
    "Plot": plots,
}

st.sidebar.title("⚡ FLASH Dashboard")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection].show()
