"""Page 1 — Monitor: live view of new files in RAW_H5_DIR and LOCAL_DAQ_DIR."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "scripts"))
import config  # noqa: E402


def _file_table(folder: Path, *, max_rows: int = 200) -> None:
    """Show up to max_rows files in *folder* sorted by modification time."""
    if not folder.exists():
        st.warning(f"Folder not found: `{folder}`")
        return

    entries = []
    for e in os.scandir(folder):
        if e.is_file():
            stat = e.stat()
            mtime = stat.st_mtime
            entries.append(
                {
                    "File": e.name,
                    "Size (MB)": round(stat.st_size / 1e6, 2),
                    "Last modified": datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "_mtime": mtime,
                }
            )

    if not entries:
        st.info("No files found.")
        return

    entries.sort(key=lambda x: x["_mtime"], reverse=True)
    total = len(entries)
    entries = entries[:max_rows]
    display = [{k: v for k, v in e.items() if k != "_mtime"} for e in entries]
    if total > max_rows:
        st.write(f"**{total} file(s)** total — showing newest {max_rows}")
    else:
        st.write(f"**{total} file(s)** — newest first")
    st.dataframe(display, width="stretch")


def _folder_summary(folder: Path, state_key: str) -> None:
    """Show count/latest-file summary and whether something changed since last refresh."""
    if not folder.exists():
        st.warning(f"Folder not found: `{folder}`")
        return

    files = []
    for e in os.scandir(folder):
        if e.is_file():
            stat = e.stat()
            files.append((e.name, stat.st_mtime))

    if not files:
        st.info("No files found.")
        return

    files.sort(key=lambda x: x[1], reverse=True)
    latest_name, latest_mtime = files[0]
    total = len(files)

    st.write(f"Files: **{total}**")
    st.write(
        "Latest: "
        f"**{latest_name}** ({datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
    )

    prev = st.session_state.get(state_key)
    current = (total, latest_mtime)
    if prev is not None and current != prev:
        st.success("New or updated file detected since last refresh.")
    else:
        st.caption("No change since last refresh.")
    st.session_state[state_key] = current


@st.fragment(run_every="60s")
def _auto_monitor() -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw H5")
        st.caption(f"`{config.RAW_H5_DIR}`")
        _folder_summary(config.RAW_H5_DIR, "monitor_raw_summary")
        with st.expander("Show latest raw files", expanded=False):
            _file_table(config.RAW_H5_DIR, max_rows=100)

    with col2:
        st.subheader("Local DAQ")
        st.caption(f"`{config.LOCAL_DAQ_DIR}`")
        if config.LOCAL_DAQ_DIR.exists():
            subfolders = sorted(
                p for p in config.LOCAL_DAQ_DIR.iterdir() if p.is_dir()
            )
            if subfolders:
                selected = st.selectbox(
                    "Measurement folder",
                    options=[p.name for p in subfolders],
                    key="monitor_local_daq_selected",
                )
                selected_path = config.LOCAL_DAQ_DIR / selected
                _folder_summary(selected_path, f"monitor_local_summary_{selected}")
                with st.expander("Show latest local DAQ files", expanded=False):
                    _file_table(selected_path, max_rows=100)
            else:
                _folder_summary(config.LOCAL_DAQ_DIR, "monitor_local_summary_root")
                with st.expander("Show latest local DAQ files", expanded=False):
                    _file_table(config.LOCAL_DAQ_DIR, max_rows=100)
        else:
            st.warning(f"Folder not found: `{config.LOCAL_DAQ_DIR}`")


def show() -> None:
    st.header("Monitor")
    st.caption(f"Environment: `{config.FLASH_ENV}` — auto-refreshes every 1 min")

    if st.button("Refresh now"):
        st.rerun()

    _auto_monitor()
