"""Page 2 — Process: run the combine pipeline for Config 1 or Config 2."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "dashboard"))
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))

import config  # noqa: E402
from combine import run_aggregates, run_config1, run_config2  # noqa: E402


def _make_progress_logger(status_area):
    """Build a compact progress logger: percentage + short status text only."""
    progress_value = 0
    total_trains_hint: int | None = None

    def _set_progress(value: int, text: str) -> None:
        nonlocal progress_value
        value = max(progress_value, min(100, int(value)))
        if value != progress_value:
            progress_value = value
        status_area.caption(text)

    def log(msg: str) -> None:
        nonlocal total_trains_hint
        msg_str = msg.strip()
        if not msg_str:
            return

        # Most writer loops emit lines like: "tID ... (1234/5678)".
        m = re.search(r"\((\d+)\/(\d+)\)", msg_str)
        if m:
            done = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                pct = min(99, int((done / total) * 100))
                _set_progress(pct, f"Processing... {pct}% ({done}/{total})")
            return

        # Aggregates prints: "trains : X total, ..." and chunk rows like
        # "[   200..   400) trains=...".
        m_total = re.search(r"trains\s*:\s*(\d+)\s+total", msg_str, flags=re.IGNORECASE)
        if m_total:
            total_trains_hint = int(m_total.group(1))
            _set_progress(3, "Initializing... 3%")
            return

        m_chunk = re.search(r"\[\s*\d+\s*\.\.\s*(\d+)\s*\)", msg_str)
        if m_chunk and total_trains_hint and total_trains_hint > 0:
            upto = int(m_chunk.group(1))
            pct = min(99, int((upto / total_trains_hint) * 100))
            _set_progress(pct, f"Processing... {pct}% ({min(upto, total_trains_hint)}/{total_trains_hint})")
            return

        # Coarse phase-based updates for steps that do not expose loop counts.
        low = msg_str.lower()
        if "measurement" in low or "input" in low:
            _set_progress(3, "Initializing... 3%")
        elif "found" in low:
            _set_progress(8, "Scanning inputs... 8%")
        elif "accumulating" in low:
            _set_progress(20, "Accumulating... 20%")
        elif "writing" in low:
            _set_progress(90, "Writing output... 90%")

    return log


def show() -> None:
    st.header("Process")
    st.caption(
        "Combine raw data streams into a single H5 file. "
        "Output is saved to the **combined** folder."
    )

    # ------------------------------------------------------------------ Config 1
    with st.expander("Config 1 — electron + ion TOF", expanded=True):
        st.markdown(
            "Aligns three streams by **train ID**: raw H5 (GMD / MPE / beam position), "
            "SDU `.txt` (delay stage), TDC `.lst` (electron + ion TOF hits)."
        )
        c1_run = st.number_input(
            "Run number", value=48346, step=1, key="c1_run"
        )
        c1_meas = st.text_input(
            "Measurement name (local DAQ subfolder)", value="delay_scan3", key="c1_meas"
        )
        c1_output = st.text_input(
            "Output file name (in combined folder)",
            value=f"combined_run{int(c1_run)}.h5",
            key="c1_output",
        )

        if st.button("Run Config 1 combine", key="c1_run_btn"):
            output_path = config.COMBINED_DIR / c1_output
            status_area = st.empty()
            status_area.caption("Queued... 0%")
            log = _make_progress_logger(status_area)

            with st.spinner("Running…"):
                try:
                    run_config1(
                        run_no=int(c1_run),
                        measurement_name=c1_meas,
                        output_path=output_path,
                        h5_folder=str(config.RAW_H5_DIR),
                        local_daq_folder=str(config.LOCAL_DAQ_DIR / c1_meas),
                        log=log,
                    )
                    status_area.caption("Done 100%")
                    st.success(f"Written: `{output_path}`")
                except Exception as exc:
                    status_area.caption("Failed")
                    st.error(f"Error: {exc}")

    # ------------------------------------------------------------------ Config 2
    with st.expander("Config 2 — liquid-jet eTOF + VLS", expanded=False):
        st.markdown(
            "Runs the real config-2 combine path from `write_h5.py` with the "
            "same train-ID alignment structure as config 1."
        )

        c2_run = st.number_input(
            "Run number", value=54609, step=1, key="c2_run"
        )
        c2_meas = st.text_input(
            "Measurement name (local DAQ subfolder)",
            value="delay_scan3",
            key="c2_meas",
        )
        c2_output = st.text_input(
            "Output file name (in combined folder)",
            value="combined_config2.h5",
            key="c2_output",
        )

        if st.button("Run Config 2 combine", key="c2_run_btn"):
            output_path = config.COMBINED_DIR / c2_output
            status_area2 = st.empty()
            status_area2.caption("Queued... 0%")
            log2 = _make_progress_logger(status_area2)

            with st.spinner("Running…"):
                try:
                    run_config2(
                        run_no=int(c2_run),
                        measurement_name=c2_meas,
                        output_path=output_path,
                        log=log2,
                    )
                    status_area2.caption("Done 100%")
                    st.success(f"Written: `{output_path}`")
                except Exception as exc:
                    status_area2.caption("Failed")
                    st.error(f"Error: {exc}")

    # ------------------------------------------------------------------ Aggregates
    with st.expander("Aggregates", expanded=False):
        st.markdown(
            "Runs the existing `analysis/scripts/compute_aggregates.py` pipeline "
            "using preset config files under `analysis/configs`."
        )

        agg_presets = {
            "Config 1 (Spectral)": _REPO_ROOT / "analysis" / "configs" / "aggregates_example_cfg1.py",
            "Config 2 (Spectral)": _REPO_ROOT / "analysis" / "configs" / "aggregates_example.py",
            "Config 2 (Time-Resolved)": _REPO_ROOT / "analysis" / "configs" / "aggregates_example_tr.py",
        }
        agg_preset = st.selectbox(
            "Aggregation preset",
            options=list(agg_presets.keys()),
            key="agg_preset",
        )

        combined_files = (
            sorted(config.COMBINED_DIR.glob("*.h5")) if config.COMBINED_DIR.exists() else []
        )
        combined_names = [p.name for p in combined_files]

        if not combined_names:
            st.info("No combined H5 files found. Run Config 1/2 combine first.")
        else:
            agg_input_name = st.selectbox(
                "Input combined H5",
                options=combined_names,
                key="agg_input_name",
            )

            suffix = "_aggregates_tr.h5" if "Time-Resolved" in agg_preset else "_aggregates.h5"
            default_out = f"{Path(agg_input_name).stem}{suffix}"
            agg_output_name = st.text_input(
                "Output file name (in combined folder)",
                value=default_out,
                key="agg_output_name",
            )

            if st.button("Run Aggregates", key="agg_run_btn"):
                agg_input = config.COMBINED_DIR / agg_input_name
                agg_output = config.COMBINED_DIR / agg_output_name
                agg_cfg = agg_presets[agg_preset]

                status_area3 = st.empty()
                status_area3.caption("Queued... 0%")
                log3 = _make_progress_logger(status_area3)

                with st.spinner("Running…"):
                    try:
                        run_aggregates(
                            input_h5=agg_input,
                            config_path=agg_cfg,
                            output_path=agg_output,
                            log=log3,
                        )
                        status_area3.caption("Done 100%")
                        st.success(f"Written: `{agg_output}`")
                    except Exception as exc:
                        status_area3.caption("Failed")
                        st.error(f"Error: {exc}")
