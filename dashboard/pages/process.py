"""Page 2 — Process: run the combine pipeline for Config 1 or Config 2."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "dashboard"))
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))

import config  # noqa: E402
from combine import run_config1, run_config2  # noqa: E402


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
            log_area = st.empty()
            lines: list[str] = []

            def log(msg: str) -> None:
                lines.append(msg)
                log_area.code("\n".join(lines))

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
                    st.success(f"Written: `{output_path}`")
                except Exception as exc:
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
            log_area2 = st.empty()
            lines2: list[str] = []

            def log2(msg: str) -> None:
                lines2.append(msg)
                log_area2.code("\n".join(lines2))

            with st.spinner("Running…"):
                try:
                    run_config2(
                        run_no=int(c2_run),
                        measurement_name=c2_meas,
                        output_path=output_path,
                        log=log2,
                    )
                    st.success(f"Written: `{output_path}`")
                except Exception as exc:
                    st.error(f"Error: {exc}")
