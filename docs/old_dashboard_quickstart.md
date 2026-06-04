# Dashboard Quick Start

This project now uses a dashboard-only workflow (no DB layer).

The dashboard is a thin UI on top of the existing processing code. Core logic
still comes from the existing scripts (especially write_h5.py,
analysis/scripts/data_loading.py, and analysis/scripts/plotting.py).

## Structure

- dashboard/app.py
  - Streamlit entry point and page navigation.
- dashboard/combine.py
  - Callable wrappers for config 1/config 2 combine + aggregates.
- dashboard/pages/monitor.py
  - Monitors RAW_H5_DIR and LOCAL_DAQ_DIR.
  - Auto-refreshes every 1 minute.
  - Shows lightweight "new/updated file" status.
- dashboard/pages/process.py
  - Config 1 combine form (run number + measurement name + output name).
  - Config 2 combine form (real write_h5 config 2 path).
  - Aggregates form with three presets (cfg1 spectral, cfg2 spectral, cfg2 TR).
- dashboard/pages/plots.py
  - Auto-detects H5 type: combined vs aggregates.
  - Auto-detects config (1 or 2) and aggregates mode (spectral or time_resolved).
  - Supports multi-panel comparison across different source files.

## How To Run

1. Install dependencies:

   pip install -r requirements.txt

2. Start the dashboard (local workstation):

   streamlit run dashboard/app.py

3. Start the dashboard on Maxwell via JupyterHub:

  FLASH_ENV=remote streamlit run dashboard/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.enableCORS false \
    --server.enableXsrfProtection false

4. Open it through the JupyterHub proxy URL (not localhost):

  https://max-jhub.desy.de/user/<your_user>/proxy/8501/

  For user `kaiyu` this is:

  https://max-jhub.desy.de/user/kaiyu/proxy/8501/

  If needed, try:

  https://max-jhub.desy.de/user/<your_user>/proxy/absolute/8501/

## Page Usage

### 1) Monitor

- Purpose: quickly see if new files appear.
- Refresh: automatic every minute, with optional manual refresh button.

### 2) Process

- Config 1:
  - Choose run number and local DAQ measurement name.
  - Writes combined H5 into COMBINED_DIR.
- Config 2:
  - Uses real config 2 combine path from write_h5.py.
  - Choose run number + measurement name.
  - Writes combined H5 into COMBINED_DIR.
- Aggregates:
  - Uses existing compute_aggregates.py with config presets in analysis/configs.
  - Presets: Config 1 (Spectral), Config 2 (Spectral), Config 2 (Time-Resolved).
  - Input: one combined H5 file; Output: aggregates H5 in COMBINED_DIR.

### 3) Plot

- Select any H5 in COMBINED_DIR (combined or aggregates).
- The page auto-detects:
  - Combined vs Aggregates
  - Config 1 vs Config 2
  - Aggregates mode (spectral vs time_resolved)
- Combined H5:
  - Set trim/downsample and load data.
  - Plot types:
    - Diagnostics
    - Linearity
    - Energy-Binned Maps
    - Delay Dependence
    - TOF inspection plots:
      - TOF Mean Spectrum (all shots)
      - TOF Spectrum vs Bunch Index
      - TOF Spectrum vs Train Index
      - TOF Spectrum per GMD Bin
    - Config 2 VLS/Gotthard inspection plots:
      - VLS Mean Spectrum (all shots)
      - VLS Spectrum vs Bunch Index
      - VLS Spectrum vs Train Index
      - GMD vs VLS Intensity
- Aggregates H5:
  - Load aggregates directly (no trim/downsample).
  - Plot types are mode/config aware:
    - Aggregate Means (all supported configs/modes)
    - Cross-Covariance per GMD bin (spectral mode)
    - Time-Resolved slice per GMD bin (config 2 time_resolved)
- Plots remain open when you load another file, so cross-file comparison is preserved.
- Each panel shows its source file, config, and data kind/mode.

## Notebook Mapping (Config + Data Type)

- Combined H5 notebooks:
  - diagnostics.ipynb: config 1 or 2 combined data (load_data)
  - demo.ipynb: config 1 or 2 combined data (load_data)
  - tof_inspect.ipynb: combined data TOF inspection; includes average TOF spectrum, spectrum vs bunch, spectrum vs train, and spectrum per GMD bin
  - vls_inspect.ipynb: config 2 combined data; includes average VLS spectrum, spectrum vs bunch, spectrum vs train, and GMD vs VLS intensity
  - check_train_alignment.ipynb: config 1/2 combined data
- Aggregates H5 notebooks:
  - covariance_inspect_cfg1.ipynb: config 1 aggregates (spectral)
  - covariance_inspect.ipynb: config 2 aggregates (spectral)
  - admm_solve.ipynb: config 2 aggregates (spectral)
  - admm_solve_tr.ipynb: config 2 aggregates (time_resolved)

## Notes

- No DB initialization is needed.
- Heavy processing and plotting only run when buttons are clicked.
- Paths are resolved through analysis/scripts/config.py.
- On Maxwell, set FLASH_ENV=remote to read cluster folders.
- On Maxwell/JupyterHub, `http://localhost:8501` from Streamlit logs is the
  compute node localhost, not your browser localhost. Always use the JupyterHub
  proxy URL.
