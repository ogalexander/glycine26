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
  - Loads one combined H5 file and runs plot_diagnostics.

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

- Select a combined H5 file.
- Choose config (1 or 2) and trim/downsample options.
- Click button to run load_data + plot_diagnostics.

## Notes

- No DB initialization is needed.
- Heavy processing and plotting only run when buttons are clicked.
- Paths are resolved through analysis/scripts/config.py.
- On Maxwell, set FLASH_ENV=remote to read cluster folders.
- On Maxwell/JupyterHub, `http://localhost:8501` from Streamlit logs is the
  compute node localhost, not your browser localhost. Always use the JupyterHub
  proxy URL.
