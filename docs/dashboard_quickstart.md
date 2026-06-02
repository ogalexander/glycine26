# Dashboard Quick Start

This project now uses a dashboard-only workflow (no DB layer).

The dashboard is a thin UI on top of the existing processing code. Core logic
still comes from the existing scripts (especially write_h5.py,
analysis/scripts/data_loading.py, and analysis/scripts/plotting.py).

## Structure

- dashboard/app.py
  - Streamlit entry point and page navigation.
- dashboard/combine.py
  - Callable wrappers for config 1 and config 2 combine steps.
- dashboard/pages/monitor.py
  - Monitors RAW_H5_DIR and LOCAL_DAQ_DIR.
  - Auto-refreshes every 1 minute.
  - Shows lightweight "new/updated file" status.
- dashboard/pages/process.py
  - Config 1 combine form (run number + measurement name + output name).
  - Config 2 combine form (config1 source + VLS source + output name).
- dashboard/pages/plots.py
  - Loads one combined H5 file and runs plot_diagnostics.

## How To Run

1. Install dependencies:

   pip install -r requirements.txt

2. Start the dashboard:

   streamlit run dashboard/app.py

## Page Usage

### 1) Monitor

- Purpose: quickly see if new files appear.
- Refresh: automatic every minute, with optional manual refresh button.

### 2) Process

- Config 1:
  - Choose run number and local DAQ measurement name.
  - Writes combined H5 into COMBINED_DIR.
- Config 2:
  - Uses a config1 combined H5 plus a VLS H5 source.
  - Writes synthetic config2 combined H5 into COMBINED_DIR.

### 3) Plot

- Select a combined H5 file.
- Choose config (1 or 2) and trim/downsample options.
- Click button to run load_data + plot_diagnostics.

## Notes

- No DB initialization is needed.
- Heavy processing and plotting only run when buttons are clicked.
- Paths are resolved through analysis/scripts/config.py.
