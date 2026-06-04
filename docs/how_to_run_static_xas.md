# How to run the static XAS pipeline

## Overview

`compute_static_xas.py` processes a raw FLASH H5 photon-energy scan and
saves **per-shot** (VLS, GMD) arrays — no GMD binning. GMD binning and all
plots are done interactively in `xas_static.ipynb`.

---

## 1. Create a config file

Copy an existing config from `analysis/configs/xas_static/` and edit:

```
analysis/configs/xas_static/1mmol_gly_03June_night.py   ← example
```

Required fields:

| Field | Example | Notes |
|---|---|---|
| `RUN_NO` | `58793` | FLASH run number |
| `NOMINAL_ENERGIES` | `np.arange(270, 286, 0.5)` | eV, one per open section |
| `CROP_ROI` | `(450, 650)` | VLS pixel ROI (half-open) |
| `SIGNAL_BUNCH_RANGE` | `(10, 40)` | Bunches with x-rays (half-open) |
| `BG_BUNCH_RANGE` | `(50, 100)` | Bunches without x-rays for per-train baseline |
| `CONFIG` | `2` | Must be 2 |
| `MODE` | `"xas_static"` | Must be `"xas_static"` |

Optional fields (shown with defaults):

```python
MAX_FILES               = None       # limit raw H5 files read
TRAIN_RATE_HZ           = 10.0
TRANSITION_TRIM_SECONDS = 3.0        # trains trimmed around shutter moves
FIRST_SECTION_STATE     = "open"     # "open" | "closed"
SHUTTER_INDEX_PATH      = "/FL2/Beamlines/Fast Shutter/shutter/index"
SHUTTER_VALUE_PATH      = "/FL2/Beamlines/Fast Shutter/shutter/value"
```

---

## 2. Run the script

```bash
# Default output path: 11022188/processed/xas_static/run<RUN_NO>_static_xas.h5
python analysis/scripts/compute_static_xas.py \
    analysis/configs/xas_static/1mmol_gly_03June_night.py

# Custom output path:
python analysis/scripts/compute_static_xas.py \
    analysis/configs/xas_static/1mmol_gly_03June_night.py \
    -o 11022188/processed/xas_static/run58793_1mmol_night.h5
```

### Output file naming convention

| Convention | Example |
|---|---|
| Default (auto) | `run58793_static_xas.h5` |
| Recommended custom | `run<RUN_NO>_<sample>_<note>.h5` |

All output files go under `11022188/processed/xas_static/`. The directory
is created automatically if it does not exist.

---

## 3. Inspect results in the notebook

Open `analysis/notebooks/xas_static.ipynb` and set:

```python
RUN_NO   = 58793
INPUT_H5 = Path(path_config.COMBINED_DIR).parent / "xas_static" / f"run{RUN_NO}_static_xas.h5"
GMD_EDGES = np.array([0.0, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
```

Run all cells to get:
1. GMD vs mean-VLS correlation (all shots, all energies pooled)
2. Shot count heatmap per (energy, GMD bin)
3. Mean GMD heatmap + per-bin vs energy lines
4. Mean VLS spectra grid
5. XAS(E) per GMD bin
