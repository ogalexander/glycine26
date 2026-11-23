# gas-vls — FLASH beamtime 11024642 (FL24, 2026)

Online and follow-up analysis for a FLASH2 gas-phase beamtime. The setup
is the **gas-phase arm only**: magnetic-bottle electron TOF + ion TOF
(TDC, local DAQ) plus the **Gotthard VLS** X-ray spectrometer, with GMD,
OPIS photon energy, beam position and the fast shutter from the FLASH DAQ.

This branch (`gas-vls`) is forked from `main` of
`github.com/ogalexander/glycine26`, the analysis for the previous
beamtime (11022188, glycine), which had a liquid-jet arm and a gas arm.
The liquid-arm code (config 2) is kept but dormant here.

---

## Beamtime goals (priority order)

1. **Good conditions.** Establish that data is taken under conditions
   where everything can be normalised and correlated: no saturation of
   the TDC hit finder (hits per bunch hitting `max_ecounts` /
   `max_icounts`) or of the detectors; sane GMD / photon-energy / beam
   position; correlations between electrons, ions, GMD and VLS.
2. **Normalisation.** A normalisation procedure good enough to identify
   small *nonlinear* signals, especially in the eTOF.
3. **Pulse-energy aggregation.** Aggregate in GMD to find nonlinear
   features.
4. **XAS maps.** Combine runs at different photon energy into maps of
   electron and ion spectra vs photon energy — using the nominal or the
   measured photon energy, depending on how good the single-shot photon
   energy metrics (OPIS / VLS) are.
5. **Covariance.** Electron–electron and electron–ion covariance, also
   as a function of photon energy and pulse energy.
6. **Time-dependent XAS**, once good conditions are established.

---

## Environment

### Maxwell (main working copy)
- Clone: `~/gas-vls` on Maxwell (`max-exfl-display`), branch `gas-vls`.
- Beamtime data: symlink `~/gas-vls/11024642 ->
  /asap3/flash/gpfs/fl24/2026/data/11024642` (git-ignored).
- Python: `/software/mamba/2026.05/bin/python3` (Python 3.13, numpy,
  scipy, h5py, pandas 3, matplotlib) — same version as the Jupyter
  kernel. The system `/usr/bin/python3` (3.9) has no pandas.
- TDC decoder (`beamtime_scripts_2021.MCS6A_decoding`) is not in the
  repo; it lives in the beamtime folder at
  `processed/analysis_tools/analysis_tools_2023/decoding_script/` and is
  put on `sys.path` by `write_h5.py` via `config.DECODER_DIR`.
- `processed/` in the beamtime folder is writable; `raw/` is read-only.
  Combined H5 files go to `processed/combined/`.

### Paths — `analysis/scripts/config.py`
The **only** place paths live. The beamtime is set by three constants:
`BEAMTIME_ID = "11024642"`, `BEAMTIME_YEAR = "2026"`,
`BEAMLINE_HALL = "fl24"`. The bunch-axis working assumptions live there
too: `TRAIN_LENGTH = 101` (bunch slots per train) and `VLS_OFFSET = 10`
(Gotthard line `b + 10` → bunch `b`) — both still to be confirmed; they
are the defaults of `write_h5.py` and `load_raw_h5` for config 1.
`FLASH_ENV` picks the data root:
- `local` (default): `<repo root>/11024642` (the symlink on Maxwell, or
  a local copy with the same layout on a workstation).
- `remote`: `/asap3/flash/gpfs/fl24/2026/data/11024642` directly.

Exports `DATA_ROOT`, `RAW_H5_DIR` (`raw/hdf/online-0/fl2user1`),
`LOCAL_DAQ_DIR` (`processed/local_DAQ`), `SDU_DIR` (`…/delay_log`),
`TDC_DIR` (`…/TDC`), `COMBINED_DIR` (`processed/combined`),
`DECODER_DIR`, `ANALYSIS_DIR`.

### Working from the Windows workstation
Maxwell is reached through WSL: `ssh maxwell` (ControlMaster socket; the
user opens it with password + OTP). Nested quoting through PowerShell →
WSL → ssh is fragile: write a script and pipe it
(`wsl -e bash -lc "ssh maxwell bash -s < /mnt/c/.../script.sh"`).

---

## Data — what has been observed so far

Raw files: `FLASH2_USER1_main_run<N>_file<k>_<timestamp>.1.h5` in
`RAW_H5_DIR` (runs from 60493). Observed per run (Nov 2026):

| Runs          | GMD pulses / train | VLS (Gotthard)        | OPIS mpe   |
|---------------|--------------------|-----------------------|------------|
| 60493–60495   | 1 slot, no signal  | (n, 70, 1280) uint16  | ~45.6 eV   |
| 60502–60508   | 29                 | 60502 only (70 lines) | 0 (OPIS off) |
| 60509         | 30                 | —                     | ~170.5 eV  |
| 60510–60514   | 60                 | (n, 100, 1280) uint16 | ~170.8 eV  |
| 60534         | 100                | —                     | ~243.8 eV  |

- **Pulses per train change between runs.** The combined H5 uses a
  fixed bunch axis (`train_length`, default 101); GMD slots beyond the
  pulses present are NaN.
- **VLS** is present only in some runs. The writer adds it for config 1
  when the run has it, lined up with the bunch axis via `--vls-offset`
  (default 10) and cropped to `train_length` (count / offset to be
  confirmed — in run 60513, 60 pulses, Gotthard lines 0–58 are the lit
  ones and the conditions notebook's offset scan peaks at offset ≈ −1).
  Raw frames: pedestal ≈ 3240 ADU; bad pixels 0–1 (stuck at 51914, the
  highest value seen), 679–681 (hot), 832–863 (flat ~4790 ADU, no noise);
  at ~171 eV the spectrum sits around pixels 510–610, ~20 ADU above the
  dark lines on average.
- **Local DAQ (TDC `.lst`, SDU delay `.txt`)**: none copied for this
  beamtime yet; they go to `processed/local_DAQ/{TDC,delay_log}`
  (confirmed). TDC channels as before: ch 1 = electron, ch 2 = ion.
  Bunch spacing is the same as last beamtime (`folding_parameter`
  39876.9).
- Other pulse-resolved raw channels worth knowing: `/Timing/bunch
  pattern`, `/Timing/set number of bunches flash2`, bunch charge and
  toroids (`/FL1/Electron Diagnostic/...`, `/uncategorised/FLASH.DIAG/
  TOROID/...`), GMD tunnel (`.../energy tunnel`).

---

## H5 File Schema (combined files written by `write_h5.py`)

First index = train. `m` = `train_length` (default 101 for config 1).

| Key                  | Shape        | Units | Description                                   |
|----------------------|--------------|-------|-----------------------------------------------|
| `tID`                | (n,)         | —     | Train IDs                                     |
| `gmd`                | (n, m)       | µJ    | Per-pulse energy (NaN beyond pulses present)  |
| `mpe`                | (n,)         | eV    | Per-train mean photon energy (OPIS)           |
| `hor_pos`, `ver_pos` | (n,)         | arb.  | Per-train beam position (GMD hall)            |
| `shutter`            | (n,)         | arb.  | Per-train fast-shutter level (NaN if absent)  |
| `z`, `z_std`         | (n, m)       | arb.  | Delay stage position (SDU) and its std        |
| `local_DAQ_running`  | (n,)         | bool  | SDU had a line for the train                  |
| `between_tdc_files`  | (n,)         | bool  | True = bad/incomplete train                   |
| `tofs_e`             | (n, m, 50)   | TDC   | Electron TOF hits, zero-padded (cfg 1)        |
| `tofs_i`             | (n, m, 120)  | TDC   | Ion TOF hits, zero-padded (cfg 1)             |
| `vls`                | (n, m, 1280) | arb.  | Gotthard spectrum, float32, NaN = no frame; cfg 1 optional (attr `vls_offset`), cfg 2 always |
| `liq_tofs_e`         | (n, m, 50)   | TDC   | Liquid-jet electron TOF (cfg 2, dormant)      |

TOF datasets are omitted when the measurement has no TDC files. The
file holds one row per train from the first to the last SDU train ID,
trimmed to the rows actually written.

### Raw-H5 source paths
| Key       | HDF5 path (`/value`, train IDs in the sibling `/index`)                     |
|-----------|-------------------------------------------------------------------------------|
| `gmd`     | `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall` — `value[:, 0, :]` (index 0 of the 8 per-pulse metadata channels = intensity) |
| `mpe`     | `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy` |
| `hor_pos` | `/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal`  |
| `ver_pos` | `/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical`    |
| `shutter` | `/FL2/Beamlines/Fast Shutter/shutter` — (n, 100), nanmean'd per train        |
| `vls`     | `/FL2/Support Infrastructure/Gotthard/images` — (n, lines, 1280) uint16      |

### Optional `vls_moments` group
`save_vls_moments` appends `vls_moments/{sums,coms,widths}` `(n, m)`
with attrs `crop_roi` / `background_roi`; `load_data` restores it.
Never write it into raw files (read-only).

---

## `write_h5.py` — combined-H5 writer (CLI)

Aligns raw FLASH H5 (GMD, MPE, beam position, shutter, VLS), SDU `.txt`
(z) and TDC `.lst` (TOFs) by train ID.

```
python write_h5.py <measurement_name> <config> <run_no> [-o OUT.h5]
    [--train-length N] [--vls | --no-vls] [--vls-offset K]
    [--data-bunches N] [--chunk-size N] [--max-ecounts N] [--max-icounts N]
    [--n-vls-pixels N] [--folding-parameter F]
```
- `config=1` (this beamtime): TDC ch 1 → `tofs_e`, ch 2 → `tofs_i`;
  VLS written automatically when the run's raw H5 has it (`--no-vls` to
  skip). Gotthard line `b + K` → bunch `b` (`K` may be negative).
- `measurement_name` may be a glob (`'scan*'`) to concatenate scans.
- Defaults: `train_length` = `config.TRAIN_LENGTH` (101), `--vls-offset`
  = `config.VLS_OFFSET` (10), chunk 1000 (200 with VLS), `max_ecounts`
  50, `max_icounts` 120, `folding_parameter` 39876.9 (TDC units; bunch
  spacing unchanged from last beamtime).
- The SDU stream is required (it defines the train range).
- Output defaults to `COMBINED_DIR/<measurement>.h5`.

`write_h5_test_config1.py` / `write_h5_test_config2.py` are legacy test
builders from the previous beamtime.

---

## Analysis Scripts API Summary (`analysis/scripts/`)

### `data_loading.py`
- `load_data(filepath, config, trim_start=2, trim_end=2, downsample_N=1,
  load_vls=True) → ExperimentData` — removes bad trains, trims edges;
  loads `vls` whenever the file has it (pass `load_vls=False` for
  TOF-only work: ~0.5 MB/train). Restores `vls_moments` if present.
- `load_raw_h5(run_no, config, raw_dir=None, train_length=None,
  n_vls_pixels=1280, max_files=None, load_vls=True, vls_offset=None,
  train_step=1) → ExperimentData` — quick look straight from raw files
  (no TDC/SDU): `gmd`, `mpe`, `shutter`, and `vls` when present.
  Config-1 defaults come from `config.TRAIN_LENGTH` / `VLS_OFFSET`;
  `train_step` keeps every N-th train and the VLS is read only for those
  (blockwise), so whole runs fit in memory. Duplicate trains at file
  boundaries are dropped.
- `save_vls_moments(data, filepath, overwrite=False)`.

### `experiment_data.py` — `ExperimentData`
Fields `config`, `tID`, `gmd`, `mpe`, `z`, `between_tdc_files`; optional
`tofs_e`, `tofs_i`, `liq_tofs_e`, `vls`, `shutter`; VLS metadata
(`vls_pixel_ax`, `vls_sums/coms/widths`, `vls_crop_roi`,
`vls_background_roi`); `shot_mask`. Properties `.n_trains`,
`.n_bunches`, `.mpe_broadcast`, `.vls_pixels`, `.valid_shots`,
`.n_valid_shots`. Transforms return a **new** object:
`crop_vls`, `subtract_background`, `auto_subtract_background`,
`auto_subtract_background_trainwise`, `roll_vls_bunches`,
`compute_vls_moments`, `filter_shots` (updates the cumulative
`shot_mask` — the right denominator for averages), `filter_trains`,
`trim_bunches`. VLS methods check `vls is None`, not `config`.

### `binning.py`
`arb_bool_ar`, `bin_bool_ar`, `bin_bool_ar_pc`, `bin_and_average`,
`bin_and_sum_ratio` (energy-normalised `sum(x)/sum(y)` per bin).

### `processing.py`
`count_hits`, `hits_to_spectrum`, `average_spectrum`,
`select_bunch_range`, `crop_to_roi`, `spectral_moments`.

### `plotting.py` (legacy quick-look, still config-1 *or* VLS)
`plot_diagnostics`, `plot_linearity`, `plot_energy_binned_maps`,
`plot_delay_dependence` — all return `(fig, axes)`. They pick
"ion vs VLS" panels by `config`; not yet updated for config 1 + VLS.

### `compute_aggregates.py`
Streams a combined H5 and writes per-GMD-bin means of the matrix
aggregates needed for covariance (`G`, `GtG`, `D`, `DtD`, `DtG`, and per
config `C`/`CtC`/`DtC`/`CtG` or `A`/`AtA`/`AtD`/`AtG`), so full per-shot
matrices never sit in memory.
- `config=1`: D = electron TOF, C = ion TOF. A `vls` dataset in the file
  is accepted but **not yet aggregated** (VLS aggregates for config 1 are
  a TODO).
- `config=2`: A = VLS, D = liquid eTOF; `mode="time_resolved"` bins by
  GMD × z.
- `load_aggregates(path) → AggregatesData` (per-bin means, NaN for empty
  bins; `.var_G = GtG − G²`).
- CLI: `python compute_aggregates.py CONFIG.py INPUT.h5 [-o OUT.h5]`;
  config modules in `analysis/configs/` (`aggregates_example_cfg1.py`
  is the config-1 template).

### Static-XAS scripts (shutter-gated photon-energy scans)
Each **open** shutter section = one nominal photon energy (assigned by
section index, so `NOMINAL_ENERGIES` must follow the scan order);
**closed** sections = background. Detection (in
`compute_static_xas_cfg1.py`): percentile threshold on `/shutter`,
polarity auto-flip from the per-train hit score, trim
`TRANSITION_TRIM_SECONDS × TRAIN_RATE_HZ` trains around each change.
- `compute_static_xas_cfg1.py` — e + i TOF per shot, per section
  (`load_data(..., load_vls=False)`).
- `compute_static_xas.py` — VLS counterpart (written for config 2).
- `compute_xas_aggregates.py` — GMD-binned VLS aggregates per section,
  straight from raw H5.
Config modules: `analysis/configs/xas_static/`, `analysis/configs/aggregates_*xas*.py`.

### `admm_ghost.py`
ADMM spectral-domain ghost-imaging solver (`solve_admm(M, B, ...)` with
`M = Cov(A)`, `B = Cov(A, D)`); was used for VLS × liquid eTOF.

---

## Repository layout notes
- `analysis/notebooks/conditions_check.ipynb` — goal 1, per run or
  combined file: machine stability, bunch pattern (`BUNCH_RANGE` /
  `shot` mask), GMD and OPIS, VLS pedestal / saturation / line-offset
  scan / VLS-vs-GMD / COM jitter, TDC hit-cap and dead-time checks,
  GMD-normalised linearity per TOF window, correlation matrix and yield
  drift. Raw mode (no TOF) until combined files exist. Deliberately flat:
  plain arrays at top level, one explicit figure per cell.
- `analysis/notebooks/inspect_vls.ipynb` — single-shot VLS spectra
  (pedestal + common mode, ROI), per-shot `shots` dict (gmd, vls_sum,
  com, width, mpe, t_min, slot), `FILTERS`, average spectra binned on
  any `shots` key with optional normalisation (`None` / `"gmd"` =
  sum/sum(GMD) / `"area"`); moment check by binning on COM and on width
  (COM-aligned).
- `analysis/notebooks/inspect_tofs.ipynb` — same pattern for `tofs_e` /
  `tofs_i` from a combined file: average TOF spectra per bin (`None` /
  `"shot"` / `"gmd"` / `"area"`) and the ratio of each bin to all shots
  (nonlinear features vs GMD).
- `analysis/notebooks/camera_shutter_difference.ipynb` — this
  beamtime: YAG camera (`/zraw/FLASH.FEL/MULTIPLX1.CAM/YAG/dGroup`)
  shutter-open minus closed, with optional filtering / downbinning.
- Everything else under `analysis/notebooks/`, `analysis/post/`
  (incl. `TASKS.md`), `analysis/configs/` (run-specific files),
  `ML/`, `dashboard/`, `old_dashboard/`, `diagnostics/`, `docs/` is from
  the glycine beamtime — reference material; paths in it point at
  11022188.

---

## Key Conventions
- **Zero-padded TOF arrays**: hits are non-zero; zeros are padding.
  Exclude zeros before histogramming or counting.
- **Normalisation**: energy-normalised averages use `sum(x)/sum(y)` per
  bin, not `mean(x/y)` (`bin_and_sum_ratio`).
- **Bunch selection**: always pass `bunch_start` / `bunch_end`; the good
  bunch range is set per dataset from diagnostics (pulses per train vary
  between runs here).
- **Config** is always passed explicitly (`config=1` for this beamtime),
  never inferred. VLS presence is detected from the data, not from
  `config`.
- **Units**: GMD µJ, photon energy eV, TOF in TDC units (as stored).
- Plot functions return `(fig, axes)`.
- Notebooks: state the question, the runs / files read and the
  parameters in a header cell; resolve paths through `config.py`.

---

## Current status
- [x] Branch `gas-vls` + Maxwell clone; beamtime symlink.
- [x] `config.py` → beamtime 11024642 (single place for ID/year/hall),
      `DECODER_DIR`.
- [x] VLS an optional part of config 1: writer (auto-detect,
      `--vls-offset`, crop to `train_length`), `load_data` /
      `load_raw_h5`, `ExperimentData`, aggregates input check.
- [x] Writer fixes: pandas-3 read-only arrays in the TDC train-ID
      correction; output trimmed to rows written (old resize was a
      no-op; old `finish` dropped the last row); CLI defaults no longer
      override per-config defaults.
- [x] Local-DAQ layout (`processed/local_DAQ/{TDC,delay_log}`) and bunch
      spacing (unchanged) confirmed.
- [ ] Confirm bunch count per train (working value 101) and VLS line
      offset (working value 10).
- [x] Conditions notebook for goal 1 (`conditions_check.ipynb`); TOF
      sections tested on previous-beamtime data only so far.
- [ ] VLS aggregates for config 1 in `compute_aggregates.py`.
- [ ] XAS maps vs photon energy (nominal vs measured).

## Open questions
- Exact bunch count and the Gotthard line ↔ FEL pulse offset (working
  values 101 / 10 in `config.py`).
- Max hits per bunch (`max_ecounts` 50 / `max_icounts` 120) adequate?
- VLS pixel → eV calibration; `z` units / sign (→ fs) for time-resolved runs.
