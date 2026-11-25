# gas-vls — FLASH beamtime 11024642 (FL24, 2026)

Online and follow-up analysis for a FLASH2 gas-phase beamtime. The setup
is the **gas-phase arm only**: magnetic-bottle electron TOF + ion TOF
(TDC, local DAQ) plus the **Gotthard VLS** X-ray spectrometer, with GMD,
OPIS photon energy, beam position and the fast shutter from the FLASH DAQ.

This branch (`gas-vls`) is forked from `main` of
`github.com/ogalexander/glycine26`, the analysis for the previous
beamtime (11022188, glycine), which had a liquid-jet arm and a gas arm.
The liquid-arm code (config 2) is kept but dormant here.

`HANDOVER.md` (25 Nov 2026) is the orientation document for colleagues
and their assistants: rules, data model, pipeline with Slurm template,
runs, results so far and open issues. Keep it in step with this file
when the state changes substantially.

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
too: `TRAIN_LENGTH = 101` (bunch slots per train) and `VLS_OFFSET = 2`
(Gotthard line `b + 2` → bunch `b`; checked on runs 60545 / 60559 with
100 pulses, `vls_alignment.ipynb`); they are the defaults of
`write_h5.py` and `load_raw_h5` for config 1.
`TDC_CHANNEL_E = 2` / `TDC_CHANNEL_I = 1` map the TDC channels to
`tofs_e` / `tofs_i` (swapped relative to beamtime 11022188).
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
| 60536         | 100                | (n, 100, 1280)        | 0          |
| 60537–60538   | 100                | —                     | 0          |
| 60559         | 100                | (n, 105, 1280) uint16 | 0 (undulator scan, set λ 4.575–4.679 nm) |

- **Pulses per train change between runs.** The combined H5 uses a
  fixed bunch axis (`train_length`, default 101); GMD slots beyond the
  pulses present are NaN.
- **VLS** is present only in some runs. The writer adds it for config 1
  when the run has it, lined up with the bunch axis via `--vls-offset`
  (default `config.VLS_OFFSET` = 2) and cropped to `train_length`.
  With 100 pulses the frame has **105 lines: 0–1 dark, 2–101 = GMD
  bunches 0–99, 102–104 dark**; the GMD ↔ line-integral correlation
  scan peaks sharply at +2 (run 60545 r = 0.95 in
  `inspect_vls_plus_energy_scan`; run 60559 r = 0.87, ≤ 0.12 at every
  other offset, `vls_alignment.ipynb`). The 101-slot crop keeps all lit
  lines, and slot 100 holds dark line 102 (no GMD, no TOF hits) — a
  per-train dark reference. Earlier runs differ (run 60513, 60 pulses:
  lines 0–58 lit, offset ≈ −1), so check runs with other bunch patterns.
  Raw frames: pedestal ≈ 3240 ADU; bad pixels 0–1 (stuck at 51914, the
  highest value seen), 291–292 (high pedestal, 291 noisy and nearly
  blind; found on run 60559), 679–681 (hot), 832–863 (flat ~4790 ADU,
  no noise);
  at ~171 eV the spectrum sits around pixels 510–610, ~20 ADU above the
  dark lines on average.
- **Local DAQ (TDC `.lst`, SDU delay `.txt`)**: so far the files land
  flat in `processed/local_DAQ/` as `<name>_<10-digit start tID>.{lst,txt}`,
  one acquisition per file (several `Argon_*` acquisitions side by side).
  The `{TDC,delay_log}` subfolders in `config.py` don't exist yet;
  the writer falls back to the flat folder. **TDC channels are swapped
  relative to last beamtime: ch 2 = electron, ch 1 = ion**
  (`config.TDC_CHANNEL_E` / `TDC_CHANNEL_I`). Bunch spacing is the same
  as last beamtime (`folding_parameter` 39876.9).
- **Crosstalk**: fast electrons from the ion spectrometer reach the
  electron TOF ~25 ns (250 TDC units) after each ion hit. They arrive
  after the electron spectrum (which ends at ~950 ns), so electron TOF is
  binned only out to 10000 TDC units (1000 ns).
- **Run 60538** (argon, `Argon_2807259185`, first e + i TOF data;
  combined `run60538_Argon_2807259185.h5`, 6057 trains, written with
  `--max-ecounts 120 --max-icounts 50`): GMD median 0.15 µJ; ~4.9 e and
  ~1.4 i hits per shot over the whole bunch period. Electrons sit at
  85–950 ns, ions at 1450–3100 ns (+ a weak light-ion group at
  250–500 ns). Ion yield per µJ falls ~6× over the first ~50 pulses of
  the train, then stays flat, while electrons per µJ stay within ±10%
  (ion detection efficiency varies along the train). 5 burst trains hit
  both hit caps (cut by `MAX_HITS_E/I`). E–i covariance is far from
  rank 1 even after GMD conditioning (`covariance_ee_ei.ipynb`).
  Ion m/q calibration (`inspect_itof_argon.ipynb`): t = −18 ns +
  481.4 ns·√(m/q), rms residual 4 ns over Ar⁺–Ar⁴⁺, N₂⁺, O₂⁺. Each peak
  has a sharp main component and a broader shoulder 2–6 % earlier;
  σ ≈ 35 ns / q. Detected Ar shares ~3 / 82 / 14 / 0.6 % (q = 1–4),
  flat across 0.03–0.5 µJ; every ion (incl. residual-gas N₂⁺) scales as
  G^0.72. Nothing above Ar⁴⁺ (Ar⁵⁺/Ar⁴⁺ < 1.4 %, 95 %).
- **Run 60559** (ethanol undulator scan, `Ethanol_undulator_scan_2807989445`;
  combined `run60559_Ethanol_undulator_scan_2807989445.h5`, 11 486 trains,
  with VLS and `nominal_energy`): 18 steps 265–282 eV in 1 eV (set
  wavelength 4.6786 → 4.3966 nm), ~60 s each; OPIS empty. The set
  wavelength is logged every ~15 s and on each change; the VLS peak
  moves 3–13 trains after the first record of a new value
  (`undulator_scan.ipynb` drops 30 trains after each change). VLS peak
  pixel 579 (265 eV) → 296 (282 eV), linear: E = 299.52 eV − 0.0597
  eV/px × pixel, rms 0.08 eV against the nominal values; a VLS ROI of
  300–640 cuts the top steps, use 200–650. Single-shot centre: σ ≈ 0.32
  eV about nominal for the 67 % of shots whose VLS signal clears 5σ of
  the dark line. ~0.1–0.4 electron and ~0.15 ion hits per shot.
  `photon_energy/` group written into the combined file (67.1 % of
  pulsed shots measured; E = 299.532 − 0.0596952·px, rms 0.080 eV).
  Energy × GMD aggregates `run60559_..._aggregates_energy.h5`
  (`aggregates_run60559_energy.py`; 39 bins of 0.5 eV 264–283 eV × 5 GMD
  percentile bins, edges 0.020 / 0.056 / 0.098 / 0.169 / 0.301 / 3.29 µJ;
  730 053 shots; with VLS `A`… over pixels 250–630, 385 MB, 6.4 min);
  `undulator_scan_aggregates.ipynb` opens it. The upper
  steps (≥ 278 eV) had lower pulse energies, so their top GMD bins hold
  few shots (e.g. 3 at 282.5 eV).
  **Residual argon** in the ethanol runs: with the calibration a slow
  electron line moves with hν at a constant E_B = 248.9–249.6 eV (Ar 2p,
  hν 265–277 eV), the ~100 ns bump is the Ar L₂,₃MM Auger, and an ion
  peak sits at m/q 19.8 (Ar²⁺ 19.98).
- **Run 60560** (ethanol undulator scan, `Ethanol_undulator_scan_2808016357`,
  started 00:33 Nov 25, running for hours; one growing `.lst`, no `.txt`
  until it ends). Written while running with `--live`:
  `run60560_..._live_novls.h5` (whole run to 02:04, no VLS, 53 865
  trains) and `run60560_..._live_seg.h5` (last 2 raw files, with VLS).
  0.5 eV steps from 265 eV, ~10 min (6030 trains) each; every train has
  TDC data; TDC ↔ GMD aligned at zero shift in every 5000-train block
  (r 0.40–0.51 vs ≤ 0.13 at ±1), incl. across the first 16-bit
  train-ID wrap; 42 merged-sweep trains; ~0.3–0.6 e and 0.2–0.3 ion hits
  per shot. VLS offset +2 and 100 lit lines confirmed; within a step the
  VLS centre wanders by ~±0.2 eV over minutes, and 268.5 / 269.0 eV sit
  at pixels 515.8 / 506.9, ~3–4 px below run 60559's positions for the
  same nominal energies. `run60560_..._live_wrap2.h5` (02:04–02:42, no
  VLS) spans the second 16-bit train-ID wrap (2808086528) and the first
  sweep-counter wrap: every train has TDC data and the alignment holds
  (r 0.48 at zero shift vs 0.09 at ±1), which the old unwrap would have
  broken. `run60560_..._live.h5` (whole run to 02:48, 80 766 trains,
  no VLS) carries `photon_energy/` measured from the raw files (88.7 %
  of pulsed shots; E = 297.953 − 0.0570834·px, rms 0.026 eV over the 14
  steps 265–271.5 eV — a different slope from 60559's −0.0597, so each
  run's measured energies sit on its own nominal scale) and the
  aggregates `run60560_..._live_aggregates_energy.h5`
  (`aggregates_run60560_energy.py`, same bins as 60559; GMD percentile
  edges 0.020 / 0.070 / 0.128 / 0.205 / 0.350 / 4.85 µJ — per run, so
  not the same as 60559's; 6.94 M shots, 264–272.5 eV populated).
  Live commands:
  `python write_h5.py Ethanol_undulator_scan_2808016357 1 60560 --live
  [--no-vls | --first-tid T] --max-ecounts 120 --max-icounts 50 -o …`.
  **Final (run complete, 42 raw files, ended 05:47):**
  `run60560_Ethanol_undulator_scan_2808016357.h5` (no VLS, written
  `--live --live-margin 0` for the read-only `.lst` snapshot; 187 322
  trains 2808016360–2808203681, 312 min; 31 nominal steps 265–280 eV;
  stopped by the SDU `.txt` written at the end) with `photon_energy/`
  from the raw files (82.7 % of pulsed shots; E = 298.987 −
  0.0590512·px, rms 0.075 eV over the 31 steps) and
  `run60560_..._aggregates_energy.h5` (14.7 M shots; GMD edges 0.020 /
  0.057 / 0.100 / 0.168 / 0.290 / 4.85 µJ). Commands in
  `aggregates_run60560_energy.py`; job `~/slurm_60560_final.sh` (5.5 min
  on a upex node). The `*_live*` files are superseded partial writes.
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
| `nominal_energy`     | (n,)         | eV    | Nominal photon energy = hc / undulator set wavelength, latest record ≤ train (NaN without one ≤ 300 trains old) |
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
| `nominal_energy` | `/Electron Diagnostic/Undulator setting/set wavelength 1` (nm; event records, not per train — no `/FL2` prefix) |
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
    [--no-sdu] [--live [--live-margin N]] [--first-tid T] [--last-tid T]
```
- **SDU optional** (`--no-sdu`, and automatic when there is no `.txt`):
  the SDU only carries the delay stage, so for runs without a delay
  scan the TDC defines the trains (first TDC train → last raw-H5 train,
  stopping when the TDC ends); `z` / `z_std` NaN, `local_DAQ_running` =
  TDC has the train. Identical output to the SDU path otherwise (checked
  on run 60559).
- **`--live`** for a measurement still being acquired: every `.lst` is
  copied into a private temp folder by `_snapshot_readonly` (source
  opened `O_RDONLY`, exactly the bytes present at that moment) and only
  the copy is decoded; the last `--live-margin` (50) TDC trains are
  dropped as possibly incomplete; raw H5 files modified < 2 min ago are
  skipped (FLASH publishes a raw file only once it is closed, ~9 min
  after it starts). The `.lst` files are writable through the beamtime
  ACL, so never point anything but read-only code at them; the decoder
  itself only uses `open(.., 'rb' / 'r')` and `np.fromfile`.
- `--first-tid` / `--last-tid`: write only part of a measurement (e.g.
  the newest raw files of a long run; earlier raw files are skipped
  cheaply).
- 16-bit TDC counters: `check_and_correct_trainIDs` unwraps sweep# and
  tag bits with `_unwrap16` (one period per wrap). The old loops added
  (i+1)·65536 at the i-th wrap, i.e. wrong from the second wrap on
  (> 65536 trains ≈ 1.8 h in one `.lst`); identical results up to one
  wrap (checked on 60559 / 60560).
- `config=1` (this beamtime): TDC ch `config.TDC_CHANNEL_E` (2) →
  `tofs_e`, ch `config.TDC_CHANNEL_I` (1) → `tofs_i`;
  VLS written automatically when the run's raw H5 has it (`--no-vls` to
  skip). Gotthard line `b + K` → bunch `b` (`K` may be negative).
- `measurement_name` may be a glob (`'scan*'`) to concatenate scans, or
  a full file stem with the 10-digit tID (`Argon_2807259185`) to take
  exactly that acquisition. Files are looked up in `SDU_DIR` / `TDC_DIR`
  (or a per-measurement subfolder), then in the flat `LOCAL_DAQ_DIR`.
- `nominal_energy` is filled after the loop from the set-wavelength
  records of every raw file of the run (`data_loading.read_nominal_energy`).
- Defaults: `train_length` = `config.TRAIN_LENGTH` (101), `--vls-offset`
  = `config.VLS_OFFSET` (2), chunk 1000 (200 with VLS), `max_ecounts`
  50, `max_icounts` 120, `folding_parameter` 39876.9 (TDC units; bunch
  spacing unchanged from last beamtime). Electrons are the busier
  channel at this beamtime, so pass `--max-ecounts 120 --max-icounts 50`
  (as for run 60538).
- Output defaults to `COMBINED_DIR/<measurement>.h5`.

`write_h5_test_config1.py` / `write_h5_test_config2.py` are legacy test
builders from the previous beamtime. The config-1 builder pins that
beamtime's channels (`TDCIterator(..., channel_e=1, channel_i=2)`);
`TDCIterator` / `extract_data_from_single_file` otherwise default to
`config.TDC_CHANNEL_E/I`.

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
  (no TDC/SDU): `gmd`, `mpe`, `shutter`, `nominal_energy`, and `vls`
  when present.
  Config-1 defaults come from `config.TRAIN_LENGTH` / `VLS_OFFSET`;
  `train_step` keeps every N-th train and the VLS is read only for those
  (blockwise), so whole runs fit in memory. Duplicate trains at file
  boundaries are dropped.
- `save_vls_moments(data, filepath, overwrite=False)`.
- `RawVLSReader(run_no, tID, train_length, vls_offset=None)` —
  `.read(rows)` gives the Gotthard frames of those combined-file rows
  from the raw files (by train ID, line `b + vls_offset` → bunch `b`,
  NaN where missing; files < 2 min old skipped); `.coverage`, `.close()`.
- `read_nominal_energy(h5_paths, tID, max_age=300) → (energy, n_records)`
  — hc / `set wavelength 1` of the latest record at or before each
  train; NaN before the first record or when it is > `max_age` trains
  old. Used by the writer and `load_raw_h5`.

### `experiment_data.py` — `ExperimentData`
Fields `config`, `tID`, `gmd`, `mpe`, `z`, `between_tdc_files`; optional
`tofs_e`, `tofs_i`, `liq_tofs_e`, `vls`, `shutter`, `nominal_energy`
(per train; sliced by `filter_trains` / `roll_tofs_trains`); VLS metadata
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
- `config=1`: D = electron TOF, C = ion TOF; with `VLS_CROP_ROI` also
  the VLS: `A`, `AtA`, `AtD`, `AtC` (n_px × n_tof_i), `AtG`, plus
  `vls_pixels` and `background`. Per shot: minus `VLS_BACKGROUND`
  (default `{"type": "auto", "roi": (100, 101)}`: mean of the dark
  slot over every `VLS_BG_STRIDE`-th (10) kept train), minus the median
  over `VLS_CM_PIXELS`, `VLS_BAD_PIXELS` → 0, then cropped. Shots
  without a frame are skipped for every aggregate (`n_skipped_no_vls`).
  `VLS_RAW_RUN = N` reads the frames from run N's raw files through
  `data_loading.RawVLSReader` (for `--no-vls` live files); identical
  aggregates to the combined-file route (checked on 60559).
- Config 1 accumulates from **hit lists** (`_accumulate_cfg1`): D, C,
  their G terms and every hit product (DtD, CtC, DtC) are scatter-adds
  over the hits / hit pairs within a shot (`np.histogram2d` bin rules),
  AtD / AtC sum the VLS rows per (cell, bin) of the hits; only AtA is a
  dense product per cell. Identical D / C / pair sums to the old dense
  loop, VLS terms ≤ 1e-9 relative (60559, 60538); G / GtG now summed in
  float64 (the old loop summed the float32 GMD in float32, ~1e-8).
  `WORKERS = N` (config) or `--workers N` splits the trains over N
  spawned processes (BLAS threads divided between them; the VLS pedestal
  pass is parallel too) and adds their sums. On a 3160-train slice of
  60560 (raw VLS): 206 s → 90 s single process; the remaining cost is
  reading the gzip Gotthard frames (~12 ms / train), which the workers
  share. Long runs: submit to a `upex` node with `sbatch` and
  `--workers` ≈ `nproc` (capped at 48); job scripts in the home
  directory, not the repo: `~/slurm_agg.sh` (aggregates only),
  `~/slurm_60560_final.sh` (write → photon energy → aggregates). The
  account has `upex` / `upex-beamtime`, mostly idle nodes with 40+ cores,
  206+ GB; on a 128-core node (48 workers) the 60560 aggregation took
  85 s (the old code was still running after 34 min on the shared
  display node; estimated > 2 h).
- `config=2`: A = VLS, D = liquid eTOF; `mode="time_resolved"` bins by
  GMD × z.
- `load_aggregates(path) → AggregatesData` (per-bin means, NaN for empty
  bins; `.var_G = GtG − G²`).
- Optional config keys `BUNCH_RANGE = (start, stop)` (half-open bunch
  slice; default all slots) and `MAX_HITS_E` / `MAX_HITS_I` (drop shots
  with more hits over the full TOF range, e.g. burst / cap-clipped
  shots, which otherwise dominate the second moments). Both are recorded
  in the output attrs (`bunch_range`, `max_hits_*`, `n_skipped_max_hits`).
- `MODE = "energy"` (config 1): bins on a per-shot photon energy ×
  GMD, every aggregate with leading axes `(N_E, N_GMD)`; config keys
  `ENERGY_EDGES` and `ENERGY_KEY` (dataset in the input file, default
  `photon_energy/measured`); shots without an energy are skipped
  (`n_skipped_energy_nan`). Output `<input>_aggregates_energy.h5` with
  `energy_edges`; `AggregatesData.mode == "energy"`,
  `.energy_centres`, `.n_energy_bins`.
- GMD edges can be percentiles instead of fixed values:
  `GMD_EDGES = None`, `GMD_PERCENTILE_BINS = k` (edges at 0, 100/k, …,
  100 % of the shots that will be binned, so equal counts overall —
  not per energy bin), optional `GMD_MIN` (noise floor left out).
- CLI: `python compute_aggregates.py CONFIG.py INPUT.h5 [-o OUT.h5]`;
  config modules in `analysis/configs/` (`aggregates_example_cfg1.py`
  is the config-1 template; `aggregates_run60538_argon.py` and
  `_b0-20.py` are the first ones for this beamtime;
  `aggregates_run60559_energy.py` the first energy-mode one).

### `calibration.py`
TOF calibrations (ns), both from argon run 60538: electrons
`ke_of_tof` / `tof_of_ke`, KE = (a/(t − t0))² + E0 with `ETOF` (t0 14.75 ns,
a 1224.7 ns√eV, E0 −0.64 eV; `etof_calibration_argon.ipynb`, all 8
lines); ions `mq_of_tof` / `tof_of_mq`, m/q = ((t − t0)/k)² with `ITOF`
(t0 −18.42 ns, k 481.37 ns; `inspect_itof_argon.ipynb`).
`per_unit(arr, *widths)` divides the trailing axes by bin widths — the
Jacobian for TOF-binned spectra (`dke = |diff(ke_of_tof(tof_edges))|`)
and, with both axes' widths, for second moments / covariances. To bin
directly in energy, histogram on `tof_of_ke(ke_edges)[::-1]` (rising
TOF = falling energy) and reverse; use bins of constant relative width
(≥ several 0.1 ns TDC steps at high energy) to avoid aliasing. Update
`ETOF` / `ITOF` here when the calibrations are redone.

### `photon_energy.py`
Per-shot photon energy from the VLS for undulator scans, stored in the
combined H5 as a `photon_energy/` group (`measured` (n, m) eV, NaN = no
pulse or VLS signal < `min_snr`·σ of the dark line; `vls_centre`,
`vls_window_sum`; calibration `cal_coef` (np.polyval, E(pixel)),
`cal_energies`, `cal_centres`, `cal_residuals`, `step_spectra`,
`pixels`; parameters as attrs). Same estimator as `undulator_scan.ipynb`:
pedestal from the dark slot 100, common mode over 650–800, peak of the
9-px-smoothed spectrum, centre of mass within ±30 px, ROI 200–650, bad
pixels incl. 291–292; calibration = linear fit of each settled nominal
step's mean-spectrum centre against its nominal energy (measured values
are on the undulator scale). `measure_photon_energy(path, **params)`,
`save_photon_energy(path, res, overwrite)`, `load_photon_energy(path)`;
CLI `python photon_energy.py COMBINED.h5 [--raw-run N] [--workers N]
[--overwrite] [--roi LO HI] …`. `workers=N` / `--workers N` splits the
trains over N spawned processes for both passes (dark-slot pedestal,
per-shot centres + per-step spectra) and merges them; the pedestal is
now averaged in float64 (was float32 `nanmean`). With `--raw-run` the frames are read from that run's
raw files (matched by train ID, line `b + VLS_OFFSET` → bunch `b`,
files < 2 min old skipped), so a combined file written `--no-vls` (live
writes of long runs) can be used; on run 60559 it reproduces the
combined-VLS result (same centres and selection; energies within
3·10⁻⁵ eV from summation order).

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
  average TOF spectra, train-averaged TOF × bunch-slot maps
  (unnormalised and `TOF_MAP_NORM` = `"gmd"` / `"area"`),
  GMD-normalised linearity per TOF window, correlation matrix and yield
  drift. TOF in ns (converted on load; electron windows stop at
  1000 ns). Deliberately flat: plain arrays at top level, one explicit
  figure per cell.
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
- `analysis/notebooks/inspect_itof_argon.ipynb` — variant of
  `inspect_tofs` for the argon ion TOF (ns): Gaussian + linear fits to
  Ar⁺–Ar⁴⁺, N₂⁺, O₂⁺ (`CAL_PEAKS`), m/q calibration `tof_of` / `mq_of`,
  spectrum on an m/q axis, GMD-binned shape comparison, background-
  subtracted yields (`YIELD_WINDOWS`) with power-law order, Ar
  charge-state shares and mean charge vs GMD, Ar^q+ search up to
  `Q_MAX` (σ_q = σ₁/q, overlaps flagged), and the same per bunch-slot
  group.
- `analysis/notebooks/covariance_ee_ei.ipynb` — goal 5, from config-1
  aggregates: pools GMD bins and merges TOF bins (exact on aggregates).
  It builds three estimates (`EST`): total, within GMD bins, and partial on
  GMD. Then: e–e / e–i maps with significance, a crosstalk check
  (e–i covariance vs t_electron − t_ion), covariance-filtered spectra
  per TOF window, a rank test (canonical correlations vs the
  (√n_e + √n_i)/√N noise edge, checked with a shifted-train null),
  the residual after the leading component, and the same per GMD bin.
  TOF parameters and plot axes in ns (TDC units × 0.1 on loading).
- `analysis/notebooks/vls_alignment.ipynb` — raw files of one run, all
  Gotthard lines: GMD ↔ line-integral offset scan, lit lines vs the GMD
  train profile, which lines the combined file keeps for
  `config.VLS_OFFSET` / `TRAIN_LENGTH`, and (optional combined file)
  the stored VLS and both TOFs on the bunch axis.
- `analysis/notebooks/etof_calibration_argon.ipynb` — electron TOF →
  KE from run 60538 (hν nominal 275.52 eV): `ASSIGNMENTS` (TOF, line,
  reading uncertainty), weighted least squares of t = t0 + a/√(KE − E0),
  fit plot with residuals, spectrum on the KE axis with the lines.
  Lines and energies follow D. Schwickert's thesis (Hamburg 2022, p. 48,
  ediss 9628; same spectrometer): E_B from its Table 4 (2p 248.4 /
  250.6, 3s 29.3, 3p 15.7 / 15.9), Auger energies at the local peaks of
  its Fig. 35b line sum (L₂,₃M₂,₃M₂,₃ 204.7, L₂,₃M₁M₁ measured 179.1 and
  calculated 147.1 eV; L₂,₃M₁M₂,₃ is only a shoulder, 190.2 eV =
  intensity-weighted centre of its lines 187.5–193.2), read from the
  vector data of the PDF. TOFs measured in the run 60538 spectrum
  (maxima of the 0.1 ns, σ 0.2 ns smoothed histogram; shoulders = minima
  of its second derivative): 3p 90.35, "3s" 93.65 (small max), M₂,₃M₂,₃
  100.15, M₁M₂,₃ 103.05 (shoulder), M₁M₁ 106.35, M₁M₁ calc. ~114.25
  (very weak shoulder), 2p₃/₂ 247.25, 2p₁/₂ 256.75 (shoulder). The fast
  lines alone cannot separate t0 from E0 (E0 ±58 eV); the 2p lines fix
  it. All lines: t0 = 14.8 ± 2.3 ns, L = 0.726 ± 0.022 m,
  E0 = −0.6 ± 1.1 eV, χ² 7.5 / 5, 3s +0.95 ns (−6 eV); without 3s:
  χ² 3.1 / 4, t0 = 13.1 ± 2.5 ns, L = 0.741 ± 0.023 m,
  E0 = −1.4 ± 1.2 eV, residuals ≤ 0.35 ns (≤ 1.5 eV) except the
  calculated M₁M₁ (−1.4 ns). (Thesis, its own setup: L = 0.750 ±
  0.014 m, E0 = 9 ± 6 eV at U_ret = −10 V.) TOFs obtained by converting
  approximate energies with an earlier calibration are not
  measurements — use the spectrum. The ~90 ns peak in the ethanol runs is their fastest
  photoelectrons (it moves between runs), not a photon peak; the ~100 ns
  bump in those runs sits on the Ar Auger.
- `analysis/notebooks/undulator_scan_aggregates.ipynb` — opens
  energy-mode aggregates (`AGG_H5`, now the final run 60560 file) as
  plain arrays (`D`, `C`,
  `DtD`, `CtC`, `DtC`, `G`, `GtG`, `DtG`, `CtG`, `n`; axes `E`,
  `gmd_edges`, `tof_e` / `tof_i` in ns), prints the photon-energy
  calibration and maps the shots per cell. Starting point for
  covariance vs photon energy. Section *XAS maps* (cells `undagg-xas*`):
  one panel per GMD bin, measured photon energy × electron KE, colour
  `D_E` (hits / shot / eV) or, with `XAS_NORM = "gmd"`, divided by each
  cell's mean GMD; `XAS_KE`, `XAS_LOG`, `XAS_SHARED_CLIM`,
  `XAS_MIN_SHOTS` (300). Section *Two-photon resonance* (cells
  `undagg-2ph-*`): looks for a core → valence-hole resonance opened by
  a first (valence-ionising) photon — fast-electron partial-yield XAS
  (`HIGH_KE` 210–285 eV, per µJ per GMD bin and relative to `REF_KE`
  10–40 eV), per-photon-energy fit ⟨N⟩ = a₁⟨G⟩ + a₂⟨G²⟩ (only where
  all GMD bins have ≥ `MIN_SHOTS`), high/low-GMD ratio (top two / bottom
  two bins pooled), KE-resolved ratio map normalised to its median over
  photon energy, partial (on GMD, within cells) e–e pair covariance in
  `HIGH_KE` with pairs < `DEAD_TIME_NS` (3 ns) apart left out, fast
  electrons × ion m/q covariance map, e–e maps at `COMPARE_E` (280 vs
  268 eV). Run 60559: yields per µJ fall ~2× across the GMD bins at all
  photon energies (a₂ < 0 everywhere); a₂/a₁ ≈ −0.3…−0.65 up to 277 eV,
  ≈ −0.9 above 277.5 eV (where the steps had weaker pulses); the pooled
  reference-normalised high/low ratio is 0.95 ± ~0.1 and flat within
  errors; pair covariance ≈ 0 ± 0.1 ⟨N⟩² at high GMD without
  photon-energy structure; fast electrons covary with m/q 1, 13–15,
  27–31, 45–46 at all photon energies. Uses only numpy (the user's
  kernel is xfel 202301, Python 3.9).
- `analysis/notebooks/undulator_scan.ipynb` — undulator-gap scans from a
  combined file with VLS + `nominal_energy`: steps (settling trains
  dropped), streamed VLS (pedestal from dark slot 100, per-shot peak
  centre, dark-noise threshold), pixel → eV fit against the nominal
  energies, scan trace, electron TOF vs nominal and vs measured photon
  energy, average-train maps (slot × e TOF / ion TOF / VLS) per step.
  Now on calibrated axes (`calibration.py`): electron energy (bins of
  constant relative width `KE_REL_BIN` 2 %, `KE_RANGE` 1–290 eV ≈ 970–87
  ns, per eV) and, in the train maps, ion m/q (per u/e). The
  aggregates notebook adds `D_E`, `DtD_E`, `DtG_E`, `C_mq`, `CtC_mq`,
  `CtG_mq`, `DtC_Emq`, `AtD_E`, `AtC_mq` (TOF order kept; `ke`, `mq`,
  `dke`, `dmq`) next to the TOF arrays. The energy-mode configs
  (`aggregates_run6055x_energy.py`) now bin the electrons in energy:
  `TOF_EDGES` = images of 1 eV bins from 140 to 290 eV and 5 % bins
  from 1 to 140 eV, snapped to half TDC ticks (252 bins, 87–970 ns;
  0.8–1.4 eV wide at 150–280 eV, 1–3 ticks each; the Jacobian uses the
  snapped widths). With the old 5 ns bins the energy bins were 22–32 eV
  wide above ~190 eV. Ions keep 10 ns (0.04–0.28 u/e).
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
- **Units**: GMD µJ, photon energy eV. TOF is stored in TDC units of
  100 ps; plots show ns.
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
- [x] Bunch spacing (unchanged) confirmed. Local-DAQ files currently
      flat in `processed/local_DAQ/` (writer falls back to it).
- [x] First TOF run processed and aggregated (60538, argon);
      e–e / e–i covariance notebook (`covariance_ee_ei.ipynb`).
- [x] Run 60559 (ethanol undulator scan) written:
      `run60559_Ethanol_undulator_scan_2807989445.h5` (11 486 trains,
      with VLS and `nominal_energy`, 3.2 GB) via `write_h5.py
      Ethanol_undulator_scan_2807989445 1 60559 --max-ecounts 120
      --max-icounts 50`; `undulator_scan.ipynb` runs on it.
- [x] Per-shot VLS photon energy (`photon_energy.py`) and energy × GMD
      aggregates (`compute_aggregates` `MODE = "energy"`) for 60559;
      `undulator_scan_aggregates.ipynb`.
- [x] VLS line offset 2 and 101 slots confirmed for 100-pulse runs
      (`vls_alignment.ipynb`); `config.VLS_OFFSET` = 2.
- [x] `nominal_energy` (undulator set wavelength) in the combined H5,
      `load_data`, `load_raw_h5`, `ExperimentData`.
- [x] Conditions notebook for goal 1 (`conditions_check.ipynb`); TOF
      sections run on run 60538.
- [x] VLS aggregates for config 1 in `compute_aggregates.py` (`A`,
      `AtA`, `AtD`, `AtC`, `AtG`); hit-list accumulation and `--workers`.
- [x] eTOF / iTOF calibrations (`calibration.py`) applied with Jacobians
      in both undulator notebooks; electron bins defined in energy.
- [x] Run 60560 complete: final combined file, photon energy and
      aggregates (Slurm, 5.5 min).
- [x] XAS maps: vs nominal energy in `undulator_scan.ipynb`, vs measured
      energy per GMD bin in `undulator_scan_aggregates.ipynb`.
- [x] Two-photon resonance section run on 60559 and 60560 (results not
      yet interpreted; see `HANDOVER.md` §7).
- [x] `HANDOVER.md` for colleagues.
- [ ] Delay scans at 268 eV (`Ethanol_delay_scan_short*_268eV`, raw runs
      from 60561): not written yet; config-1 `time_resolved` hit-list
      path unverified.

## Open questions
- Gotthard line ↔ pulse offset for runs with other bunch patterns
  (+2 / 101 slots hold for the 100-pulse runs checked).
- A few trains per run (60538: 5 of 6057; 60559: 20 of 11 486) carry
  about twice the usual hits, and one shot in each holds TOF values
  beyond the bunch period (> 40000 TDC; in 60559 also negative ones,
  stored wrapped near 2³²) — looks like two sweeps merged under one
  train ID, which breaks the sorted-stream assumption in
  `_fold_into_bunches`. Cut trains with any hit > 40000 until the
  writer handles it.
- Max hits per bunch adequate? With `--max-ecounts 120 --max-icounts 50`
  only burst shots reach them in run 60538.
- Source of the ion-efficiency drop along the train (run 60538).
- Why every ion yield (and the electrons) grows as ~G^0.7 rather than
  linearly in run 60538 — common to all species, so GMD scale / offset
  or detector response rather than the ionisation.
- `inspect_tofs` still plots TDC units and defaults to the whole bunch
  (0–40000), so its electron spectra include the crosstalk after
  1000 ns.
- VLS pixel → eV: only relative to the undulator set values so far
  (run 60559 fit); no independent absolute reference. `z` units / sign
  (→ fs) for time-resolved runs.
