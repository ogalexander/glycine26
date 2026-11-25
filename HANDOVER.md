# Handover — `gas-vls` analysis, FLASH2 beamtime 11024642 (Nov 2026)

Written 25 Nov 2026 for a colleague and the AI assistant they work with
(e.g. ChatGPT). It is a snapshot: it tells you where things are, what has
been done, what was found and what is open. For the details of every
function, dataset and run, read **`CLAUDE.md`** at the repo root. It is
the maintained project log and is updated with the code. Where the two
disagree, the code is right, then `CLAUDE.md`, then this file.

Repository: `github.com/ogalexander/glycine26`, branch **`gas-vls`**. The
`main` branch is the previous beamtime (11022188, glycine, liquid jet +
gas). Much of that code and many of its notebooks are still on this
branch as reference material (see §8).

---

## 0. Ground rules (read first)

1. **Raw data are read-only, and must stay that way.** Never write into
   `raw/` (FLASH HDF5) and never write any group or attribute back into a
   raw file.
2. **The TDC `.lst` files in `processed/local_DAQ/` are writable through
   the beamtime ACL, but nothing may ever write to them.** Only point
   read-only code at them. `write_h5.py --live` decodes a private
   read-only snapshot. The decoder itself opens the files `'rb'` / `'r'`.
   Never open them in an editor that might save, and never `touch`,
   move or rename them.
3. **All paths go through `analysis/scripts/config.py`.** Never hard-code
   absolute paths in scripts or notebooks.
4. **Outputs go to `processed/combined/`** in the beamtime folder:
   combined H5 files, `photon_energy/` groups added to them, and aggregate
   files.
5. **Report observations as observations.** Many signals here are small
   and the detectors respond nonlinearly (see §7). Keep "we see X" apart
   from "X means Y", and quote errors and shot counts.
6. **If you change numerical code, compare it with the previous
   output.** Rerun a processed run (60559 is small) and diff the new file
   against the existing one before replacing anything. Every speed-up
   so far was checked this way (§5).
7. **Commit messages:** short, informal, kebab-case subjects
   (`photon-energy-workers-run60560-final`). Keep `CLAUDE.md` current
   when you change behaviour.
8. **Notebooks are edited live in Jupyter on Maxwell.** If someone has a
   notebook open, coordinate before changing the file on disk, or their
   next save overwrites your change.

---

## 1. The experiment in one page

- **Facility / setup:** FLASH2, FL24, beamtime **11024642**, November
  2026. The setup is a gas-phase arm only:
  - magnetic-bottle **electron TOF** and **ion TOF**, read by a TDC
    (MCS6A) on a separate *local DAQ*, which writes `.lst` files (plus
    SDU `.txt` delay-stage logs);
  - **Gotthard VLS** X-ray spectrometer, one spectrum per pulse;
  - **GMD** pulse-resolved pulse energy;
  - **OPIS** photon energy (mostly off);
  - **undulator set wavelength**;
  - beam position and the fast shutter, all from the FLASH DAQ (raw HDF5).
- **Time structure:** trains at 10 Hz with up to 100 pulses per train;
  the number of pulses changes between runs. The analysis uses a fixed
  bunch axis of **101 slots**. Slots 0–99 are the pulses; slot 100 holds
  the VLS dark line (no GMD, no TOF hits).
- **Samples so far:**
  - argon (calibration, run 60538);
  - neon and residual gas (not processed);
  - short ethanol acquisitions (runs 60551–60557);
  - ethanol undulator scans (runs 60559, 60560), with residual argon in
    the chamber;
  - ethanol delay scans at 268 eV, started on the morning of 25 Nov and
    not processed yet (§9).
- **Beamtime goals, in priority order** (from `CLAUDE.md`):
  1. Good conditions: no TDC hit-finder or detector saturation; sane GMD,
     photon energy and pointing; correlations between e, i, GMD and VLS.
  2. A normalisation good enough to see small **nonlinear** signals,
     especially in the eTOF.
  3. Aggregation in pulse energy (GMD) to find nonlinear features.
  4. XAS maps: electron and ion spectra vs photon energy, nominal or
     measured.
  5. Electron–electron and electron–ion covariance vs photon energy and
     pulse energy.
  6. Time-dependent XAS.
- **Physics question currently being pursued:** a **sequential two-photon
  resonance** within one pulse. A first photon ionises the valence shell
  of ethanol, and a second photon promotes a C 1s electron into the
  valence hole. Such a cation resonance is expected below the neutral
  C 1s absorption edge; the scans cover 265–282 eV, and the notebook
  takes 280 eV as "near the expected resonance". It would show up as a
  photon-energy-dependent, superlinear-in-GMD increase of fast electrons
  (resonant Auger or participator), or in the e–e / e–i covariance. The
  notebook section and the results so far are in §6–§7.

---

## 2. Where things live

### Computing
- **Maxwell** (DESY HPC). Interactive work runs on `max-exfl-display`
  nodes. These are shared with other users, so heavy jobs go to Slurm (§5.4).
- **Beamtime folder:** `/asap3/flash/gpfs/fl24/2026/data/11024642`. In a
  clone, make a symlink `<clone>/11024642` pointing to it (git-ignored);
  `config.py` resolves paths through it by default.

  | Subfolder | Contents |
  |-----------|----------|
  | `raw/hdf/online-0/fl2user1/` | FLASH raw HDF5, `FLASH2_USER1_main_run<N>_file<k>_<timestamp>.1.h5`, about 9 min per file; read-only |
  | `processed/local_DAQ/` | TDC `<name>_<10-digit start train ID>.lst`, SDU `.txt` (flat folder, one acquisition per file pair) |
  | `processed/combined/` | combined H5 files and aggregates (ours) |
  | `processed/analysis_tools/analysis_tools_2023/decoding_script/` | TDC decoder `beamtime_scripts_2021.MCS6A_decoding`, not in the repo; `config.DECODER_DIR` puts it on `sys.path` |

- **Python:**
  - Scripts: `/software/mamba/2026.05/bin/python3` (3.13: numpy,
    scipy, h5py, pandas 3, matplotlib; *no seaborn*).
  - The owner's Jupyter kernel for the undulator notebooks is **"xfel
    (202301; Python 3.9)"**, which has seaborn. Keep notebook code
    Python 3.9 compatible: no `match`, no `X | Y` type unions, and no
    numpy/matplotlib features newer than early 2023.
  - The system `/usr/bin/python3` (3.9) lacks pandas.
- **Slurm partitions** used so far: `upex` and `upex-beamtime` (check
  `sinfo` for your account). Most nodes are idle and have 40–128 cores
  and 206+ GB.

### `config.py` (the only place paths and beamtime constants live)
- `BEAMTIME_ID = "11024642"`, `BEAMTIME_YEAR = "2026"`, `BEAMLINE_HALL = "fl24"`.
- `TRAIN_LENGTH = 101`: bunch slots in the combined files.
- `VLS_OFFSET = 2`: Gotthard line `b + 2` → bunch `b` (checked on runs
  60545, 60559, 60560 with 100 pulses).
- `TDC_CHANNEL_E = 2`, `TDC_CHANNEL_I = 1`: swapped relative to the
  glycine beamtime.
- `FLASH_ENV=local` (default) uses `<repo>/11024642`; `FLASH_ENV=remote`
  uses the `/asap3/...` path directly.
- Exports `RAW_H5_DIR`, `LOCAL_DAQ_DIR`, `COMBINED_DIR`, `DECODER_DIR`, etc.

### Code relevant to this beamtime
| Path | What |
|------|------|
| `write_h5.py` | CLI: raw H5 + `.lst` (+ SDU `.txt`) → combined H5, aligned by train ID (§5.1) |
| `analysis/scripts/config.py` | paths and constants |
| `analysis/scripts/data_loading.py` | `load_data`, `load_raw_h5`, `RawVLSReader`, `read_nominal_energy`, `save_vls_moments` |
| `analysis/scripts/experiment_data.py` | `ExperimentData` container with non-mutating transforms |
| `analysis/scripts/photon_energy.py` | per-shot photon energy from the VLS → `photon_energy/` group (§5.2) |
| `analysis/scripts/compute_aggregates.py` | streamed binned moments for covariance (§5.3); `load_aggregates` |
| `analysis/scripts/calibration.py` | eTOF → kinetic energy, iTOF → m/q, Jacobian helper `per_unit` (§4) |
| `analysis/scripts/binning.py`, `processing.py`, `plotting.py` | older helpers (plotting not updated for config 1 + VLS) |
| `analysis/configs/aggregates_run*.py` | one config module per aggregation (bins, VLS crop, caps…) |
| `analysis/notebooks/*.ipynb` | see §6 |

---

## 3. Data model

### 3.1 Raw sources (FLASH HDF5; `/value`, with train IDs in the sibling `/index`)
| Quantity | HDF5 path | Notes |
|----------|-----------|-------|
| GMD per pulse | `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall` → `value[:, 0, :]` | index 0 of 8 per-pulse metadata channels = intensity, µJ |
| OPIS mean photon energy | `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy` | 0 / empty in the scans |
| beam position | `/FL2/Photon Diagnostic/GMD/Average beam position/position hall {horizontal,vertical}` | per train |
| fast shutter | `/FL2/Beamlines/Fast Shutter/shutter` | (n, 100), nanmean'd per train |
| undulator set wavelength | `/Electron Diagnostic/Undulator setting/set wavelength 1` | nm; *event records* (every ~15 s and on change), no `/FL2` prefix |
| Gotthard VLS | `/FL2/Support Infrastructure/Gotthard/images` | (n, lines, 1280) uint16, gzip; ~12 ms/train to read |

**Gotthard frame (100-pulse runs):** 105 lines. Lines 0–1 are dark,
2–101 are bunches 0–99, and 102–104 are dark. The pedestal is about
3240 ADU. Bad pixels:
- 0–1: stuck;
- 291–292: high pedestal; 291 nearly blind;
- 679–681: hot;
- 832–863: flat.

Other bunch patterns (e.g. 60 pulses, run 60513) have a different line
offset; check them with `vls_alignment.ipynb`.

### 3.2 Local DAQ (TDC)
- TOF values are in **TDC units of 100 ps** (10 units = 1 ns). Channel 2
  is electrons and channel 1 is ions. The bunch period ("folding
  parameter") is 39876.9 TDC units.
- Electrons arrive at 85–970 ns, ions at 1450–3100 ns (plus a weak light
  group at 250–500 ns).
- **Crosstalk:** fast electrons from the ion spectrometer reach the eTOF
  ~25 ns after each ion hit, beyond the electron spectrum. The electron
  analysis therefore stops at 1000 ns.
- 16-bit sweep and tag counters wrap (about every 1.8 h). `write_h5.py`
  unwraps them correctly (`_unwrap16`); this was checked across two wraps
  in run 60560.
- **Merged-sweep trains:** a few trains per run carry about twice the
  usual hits, and TOF values beyond the bunch period (> 40000 TDC, some
  stored wrapped near 2³²). The writer does not handle them. Cut trains
  with any hit > 40000, or rely on the `MAX_HITS_*` caps in the
  aggregates.
- The hit caps per bunch in the combined files are set with
  `--max-ecounts 120 --max-icounts 50` at this beamtime (electrons are
  the busier channel). Only burst shots reach them.

### 3.3 Combined H5 (`write_h5.py`; first axis = train, `m` = 101 slots)
| Key | Shape | Units | Notes |
|-----|-------|-------|-------|
| `tID` | (n,) | — | train IDs |
| `gmd` | (n, m) | µJ | NaN beyond the pulses present |
| `mpe` | (n,) | eV | OPIS |
| `hor_pos`, `ver_pos`, `shutter` | (n,) | arb. | |
| `nominal_energy` | (n,) | eV | hc / set wavelength, latest record ≤ train, NaN if > 300 trains old |
| `z`, `z_std` | (n, m) | arb. | delay stage (SDU); NaN without SDU |
| `local_DAQ_running`, `between_tdc_files` | (n,) | bool | `between_tdc_files` = bad / incomplete train |
| `tofs_e` / `tofs_i` | (n, m, cap) | TDC | **zero-padded hit lists**: zeros are padding, exclude them |
| `vls` | (n, m, 1280) | ADU | float32, NaN = no frame; optional (`--no-vls` for long runs) |
| `photon_energy/` (group) | | | added by `photon_energy.py`: `measured` (n, m) eV, `vls_centre`, `vls_window_sum`, calibration `cal_coef` … |

### 3.4 Aggregates (`compute_aggregates.py`, config 1, `MODE = "energy"`)
Every dataset is a **per-cell mean over shots** (empty cells are NaN). A
cell is one photon-energy bin × one GMD bin, so the leading axes are
`(N_E, N_GMD)`.

| Dataset | Shape (after the leading axes) | Meaning |
|---------|-------------------------------|---------|
| `n_per_bin` | — | shots per cell |
| `G`, `GtG` | — | ⟨G⟩, ⟨G²⟩ (µJ, µJ²) |
| `D`, `DtD`, `DtG` | (n_e,), (n_e, n_e), (n_e,) | electron hits per bin per shot; ⟨DDᵀ⟩; ⟨DG⟩ |
| `C`, `CtC`, `CtG` | (n_i,), (n_i, n_i), (n_i,) | the same for ions |
| `DtC` | (n_e, n_i) | ⟨DCᵀ⟩ |
| `A`, `AtA`, `AtD`, `AtC`, `AtG` | (n_px,), (n_px, n_px), (n_px, n_e), (n_px, n_i), (n_px,) | VLS (cropped, background-subtracted) and its products |
| `energy_edges`, `gmd_edges`, `tof_edges`, `ion_tof_edges`, `vls_pixels`, `background` | | axes (TOF edges in TDC units) |

Attributes record the input file, config path, bunch range, caps, skip
counters and the VLS preprocessing. `load_aggregates(path)` returns an
`AggregatesData` object with all of these and with `.energy_centres`,
`.gmd_centres`, `.var_G`, and so on.

**Covariances from aggregates** (within one cell, or after pooling cells
weighted by `n`):
- `Cov(X, Y) = XtY − X Yᵀ`
- `Var(G) = GtG − G²`
- partial covariance on pulse energy:
  `pCov(X, Y; G) = Cov(X, Y) − Cov(X, G) Cov(Y, G)ᵀ / Var(G)`, with
  `Cov(X, G) = XtG − X G`.

When pooling cells, form the moments by weighted averaging *before*
subtracting the products of means; "within-cell" and "total" covariance
differ, and `covariance_ee_ei.ipynb` shows both.

---

## 4. Calibrations

### 4.1 Electron TOF → kinetic energy (`calibration.ETOF`)
- Model: `KE = (a / (t − t0))² + E0`, with `t0 = 14.748 ns`,
  `a = 1224.73 ns·√eV` and `E0 = −0.644 eV`. This corresponds to a flight
  length of 0.726 ± 0.022 m.
- **Source:** argon run 60538 (nominal hν 275.52 eV),
  `etof_calibration_argon.ipynb`. The fit uses 8 lines: Ar 3p and 3s
  photolines, four L₂,₃MM Auger groups and the 2p₃/₂, 2p₁/₂ photolines.
- **Line energies:** from D. Schwickert's PhD thesis (Univ. Hamburg 2022,
  ediss 9628, p. 48: Table 4 and Fig. 35b; same spectrometer type). The
  peak TOFs were measured in the spectrum: maxima, and shoulders as
  minima of the second derivative.
- **Fit quality:** χ² = 7.5 / 5. The 3s line is the outlier (+0.95 ns ≈
  −6 eV); without it χ² = 3.1 / 4 and the parameters move within errors.
  Uncertainties: t0 ± 2.3 ns, E0 ± 1.1 eV. The fast lines alone cannot
  separate t0 from E0; the 2p lines fix them.
- In the fit without 3s, every line lies within 0.35 ns (≤ 1.5 eV) of
  the curve, except the weak calculated M₁M₁ shoulder (−1.4 ns). The
  calibration comes from one run (60538): recheck it if the spectrometer
  settings changed.

### 4.2 Ion TOF → m/q (`calibration.ITOF`)
- Model: `m/q = ((t − t0) / k)²`, with `t0 = −18.42 ns` and
  `k = 481.37 ns`. Fitted to Ar⁺–Ar⁴⁺, N₂⁺ and O₂⁺
  (`inspect_itof_argon.ipynb`); rms residual 4 ns.
- Each peak has a sharp core and a broader shoulder 2–6 % earlier.

### 4.3 Jacobian
TOF-binned counts become counts per eV (or per u/e) by dividing by each
bin's width on the new axis: `calibration.per_unit(arr, *widths)`, with
`dke = |diff(ke_of_tof(tof_edges))|`. Second moments and covariances are
divided by the widths of both axes. The arrays stay in TOF order, i.e.
**energy falls along the electron axis**; reverse them for plots on
rising energy. To bin directly in energy, histogram on
`tof_of_ke(ke_edges)[::-1]` and use bins at least a few TDC ticks
(0.1 ns) wide, or the TDC quantisation aliases into the spectrum. The
energy-mode configs do exactly this (§5.3).

### 4.4 Photon energy (VLS) — per run, relative to the undulator
`photon_energy.py`:
1. Pedestal from dark slot 100, then common mode (median over pixels
   650–800), bad pixels zeroed.
2. Peak of the 9-px-smoothed spectrum, then centre of mass within
   ±30 px (ROI 200–650).
3. A shot is kept if its VLS signal exceeds 5σ of the dark line.
4. **Calibration:** linear fit of each settled nominal step's mean
   spectrum centre against the nominal energy (trains < 30 after a
   set-wavelength change are dropped).

So "measured" energies are on **each run's undulator (nominal) scale**:
- 60559: E = 299.532 − 0.0596952·px (rms 0.080 eV, 18 steps)
- 60560: E = 298.987 − 0.0590512·px (rms 0.075 eV, 31 steps)

The slopes differ by 1 % and the same nominal energy sits 3–4 px apart
in the two runs. Within a step the centre wanders by about ±0.2 eV over
minutes. Single-shot resolution is σ ≈ 0.3 eV. **Do not merge runs on
measured photon energy without cross-calibrating them.** There is no
independent absolute photon-energy reference yet.

---

## 5. Processing pipeline (how each product was made)

### 5.1 Combined H5 — `write_h5.py`
```
python write_h5.py <measurement> 1 <run_no> --max-ecounts 120 --max-icounts 50 [-o OUT.h5]
    [--no-vls] [--no-sdu] [--live [--live-margin N]] [--first-tid T] [--last-tid T]
```
- `<measurement>` is the `.lst` stem with its 10-digit train ID (e.g.
  `Ethanol_undulator_scan_2808016357`), so exactly that acquisition is
  taken. A glob concatenates several.
- Without an SDU `.txt` (no delay scan) the TDC defines the trains
  (automatic, or `--no-sdu`).
- **`--live`** for runs still being acquired:
  - the `.lst` is copied to a private read-only snapshot;
  - the last `--live-margin` (50) TDC trains are dropped;
  - raw files younger than 2 min are skipped.

  It is also fine for finished runs with `--live-margin 0`; run 60560
  was written that way.
- Long runs: write `--no-vls` (VLS is ~0.28 MB/train). `photon_energy.py`
  and `compute_aggregates.py` then read the Gotthard frames straight from
  the raw files.
- Runtime: run 60560 (187k trains) took 4 min on a compute node.

### 5.2 Photon energy — `photon_energy.py`
```
python analysis/scripts/photon_energy.py COMBINED.h5 [--raw-run N] [--workers W] [--overwrite]
```
- Writes the `photon_energy/` group into the combined file.
- `--raw-run N` reads frames from run N's raw files by train ID.
- `--workers` splits the work over processes. Run 60560 took 23 s with
  48 workers; one process was estimated at 15 min or more.
- Parallel and serial results agree to 3·10⁻⁵ eV with an identical shot
  selection (checked on 60559).

### 5.3 Aggregates — `compute_aggregates.py`
```
python analysis/scripts/compute_aggregates.py analysis/configs/<config>.py INPUT.h5 [-o OUT.h5] [--workers W]
```
Output defaults to `<input>_aggregates_energy.h5` in energy mode.

The config module sets (see `aggregates_run60559_energy.py`, the
template for undulator scans):
- `CONFIG = 1`, `MODE = "energy"`.
- `ENERGY_EDGES = np.arange(263.75, 283.26, 0.5)`, i.e. 39 bins of
  0.5 eV, binned on `photon_energy/measured`.
- `GMD_EDGES = None`, `GMD_PERCENTILE_BINS = 5`, `GMD_MIN = 0.02`. The
  GMD edges are **percentiles of each run**, so bins differ between runs.
  Set fixed `GMD_EDGES` to compare runs.
- `TOF_EDGES`: electron bins defined in kinetic energy, 5 % wide from 1
  to 140 eV and 1 eV from 140 to 290 eV. They are mapped to TOF and
  snapped to half TDC ticks (252 bins, 87–970 ns; 0.7–1.4 eV wide at
  high KE).
- `ION_TOF_EDGES`: 10 ns bins, 200–3400 ns.
- `BUNCH_RANGE = (0, 100)`, `MAX_HITS_E = 60`, `MAX_HITS_I = 30`,
  `TRIM_START/END = 3`.
- VLS: `VLS_CROP_ROI = (250, 630)`, `VLS_BACKGROUND` = auto (dark slot
  100), `VLS_CM_PIXELS = (650, 800)`, `VLS_BAD_PIXELS`, and optionally
  `VLS_RAW_RUN = N`.

How it works:
- Config 1 accumulates from **hit lists**: every product involving D or
  C is a scatter-add over the hits or hit pairs actually present, so only
  the dense VLS `AtA` needs matrix products. `--workers` splits trains
  across processes and adds their partial sums.
- **Verification:** D/C sums and pair products are bit-identical to the
  previous dense code; the VLS terms agree to ≤ 1e-9 relative (runs 60538
  and 60559). Only the spectral and energy modes were verified; see §9
  for the time-resolved mode.
- Runtime for run 60560 (14.7 M shots): 73 s on a 128-core node. The
  previous code was estimated at more than 2 h on the shared display
  node.

### 5.4 Running on Slurm
The job scripts used so far live in the owner's home directory, not in
the repo. This is the pattern (a finished undulator-scan run, end to
end):

```bash
#!/bin/bash
#SBATCH --partition=upex
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=undscan
#SBATCH --output=%x_%j.log
set -eo pipefail
cd <your clone>                      # with the 11024642 symlink
P=/software/mamba/2026.05/bin/python3
C=11024642/processed/combined
W=$(( $(nproc) < 48 ? $(nproc) : 48 ))
RUN=60560; MEAS=Ethanol_undulator_scan_2808016357
F=$C/run${RUN}_${MEAS}.h5
$P -u write_h5.py $MEAS 1 $RUN --live --live-margin 0 --no-vls --max-ecounts 120 --max-icounts 50 -o $F
$P analysis/scripts/photon_energy.py --raw-run $RUN --workers $W $F
$P analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run${RUN}_energy.py $F --workers $W
```
Submit with `sbatch job.sh`; it typically starts at once. The whole 60560
chain took 5.5 min, with a peak of about 40 GB RAM at 48 workers.

**Pitfall:** the workers are *spawned* processes, so any script that
uses `--workers` / `workers=` must run from a file with an
`if __name__ == "__main__":` guard. Never pipe it to
`python - <<EOF`: the workers cannot start and the pool raises
`BrokenProcessPool`. Calls from a Jupyter notebook work.

---

## 6. Runs, files and notebooks

### 6.1 Processed runs (`processed/combined/`)
| Run | Acquisition (`.lst` stem) | Content | Combined file | Aggregates |
|-----|---------------------------|---------|---------------|-----------|
| 60538 | `Argon_2807259185` | argon, hν 275.5 eV nominal; calibration run | `run60538_Argon_2807259185.h5` (6057 trains) | `…_aggregates.h5` (GMD bins), `…_aggregates_b0-20.h5` (bunch slots 0–20 only) |
| 60551–60557 | `Ethanol_28078…`–`Ethanol_28079…` | short ethanol acquisitions | `run6055x_Ethanol_*.h5` (3–4 MB each) | `…_aggregates.h5`. Parameters are not recorded in `CLAUDE.md`: read the file attrs or ask the owner |
| 60559 | `Ethanol_undulator_scan_2807989445` | ethanol undulator scan, 18 steps 265–282 eV (1 eV, ~60 s each) | `run60559_…2807989445.h5` (11 486 trains, **with VLS**, 3.2 GB) + `photon_energy/` (67 % of pulsed shots) | `…_aggregates_energy.h5`: 730k shots; GMD edges 0.020/0.056/0.098/0.169/0.301/3.29 µJ |
| 60560 | `Ethanol_undulator_scan_2808016357` | ethanol undulator scan, 31 steps 265–280 eV (0.5 eV, ~10 min each), 312 min | `run60560_…2808016357.h5` (187 322 trains, no VLS, 0.6 GB) + `photon_energy/` from raw files (82.7 %) | `…_aggregates_energy.h5`: 14.7 M shots; GMD edges 0.020/0.057/0.100/0.168/0.290/4.85 µJ |

`run60560_*_live*.h5` (5 files, about 3.4 GB) are **superseded** partial
writes made while 60560 was running. They can be deleted once the owner
agrees.

### 6.2 Raw data not yet processed
- Local-DAQ acquisitions: `Argon_2807246386`, `Argon_2807249362`,
  `Neon+residual_gas_2807204453`, `residual_gas_2807853910`, two tiny
  `Ethanol_28079489*`.
- **`Ethanol_delay_scan_short_268eV_2808213134`** (with an SDU `.txt`,
  i.e. a delay scan) and `Ethanol_delay_scan_short_2_268eV_2808227101`,
  which was still being acquired on the morning of 25 Nov. Their raw
  FLASH runs start at 60561; match them by train ID.
- Raw runs 60493–60550 (apart from 60538) have no combined files. Many
  are setup and alignment runs (pulses per train and VLS presence vary;
  table in `CLAUDE.md`). Run 60545 was used for the VLS line-offset check.

### 6.3 Notebooks for this beamtime (`analysis/notebooks/`)
| Notebook | Purpose | Reads |
|----------|---------|-------|
| `conditions_check.ipynb` | goal 1: stability, bunch pattern, GMD, VLS pedestal / saturation / offset, TDC caps and dead time, TOF maps, linearity, correlations | a run or a combined file |
| `vls_alignment.ipynb` | Gotthard line ↔ pulse offset, lit lines, crop check | raw files (+ combined) |
| `inspect_vls.ipynb`, `inspect_tofs.ipynb`, `inspect_vls_plus_energy_scan.ipynb` | quick per-shot inspection (`inspect_tofs` still in TDC units) | combined / raw |
| `inspect_itof_argon.ipynb` | ion m/q calibration, charge-state shares, yields vs GMD | 60538 |
| `etof_calibration_argon.ipynb` | electron TOF → KE fit (§4.1) | 60538 |
| `covariance_ee_ei.ipynb` | e–e and e–i covariance (total / within-cell / partial), crosstalk check, rank test | config-1 GMD aggregates (60538) |
| `undulator_scan.ipynb` | per-step VLS centres, pixel → eV fit, eTOF vs nominal and measured photon energy, average-train maps per step; calibrated axes | combined file with VLS (60559) |
| `undulator_scan_aggregates.ipynb` | **main analysis notebook**: opens energy-mode aggregates as plain arrays, calibrated axes (`D_E`, `C_mq`, …), **XAS maps** (photon energy × KE per GMD bin), **two-photon resonance** section | `AGG_H5` (currently final 60560) |
| `camera_shutter_difference.ipynb` | YAG camera, shutter open − closed | raw |

Notebook conventions:
- A header cell states the question, the runs or files read, and the
  parameters (`UPPER_CASE` in a parameter cell).
- Code is inline at top level with plain arrays, one figure per cell.
- No "summary for the logbook" cells; the logbook is kept by hand.

---

## 7. Results so far (observations)

**Conditions (run 60538, argon):**
- About 4.9 electron and 1.4 ion hits per shot.
- Ion yield per µJ falls about 6× over the first ~50 pulses of a train,
  then stays flat. Electrons per µJ stay within ±10 %. This is an
  ion-detection effect along the train whose cause is not identified.
- Every ion species, residual N₂⁺ included, and the electrons scale as
  ~G^0.7 with pulse energy. Because all species share it, this points to
  the GMD scale or offset or the detector response rather than the
  ionisation. **This sublinearity is the main systematic for any
  nonlinear-signal claim.**
- Argon charge-state shares (q = 1–4): about 3 / 82 / 14 / 0.6 %, flat
  over 0.03–0.5 µJ.
- The e–i covariance is far from rank 1 even after GMD conditioning.

**Undulator scans (ethanol):**
- About 0.1–0.6 electron and 0.15–0.3 ion hits per shot.
- TDC ↔ GMD alignment holds at zero shift throughout, including across
  counter wraps.
- **Residual argon:**
  - a slow electron line moves with hν at a constant binding energy of
    248.9–249.6 eV (Ar 2p);
  - a bump near 100 ns is the Ar L₂,₃MM Auger;
  - an ion peak at m/q 19.8 is Ar²⁺.
- The fastest photoelectrons (~90 ns) are ethanol valence photolines and
  move with hν.

**Two-photon resonance search (`undulator_scan_aggregates.ipynb`, section
"Two-photon resonance"):**
- The quantity studied is fast electrons in `HIGH_KE` (210–285 eV),
  normalised to a one-photon reference window `REF_KE` (10–40 eV).
- **Fit per photon-energy bin:** ⟨N⟩ = a₁⟨G⟩ + a₂⟨G²⟩, using only cells
  where all GMD bins have ≥ 300 shots.
- **Ratio:** reference-normalised yield per µJ, high-GMD bins over
  low-GMD bins.
- **60559** (730k shots):
  - a₂ < 0 at every photon energy, the yields per µJ falling about 2×
    across the GMD bins;
  - a₂/a₁ ≈ −0.3…−0.65 up to 277 eV and ≈ −0.9 above;
  - pooled high/low ratio 0.95 ± ~0.1, flat;
  - fast e–e pair covariance ≈ 0 ± 0.1 ⟨N⟩² with no photon-energy
    structure;
  - fast electrons covary with m/q 1, 13–15, 27–31 and 45–46 at all
    photon energies.
- **60560 final** (14.7 M shots, the owner's notebook output on 25 Nov,
  `DEAD_TIME_NS = 1.0`; 172 of 195 cells have ≥ 300 shots):
  - a₂/a₁ runs from about −0.42 to −0.62 between 264 and 274.5 eV, then
    becomes more negative: −0.65 to −0.84 at 275–279.5 eV, −0.91 at
    280.0 eV and −1.55 at 280.5 eV. The top steps had weaker pulses, as
    in 60559.
  - The reference-normalised high/low ratio is 0.91–1.00 at
    265–271.5 eV (lowest, 0.91–0.93, at 270–271.5 eV) and 0.96–1.04 at
    272–280 eV. Errors are ±0.02–0.04 (±0.09 at 280.5 eV).
  - These numbers have not been interpreted yet. Before reading
    photon-energy structure into a₂ or the ratio, account for the
    different GMD distribution of each step and for the global ~G^0.7
    response.
- **XAS maps** (same notebook, section "XAS maps") show the per-µJ
  electron spectrum vs measured photon energy and KE for each GMD bin.
  Photolines run diagonally, Auger lines vertically.

---

## 8. Legacy material on this branch
These come from the glycine beamtime (11022188); paths inside them point
at that beamtime. Use them as reference only:
- `analysis/post/`, including `TASKS.md`, the glycine-era task list;
- `analysis/notebooks/legacy/` and `analysis/notebooks/xas_static/`;
- the `static_xas_*` scripts;
- `ML/`, `dashboard/`, `old_dashboard/`, `diagnostics/`, `docs/`,
  `miguel_notebooks_maybeobsoletenow/`;
- the glycine configs in `analysis/configs/`;
- `admm_ghost.py`, and the config-2 (liquid-jet, `liq_tofs_e`) paths in
  the writer and aggregates.

---

## 9. Open issues and suggested next steps

Roughly in priority order:

1. **Interpret the two-photon search on 60560**, with the systematics
   under control:
   - the global sublinear GMD response;
   - per-step GMD distributions;
   - the reference window, which contains Ar 2p from residual argon.

   Options: fixed GMD edges common to all steps; a reference window free
   of argon; comparing 60559 and 60560 on a common GMD binning.
2. **Delay scans at 268 eV** (`Ethanol_delay_scan_short*_268eV`, run 60561
   onwards) still need writing (they have an SDU, so `z` is filled).
   - The config-1 `time_resolved` aggregation mode (GMD × z; example
     `aggregates_example_cfg1_tr.py`) has **not been run on this
     beamtime's data**.
   - Its hit-list code path was **not verified** against the old dense
     loop: do that first, on a short run.
   - The units and sign of `z` (→ fs) are unknown.
3. **Photon-energy cross-calibration between runs.** The VLS slopes and
   offsets differ, and there is no absolute reference (an atomic line
   would help: Ar 2p binding energy vs hν is a candidate).
4. **Merged-sweep trains** (§3.2): make the writer handle them, or cut
   them at write time.
5. **eTOF calibration:** the 3s outlier, a single retardation setting,
   and t0 / E0 that are only constrained by the 2p lines.
6. **The ion-efficiency drop along the train** and the **G^0.7 scaling**
   (run 60538): understand both before using absolute yields.
7. Gotthard line offset for bunch patterns other than 100 pulses.
8. **Housekeeping:**
   - delete the superseded `run60560_*_live*` files (owner's call);
   - `inspect_tofs.ipynb` still works in TDC units and integrates past
     1000 ns (crosstalk);
   - `plotting.py` is not updated for config 1 + VLS;
   - the markdown of the "Calibrated axes" section in
     `undulator_scan_aggregates.ipynb` still describes the old 5 ns bins;
   - several of the owner's exploratory notebooks are uncommitted, on the
     Maxwell clone only (`inspect_*` edits, `camera_run_difference`, …).

---

## 10. Glossary
- **Train / bunch (slot) / shot:** 10 Hz train; up to 100 pulses per
  train (slots 0–99, plus dark slot 100 in the combined files). A
  *shot* is one pulse.
- **Cell:** one photon-energy bin × one GMD bin in the energy-mode
  aggregates.
- **D / C / A / G:** electron TOF histogram, ion TOF histogram, VLS
  spectrum, GMD pulse energy, per shot. `XtY` is ⟨XYᵀ⟩ per cell.
- **Nominal vs measured photon energy:** nominal = hc / undulator set
  wavelength (per train); measured = per-shot VLS centre, calibrated
  to the nominal steps of the same run.
- **Live write:** `write_h5.py --live`, which decodes a read-only snapshot
  of a `.lst` that is still growing.
- **Partial covariance on GMD:** covariance with the linear dependence on
  pulse energy removed (formula in §3.4).
