# Handover — `gas-vls` analysis, FLASH2 beamtime 11024642 (Nov 2026)

Written 25 Nov 2026 and updated 27 Nov 2026 for a colleague and the AI
assistant they work with (e.g. ChatGPT). It is a snapshot: it tells you
where things are, what has been done, what was found and what is open.
What changed on 27 Nov is listed in §0a. For the details of every
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

## 0a. What changed since the 25 Nov version

- **More undulator scans processed** (§6.1), all written **with the VLS
  in the combined file**:
  - ethanol 60586 (265–272 eV) and 60591 (quick scan, 265–272 eV);
  - **acetaldehyde (AcH)** in two runs: 60611 (part 1, 265–271 eV) and
    60612 (part 2, 270.5–280 eV);
  - the two AcH parts **combined** into one file
    `run60611-60612_AcH_undulator_scan_265-280eV.h5` with its
    aggregates. `undulator_scan_aggregates.ipynb` now opens this file.
- **Dead-time correction** (§5.5): `compute_ready_counts.py` writes a
  `<aggregates>_ready.h5` file next to an energy-mode aggregates file.
  It holds the electron spectrum corrected for the detector dead time
  (a Coates-type "ready shot" correction, τ = 6 ns chosen). The
  aggregates notebook uses it by default.
- **Normalisation** (§7): the fast-electron yield per µJ rises in the
  lowest GMD bins. Most of that comes from detector dead time, GMD noise
  and a selection bias (`yield_per_uj_vs_gmd.ipynb`). The aggregates
  notebook now divides by a chosen monitor: the VLS line sum (default),
  the GMD, or the Ar 2p photoline of the residual argon.
- **Concatenating runs** (§5.6): `concat_combined_h5.py` joins the
  combined files of consecutive runs. Each run keeps its own
  photon-energy calibration.
- **OPIS** (§3.1, §4.4): where it worked, it reads 0.3–1.8 eV above the
  undulator set energy near 265–272 eV and 6.4 eV below it at 177 eV.
  The VLS step sizes follow the set values, not OPIS.
- **Kernel** (§2): the owner's notebooks now run on the default
  "Python 3 (ipykernel)" kernel (Python 3.13).
- Colleagues' work since 25 Nov (delay-scan combined files and
  aggregates, `write_h5.py --lst-name`, new notebooks) is in the git log
  and the notebooks' own header cells; this file does not cover it.

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
  - ethanol undulator scans (runs 60559, 60560, 60586, 60591), with
    residual argon in the chamber;
  - ethanol delay scans at 266, 268, 270 and 271.5 eV (25–26 Nov;
    combined files written by colleagues, see §6.2);
  - **acetaldehyde (AcH)**: an undulator scan in two runs, 60611 and
    60612 (26 Nov 23:24 to 27 Nov 05:07), then delay scans at 268.7 and
    268.2 eV
    (not processed here).
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
  - The owner's notebooks now run on the default **"Python 3
    (ipykernel)"** kernel: the same mamba 2026.05 Python 3.13, plus
    packages in the owner's `~/.local` (seaborn is installed there).
    Earlier they used "xfel (202301; Python 3.9)"; older notebook code
    may still be written for that.
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
| `analysis/scripts/compute_ready_counts.py` | dead-time "ready" counts for an energy-mode aggregates file → `<aggregates>_ready.h5` (§5.5); `load_ready_counts` |
| `analysis/scripts/concat_combined_h5.py` | joins combined H5 files of consecutive runs along the train axis (§5.6) |
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
| OPIS mean photon energy | `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy` | event records (~every 10 trains when live); live only in runs 60509–60514, 60539–60544, 60569–60587 and 60597–60607; 0 in 60545–60568, 60588–60596 and 60608–60609 (incl. the scans 60545, 60559, 60560, 60591); stuck at 273.375 eV from 60610 (incl. 60611, 60612). The combined `mpe` takes it only on the trains with a record. See §4.4 |
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

**Concatenated files** (`concat_combined_h5.py`, e.g. runs 60611 + 60612):
- Same datasets, with the trains of all inputs in train order.
- Root attributes `sources` (input file names), `source_first_index` and
  `source_n_trains` give the rows of each input.
- In `photon_energy/` the per-shot datasets are concatenated, and **each
  input keeps its own calibration**:
  - `cal_coef` is `(n_inputs, 2)`, row *i* for input *i*;
  - the step datasets (`cal_energies`, `cal_centres`, `cal_residuals`,
    `step_spectra`) are concatenated, and `cal_source` names the input
    of each step;
  - attributes that differ between the inputs (`dark_median`,
    `dark_sigma`, `threshold`) hold one value per input.
- Code that expects a 1-D `cal_coef` or a scalar `dark_median` must
  handle this.

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

### 3.5 Ready counts (`compute_ready_counts.py`, next to an aggregates file)
`<aggregates stem>_ready.h5` holds dead-time-corrected electron means for
the same cells (leading axes as the aggregates):

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `taus_ns` | (n_tau,) | dead times tried: 1, 2, 3, 4, 5, 6, 8, 10 ns |
| `n_per_bin`, `D` | cells, (cells, n_e) | as in the aggregates (checked equal: attrs `check_n_equal`, `check_D_max_abs_diff`) |
| `D_live` | (n_tau, cells, n_e) | electrons per shot per bin, corrected for dead time τ |
| `ready_frac` | (n_tau, cells, n_e) | fraction of shots "ready" (no recorded hit in the preceding τ) |
| `p0_windows_ke`, `p0`, `mu0` | (n_w, 2), (cells, n_w) | KE windows (default 210–285 and 10–40 eV); fraction of shots with no hit in the window; μ₀ = −ln p0 |

Only the means are corrected. The second moments (`DtD`, `DtC`, `DtG`)
stay as recorded, so covariances use the recorded `D`, not `D_live`.

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
- 60586: E = 297.441 − 0.0560721·px (rms 0.058 eV, 15 steps)
- 60591: E = 298.639 − 0.0577877·px (rms 0.039 eV, 8 steps)
- 60611: E = 299.331 − 0.0591219·px (rms 0.032 eV, 13 steps)
- 60612: E = 299.885 − 0.0600692·px (rms 0.053 eV, 20 steps)

The slopes differ by up to 6 % between runs, and the same nominal
energy sits several pixels apart: 3–4 px between 60559 and 60560, and
1.5 px (0.09 eV) at 270.5 eV between 60611 and 60612. Within a step the
centre wanders by about ±0.2 eV over minutes. Single-shot resolution is
σ ≈ 0.3 eV. **Do not merge runs on measured photon energy without
cross-calibrating them.** The combined 60611 + 60612 file does this
through the nominal scale: each part keeps its own calibration (§3.3).
There is no independent absolute photon-energy reference yet.

**Along the train** (60586): pulses 0–20 read about 0.15 eV above the
mean of all 100 pulses, and the measured energy falls by about 0.4 eV
from the first to the last pulse.

**OPIS vs nominal** (`opis_vs_nominal_photon_energy.ipynb`, on the
Maxwell clone, not committed). This uses the runs where OPIS gave a
result (§3.1):
- Scan 60586: OPIS − nominal is +0.34 to +0.91 eV at 265–270.5 eV, then
  +1.4 to +1.8 eV at 271–272 eV.
- Scan 60513 (177–181 eV): −6.1 to −6.6 eV, slope 1.007.
- Fixed-energy runs at 266–271.5 eV: +0.5 to +1.4 eV, varying by about
  ±0.3 eV between runs at the same nominal energy.
- After every nominal change in 60586, OPIS holds its old value (or 0)
  for about 1250 trains. In the fast scan 60542 the steps were too short
  for OPIS to settle.
- **VLS check on 60586.** The VLS step centres fit the nominal energies
  to 0.06 eV rms, but the OPIS step values to only 0.32 eV rms. For
  270.5 → 271 eV the steps were 0.51 eV (VLS), 0.50 eV (nominal) and
  1.77 eV (OPIS).
- Within a step OPIS wanders by 0.46 eV rms, while the 60-train VLS mean
  moves by 0.04 eV, with correlation 0.10.

So the VLS agrees with the undulator setting on step sizes and does not
see the OPIS jumps or wander. Neither OPIS nor the VLS gives the
absolute energy here.

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
- The VLS takes ~0.28 MB per train (60611: 21.7 GB for 76 668 trains).
  60591, 60611 and 60612 were written with it. 60560 and 60586 were
  written `--no-vls`; `photon_energy.py --raw-run` and
  `compute_aggregates.py` (`VLS_RAW_RUN`) then read the Gotthard frames
  straight from the raw files.
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
The job scripts used so far live in the owner's home directory
(`~/slurm_undscan.sh`, `~/slurm_concat.sh`), not in the repo. This is
the pattern for a finished undulator-scan run, end to end, with the VLS
in the combined file (one config per run,
`analysis/configs/aggregates_run<RUN>_energy.py`, which also lists the
commands):

```bash
#!/bin/bash
#SBATCH --partition=upex
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=undscan
#SBATCH --output=%x_%j.log
set -eo pipefail
cd <your clone>                      # with the 11024642 symlink
P=/software/mamba/2026.05/bin/python3
C=11024642/processed/combined
W=$(( $(nproc) < 48 ? $(nproc) : 48 ))
RUN=60612; MEAS=AcH_undulator_scan_part2_265-280eV_2809782740
F=$C/run${RUN}_${MEAS}.h5
$P -u write_h5.py $MEAS 1 $RUN --live --live-margin 0 --max-ecounts 120 --max-icounts 50 -o $F
$P analysis/scripts/photon_energy.py --workers $W $F
$P analysis/scripts/compute_aggregates.py analysis/configs/aggregates_run${RUN}_energy.py $F --workers $W
$P analysis/scripts/compute_ready_counts.py ${F%.h5}_aggregates_energy.h5 --workers 16
```
Submit with `sbatch job.sh`; it typically starts at once.
- Run 60560 (no VLS, 187k trains): 5.5 min, peak about 40 GB RAM at
  48 workers.
- Run 60612 (with VLS, 125 649 trains): 18.6 min (write 15 min, photon energy 37 s, aggregates 108 s, ready counts 55 s).

For a run still acquiring, use `--live` (default margin) and a
`_live` file name. Rewrite the final file once the run has ended.

**Pitfall:** the workers are *spawned* processes, so any script that
uses `--workers` / `workers=` must run from a file with an
`if __name__ == "__main__":` guard. Never pipe it to
`python - <<EOF`: the workers cannot start and the pool raises
`BrokenProcessPool`. Calls from a Jupyter notebook work.

### 5.5 Dead-time ready counts — `compute_ready_counts.py`
```
python analysis/scripts/compute_ready_counts.py AGG.h5 [--taus 1 2 3 4 5 6 8 10] [--windows 210:285 10:40]
       [--gmd-sub 4] [--workers 16] [-o OUT.h5]
```
The electron detector / TDC records no second hit within its dead time:
- no hit pairs closer than 1.5 ns;
- recorded hits are partly lost up to about 6 ns after a hit.

The loss grows with the hit density, so it is largest on the fast
photolines at high GMD.

How the correction works:
- The script goes back to the combined file the aggregates came from and
  puts every shot into the same cell as `compute_aggregates.py` does.
- Per cell and per TDC tick (0.1 ns) it counts R, the shots "ready" (no
  recorded hit in the preceding τ), and H, the hits recorded at that tick
  in ready shots.
- The electrons per shot in a TOF bin are D_live = Σ_ticks −ln(1 − H/R).
  This is the Coates correction for pile-up in TOF spectra.
- R and H are counted in 4 GMD sub-bins per GMD bin, because mixing
  pulse energies under-corrects (synthetic test: 7 % low with one
  sub-bin, within 2 % with 3–6).

Checks:
- The zero-hit estimate μ₀ = −ln P(no hit in a window) is not affected by
  dead time, since the first hit of a shot is never lost.
- On run 60611 the fast-window total of D_live reaches a plateau from
  τ = 6 ns. There it equals μ₀ to 0.1 % in every GMD bin, which is
  1.56× the recorded hits in the top bin. The notebooks use
  `TAU_NS = 6`.
- `n_per_bin` and `D` reproduce the aggregates exactly
  (`check_n_equal`, `check_D_max_abs_diff = 0`).

A ready file exists for 60591, 60611, 60612 and the combined 60611 +
60612. It runs in under a minute at 16 workers (39 s for the joined
AcH file).

### 5.6 Joining runs — `concat_combined_h5.py`
```
python analysis/scripts/concat_combined_h5.py OUT.h5 IN1.h5 IN2.h5 [...] [--check N]
```
- Concatenates combined files along the train axis, in train order.
- The inputs must hold the same datasets and attributes and must not
  overlap in train ID.
- Datasets stored one train per chunk (the VLS) are copied without
  recompression.
- `photon_energy/` is carried over with each input's own calibration
  (§3.3).
- The script refuses to overwrite, writes `OUT.h5.partial` and renames
  it when complete, then spot-checks random trains of every input.

Then run `compute_aggregates.py` (with its own config) and
`compute_ready_counts.py` on the output. `~/slurm_concat.sh OUT CONFIG
IN1 IN2 …` does all three. The GMD percentile edges of the joined file
differ from those of the parts.
Joining 60611 + 60612 (202 317 trains, 57 GB) took 107 s, its
aggregates 73 s and the ready counts 39 s.

---

## 6. Runs, files and notebooks

### 6.1 Processed runs (`processed/combined/`)
| Run | Acquisition (`.lst` stem) | Content | Combined file | Aggregates |
|-----|---------------------------|---------|---------------|-----------|
| 60538 | `Argon_2807259185` | argon, hν 275.5 eV nominal; calibration run | `run60538_Argon_2807259185.h5` (6057 trains) | `…_aggregates.h5` (GMD bins), `…_aggregates_b0-20.h5` (bunch slots 0–20 only) |
| 60551–60557 | `Ethanol_28078…`–`Ethanol_28079…` | short ethanol acquisitions | `run6055x_Ethanol_*.h5` (3–4 MB each) | `…_aggregates.h5`. Parameters are not recorded in `CLAUDE.md`: read the file attrs or ask the owner |
| 60559 | `Ethanol_undulator_scan_2807989445` | ethanol undulator scan, 18 steps 265–282 eV (1 eV, ~60 s each) | `run60559_…2807989445.h5` (11 486 trains, **with VLS**, 3.2 GB) + `photon_energy/` (67 % of pulsed shots) | `…_aggregates_energy.h5`: 730k shots; GMD edges 0.020/0.056/0.098/0.169/0.301/3.29 µJ |
| 60560 | `Ethanol_undulator_scan_2808016357` | ethanol undulator scan, 31 steps 265–280 eV (0.5 eV, ~10 min each), 312 min | `run60560_…2808016357.h5` (187 322 trains, no VLS, 0.6 GB) + `photon_energy/` from raw files (82.7 %) | `…_aggregates_energy.h5`: 14.7 M shots; GMD edges 0.020/0.057/0.100/0.168/0.290/4.85 µJ |
| 60586 | `Ethanol_undulator_scan_265-272eV_2808708850` | ethanol undulator scan, 15 steps 265–272 eV (0.5 eV, ~10 min each); OPIS live | `run60586_…2808708850.h5` (92 074 trains, no VLS, 0.3 GB) + `photon_energy/` from raw files (97.6 %) | `…_aggregates_energy.h5`: 8.83 M shots; GMD edges 0.020/0.108/0.197/0.323/0.544/4.80 µJ. (`…_aggregates_energy_comparison.h5` is a colleague's file) |
| 60591 | `Ethanol_quick_undulator_scan_265-272eV_2808989151` | quick ethanol scan, 8 steps 265–272 eV (1 eV, ~5 min each) | `run60591_…2808989151.h5` (24 791 trains, **with VLS**, 7.0 GB) + `photon_energy/` (93.9 %) | `…_aggregates_energy.h5`: 2.28 M shots; GMD edges 0.020/0.085/0.151/0.235/0.391/4.28 µJ; `…_ready.h5` |
| 60611 | `AcH_undulator_scan_265-280eV_2809702542` | **AcH** undulator scan part 1, 13 steps 265–271 eV (0.5 eV, ~10 min each) | `run60611_…2809702542.h5` (76 668 trains, **with VLS**, 21.7 GB) + `photon_energy/` (92.2 %) | `…_aggregates_energy.h5`: 6.91 M shots; GMD edges 0.020/0.109/0.217/0.383/0.668/5.01 µJ; `…_ready.h5` |
| 60612 | `AcH_undulator_scan_part2_265-280eV_2809782740` | **AcH** part 2, 20 steps 270.5–280 eV (0.5 eV, ~10 min each), 209 min | `run60612_…2809782740.h5` (125 649 trains, **with VLS**, 35.4 GB) + `photon_energy/` (83.2 %) | `…_aggregates_energy.h5`: 10.16 M shots; GMD edges 0.020/0.080/0.163/0.297/0.573/5.49 µJ; `…_ready.h5` |
| 60611 + 60612 | both of the above | **AcH, both parts joined** (`concat_combined_h5.py`), 265–280 eV; 270.5 and 271 eV in both parts | `run60611-60612_AcH_undulator_scan_265-280eV.h5` (202 317 trains, with VLS, 57.1 GB); per-part `photon_energy/` calibrations | `…_aggregates_energy.h5` (`aggregates_run60611_60612_energy.py`): 17.07 M shots; GMD edges 0.020/0.089/0.183/0.333/0.615/5.49 µJ; `…_ready.h5` |

`_live` files are partial writes made while a run was acquiring. These
are **superseded** but still on disk; they can be deleted once the owner
agrees:
- `run60560_*_live*.h5`: 5 files, about 3.4 GB;
- `run60586_…_265-275eV_…_live.h5` and its aggregates;
- `run60612_…_live.h5` and its aggregates and ready file, about 27 GB.

The run-60611 live files were deleted on 27 Nov.

### 6.2 Raw data not processed here
- Local-DAQ acquisitions: `Argon_2807246386`, `Argon_2807249362`,
  `Neon+residual_gas_2807204453`, `residual_gas_2807853910`, two tiny
  `Ethanol_28079489*`.
- **Delay scans.** Colleagues wrote the ethanol delay-scan combined files
  (`Ethanol_delay_scan_*` and `Ethanol_*_delay_scan_*` at 266, 268, 270
  and 271.5 eV, raw runs from 60561) and the time-delay aggregates
  (commit "time delay aggregates", `inspect_delay_scan.ipynb`,
  `delay_scan_covariance.ipynb`). Ask them for the details.
- The AcH delay scans (`AcH_*delay_scan_268.7eV*`, `…_268.2eV…`, raw
  runs from 60613) are not processed.
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
| `undulator_scan_aggregates.ipynb` | **main analysis notebook**. Opens energy-mode aggregates as plain arrays with calibrated axes (`D_E`, `C_mq`, …). Dead-time correction from the ready file (`READY_H5 = 'auto'`, `TAU_NS = 6`; `Dm` = corrected means, with a τ check plot). Normalisation monitor `MONITOR` = `'vls'` (VLS line sum, default), `'gmd'` or `'ar'` (Ar 2p photoline, `AR_BE` 246.8–251.8 eV binding energy with linear sidebands of `AR_SIDE` = 2 eV). Sections: **XAS maps** (photon energy × KE per GMD bin) and the **two-photon resonance** fit of fast electrons against the monitor | `AGG_H5` (now the joined 60611 + 60612; earlier runs listed as comments) |
| `yield_per_uj_vs_gmd.ipynb` | why the per-µJ yield rises at low GMD: GMD noise (hall vs tunnel), dead time (hit spacing, Poisson, −ln P0), VLS as monitor, selection on a measured photon energy, train position | a combined file with `photon_energy/` + the raw run (tunnel GMD); 60611 |
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

**Normalisation: yield per µJ vs GMD (`yield_per_uj_vs_gmd.ipynb`, run
60611 single shots).** In every run the fast electrons per µJ were
higher in the low GMD bins, while hits against GMD looked linear. Both
hold because the line has an intercept: fast-electron hits =
1.107·G + 0.243. Per µJ that is a + b/G, with b/a = 0.22 µJ. The slow
window (10–40 eV) has b/a = 0.026 µJ and the ions 0.009 µJ. The
contributions found:
- **GMD noise.** The hall GMD (`gmd`) has 0.03–0.05 µJ rms single-shot
  noise against the tunnel GMD. Binning on it moves under-read pulses
  into the lowest bins: pulses with hall < 0.01 µJ carry 0.036 µJ by the
  tunnel GMD.
- **Dead time** in the fast window, which loses hits at high GMD (up to
  1.56× in the top bin of 60611; §5.5).
- **Selection.** The aggregates keep only shots with a measured photon
  energy, which favours weak pulses with a strong VLS signal.
- **Train position**: the yield per µJ changes along the train at fixed
  pulse energy.
- After these, a common fall-off of about 1.4× across the GMD range
  remains in every channel; its cause is not identified.

With the dead-time correction and the VLS line sum as monitor (the
current notebook default), the fast-electron yields of the five GMD bins
agree within about ±6 % on the joined AcH scan (below). Per µJ, without
the correction, they had spread by a factor 2–3. The 60559 / 60560
numbers below were made before this: GMD-normalised, without dead-time
correction.

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
- **AcH, 60611 + 60612 joined** (17.07 M shots, 171 of 195 cells with
  ≥ 300 shots). This is the notebook output of 27 Nov with dead-time
  correction (τ = 6 ns) and `MONITOR = 'vls'`, so here a₁, a₂ are per
  10⁴ ADU of VLS line sum and the fit is ⟨N⟩ = a₁⟨M⟩ + a₂⟨M²⟩:
  - **Dead time.** At τ = 6 ns the corrected fast-window total equals
    −ln P0 to within ~1 % in every GMD bin. The recorded hits are 0.97
    (lowest GMD bin) to 0.60 (top bin, ⟨G⟩ ≈ 1 µJ) of it.
  - **Fast electrons per VLS unit.** The five GMD bins agree within
    about ±6 %. At 265–271 eV the top GMD bin is the lowest; at
    272–279 eV the lowest GMD bin is. The yield rises from ~0.72 to
    ~0.85 per 10⁴ ADU between 272 and 280 eV in all bins.
  - **a₂/a₁** rises smoothly with photon energy (errors ±0.001, ±0.005
    at 279.5–280 eV): −0.037 to −0.017 at 264.5–271.5 eV, −0.010 to
    +0.002 at 272–277 eV, and +0.005 to +0.018 at 277.5–280 eV. There is
    no step where the parts meet: 270.5–271.5 eV was taken in both.
  - **Reference-normalised high/low ratio:** 1.02–1.11 at every photon
    energy (median 1.054, errors ±0.015–0.04), with no clear
    photon-energy structure.
  - Not interpreted yet. The same caveats apply (per-step GMD
    distributions, edge bins). With the VLS as monitor, check also that
    the VLS line sum is linear in pulse energy over the top GMD bins.
- **XAS maps** (same notebook, section "XAS maps") show the per-µJ
  electron spectrum vs measured photon energy and KE for each GMD bin.
  Photolines run diagonally, Auger lines vertically. They are now
  divided by the chosen monitor instead (`XAS_NORM = "monitor"`).

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

1. **Interpret the two-photon search** on the joined AcH scan and on
   the ethanol runs, with the corrected normalisation (§7):
   - rerun 60559 / 60560 / 60586 with the dead-time correction and the
     VLS monitor (ready files are missing for 60559, 60560 and 60586);
   - per-step GMD distributions;
   - the edge bins of each scan.

   Options: fixed GMD edges common to all steps; comparing runs on a
   common GMD binning.
2. **Dead time beyond the means.**
   - The pair covariance in the notebook leaves out pairs closer than
     `DEAD_TIME_NS = 1.0` ns, below the ~6 ns the ready counts show.
   - Second moments are not corrected.
   - The ions are not corrected.
   - The delay-scan notebooks do not use the correction yet.
3. **The remaining ~1.4× fall-off** of the per-µJ yield across the GMD
   range (all channels, after the known corrections), and whether the
   VLS line sum saturates in the top GMD bins (not checked).
4. **Delay scans**: colleagues have written the ethanol delay scans and
   time-delay aggregates. Ask them whether the config-1
   `time_resolved` hit-list path has been checked against the old dense
   loop, and about the units and sign of `z` (→ fs). The AcH delay scans
   are not processed.
5. **Photon-energy cross-calibration between runs.** The VLS slopes and
   offsets differ. There is no absolute reference: OPIS disagrees with
   both the undulator and the VLS (§4.4). An atomic line would help; the
   Ar 2p binding energy vs hν is a candidate.
6. **Merged-sweep trains** (§3.2): make the writer handle them, or cut
   them at write time.
7. **eTOF calibration:** the 3s outlier, a single retardation setting,
   and t0 / E0 that are only constrained by the 2p lines.
8. **The ion-efficiency drop along the train** and the **G^0.7 scaling**
   (run 60538): understand both before using absolute yields.
9. Gotthard line offset for bunch patterns other than 100 pulses.
10. **Housekeeping:**
   - delete the superseded `_live` files listed in §6.1 (owner's call);
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
- **Ready shot / ready counts:** a shot with no recorded electron hit in
  the τ before a given TDC tick, so the detector could record a hit
  there. `D_live` is the dead-time-corrected electron mean (§5.5).
- **Monitor:** the per-cell intensity the yields are divided by, chosen
  in the aggregates notebook: GMD, VLS line sum or Ar 2p photoline.
