# Handover — AcH delay scans, argon controls and ethanol comparison

FLASH2 beamtime **11024642**, `gas-vls`, **27 Nov 2026**.

This is the daytime-to-night-shift analysis handover. It covers the
three notebooks below, rather than repeating the previous `HANDOVER.md`.
It is written mainly for the receiving Claude Code agent, with enough
context for a colleague to inspect the calculations and continue them.

**Evidence:** the notebook source, saved text outputs and selected saved
figures were inspected. Beamtime H5 files are absent from this checkout;
the analyses were not rerun on real data for this handover. Saved outputs
include several different executions and file selections. Treat the
numbers below as labelled snapshots, not one consistent Run All result.
The source defines what the next execution will calculate.

**To add before sending — scientist's conclusions / shift priorities:**

- [Add physical conclusions and preferred interpretation here.]
- [Add acquisition priorities, setup changes and any confirmed time zero.]

---

## 0. Read this before restarting

1. **Read `AGENTS.md`.** It records the requested notebook style:
   sequential, explicit calculations and plots, adjustable parameters
   near the beginning, few functions, no unnecessary abstractions.
   Coordinate before editing a notebook that is open in Jupyter.
2. Keep raw H5 and local-DAQ `.lst` inputs read-only. Paths come from
   `analysis/scripts/config.py`; the notebooks import `COMBINED_DIR`.
3. **The updated delay notebook currently has an execution-order issue:**
   two TOF covariance cells request `partial on VLS` before the cell that
   creates it. See §6. Saved output does not demonstrate a clean restart.
4. **The argon notebook selects 269.7 eV scans but sets
   `PHOTON_ENERGY = 268`.** This determines its Ar photoelectron gate.
   Reconcile the energy setting, calibration and observed peak before
   interpreting or extending those controls.
5. **The matched-intensity part of `acet_vs_ethanol.ipynb` includes the
   third-harmonic ethanol scan excluded earlier in that notebook.** Its
   current plots are not a comparison of the same acquisitions. See §4.
6. **Electron means can be dead-time corrected; covariance cannot.**
   A 6 ns pair exclusion is not a correction for lost second moments.
   Retain recorded means when forming covariance from recorded moments.

---

## 1. Which notebook does what

All three are in `analysis/notebooks/` and currently specify the
**Python (FLASH Remote)** kernel (`flash-remote`). They locate the repo
from the current directory, so start from the root or that notebook
folder. Their cells read data; they do not generate combined files,
aggregates or ready files.

| Notebook | Purpose | Main working arrays |
|---|---|---|
| [`inspect_delay_scan_acet_updated.ipynb`](analysis/notebooks/inspect_delay_scan_acet_updated.ipynb) | Main delay-scan inspection: statistics, FEL stability, corrected electron spectra/traces, ion plots, covariance and gates | `scans`, `combined` = recorded sums; `electron_scans`, `electron_combined` = selected electron means reconstructed as sums; `ready_scans` = GMD-resolved correction diagnostics; `EST`, `EST_AV` = recorded covariance |
| [`inspect_delay_scan_argon_reference.ipynb`](analysis/notebooks/inspect_delay_scan_argon_reference.ipynb) | Argon yield/covariance controls, molecular-to-Ar covariance ratios, energy-reference tests, single- and multiple-monitor partial covariance; optional pure-Ar template | `roi_defs`, `roi_masks`; `methods`, `EST`, `EST_AV`; `n_by`, `g_by`, `m_by`; `cell_records`, `monitor_records`; `energy_corrected_rois` |
| [`acet_vs_ethanol.ipynb`](analysis/notebooks/acet_vs_ethanol.ipynb) | Integrated molecular yield/spectrum comparison, dead-time correction, GMD versus VLS normalisation, then sampled shot-level intensity matching | `molecules`, `corrected_spectra`, `relative_metrics`; later `intensity_data` |

Use the **updated** AcH notebook for the work described here. The earlier
`inspect_delay_scan_acet.ipynb` remains in the repo but its normalisation
behaviour differs: the newer notebook propagates `MONITOR` and
`USE_READY` through the main electron analysis.

---

## 2. Files and conventions needed to continue

### 2.1 Selected acquisitions in the saved outputs

All filenames below are relative to `COMBINED_DIR`. The stems in this
table take the suffix **`_aggregates_tr.h5`**; their companion correction
files take **`_aggregates_tr_ready.h5`**.

| Acquisition stem | Selected shots | Where used |
|---|---:|---|
| `AcH_long_delay_scan_1_269.7eV_2810289120` | 2,459,088 | Updated delay notebook and argon notebook, current top-level selection |
| `AcH_long_delay_scan_2_269.7eV_2810315452` | 2,457,818 | Same |
| `AcH_long_delay_scan_3_269.7eV_2810342023` | 2,453,743 | Same |
| `AcH_short_delay_scan_2_268.2eV_2810107042` | 1,169,112 | Molecular comparison; older saved covariance in the updated delay notebook |
| `AcH_short_delay_scan_3_268.2eV_2810120192` | 1,171,918 | Same |
| `AcH_short_delay_scan_4_268.2eV_2810135905` | 1,173,952 | Same |
| `Ethanol_short_delay_scan_1_271_5eV_2809266777` | 1,152,242 | Molecular comparison |
| `Ethanol_short_delay_scan_2_271_5eV_2809279486` | 863,467 | Molecular comparison |
| `Ethanol_short_delay_scan_3_271_5eV_2809292235` | 1,155,314 | Molecular comparison |
| `AcH_short_delay_scan_268.7eV_2809924372` | 1,129,259 | Separate linearity diagnostic in the updated delay notebook |

The three long scans total **7,370,649 selected shots**, about **125.4 min**.
Their saved delay axis is **−4.889 to 59.333 fs**, 32 bins, **2.006968 fs/bin**.
The three 268.2 eV short scans total **3,514,982 shots**, 60 min; their
axis is **−15.356 to 15.356 fs**, 31 bins, **0.990710 fs/bin**.
These are analysis axes under the current conversion, not a confirmed
absolute time zero or instrument response.

The VLS calibration file used is:

```
run60612_AcH_undulator_scan_part2_265-280eV_2809782740.h5
```

Its saved coefficients are approximately `[-0.0600691854, 299.885384]`
for `np.polyval(cal_coef, pixels)`, with calibration residual RMS
**0.05346 eV**. This is a transferred relative calibration; do not equate
that fit residual with absolute photon-energy accuracy.

### 2.2 Data model and weighting

- Config 1 `time_resolved` moments have leading axes **(GMD, delay)**.
  `D`, `C`, `A`, `G` and the second moments are **means per shot**;
  `n_per_bin` is the population. Reconstruct sums with `mean * n`, then
  pool sums and exposures before division. Empty cells contain NaNs.
- `D` is the electron hit histogram, `C` the ion histogram and `A` the
  cropped, background-subtracted VLS spectrum. `DtD = <DDᵀ>`,
  `DtC = <DCᵀ>`, `AtD = <ADᵀ>`, etc.; these are not centred covariances.
- TOF edges are in **100 ps units**. Divide by 10 before passing them
  to the calibration functions, which take ns. Reverse the electron
  axis for increasing KE. Divide spectra by actual transformed bin
  widths for densities; integrate **counts/bin**, not densities, in ROIs.
- The delay conversion is `DELAY_SIGN * (z - Z_ZERO) * 0.349 / 1000` fs.
  Current settings are `DELAY_SIGN = 1`, `Z_ZERO = 0`.
- VLS monitor **M = sum(A over pixels) / 1e4**, in units of 10⁴ ADU/shot.
  Its pooled exposure is the sum of `M*n`, not the sum of cell means.
- GMD percentile edges differ by acquisition. Selecting the same bin
  indices is not selecting the same absolute pulse-energy interval.
- Pool only files with compatible axes. In particular, do not pool
  the long and short delay grids directly.

Relevant existing code: `config.py`, `calibration.py`,
`compute_aggregates.py` and `compute_ready_counts.py` in
`analysis/scripts/`. The current ready-count code supports time-resolved
cells. To generate a missing correction file separately:

```bash
python analysis/scripts/compute_ready_counts.py <path_to_aggregates_tr.h5> --workers 16
```

Use the configured beamtime Python environment and appropriate compute
resources. The notebook validates matching shot counts, TOF edges and
recorded `D` before using `D_live` at the selected tau.

---

## 3. Updated AcH delay notebook

### 3.1 Structure and switches

The original inspection sequence is largely retained:

| Section | Calculation |
|---|---|
| 1–3 | Selection, first moments, calibrated axes, delay reference and detected-hit statistics |
| 4 / 4.1 | Recorded stability checks; VLS spectra; all ready-file taus; correction versus GMD, delay and KE; comparison of GMD and VLS normalisation |
| 4.2 | Recorded-moment uncertainty, cumulative convergence, independent acquisitions, time estimates and coarser delay bins |
| 5–6 | Electron and ion spectra, absolute/relative delay maps and spectral shape |
| 7–8 | Electron/ion temporal traces, GMD-bin comparisons, scan comparisons; extra single-run linearity cells |
| 9 | Recorded total/within-cell/partial covariance, pair-separation diagnostic, KE and TOF maps, gated spectra and ROI traces |

Current main settings select the three **269.7 eV long scans**, all GMD
bins, `MONITOR = 'vls'`, `USE_READY = True`, `TAU_NS = 6`.

- **`MONITOR`** (`vls`, `gmd`, `shot`) and **`USE_READY`** control main
  electron spectra/maps/traces in sections 5, 7 and 8.
- **`NORMALISE`** (`per_uJ`, `per_shot`) controls ion plots. Ions are
  uncorrected.
- `combined` / `scans` retain recorded sums. `electron_combined['e']`
  and `electron_scans[name]['e']` contain the selected electron sums;
  `e_recorded` and `e_corrected` preserve the alternatives.
- Ready files are inspected even with `USE_READY=False`.
  `USE_READY=True` requires valid correction data for every selected scan;
  it raises rather than silently substituting recorded means.
- `ready_scans` retains all GMD bins and, in this notebook, **all taus**.
  The correction diagnostics intentionally use those bins independently
  of the main `GMD_BINS` selection. These are histogram arrays, not
  full train × bunch × detector data.
- Acquisition counts, recorded GMD diagnostics, section 4.2 and all
  covariance retain recorded moments. Their errors are not errors on the
  corrected/VLS-normalised electron traces.
- `DELAY_BASELINE=None` uses all delays as the reference, not pump-off.
  Relative maps are deviations from that exposure-weighted reference.
- `RUN_TIME_DIAGNOSTIC` and `RUN_FINE_IONS` are off. When enabled they
  reread combined files in chunks with separately stated cuts; those
  diagnostics do not reproduce every aggregate selection cut.

### 3.2 Observations from saved output

**Dead time and monitor dependence.** For long scan 1, the saved 6 ns
correction in 210–285 eV rises from about **1.02× to 1.24×** across
the GMD bins; around the strongest fast photoline it reaches roughly
**1.4×** in the top bin. The 10–40 eV correction stays close to unity.
The fast-window correction also varies with delay, roughly 1.12–1.17×
in that scan after pooling GMD bins. Do not assume a constant scale factor.

The before/after normalisation plots still show several-percent delay
variation in multiple electron windows. Correcting the mean does not
remove that shared structure. Its physical origin is not established.

**Separate 268.7 eV linearity test**, from the file named in §2.1,
155 GMD-delay cells and 1,129,259 selected shots:

| Window / monitor | Recorded fit intercept | 6 ns corrected intercept | Pearson r, recorded → corrected |
|---|---:|---:|---|
| 210–285 eV / GMD | +0.0559 | +0.0150 | 0.9965 → 0.9984 |
| 210–285 eV / VLS | +0.0431 | −0.0033 | 0.9951 → 0.9992 |

These are free-intercept fits of **mean electrons/shot against mean
monitor in each cell**, with the printed unweighted cell fits. The
10–40 eV fits change very little. The fast-electron intercept becomes
much smaller after correction, especially against VLS. This supports
checking detector response and monitor choice before interpreting a
nonlinear yield; it does not prove that VLS is an absolute linear monitor
or measure photon order from cell means.

**Long-scan statistics**, using recorded moments and GMD normalisation:

| KE window | Recorded electrons | Median Poisson relative error | Median measured ratio error |
|---|---:|---:|---:|
| 220–263 eV | 2,422,127 | 0.363% | 0.322% |
| 220–240 eV | 970,049 | 0.575% | 0.533% |
| 242–246 eV | 249,085 | 1.133% | 1.106% |
| 250–263 eV | 804,095 | 0.631% | 0.576% |

The measured ratio error uses the stored `XX`, `XG`, `GG` moments of
`X - yG`, rather than treating exposure as fixed Poisson counts. It still
does not include all acquisition drift, reference uncertainty or
detector systematics. The comparison of scan 1 against scans 2+3 gives
RMS differences **4.17, 2.69, 1.51 and 2.56** times the combined quoted
statistical error, respectively, for the four rows above. There is
substantial between-acquisition disagreement under that error model.
The time estimates assume statistical `1/sqrt(time)` scaling; they are
not a prescription to acquire longer without resolving that disagreement.

The saved averaged VLS spectrum peaks at **268.890 eV**, FWHM
**2.003 eV** (0.745%). This is an averaged spectral width, not the
single-shot bandwidth or temporal resolution.

---

## 4. Acetaldehyde versus ethanol

### 4.1 What is compared

`COMPARE_PATTERNS` selects the three short ethanol scans at **271.5 eV**
and the three short AcH scans at **268.2 eV** in §2.1. The main comparison
uses **all GMD bins**, independently of the preliminary `FILES` /
`GMD_BINS` loading cells. `COMPARE_ROI = (220, 263)` eV.

It compares detected total electrons, ROI electrons and ions per shot,
per GMD energy, per VLS exposure and optionally per acquisition minute.
Spectral ratios use the common KE grid. The correction section loads
every matching ready file at **6 ns**, regardless of the preliminary
`USE_READY=False` setting; missing files raise. It checks that recorded
sums reproduce the earlier molecular totals.

### 4.2 Saved integrated observations

| Quantity | Ethanol | AcH |
|---|---:|---:|
| Selected shots | 3,171,023 | 3,514,982 |
| Acquisition time | 54.20 min | 60.00 min |
| Mean GMD | 0.3151 µJ/shot | 0.3486 µJ/shot |
| Mean VLS | 0.7759 × 10⁴ ADU/shot | 0.6465 × 10⁴ ADU/shot |
| Recorded total electrons/shot | 0.9761 | 0.9466 |
| Recorded 220–263 eV electrons/shot | 0.2135 | 0.2963 |
| Recorded ions/shot | 0.3096 | 0.4588 |

The saved bar plot gives these **AcH / ethanol** ratios (rounded):

| Exposure | Total electrons, recorded → corrected | ROI electrons, recorded → corrected | Ions, uncorrected |
|---|---:|---:|---:|
| Per shot | 0.97 → 0.99 | 1.39 → 1.44 | 1.48 |
| Per GMD µJ | 0.88 → 0.90 | 1.25 → 1.31 | 1.34 |
| Per VLS unit | 1.16 → 1.19 | 1.67 → 1.73 | 1.78 |

The total-electron correction is about **1.019–1.023×** for ethanol
and **1.043–1.044×** for AcH; the fast ROI correction is larger:
**1.087–1.097×** and **1.127–1.145×**, respectively.

The monitor choice changes the apparent molecular ratio more than the
integrated dead-time correction here. VLS/GMD is **2.4627** for ethanol
and **1.8545** for AcH in the notebook's scaled units. The fast ROI yield
is larger for AcH under all three normalisations in this snapshot.
These are measured yield ratios at different photon energies and
acquisition conditions, not molecular cross-section ratios or evidence
by themselves for an enhanced two-photon channel. Gas density, residual
argon, spectra, FEL intensity distributions and detector response all
need to be accounted for.

### 4.3 Shot-resolved intensity matching: useful, but selection needs fixing

The later cells reopen combined files and reuse
`compute_aggregates._build_keep_mask`, `_vls_preprocess_block` and
`_pedestal_part`. They reconstruct the VLS pedestal/common-mode/bad-pixel
treatment and the aggregate GMD, delay, hit-cap and VLS-validity cuts.
They store five columns in `intensity_data[molecule]`:
**GMD, VLS, total recorded electrons, ROI recorded electrons, total ions**.

Current sampling is **every 20th accepted train**, chunks of 32 selected
trains. It is not an independent random sample of shots. `TRAIN_STEP=1`
enables a count check against the aggregate population but increases
I/O and retained sample memory. Full VLS frames are read in chunks.

Thirty common absolute intensity bins span zero to the larger 99th
percentile; both molecules need **500 shots/bin**. The plots compare
mean recorded yield at similar measured GMD or VLS intensity. These
shot-level yields are **not dead-time corrected**.

**Current inconsistency:** the shot-level loader uses a fresh
`sorted(COMBINED_DIR.glob(pattern))`, without the earlier `EXCLUDE_FILES`
and `SCAN_TYPE` filtering. Its saved output includes
`Ethanol_short_delay_scan_271_5eV_3rd_harmonic_2809252596_aggregates_tr.h5`.
Thus it samples **four ethanol scans** (216,542 shots), versus the three
used for the integrated results; the AcH sample has 175,842 shots.
Reuse the identical selected file lists and rerun before drawing
conclusions from the matched-intensity ratios. In particular, the very
large ratios in the lowest VLS bin are not established molecular physics.

---

## 5. Argon reference notebook

### 5.1 Question and calculation structure

The purpose is to test whether residual Ar can help diagnose shared
intensity/energy effects and covariance backgrounds in the AcH scans.
It is a companion control notebook, not an automatically validated
argon-subtraction pipeline.

| Section | Test |
|---|---|
| 0–1 | Same recorded/ready yield machinery; candidate Ar photoelectron/Auger gates; GMD, VLS and per-shot yield normalisation |
| Shared covariance preparation | Rebin axes, construct ROI masks, read cell moments once, compute alternative estimators on matched accepted cells |
| 2 | Candidate Ar photo–Auger EE and photo–Ar²⁺ EI islands; absolute/relative traces and separate acquisitions |
| 3 | Per-acquisition covariance versus mean intensity; molecular/Ar covariance ratios |
| 4 | Electrons and ions covarying with the candidate Ar photoelectron |
| 5 / 5.1 / 5.2 | Ordinary window centroids/widths, VLS centroid, Ar²⁺-gated photoelectron and photoelectron-gated Auger centroids |
| 5.3 | Exploratory VLS energy-shift compensation in molecular spectra and covariance ROIs |
| 6–7 | Partial covariance on Ar and joint GMD/VLS/Ar monitors; independent-run comparisons |
| 8 | Opt-in independent pure-Ar template subtraction; currently skipped |

Current source selects the three **269.7 eV long scans**, but uses
`PHOTON_ENERGY=268`, `MONITOR='vls'`, **`USE_READY=False`**. Its main
electron yield is therefore **recorded/VLS**, unlike the updated delay
notebook's corrected/VLS selection. Ready-corrected comparison curves
are still available when ready files validate.

Candidate gates in the current source:

- `AR_PHOTO = (PHOTON_ENERGY - 253.2, PHOTON_ENERGY - 245.2)`:
  currently **14.8–22.8 eV**. With a 269.7 eV parameter this expression
  would give **16.5–24.5 eV**. Inspect the observed doublet and calibration
  rather than treating either gate as a confirmed assignment.
- `AR_AUGER = (200, 208)` eV; `AR_ION = (19, 21)` u/e.
  Ion overlap with molecular fragments remains possible.
- `AR_CONDITION = AR_PHOTO` is the **recorded electron count** used for
  Ar partial covariance. `HOLDOUT_EE` and `HOLDOUT_EI` are unset.

The photoelectron is expected to follow photon energy; an ordinary
Auger line provides a candidate approximately fixed-KE comparison.
Window centroids can also shift through overlap, relative line strengths
and clipping. They are not fitted isolated peak energies.

### 5.2 Covariance definitions and monitor controls

Per GMD-delay cell:

```
cov(X,Y) = <XY> - <X><Y>
partial(X,Y | Z) = cov(X,Y) - cov(X,Z) inv(cov(Z,Z)) cov(Z,Y)
```

`cell_moments` constructs recorded first/second moments and the needed
monitor cross-moments. `partial_pair` standardises monitor variances,
checks positive definiteness/condition number, then solves the joint
system. It does not sum independent single-monitor corrections.

Available saved methods are total, within GMD bins, partial on GMD,
VLS, Ar, GMD+Ar, VLS+Ar, and GMD+VLS+Ar. VLS intensity moments are built
by summing `AtA`, `AtD`, `AtC`; the triple estimator also needs the
uncentred GMD–VLS moment `AtG`, scaled consistently with M.

`COMMON_COV_SELECTION=True` keeps identical accepted cells for all
enabled methods. `MIN_COV_SHOTS=300`, `MONITOR_COND_MAX=1e6`, and the
pair mask is `DEAD_TIME_NS=6` using TOF-centre separation. The saved
long-scan calculation accepts **all 7,370,649 eligible shots**, with
zero rejections by individual monitor tests. Matrix storage is about
**212.6 MB**, plus temporary cell arrays; `MAX_MATRIX_MB=1800` guards
allocation. The notebook stores DD and DC matrices, not CC matrices.

`EST` holds covariance per shot versus delay; `EST_AV` holds
shot-weighted delay averages. Total covariance is formed by subtracting
the product of pooled first moments; it is not the weighted mean of
within-cell covariance. `cell_records` keeps small per-cell ROI values
for the other estimators, and `monitor_records` stores within-cell Ar
correlations. Those small records support the run/intensity checks
without rereading every matrix.

`COV_NORMALISE_GMD=True` divides displayed traces by mean GMD; it is
not itself a nonlinear-response correction. Signed covariance relative
traces use `(y - reference)/abs(reference)`, not ordinary yield ratios.
The molecular/Ar ratio diagnostic explicitly uses raw per-shot
covariance, with `RATIO_POWER=1`, and masks weak/non-positive Ar reference
bins. A fitted covariance-versus-mean-intensity power is descriptive,
not a photon-order measurement of a two-pulse process.

Conditioning on an Ar gate suppresses covariance with that gate by
construction. This cannot independently validate the correction, and
can remove real signal when channels overlap. A useful next check is
an independently identified holdout channel plus acquisition-to-
acquisition reproducibility. Second moments alone do not supply reliable
uncertainties for these ratios and residuals.

### 5.3 Saved energy-reference observations and limitations

For the long-scan snapshot, all 32 delay bins pass the covariance-gated
centroid quality cuts (positive net area and negative covariance fraction
at most 0.25). Against the matched VLS centroid shifts:

| Electron centroid | Slope versus VLS shift | Pearson r |
|---|---:|---:|
| Ordinary candidate Ar photoelectron | 0.284 | 0.420 |
| Ar²⁺-gated candidate photoelectron | 0.178 | 0.075 |
| Ordinary candidate Auger | 0.012 | 0.013 |
| Photoelectron-gated candidate Auger | −0.288 | −0.109 |

These diagnostics do not establish a clean unit-slope photon-energy
reference. Revisit the gate/energy mismatch before assigning physical
meaning to the slopes. The ordinary and covariance-gated spectra also
weight events differently.

Section 5.3 uses a saved VLS reference centroid **268.7510 eV** and
`SHIFT_SCALE=1`. For each acquisition/delay it shifts ordinary electron
histograms by the VLS centroid offset using cumulative-count
interpolation, then re-differences the counts. Positive photon-energy
drift moves the corrected photoelectron spectrum to lower KE. It is
not a new absolute calibration or a shot-by-shot correction, and moving
the whole spectrum is not physically appropriate for every Auger feature.

The molecular covariance diagnostic instead shifts **only the first,
candidate photoelectron gate**, leaving the partner gate fixed. It
uses already pooled covariance and the VLS centroid matched to the
accepted covariance cells; it does not undo run-by-run spectral
misalignment before pooling. Both before/after ROI integrals use
fractional-bin overlap weights, so compare those curves with each other
rather than assuming equality with the earlier centre-selected gates.

The saved ordinary/covariance before-and-after traces change little
visually; this test does not remove the broad delay structure. The
ordinary shift retains **99.214%** of the original full-range counts;
boundary losses are therefore present. Molecular KE bins are about
**1.119 eV** wide in the printed region. This is an exploratory finite-bin
test, not proof that energy drift has been fully excluded.

### 5.4 Template subtraction is not yet established

`AR_TEMPLATE_FILE=None`, `TEMPLATE_COMPATIBLE=False`: the saved execution
**skips** template subtraction. Supply an independent pure-Ar aggregate
with matching detector/TOF settings, photon energy and compatible FEL
and gas conditions before enabling it. The notebook notes that the
existing ~281.8 eV pure-Ar reference is not directly compatible with the
lower-energy delay scans; file compatibility alone is insufficient.

The intended operation scales a within-cell Ar template to the positive
Ar EE island at each delay, then subtracts it. A zero in that same island
is forced by the scaling and is not validation. EI subtraction is
separately off (`TEMPLATE_SUBTRACT_EI=False`). Any residual is exploratory,
not an isolated molecular covariance. There is also an opt-in code issue
to address first (§6).

---

## 6. Known restart and provenance issues

These were found by source inspection, not fixed during preparation of
this handover. Cell numbers here are **zero-based JSON indices**; use the
quoted heading/comment to locate them if cells are added.

1. **Updated AcH notebook, cells 77–78:** both “Electron-electron
   covariance in TOF coordinates” cells request `partial on VLS`.
   Section 9.1 initially creates only total, within-GMD and GMD-partial
   estimators. The VLS-partial calculation is currently **later, cell 81**,
   under “Partial covariance on integrated VLS intensity”. A fresh
   top-to-bottom execution should fail at the first missing VLS entry.
   Move the VLS calculation ahead of the consuming TOF plots, or change
   their explicit method selection, then rerun the covariance sequence.
2. **Updated AcH notebook, cells 49–52:** linearity has its own
   `agg_path` fixed to the **268.7 eV short scan**, and explicitly selects
   **6 ns**, independently of top-level `FILES` and `TAU_NS`. Keep that
   labelled as a separate diagnostic or retarget it deliberately.
   Its later `axis=1` reduction assumes time-resolved (GMD, delay) layout.
3. **Mixed saved selections:** updated notebook loading/statistics show
   the 269.7 eV long scans, while much of saved covariance shows the
   three 268.2 eV short scans; the final pair-statistics cell shows the
   single 268.7 eV scan. Those figures must not be interpreted as one
   run. Some argon ratio/intensity outputs also predate its current
   preparation cells. Restart and rerun after selecting a dataset.
4. **Argon parameter mismatch:** fix/verify `PHOTON_ENERGY` versus the
   selected scan and resulting Ar gate (§5.1); old heading comments still
   mention short scans/268.2 eV in places.
5. **Argon template branch:** the centroid-correlation cell 27 assigns
   `fig, axes = plt.subplots(...)`, overwriting the earlier `axes`
   dictionary of H5 bin edges. The later opt-in template branch uses
   `axes[key]`. Rename the plot variable or preserve the edge dictionary
   before testing template subtraction. The default skipped branch
   hides this issue. Also, later centroid cells assume `vcentre` exists
   even though the initial VLS calibration load is conditional.
6. **Molecular comparison selection:** apply the same exclusions/file
   lists in the shot-level loader as in the integrated comparison
   before using the matched-intensity figures (§4.3).

No clean restart on the real beamtime data is claimed for these current
notebook versions. Earlier tests of the original AcH adaptation do not
validate subsequent live edits to the updated notebook.

---

## 7. Suggested continuation

1. Resolve the selection/energy settings and execution-order issues in
   §6, coordinating notebook edits with the person running Jupyter.
   Restart on one acquisition first, then the intended compatible set.
   Keep the selected files, normalisation and recorded/corrected label
   attached to the resulting figures.
2. Check the tau plateau/zero-hit diagnostic for the new acquisitions
   and match ready files to their aggregates. Near-unity or agreeing
   zero-hit estimates are model checks, not an independent measurement
   of the true yield. Do not extend the mean correction to covariance.
3. Establish whether common delay structure is reproducible across
   separate acquisitions, and examine acquisition-time/FEL drift before
   fitting molecular dynamics. Compare recorded/corrected and GMD/VLS
   traces on the same shots; establish delay sign and time zero.
4. Recheck the candidate Ar gates and holdout features, then compare
   GMD, VLS, Ar and joint conditioning on identical accepted cells.
   Inspect what spectral structure each correction removes. Avoid using
   the conditioned or template-scaling gate as its own validation.
5. Rerun the matched-intensity molecular comparison with the third-
   harmonic scan consistently excluded. Keep the different photon
   energies and uncorrected shot-level hit response explicit.
6. Use independent acquisitions or train-block resampling for uncertainty
   tests of weak covariance ratios/residuals. The displayed approximate
   errors and the naive acquisition-time extrapolations are not enough
   to establish a dynamical signal.

At drafting time the three notebooks and `AGENTS.md` are **untracked in
this checkout**. Include them explicitly when preparing the push; a
tracked-files-only commit would omit them. This handover preparation did
not modify the notebooks, pipeline or beamtime data and did not commit
or push anything.
