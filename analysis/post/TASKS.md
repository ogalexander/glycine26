# Post-Analysis Task Tracker

Maintained task list for the **publishable-results** phase. The beamtime
is complete — no new raw data. All notebooks live in `analysis/post/`,
one notebook per task, commented throughout with the runs and steps they
use (see the "Post-Analysis Phase" section of `../../CLAUDE.md`).

**This file is the source of truth for what is done / in progress / next.**
Edit it freely — update status, add subtasks, drop notes.

Status legend: ☐ not started · ◐ in progress · ☑ done · ✗ dropped/blocked

Run reference (fill in / correct as confirmed):

| Run         | Sample / purpose                                        |
|-------------|---------------------------------------------------------|
| 58825       | Residual gas (background for Ar calibration)            |
| 58826       | Argon — eTOF kinetic-energy calibration                 |
| 58827–58853 | Glycine, photon-energy series (peak assignment)         |
| 58890       | Residual gas at different photon energies (gly background) |

---

## 1. eTOF kinetic-energy calibration  ◐  *(in progress — scaffold ready)*
**Notebook:** `analysis/post/etof_energy_calibration.ipynb`  *(scaffold created; run once 58825/58826 combined H5 exist)*
**Objective:** Map electron time-of-flight → electron kinetic energy
(TOF → KE), the calibration every downstream electron analysis depends on.

**Data:**
- Argon calibration: **Run 58826** (known Ar photo/Auger lines).
- Residual-gas background: **Run 58825** — subtract to isolate the Ar
  contribution.

**Subtasks:**
- [ ] ☐ Load both runs (combined H5), select good bunch range, exclude
      zero-padding in TOF arrays.
- [ ] ☐ Histogram eTOF spectra for Ar (58826) and residual gas (58825);
      normalise consistently (per shot / per GMD) before subtracting.
- [ ] ☐ Background-subtract 58825 from 58826 to isolate Ar lines.
- [ ] ☐ Identify known Ar lines (assign literature KE to TOF peaks).
- [ ] ☐ Fit TOF→KE model (e.g. KE = a/(t − t0)² + E0); record fit params.
- [ ] ☐ Provide a reusable `tof_to_ke()` calibration (persist params;
      consider promoting into `processing.py`).
- [ ] ☐ Validate against a second known line / second photon energy.

**Notes:**
- Confirm photon energy used for the Ar run and the expected Ar 2p/3p
  photolines + LMM Auger positions.
- Decide where calibration params are stored (small JSON/NPZ in `post/`?).

---

## 2. Residual-gas background vs photon energy  ☐
**Notebook:** `analysis/post/residual_gas_background.ipynb`
**Objective:** Characterise the residual-gas contribution (background to
the glycine scans) and how it changes with photon energy.

**Data:** **Run 58890** (residual gas across photon energies).

**Subtasks:**
- [ ] ☐ Detect the photon-energy sections (shutter / mpe).
- [ ] ☐ Average eTOF (and iTOF) spectrum per photon energy.
- [ ] ☐ Quantify how the background shape/intensity scales with energy.
- [ ] ☐ Produce a per-energy background usable for glycine subtraction.

**Notes:**
- Cross-check against 58825 (Ar-run background) for consistency.

---

## 3. Glycine peak assignment vs photon energy  ☐
**Notebook:** `analysis/post/glycine_peak_assignment.ipynb`
**Objective:** Assign glycine features by tracking peak KE vs photon
energy across a range of energies, separating:
- **dispersive** lines — direct photolines (KE shifts 1:1 with photon E),
- **non-dispersive** lines — non-resonant Auger (KE fixed),
- **resonant Auger** features (near absorption edges).

**Data:** **Runs 58827–58853** (glycine photon-energy series).

**Subtasks:**
- [ ] ☐ Build a (photon energy × electron KE) map using the Task-1
      TOF→KE calibration and the Task-2 background.
- [ ] ☐ Track peak positions vs photon energy; classify dispersive vs
      non-dispersive by slope (dKE/dhν ≈ 1 vs ≈ 0).
- [ ] ☐ Flag resonant-Auger behaviour near edges.
- [ ] ☐ Tabulate assignments with KE / binding energy.

**Notes:**
- Depends on Tasks 1 (calibration) and 2 (background).

---

## 4. Charge / pile-up sensitivity across the train  ☐
**Notebook:** `analysis/post/pileup_train_sensitivity.ipynb`
**Objective:** Check whether the electron and ion detectors keep constant
sensitivity (electrons / ions per µJ) across the bunch train, or whether
late bunches are less sensitive (charging / pile-up).

**Data:** TBD — a high-statistics run (Argon 58826 and/or glycine).

**Subtasks:**
- [ ] ☐ Compute counts-per-µJ (hits / GMD) per bunch index, for both
      eTOF and iTOF.
- [ ] ☐ Plot sensitivity vs bunch index; test for a late-train fall-off.
- [ ] ☐ Compare electron vs ion behaviour.

**Notes:**
- Informs which `SIGNAL_BUNCH_RANGE` is safe to use elsewhere.

---

## 5. Hit-finder saturation / linearity checks  ☐
**Notebook:** `analysis/post/hitfinder_saturation.ipynb`
**Objective:** Check (a) whether counts-per-GMD is linear with GMD across
different photon energies (saturation onset), and (b) whether
close-together electrons are still resolved by the hit finder.

**Data:** TBD — runs spanning a GMD range at several photon energies.

**Subtasks:**
- [ ] ☐ Counts vs GMD per photon energy; look for sub-linearity at high
      GMD (extends `plot_linearity`).
- [ ] ☐ Inspect inter-hit TOF spacing distribution for a dead-time /
      merging signature at small separations.
- [ ] ☐ Recommend a usable GMD range for quantitative analysis.

**Notes:**

---

## 6. Argon covariance (detector volume overlap)  ☐
**Notebook:** `analysis/post/argon_covariance.ipynb`
**Objective:** Electron–electron, ion–ion, and electron–ion covariance on
Argon to interpret features and confirm the eTOF and iTOF collection
volumes overlap (coincident signal ⇒ shared volume).

**Data:** **Run 58826** (Argon).

**Subtasks:**
- [ ] ☐ Histogram per-shot eTOF and iTOF; build Cov(e,e), Cov(i,i),
      Cov(e,i) (reuse / extend `covariance_inspect_cfg1`).
- [ ] ☐ GMD-controlled partial covariance Cov(e,i|G) to suppress the
      common-pulse-energy correlation.
- [ ] ☐ Interpret coincidence features; assess volume overlap.

**Notes:**
- Builds on the existing config-1 covariance machinery.

---

## 7. Train uniformity of pulse energy & spectrum (VLS)  ☐
**Notebook:** `analysis/post/train_uniformity_vls.ipynb`
**Objective:** Verify the per-bunch pulse-energy distribution and the
X-ray spectrum (VLS arm) are constant across the bunch train.

**Data:** TBD — config-2 run(s) with VLS.

**Subtasks:**
- [ ] ☐ GMD distribution per bunch index across the train.
- [ ] ☐ VLS mean spectrum / spectral moments (COM, width) per bunch index.
- [ ] ☐ Quantify any drift along the train.

**Notes:**
- Uses `compute_vls_moments` / `vls_inspect` machinery.

---

## Backlog / cross-cutting
- [ ] ☐ Confirm run-number ↔ sample mapping table above against the logbook.
- [ ] ☐ Decide storage format + location for the TOF→KE calibration so all
      notebooks load the same one.
- [ ] ☐ Promote any helper reused across ≥2 post notebooks into the
      `analysis/scripts/` package.

## Incident-energy intensity dips: 150C versus 7 June / 147C  ◐
**Notebook:** appended comparison section in `Bunch investigations.ipynb`.
**Reader:** `incident_yield_comparison.py` (chunked, read-only; Slurm CLI).

- [x] Shared-scale maps and lineouts at overlapping energies; exposure/yield comparison.
- [x] Identical broad/fixed GMD selections, fixed/tracking feature windows, e/i ratios.
- [x] Acquisition-block, bunch, delay and available beam/DAQ monitor diagnostics.
- [x] Synthetic numerical checks and four-train smoke checks in each 274 eV source.
- [ ] Run the full Slurm comparison, inspect its results and establish which causes
      remain consistent with the intensity dips. Preview outputs are not evidence.

Selections and calibration are recorded in a timestamped manifest; original
notebook analysis and raw data are unchanged. The 147C scan does not cover the
150C points below 274 eV, so direct family comparisons use shared energies only.

## Full-run chronology and candidate ion-monitor diagnostics  ◐
**Notebook:** new final section in `Bunch investigations.ipynb`.
**Reader:** `incident_chronology.py` (full data, chunked; notebook-local or Slurm execution).
The chronology section defaults to `CHRON_EXECUTION_MODE = "local"`; `"slurm"`
retains batch preparation/loading. `CHRON_RESULT_FILE` can reuse results in either mode.

- [x] Actual tID chronology/provenance and matched energy/order/tID observable plots.
- [x] Eight equal valid-train blocks, monitor mean/std/median/MAD, Pearson tables.
- [x] Diagnostic E/I, A/I, B/I, overlap ratios and strict JSON per-run summary;
      no ion-based correction is applied.
- [x] Synthetic alignment/count/uncertainty checks and appended-cell execution checks.
- [x] Lightweight full tID inventory: both series increase in energy with chronology.
      The 147C 276.5 eV source has 53,293 retained rows with tID=0; flag chronology
      as unavailable, keep energy diagnostics, label source-row blocks and suppress
      unverified train-cluster errors. No IDs or raw data are repaired or removed.
- [ ] Run the new full-data Slurm job and assess common modulation versus
      detector-specific effects. Chronology and photon energy remain confounded.


## Weak Auger search: 7 June / 147 C electron–electron and ion gates  ◐
**Notebook:** `ee_ion_covariance_auger_search.ipynb`.
**Reader:** `ee_ion_covariance.py` (read-only train chunks and selective aggregate rows).

- [x] Verified all six 274.0–276.5 eV sources and fine config-1 aggregate schemas.
- [x] Full recorded coverage, ion spectra, unique pairs, pooled covariance/GMD partial covariance.
- [x] Configurable moving-PE projections, Boolean ion/control gates, final comparison panels.
- [x] Synthetic comparison with existing loader/aggregate writer and direct numerical moments.
- [x] Notebook-cell smoke check: eight trains/file and small aggregate KE subsets; optional branches exercised with unassigned validation ROIs.
- [ ] Supply a justified 10a′ BE window and establish ion mass/charge assignments and sidebands.
- [ ] Run full diagnostics with a Slurm compute-node kernel and assess backgrounds/uncertainties.

Defaults remain small previews, with PE/ion gates unset and Cb disabled. Raw `[1,12)`
GMD selection cannot be recovered from existing aggregate bin boundaries; selected
aggregate cells/populations and native-bin KE coarsening are reported explicitly.
