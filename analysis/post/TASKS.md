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

## 6b. Scaled partial covariance (e–e on KE axis)  ◐  *(scaffold ready)*
**Notebook:** `analysis/post/ee_covariance_ke_scaled_partial.ipynb`
**Objective:** Plot the electron–electron covariance on a KE axis (Task-1
TOF→KE calibration) and test how scaling the GMD common-mode subtraction —
`pCov(α) = Cov(D,D) − α·Cov(D,G)Cov(D,G)ᵀ/Var(G)` with α chosen to
minimise the residual in an a-priori-uncorrelated region of the map —
changes the covariance map shape relative to the standard α = 1 partial.

**Data:** any config-1 aggregates H5 (defaults to the glycine 272.0 eV
delay-scan aggregates) + the Task-1 calibration JSON.

**Subtasks:**
- [x] ☑ Notebook scaffold: KE mapping (Jacobian-corrected density),
      closed-form α* per GMD bin, α scan, map comparisons.
- [ ] ☐ Choose the a-priori-uncorrelated KE rectangles from the raw map
      (placeholder `(30–60) × (150–250)` eV currently).
- [ ] ☐ Run on real aggregates; interpret α* vs GMD bin (α*>1 ⇒
      under-subtraction / nonlinear GMD response).

**Notes:**
- Depends on Task 1's persisted calibration
  (`analysis/post/calibration/etof_tof_to_ke_argon_nozzle_in.json`).
- Same machinery applies to i–i and e–i covariance if useful.

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
