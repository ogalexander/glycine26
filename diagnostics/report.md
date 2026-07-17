# Static XAS GMD-Dependence Diagnostics

Mode: full run
Sample: `/Users/hukaiyu/Desktop/PhD/FLASH Beamtime 202605/glycine26/11022188/processed/xas_static/run58794_static_xas.h5`
Reference: `/Users/hukaiyu/Desktop/PhD/FLASH Beamtime 202605/glycine26/11022188/processed/xas_static/reference_frame.h5`
Max shots per energy: `None`
GMD edges: `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]`
VLS peak threshold: `1000.0`
Reject double peaks: `False`
Intensity ROI: `None`

## Selected Energies

- off: requested 272.000 eV, nearest sample nominal 272.000 eV
- peak1: requested 273.500 eV, nearest sample nominal 273.500 eV
- peak2: requested 275.000 eV, nearest sample nominal 275.000 eV
- peak3: requested 278.000 eV, nearest sample nominal 278.000 eV

## Check 1 - Reference-Free Curvature

Figure: `check1_reference_free_curvature.png`
- off: b=-3925.03, se=45.1768, t=-86.88, n=24769 -> ARTIFACT/BEAM
- peak1: b=-4540.05, se=65.3146, t=-69.51, n=15368 -> PHYSICS-CANDIDATE
- peak2: b=-2606.42, se=32.0622, t=-81.29, n=19277 -> PHYSICS-CANDIDATE
- peak3: b=-2813.49, se=40.6405, t=-69.23, n=11655 -> PHYSICS-CANDIDATE

## Check 2 - Is GMD A Valid Fluence Proxy?

Figure: `check2_gmd_proxy_shape_metrics.png`
- centroid: Pearson r=0.0224 at nominal 284.00 eV -> INCONCLUSIVE/OK
- width: Pearson r=0.0084 at nominal 284.00 eV -> INCONCLUSIVE/OK
- peak: Pearson r=0.0541 at nominal 284.00 eV -> INCONCLUSIVE/OK

## Check 3 - I_trans, I_inc, Ratio By GMD

Figure: `check3_itrans_iinc_ratio_by_gmd.png`
- Verdict: ARTIFACT if the off-resonant ratio baseline or I_inc shape moves strongly with GMD; otherwise inspect resonance-specific ratio changes.

## Check 4 - Nominal vs Actual Binning

Figure: `check4_nominal_vs_actual_binning.png`
- Verdict: ARTIFACT if the GMD ordering changes or disappears only under transmitted-center actual-energy binning.

## Check 5 - Negative-Clipping Sensitivity

Figure: `check5_negative_clipping_sensitivity.png`
- Max peak absorbance difference between clip ON and OFF: 0.00206141
- Verdict: ARTIFACT if clipping changes the GMD ordering or amplitude materially.

## Check 6 - Counts And Actual-Energy Distributions

Figure: `check6_counts_and_actual_energy_distributions.png`
- Verdict: ARTIFACT if reference counts collapse or sample/reference actual-energy distributions diverge in the affected cells.

## Check 7 - ROI And Filter Robustness

Figure: `check7_roi_filter_robustness.png`
- Verdict: PHYSICS only if the GMD ordering is stable across ROIs and filtering choices.
- Double-peak robustness pass run: `False`

### Survival Counts By GMD Bin

| run | GMD bin | surviving shots | fraction |
|---|---:|---:|---:|
| sample | [0.0, 1.0) | 34279 | 0.0886 |
| sample | [1.0, 2.0) | 68393 | 0.1768 |
| sample | [2.0, 3.0) | 69654 | 0.1801 |
| sample | [3.0, 4.0) | 60199 | 0.1556 |
| sample | [4.0, 5.0) | 49462 | 0.1279 |
| sample | [5.0, 6.0) | 37978 | 0.0982 |
| sample | [6.0, 7.0) | 27138 | 0.0702 |
| sample | [7.0, 8.0) | 19021 | 0.0492 |
| sample | [8.0, 9.0) | 12758 | 0.0330 |
| sample | [9.0, 10.0] | 7962 | 0.0206 |
| reference | [0.0, 1.0) | 42881 | 0.0937 |
| reference | [1.0, 2.0) | 61101 | 0.1335 |
| reference | [2.0, 3.0) | 60797 | 0.1328 |
| reference | [3.0, 4.0) | 59661 | 0.1303 |
| reference | [4.0, 5.0) | 54855 | 0.1198 |
| reference | [5.0, 6.0) | 47860 | 0.1046 |
| reference | [6.0, 7.0) | 40890 | 0.0893 |
| reference | [7.0, 8.0) | 35373 | 0.0773 |
| reference | [8.0, 9.0) | 29696 | 0.0649 |
| reference | [9.0, 10.0] | 24609 | 0.0538 |

## Check 8 - Fluence vs Saturation Sanity

Figure: `check8_fluence_saturation_sanity.png`
- photon energy: 288.000 eV
- cross section: 1e-18 cm^2
- F_sat = 23.1 J/cm^2
- verdict: INCONCLUSIVE: no focus size configured

## Decision

ARTIFACT/BEAM: off-resonant I_trans also curves with GMD.

A nonlinear-physics claim is credible only if the effect survives nominal-energy binning, survives clipping-off, lives in I_trans rather than I_inc, shows energy-selective reference-free curvature, and is compatible with the fluence scale.
