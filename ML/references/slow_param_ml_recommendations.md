# Slow Parameter ML Recommendations

Last checked: 2026-07-01

Scope: slow / sparse train-indexed raw parameters in `run58780_file1` that may
be useful for ML features after alignment to Gotthard/VLS shots.

This is not a full dictionary. For meanings by device family, see
`low_frequency_param_reference.md`. For shot-resolved GMD channels, see
`flashuser_param_reference.md`.

## Source Basis

- FLASHUSER HDF5 structure page: train-ID alignment, slow-data behavior,
  attenuator/filter/aperture/mirror, GMD, OPIS, undulator, timing, and
  `/uncategorized/` notes.
- DESY photon diagnostics pages for GMD, OPIS, and VLS.
- Raw H5 metadata and run58780 statistics from
  `sparse_train_indexed_params.csv`.

## Main Decision

Do not use slow average GMD energy/position as a first-choice ML feature for
this project. The raw file already has shot-resolved GMD with 8 channels:

- channel 0: pulse intensity,
- channels 2 and 3: horizontal/vertical beam position,
- channel 7: warning/error flags.

Those are closer to the shot-level VLS/Gotthard samples than slow average GMD.
Keep slow GMD average/expert channels for detector-health QC only.

## Recommended Slow Features

### 1. Photon-energy / undulator state

Use when the model should know the scan coordinate or machine photon-energy
setting.

Recommended paths:

- `/Electron Diagnostic/Undulator setting/set wavelength 1`
- `/Electron Diagnostic/Undulator setting/set wavelength 2` only if two-color
  operation is active; it is constant in `run58780_file1`
- `/Electron Diagnostic/Undulator setting/SASE02 gap`
- `/FL2/Electron Diagnostic/Undulator setting/SASE03 gap` through
  `/FL2/Electron Diagnostic/Undulator setting/SASE13 gap`
- `/Electron Diagnostic/Undulator setting/SASE02 k value` through
  `/Electron Diagnostic/Undulator setting/SASE13 k value`
- `/Electron Diagnostic/Undulator setting/gap error`

Feature engineering:

- `set_wavelength_1_nm`
- mean/std/slope of SASE gaps, instead of all gaps blindly
- mean/std/slope of K values, or choose either gap or K to avoid duplication
- `gap_error` as QC/exclusion flag

Run58780 evidence:

- `set wavelength 1` has 8 unique values.
- SASE gaps/K values have 4-5 unique values.

Warning: these variables encode the scan / photon-energy condition. They are
good for conditioning or grouping, but they can be leakage if the goal is to
predict absorption independent of nominal photon energy.

### 2. FL2 gas attenuator state

Use to describe upstream attenuation/transmission setting.

Recommended paths:

- `/FL2/Beamlines/Attenuator/pressure`
- `/FL2/Beamlines/Attenuator/gas type`

Feature engineering:

- previous-value-held `attenuator_pressure_mbar`
- `attenuator_gas_type` as categorical code
- optionally use pressure only for QC if shot-resolved hall GMD already captures
  downstream pulse-energy fluctuations

Run58780 evidence:

- pressure has 411 samples and 78 unique values.
- gas type is constant in this file.

### 3. Beamline filters

Use when comparing multiple runs or when filter wheels change.

Recommended paths:

- `/FL2/Beamlines/Filters/position filter 1`
- `/FL2/Beamlines/Filters/position filter 2`
- `/FL2/Beamlines/Filters/fundamental transmission filter wheel 1`
- `/FL2/Beamlines/Filters/fundamental transmission filter wheel 2`
- `/FL2/Beamlines/Filters/3rd harmonic transmission filter wheel 1`
- `/FL2/Beamlines/Filters/3rd harmonic transmission filter wheel 2`

Feature engineering:

- categorical filter-wheel positions
- map wheel position to material/transmission only if the filter reference or
  logbook mapping is available

Run58780 evidence:

- these are constant in `run58780_file1`, so they are not useful within this
  single file.

### 4. Beamline geometry: apertures and mirrors

Use as run-state/context features when combining files/runs.

Recommended families:

- `/FL2/Beamlines/Tunnel Apertures/position aperture1 horizontal`
- `/FL2/Beamlines/Tunnel Apertures/position aperture1 vertical`
- `/FL2/Beamlines/Tunnel Apertures/position aperture2 horizontal`
- `/FL2/Beamlines/Tunnel Apertures/position aperture2 vertical`
- `/FL2/Beamlines/Hall Apertures/position aperture3-*` through `aperture6-*`
- `/FL2/Beamlines/Tunnel Mirrors/position Mirror1-*`
- `/FL2/Beamlines/Tunnel Mirrors/position Mirror2-*`
- `/FL2/Beamlines/Hall Mirrors/position FL20M3-*`

Feature engineering:

- use as categorical/run-level beamline geometry settings,
- include only if they vary across the training set,
- avoid interpreting absolute motor units without beamline geometry/logbook.

Run58780 evidence:

- all listed aperture/mirror positions are constant in this file.

### 5. Sample / vacuum / gas environment

Use as QC and possible physical covariates if sample/background is sensitive to
gas pressure or valve state.

Recommended paths:

- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/OPCUA.fbPGas1_BL_8_3.fPressureMBar`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/OPCUA.fbPGas2_BL_8_3.fPressureMBar`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/OPCUA.fbV_BL_1.bOpen`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BEAMLINE/OPCUA.fbV_BL_1.bClosed`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.BL5/OPCUA.stUHVG_BL_5_1.fPressure`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.HG0/OPCUA.fbUHVG_HG_0_1.fPressureMBar`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.JS0/OPCUA.fbUHVG_JS_0_1.fPressureMBar`
- `/uncategorised/FLASH.FEL/FL26.VACUUM/REMI.JS3/OPCUA.stUHVG_JS_3_1.fPressure`

Feature engineering:

- previous-value-held pressure values,
- valve open/closed as binary QC state,
- log-transform pressure-like values if used as numeric features.

Run58780 evidence:

- several pressure channels have 300-450 samples and 5-80 unique values.
- valve states are constant, useful as QC.

### 6. OPIS processed photon energy, only if valid

Use only when OPIS online values are nonzero and physically valid.

Candidate paths:

- `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy`
- `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean wavelength`
- `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/number of analysed bunch`

Run58780 decision:

- Do not use these from `run58780_file1`: mean photon energy and wavelength are
  constant zero here.

Across other runs:

- if nonzero, OPIS mean photon energy can be a useful independent photon-energy
  diagnostic, separate from undulator set wavelength.

### 7. Gotthard detector settings

Use for detector configuration consistency, not as a per-shot physical feature.

Candidate paths:

- `/FL2/Support Infrastructure/Gotthard/delay`
- `/FL2/Support Infrastructure/Gotthard/exposure`
- `/FL2/Support Infrastructure/Gotthard/frames`
- `/FL2/Support Infrastructure/Gotthard/period`

Run58780 evidence:

- all are constant in this file.

Use case:

- include in metadata tables or QC checks when combining files with different
  detector settings.

## Conditional / QC Features

Use these for filtering or metadata, not as ordinary predictors:

- `/FL2/Beamlines/Photon Shutter/status PS0`
- `/FL2/Beamlines/Photon Shutter/status PS1`
- `/FL2/Beamlines/Photon Shutter/status PS2`
- OPIS eTOF warning/error status and HV settings
- OPIS gas dosing/supply values
- GMD/XGM gas pressure, gas dosing, range codes, warning/error internals
- DAQ distributor event counters and event masks
- ADC trigger delays and timer trigger delays
- `/FL2/Experiment/FL24/User store/value.*` unless mapped in the experiment
  logbook

## Not Recommended As ML Predictors

Do not use these as first-pass predictors:

- slow GMD average energy / average position, because shot-resolved GMD channels
  are already available,
- GMD/XGM expert current/photon-flux internals, except for detector-health QC,
- raw event/train IDs,
- constants within the training split,
- opaque user-store values without logbook mapping,
- shutter or photon shutter states after selecting open-shutter data.

## Suggested First Feature Set

For a first slow-feature table aligned to Gotthard train IDs, use:

1. `set_wavelength_1_nm`
2. undulator gap summary: mean, std, and slope across SASE02-SASE13
3. undulator K summary: mean, std, and slope across SASE02-SASE13, or omit if
   redundant with gaps
4. `gap_error`
5. `attenuator_pressure_mbar`
6. `attenuator_gas_type`
7. filter wheel categorical positions, only if nonconstant across files/runs
8. selected aperture/mirror positions, only if nonconstant across files/runs
9. selected FL26 vacuum/gas pressures and valve states
10. OPIS mean photon energy/wavelength only for runs where nonzero

Alignment recommendation:

- Use previous-value hold from each slow parameter's `index` to the Gotthard
  train IDs.
- Record `source_index_train_id`, `age_in_trains`, and missing flags for each
  slow feature family.
- Keep slow features separate from shot-resolved GMD/Gotthard shot features so
  the model design can decide whether to broadcast them to shots.

