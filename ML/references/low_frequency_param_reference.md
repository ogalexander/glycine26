# Low-Frequency Raw Parameter Reference

Last checked: 2026-07-01

Scope: low-frequency / sparse train-indexed parameters from
`analysis/outputs/ml_feature_eda/run58780_file1/sparse_train_indexed_params.csv`.
There are 325 such rows in `run58780_file1`. They are slow-control or partial
train-ID monitor channels, not shot-resolved features.

Sources checked:

- FLASHUSER HDF5 structure page.
- DESY photon diagnostics pages for GMD, OPIS, and VLS.
- Raw H5 metadata via `h5dump -A -d` on
  `FLASH2_USER1_main_run58780_file1_20260603T100002.1.h5`.

Use the exact `path`, `index_path`, and `value_path` from
`sparse_train_indexed_params.csv` for the full per-channel list. This file gives
the official meaning by device family.

## Low-Frequency Rule

The XWiki HDF5 page states that each DAQ channel has its own train-ID `index`;
slow channels are often saved around 1 Hz or only when values change. For this
run, most slow-control channels have about 62 samples, while GMD average/current
channels have about 470-492 samples. Align by train ID or previous-value hold;
do not treat these as pulse-resolved measurements.

## Family Summary

| family | rows | official meaning | unit / value type | EDA guidance |
|---|---:|---|---|---|
| Undulator settings | 29 | Set undulator gap, K value, afterburner gap/shift, gap error, and set wavelength. XWiki says set wavelength is the anticipated wavelength used to set the undulator gap and may differ from the actual wavelength by several percent. Gap values show how many undulators were closed and taper. | gap `mm`; wavelength `nm`; K value dimensionless | Good run-state metadata. Use to identify wavelength scan sections or undulator drift, not shot noise. |
| FL2 beamline attenuator | 3 | Gas attenuator pressure and gas type. XWiki: pressure is the set pressure in the gas attenuator. | pressure `mbar`; gas type code | Important for transmission context. Slow and not shot-resolved. |
| FL2 filters | 6 | Filter wheel positions and harmonic/fundamental transmission filter wheel state. XWiki documents filter wheel position and says material mapping requires the filter reference. | position usually degree or code; transmission wheel code | Use as section/run-state variables. Need filter mapping before physics interpretation. |
| FL2 apertures | 12 | Hall/tunnel aperture motor positions. XWiki notes aperture positions are saved and to ask local contact for detailed geometry. | motor position, raw metadata often no explicit unit | Use for beamline geometry state. Avoid over-interpreting without beamline logbook. |
| FL2 mirrors | 12 | Hall/tunnel mirror positions: horizontal, vertical, roll, rotation. XWiki says beam steering mirror positions are saved; details require local contact. | motor position, raw metadata often no explicit unit | Use for beamline geometry state and change detection. |
| FL2 photon shutter status | 3 | Photon shutter status PS0/PS1/PS2. Raw metadata: `1=closed, 2=open`. | integer status code | Distinct from fast shutter waveform. Use for beamline open/closed state checks. |
| GMD/XGM average diagnostics | 6 | Average pulse energy and average beam position for tunnel/hall GMD. XWiki: average energy is calibrated SASE energy per pulse with about 20 s averaging; average position is photon beam position determined by GMD. | energy `uJ`; position `mm` | Useful slow energy/position context. Do not replace pulse-resolved GMD for shot-level normalization. |
| GMD/XGM beam-position currents | 4 | Expert beam-position current sums such as `IX.SUM` and `IY.SUM` for tunnel/hall GMD. | current-like raw values, often very small | Expert diagnostic behind the average position calculation. Prefer processed average position unless debugging GMD. |
| GMD/XGM electrometer currents | 20 | Keithley/K2700 and XGM current channels for GMD/XGM. GMD page explains the detector uses electron/ion currents from gas ionization to infer intensity. | current-like values; range/status codes where named `RANGE.CODE` | Expert diagnostic. Use only for detector-health checks, not as first-pass ML features. |
| GMD/XGM photon-flux expert | 36 | XGM photon-flux internals: detector currents, cross sections, pulse energy, number of bunches, and status-like fields. XWiki documents average energy and GMD pulse-energy principles; raw metadata gives exact DOOCS names. | mixed: `uJ`, current, cross-section, count/status | Expert GMD internals. Prefer high-level average energy or pulse-resolved GMD unless troubleshooting calibration. |
| GMD/XGM pressure and gas monitors | 30 | XGM/GMD gas dosing, gas supply, and pressure/readback channels for tunnel/hall devices. Raw metadata includes computed pressure and gas type. | pressure `mbar`; gas IDs/codes; valve/status codes | Detector operating condition. Good for QC, not shot-resolved. |
| GMD two-color metadata | 4 | Two-color operation pulse-energy and wavelength placeholders/readbacks. | wavelength `nm`; pulse energy may be NaN in this run | Mostly constant or missing here. Use only if two-color operation is active. |
| OPIS processed | 6 | Online photoionization spectrometer processed outputs. XWiki: mean photon energy and wavelength are about 1 s averages for a selected bunch from the bunch train; number of analysed bunch is the bunch number used. | photon energy `eV`; wavelength `nm`; bunch number count | In this run processed OPIS values are mostly zero/constant. Use only after confirming OPIS was operating. |
| OPIS eTOF expert | 56 | Electron time-of-flight analysis values: peak/feature ToF, wavelength, prompt position, eTOF voltage status and nominal HV. OPIS page explains electron ToF spectra reflect photoelectron kinetic energy and are used for wavelength determination. | ToF / wavelength / voltage/status, path-dependent | Expert OPIS internals. Most are zero in this run; likely not useful unless OPIS online analysis was active. |
| OPIS gas dosing/supply | 20 | OPIS target gas dosing and supply readbacks for OPIS1/OPIS2. OPIS page explains wavelength measurements use gas targets such as rare gases or small molecules. | gas ID/code; pressure-related values; digital input/output bits | Detector operating condition. Use for OPIS validity/QC. |
| OPIS general/status | 13 | OPIS check interval, active checks, eTOF warning/error status, fixed prompt positions, HV for ROI setting, run-control status. | status/code; interval; voltage-like setting | QC/status only. Do not use as physical predictor. |
| OPIS raw ADC trigger | 2 | Backplane trigger delays for OPIS raw ADC. XWiki says OPIS single-bunch information may require saving full ADC traces. | trigger delay setting | Acquisition timing metadata. |
| Gotthard settings | 4 | Gotthard detector delay, exposure, frames, and period. Raw metadata identifies detector settings; VLS page describes pulse-resolved spectra acquired with MHz line detector/KALYPSO. | exposure `nanosec`; frames count; delay/period settings | Detector configuration; should be constant within a run. |
| User/store values | 31 | User DAQ store values, including `/FL2/Experiment/FL24/User store/value.*` and `FLASH.UTIL/STORE/FL2.TUNNEL.OPIS/VAL*`. XWiki warns short-notice user parameters may appear in `/uncategorized/` with DOOCS names. | unknown or path-specific | Treat as opaque unless mapped from experiment logbook or local contact. |
| Experiment ADC/timing/valve | 7 | ADC backplane trigger delays and Even-Lavie valve delay. XWiki ADC section describes GHz/MHz ADC traces and trigger/sample metadata. | delay/timing setting | Acquisition timing metadata, not beam physics by itself. |
| FL26 vacuum/valves | 13 | REMI/FL26 vacuum pressures and TAS gas valve open/closed states. Raw metadata gives pressure in TAS gas lines and valve status. | pressure `mBar` or raw pressure; valve boolean/integer | Experiment/sample environment and safety state. Useful QC for gas/beamline condition. |
| Timer trigger delays | 4 | PPLASERA front trigger delay/fine-delay settings from `FLASH.FEL/TIMER`. | timing setting | Pump-probe/laser timing metadata; zero in this run. |
| DAQ/distributor state | 3 | DAQ finite-state-machine or distributor event-mask/event-counter channels. | status/counter | DAQ bookkeeping. Avoid as ML feature except QC. |
| FL24 motor | 1 | Motor odometer/readback for `FL24.MOTOR/MOTOR3.MOT1`. | motor position/readback | Experiment geometry metadata; needs logbook for physical meaning. |

## Important Official Details

### Beamline and undulator

- FL2 attenuator pressure is the gas attenuator set pressure, unit `mbar`.
- FL2 filter wheel position is documented as filter-wheel position; filter
  material mapping requires the separate filter reference.
- Aperture and steering-mirror positions are saved, but XWiki explicitly says to
  ask the local contact for more detailed geometry.
- FL2 set wavelength is the anticipated wavelength used to set the undulator
  gap and may differ from actual wavelength by several percent.
- FL2 undulator gap values are saved for SASE undulators and can show closure
  and taper; gap unit is `mm`.

### GMD/XGM

- GMD is a non-invasive intensity and position monitor using gas ionization.
- Average GMD energy is calibrated average SASE energy per pulse, about 20 s
  averaging time, unit `uJ`.
- Average GMD beam position is photon beam position from GMD, unit `mm`.
- Hall GMD is downstream of the gas attenuator. The hall/tunnel ratio can show
  attenuation, but filters and aperture 4 are downstream of the hall GMD and are
  not measured by it.
- Expert current, pressure, gas dosing, and photon-flux channels describe GMD
  detector operating conditions/calibration internals.

### OPIS

- OPIS at FLASH2 uses electron and ion time-of-flight spectrometers to monitor
  photon wavelength.
- Processed OPIS mean photon energy is an about 1 s average for a selected bunch,
  unit `eV`; mean wavelength unit is `nm`.
- OPIS can provide single-shot information from full ADC traces, but that is
  large raw data and requires post-processing.
- OPIS gas dosing/supply and eTOF warning/error channels should be treated as
  detector status/QC variables.

### Experiment/user and vacuum channels

- User/store values and `/uncategorized/` values are often custom or late-added
  channels. XWiki says the correct HDF5 naming may not be available in time, so
  DOOCS names appear instead.
- FL26 vacuum channels are official raw metadata names for REMI/TAS gas-line
  pressure and valve state. Use them as sample-environment/QC metadata.

## Practical ML Guidance

- These low-frequency channels are not shot-resolved.
- For first-pass ML features, prefer physically interpretable slow states:
  undulator wavelength/gap, attenuator pressure, filter position, GMD average
  energy/position, OPIS mean photon energy if nonzero, and vacuum/gas status.
- Treat expert internals, warning/error bits, DAQ counters, and opaque user-store
  values as QC or exclusion variables until mapped to a physical logbook entry.
- Always record alignment method. Previous-value hold is reasonable for slow
  controls; exact train matching alone will mark most Gotthard trains missing.

