# Train-Resolved Parameter Meanings

Source table:
`analysis/outputs/ml_feature_eda/run58780_file1/train_resolved_params.csv`

In this file, "train-resolved" means the parameter index covers every Gotthard
train ID in `run58780_file1`. It does not always mean shot-resolved. Only
parameters with an explicit bunch axis can be mapped to individual FEL shots.

## Detector And FEL Pulse Diagnostics

| path | shape | physical meaning | use in EDA |
|---|---:|---|---|
| `/FL2/Support Infrastructure/Gotthard/images` | `(4330, 110, 1280)` | Gotthard detector images. For this static-XAS work this is the VLS spectrum stack: train, Gotthard line/bunch slot, pixel. | Master detector axis. VLS selected bunches `10:40` are the 30 shot spectra used for XAS. |
| `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall` | `(4513, 8, 30)` | Pulse-resolved gas-monitor detector signal in the experimental hall. Axis meaning is train, GMD channel, GMD raw bunch. | Main shot-resolved FEL pulse energy monitor. Current pipeline uses channel 0. VLS `10:40` maps to GMD `0:30`. |
| `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel` | `(4513, 8, 30)` | Pulse-resolved gas-monitor detector signal in the tunnel, upstream of the hall measurement. Axis meaning is train, GMD channel, GMD raw bunch. | Shot-resolved upstream pulse-energy monitor. Useful for hall/tunnel comparison, transmission checks, and bad-shot flags. |

GMD channel interpretation from this file:

- Channel 0 is the pulse-energy-like channel and is finite for all aligned GMD shots.
- Channels 1, 3, 5, and 6 are constant zero in this file.
- Channels 2 and 4 are nonzero diagnostic-like channels.
- Channel 7 has sparse status/flag-like values, with finite fraction about `1/30`.

## Beamline State

| path | shape | physical meaning | use in EDA |
|---|---:|---|---|
| `/FL2/Beamlines/Fast Shutter/shutter` | `(4513, 100)` | Fast-shutter readback sampled as a short vector per train. It represents shutter state/transition behavior, not FEL intensity. | Section detection and open/closed masking. Do not treat as a normal physical predictor after selecting open-shutter shots. |

## Timing And Bunch Pattern

| path | shape | physical meaning | use in EDA |
|---|---:|---|---|
| `/Timing/event ID` | `(4513,)` | Event/train identifier from the timing system. It is effectively the machine event counter. | Alignment, ordering, grouping, drift plots. Do not use directly as an ML feature. |
| `/Timing/bunch pattern` | `(4513, 7222)` | Full timing-system bunch-pattern array per train. The long second axis encodes the machine bunch pattern over timing slots. | Decode or reduce to selected bunch-slot indicators. Too high-dimensional to use raw as a feature. |
| `/Timing/set number of bunches flash1` | `(4513, 4)` | Timing-system setting for the requested number of bunches on FLASH1, stored as a small multi-field vector. | Run-state check. In this file most values are zero or a fixed setting. |
| `/Timing/set number of bunches flash2` | `(4513, 4)` | Timing-system setting for the requested number of bunches on FLASH2, stored as a small multi-field vector. | Confirms the FLASH2 bunch-pattern program. Contains values including `30`, consistent with the GMD raw bunch count. |
| `/Timing/set repetition rate flash1` | `(4513,)` | Requested FLASH1 repetition-rate setting. | Run-state check only; constant zero in this file. |
| `/Timing/set repetition rate flash2` | `(4513,)` | Requested FLASH2 repetition-rate setting. | Run-state check only; constant in this file. |
| `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH_PATTERN_NATIVE` | `(4513,)` | Native timing-system bunch-pattern code or packed bit-pattern readback. | Machine-state check; constant in this file. Needs decoding before physical use. |
| `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.REP_RATE_KHZ.3` | `(4513,)` | Native timing-system repetition-rate readback for one timing channel. | Machine-state check only; constant zero in this file. |
| `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.1.NATIVE` | `(4513, 4)` | Native timing-system bunch-position readback for bunch-position set 1. | Defines or verifies bunch timing slots. Use only after decoding the 4-vector fields. |
| `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.2.NATIVE` | `(4513, 4)` | Native timing-system bunch-position readback for bunch-position set 2. | Defines or verifies bunch timing slots. Use only after decoding the 4-vector fields. |
| `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.3.NATIVE` | `(4513, 4)` | Native timing-system bunch-position readback for bunch-position set 3. | Defines or verifies bunch timing slots. Use only after decoding the 4-vector fields. |

## Electron-Beam Charge And Toroids

These arrays are train-resolved and have shape `(4513, 802)`. The first axis is
train ID. The second axis is a bunch-slot or timing-slot vector from the
electron-beam diagnostics, not the Gotthard pixel axis.

| path | shape | physical meaning | use in EDA |
|---|---:|---|---|
| `/FL1/Electron Diagnostic/Bunch charge/at gun` | `(4513, 802)` | Electron bunch charge measured near the gun. This is upstream of FLASH2 and records the bunch-charge pattern. | Potential upstream electron-beam predictor after selecting or reducing the relevant bunch slots. |
| `/uncategorised/FLASH.DIAG/TOROID/1FL0UBC2` | `(4513, 802)` | Toroid charge/current monitor near the FL0 upstream bunch-compressor area. | Upstream electron-beam charge monitor. Use reduced bunch-slot features, not the raw 802-vector. |
| `/uncategorised/FLASH.DIAG/TOROID/4FL0EXTR` | `(4513, 802)` | Toroid charge/current monitor in the FL0 extraction/transport area. | Upstream electron-beam charge monitor. Useful for charge transmission/drift checks. |
| `/uncategorised/FLASH.DIAG/TOROID/8FL0DBC1` | `(4513, 802)` | Toroid charge/current monitor in the FL0 diagnostic/bunch-compressor area. | Upstream electron-beam charge monitor. Useful for bunch-pattern and charge stability checks. |
| `/uncategorised/FLASH.DIAG/TOROID/1FL1DIAG` | `(4513, 802)` | Toroid charge/current monitor in an FL1 diagnostic section. | Cross-check of electron-beam charge downstream of early acceleration/transport. |
| `/uncategorised/FLASH.DIAG/TOROID/4FL1LOLA` | `(4513, 802)` | Toroid charge/current monitor near the FL1 LOLA diagnostic section. | Electron-beam diagnostic; likely useful for charge stability, but in this file its range is near zero. |
| `/uncategorised/FLASH.DIAG/TOROID/6FL1DIAG` | `(4513, 802)` | Toroid charge/current monitor in another FL1 diagnostic section. | Electron-beam diagnostic; likely useful for consistency checks, but in this file its range is near zero. |
| `/uncategorised/FLASH.DIAG/TOROID/18FL2EXTR` | `(4513, 802)` | Toroid charge/current monitor in the FLASH2 extraction/transport line. | More directly relevant to FLASH2 beam delivery than FL0/FL1 monitors. Use selected bunch-slot reductions. |
| `/uncategorised/FLASH.DIAG/TOROID/4FL2EXTR` | `(4513, 802)` | Toroid charge/current monitor in the FLASH2 extraction/transport line. | Electron-beam charge monitor close to FLASH2 transport; candidate predictor after bunch-slot reduction. |

The toroid device names encode approximate machine locations. Exact beamline
positions should be confirmed from the FLASH device database or beamline logbook
before making a strong physics claim.

## Practical Feature Guidance

- Direct shot-resolved candidate: GMD hall/tunnel channel 0 with GMD raw bunches
  `0:30`, mapped to VLS bunches `10:40`.
- Train-level or train-array candidates: shutter, timing, charge, and toroid
  diagnostics. These must be reduced or mapped to the selected bunch slots before
  they can be combined with VLS shots.
- Alignment-only or leakage-risk variables: event ID, timing counters, raw
  repetition-rate constants, and shutter after open-shot selection.
- High-dimensional arrays (`7222` timing slots or `802` charge/toroid slots)
  should not be passed raw into a first ML model. First derive finite-count,
  selected-slot, mean/sum, or status features with explicit provenance.
