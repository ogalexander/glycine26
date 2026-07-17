# FLASHUSER Parameter Notes For ML Feature EDA

Last checked: 2026-07-01

Scope: concise reference for the train-resolved raw parameters in
`run58780_file1`. Sources checked:

- FLASHUSER main page and Controls pages, including Train ID readout.
- FLASH HDF5 structure page, especially the FLASH2 GMD, shutter, timing, train
  ID, User DAQ, and `/uncategorized/` sections.
- Linked DESY photon-diagnostics pages for GMD, OPIS, and VLS.
- Raw H5 metadata with `h5dump -A -d` on
  `FLASH2_USER1_main_run58780_file1_20260603T100002.1.h5`.

Searches for exact public definitions of `BUNCH_FIRST_INDEX`,
`BUNCH.POSITION.*.NATIVE`, and `FLASH.DIAG/TOROID/*` did not find a better
public source than the XWiki HDF5 page plus the raw H5 metadata.

## General Rules

- HDF5 datasets are aligned by train ID. Each parameter has its own `index` and
  `value`.
- `zraw` keeps the original DAQ/DOOCS names and is the authority when an HDF5
  path is uncategorised.
- Fast pulse-synchronous data should be near 10 Hz / near full train coverage;
  slow controls may only cover about 10-20% of trains.
- A FLASH train is a 10 Hz burst. A burst can contain 1-800 pulses, separated by
  1-20 usec, with total train length below 800 usec.

## Covered Exact Paths

This note covers these run58780 train-resolved paths:

- `/FL2/Support Infrastructure/Gotthard/images`
- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall`
- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel`
- `/FL2/Beamlines/Fast Shutter/shutter`
- `/Timing/event ID`
- `/Timing/bunch pattern`
- `/Timing/set number of bunches flash1`
- `/Timing/set number of bunches flash2`
- `/Timing/set repetition rate flash1`
- `/Timing/set repetition rate flash2`
- `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH_PATTERN_NATIVE`
- `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.REP_RATE_KHZ.3`
- `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.1.NATIVE`
- `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.2.NATIVE`
- `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.3.NATIVE`
- `/FL1/Electron Diagnostic/Bunch charge/at gun`
- `/uncategorised/FLASH.DIAG/TOROID/1FL0UBC2`
- `/uncategorised/FLASH.DIAG/TOROID/4FL0EXTR`
- `/uncategorised/FLASH.DIAG/TOROID/8FL0DBC1`
- `/uncategorised/FLASH.DIAG/TOROID/1FL1DIAG`
- `/uncategorised/FLASH.DIAG/TOROID/4FL1LOLA`
- `/uncategorised/FLASH.DIAG/TOROID/6FL1DIAG`
- `/uncategorised/FLASH.DIAG/TOROID/18FL2EXTR`
- `/uncategorised/FLASH.DIAG/TOROID/4FL2EXTR`

## Timing 4-Vectors

There are two different timing 4-vectors. They are related, but they are not the
same quantity.

### `/uncategorised/FLASH.DIAG/TIMINGINFO/TIME1.BUNCH.POSITION.{1-3}.NATIVE`

Raw metadata description:

`FLASH{1,2,3} bunch positions On/Off Start Duration`

Field interpretation for this run:

| field | meaning | unit | evidence |
|---:|---|---|---|
| 0 | on/off enable flag | none | values are `1` for active FLASH1/FLASH2, `0` for inactive FLASH3 |
| 1 | timing-window start | usec | raw metadata unit is `usec`; values are `700`, `834.585`, `1499.975` |
| 2 | timing-window duration | usec | values plus start define contiguous windows |
| 3 | reserved / unused here | unknown | all inspected values are `0`; no source says this is pulse count |

Run58780 examples:

| path suffix | first value vector | interpretation |
|---|---:|---|
| `.1.NATIVE` | `[1, 700, 134.585, 0]` | FLASH1 window enabled, from 700 to 834.585 usec |
| `.2.NATIVE` | `[1, 834.585, 665.391, 0]` | FLASH2 window enabled, from 834.585 to about 1499.976 usec |
| `.3.NATIVE` | `[0, 1499.975, 0, 0]` | FLASH3 window disabled in this run |

### `/Timing/set number of bunches flash1/flash2`

Raw DAQ names:

- `/Timing/set number of bunches flash1` ->
  `FLASH.DIAG/TIMINGINFO/TIME1.BUNCH_FIRST_INDEX.1`
- `/Timing/set number of bunches flash2` ->
  `FLASH.DIAG/TIMINGINFO/TIME1.BUNCH_FIRST_INDEX.2`

XWiki explicitly says the 4th number is the set number of pulses. Raw values and
the `BUNCH.POSITION` vectors show the full 4-vector is:

| field | meaning | unit |
|---:|---|---|
| 0 | start index in the native timing grid | native tick index |
| 1 | timing-window length in native ticks | native ticks |
| 2 | bunch spacing in native ticks | native ticks |
| 3 | set number of pulses / bunches | count |

Run58780 examples:

| path | first value vector | interpretation |
|---|---:|---|
| `/Timing/set number of bunches flash1` | `[0, 1215, 0, 0]` | FLASH1 window starts at native tick 0, length 1215 ticks, no requested pulses |
| `/Timing/set number of bunches flash2` | `[1215, 6007, 36, 30]` | FLASH2 window starts at native tick 1215, length 6007 ticks, pulse spacing 36 ticks, 30 requested pulses |

For this run, the native timing tick is about `0.11077 usec`:

- `1215 * 0.11077 usec = 134.585 usec`, matching the FLASH1 window duration.
- `6007 * 0.11077 usec = 665.391 usec`, matching the FLASH2 window duration.
- `700 + 1215 * 0.11077 usec = 834.585 usec`, matching FLASH2 window start.
- `36 * 0.11077 usec = 3.9877 usec`, matching
  `/Timing/set repetition rate flash2 = 250.771 kHz`.

So for ML EDA, use field 3 as the requested bunch count, field 2 as the bunch
spacing in native ticks, and only convert fields 0-2 to time after recording the
native tick conversion used for that run.

## Axis Quantity And Unit

First axis is always train sample, aligned by the dataset `index` train IDs.

| path or family | shape | second axis | third axis | value unit |
|---|---:|---|---|---|
| `/FL2/Support Infrastructure/Gotthard/images` | `(train, 110, 1280)` | Gotthard image row / VLS line slot, index | spectral detector pixel, index | `a.u.` |
| `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/*` | `(train, 8, 30)` | GMD channel field | pulse/bunch index in train | channel-dependent |
| `/FL2/Beamlines/Fast Shutter/shutter` | `(train, 100)` | technical ADC samples; saved for technical reasons | none | `V`, but physical state is 1=open, 0=closed |
| `/Timing/bunch pattern` | `(train, 7222)` | native timing-pattern array element | none | native/none |
| `/Timing/set number of bunches flash1/flash2` | `(train, 4)` | `[start_tick, window_ticks, spacing_ticks, n_pulses]` | none | mixed native/count |
| `/uncategorised/...BUNCH.POSITION.*.NATIVE` | `(train, 4)` | `[enabled, start_usec, duration_usec, reserved]` | none | mixed flag/usec |
| `/FL1/Electron Diagnostic/Bunch charge/at gun` | `(train, 802)` | timing slot in charge trace; coordinate approx `700 + i * 0.996923 usec` | none | `nC` |
| `/uncategorised/FLASH.DIAG/TOROID/*` | `(train, 802)` | timing slot in toroid charge trace; coordinate approx `700 + i * 0.996923 usec` | none | `nC` |

## GMD Channel Meaning

For GMD pulse-resolved arrays, raw metadata gives:

| channel | raw column | quantity | unit |
|---:|---|---|---|
| 0 | `INTENSITY.TD` | pulse intensity | `au` |
| 1 | `INTENSITY_AUX.TD` | auxiliary GMD intensity | `au` |
| 2 | `X.TD` | horizontal beam position | `mm` |
| 3 | `Y.TD` | vertical beam position | `mm` |
| 4 | `INTENSITY.SIGMA.TD` | pulse-intensity sigma / error indicator | `au` |
| 5 | `X.SIGMA.TD` | horizontal-position sigma | `mm` |
| 6 | `Y.SIGMA.TD` | vertical-position sigma | `mm` |
| 7 | `TSS` | combined warning/error flags | `none` |

The current static-XAS code uses GMD channel 0. Do not average the channel axis
unless there is an explicit physical reason.

## Charge / Toroid Parameters

`/FL1/Electron Diagnostic/Bunch charge/at gun` and
`/uncategorised/FLASH.DIAG/TOROID/*` are all charge traces:

- raw DOOCS column: `CHARGE.TD`
- value unit: `nC`
- second axis: timing slot across the train
- `at gun` raw DAQ name: `FLASH.DIAG/TOROID/3GUN`

The individual `TOROID/*` names encode different accelerator locations. Public
XWiki does not give enough detail to map every device name to a precise physical
position; use the device database/logbook for strong location claims.

## Practical Feature Guidance

- Direct shot-resolved feature: GMD channel 0 with explicit GMD bunch selection.
- VLS/Gotthard is the master detector axis; in current configs VLS rows `10:40`
  map to GMD raw bunches `0:30`.
- Timing and charge vectors are useful diagnostics, but should be reduced to
  selected physical slots or decoded fields before ML use.
- Train ID, timing counters, raw native codes, and shutter state are alignment
  or run-state variables, not ordinary predictors.
