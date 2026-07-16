# run58780 file1 raw-H5 feature EDA

Raw file:
`11022188/raw/hdf/online-0/fl2user1/FLASH2_USER1_main_run58780_file1_20260603T100002.1.h5`

This pass was generated from the raw H5 file itself. The `references/keys` file
was not used as the source of truth for shapes or counts.

## Master Axis And Bunch Mapping

- Gotthard master index: `/FL2/Support Infrastructure/Gotthard/images/index`
- Gotthard image shape: `(4330, 110, 1280)`
- Gotthard train IDs: `2657120149 .. 2657124661`
- GMD hall/tunnel raw shape: `(4513, 8, 30)`
- All 4330 Gotthard train IDs are present in the GMD index.
- The VLS selected bunch range is `10:40` in the Gotthard/VLS bunch coordinate.
- That selected 30-shot VLS range maps to GMD raw bunches `0:30`, not `10:30`.

## Time-Resolution Classification

The script found 349 raw `index/value` parameter groups:

| class | count | meaning |
|---|---:|---|
| `gotthard_master_image` | 1 | Gotthard/VLS master data |
| `pulse_resolved_bunch_multi_channel` | 2 | GMD hall/tunnel, `(train, channel, bunch)` |
| `train_resolved_scalar` | 5 | full Gotthard train coverage, scalar per train |
| `train_resolved_matrix_small_axis` | 6 | full Gotthard train coverage, small second axis |
| `train_resolved_array` | 10 | full Gotthard train coverage, long array per train |
| `sparse_train_indexed_monitor` | 325 | partial train-ID overlap; low-rate monitor/slow parameter |

So the strict full-coverage train-resolved set is 24 rows including Gotthard and
GMD. The larger sparse train-indexed set has 325 rows; these do have train-ID
indices, but they do not sample every Gotthard train in this file.

## GMD Channel Summary

Statistics below use exact train-ID alignment to Gotthard and the corrected
mapping `VLS 10:40 -> GMD 0:30`.

| signal | channel | finite fraction | median | p1 | p99 | note |
|---|---:|---:|---:|---:|---:|---|
| hall | 0 | 1.000 | 3.116 | 0.519 | 11.878 | main pulse-energy-like channel |
| tunnel | 0 | 1.000 | 4.583 | 0.583 | 17.494 | main pulse-energy-like channel |
| hall/tunnel | 1, 3, 5, 6 | 1.000 | 0.000 | 0.000 | 0.000 | constant zero in this file |
| hall | 2 | 1.000 | 0.051 | 0.019 | 0.107 | nonzero diagnostic-like channel |
| tunnel | 2 | 1.000 | 0.056 | -1.861 | 1.554 | nonzero diagnostic-like channel |
| hall | 4 | 1.000 | 0.070 | 0.056 | 0.098 | nonzero diagnostic-like channel |
| tunnel | 4 | 1.000 | 0.369 | 0.305 | 0.467 | nonzero diagnostic-like channel |
| hall/tunnel | 7 | 0.033 | 0.000 | 0.000 | 4096.000 | sparse status/flag-like values |

Channel 0 remains the reasonable default pulse-energy channel, but the output
keeps all channels so this assumption is explicit.

## Examples Of Full-Coverage Train-Resolved Parameters

- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall`, `(4513, 8, 30)`
- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel`, `(4513, 8, 30)`
- `/FL2/Beamlines/Fast Shutter/shutter`, `(4513, 100)`
- `/Timing/event ID`, `(4513,)`
- `/Timing/set number of bunches flash1`, `(4513, 4)`
- `/Timing/set number of bunches flash2`, `(4513, 4)`
- `/Timing/bunch pattern`, `(4513, 7222)`
- `/FL1/Electron Diagnostic/Bunch charge/at gun`, `(4513, 802)`
- Several toroid arrays under `/uncategorised/FLASH.DIAG/TOROID/...`, `(4513, 802)`

The complete list is in `train_resolved_params.csv`.

## Examples Of Sparse Train-Indexed Monitors

- OPIS processed mean photon energy: 62 samples, 57 matched Gotthard trains,
  finite range `0 .. 0` in this file.
- Undulator set wavelength 1: 68 samples, 63 matched Gotthard trains, p1-p99
  `4.575 .. 4.592`.
- GMD average energy hall/tunnel: 487/469 samples, about 10 percent Gotthard
  coverage.
- Gotthard delay/exposure/frames/period: 62 samples, effectively run-state
  constants for this file.

The complete list is in `sparse_train_indexed_params.csv`.

## Output Files

- `raw_param_inventory.csv`: every raw `index/value` parameter group with shape,
  train-ID overlap, classification, and sampled robust ranges.
- `train_resolved_params.csv`: full-coverage Gotthard/GMD/train-resolved rows.
- `sparse_train_indexed_params.csv`: partial train-indexed monitor rows.
- `gmd_channel_stats.csv`: GMD hall/tunnel channel ranges using `VLS 10:40 -> GMD 0:30`.
- `gmd_bunch_stats.csv`: GMD hall/tunnel per-channel, per-bunch ranges.
- `keys_extraction_audit.md`: coverage check proving that all raw H5
  `index/value` parameter groups are present in both `references/keys` and
  `raw_param_inventory.csv`.
- `train_resolved_2d_param_plot_index.md`: index of diagnostic plots for every
  2D train-resolved parameter with shape `(train, second_axis)`. VLS/Gotthard
  and GMD 3D arrays are excluded from this focused plot set.
- `figures/`: quicklook plots for class counts and GMD channel/bunch behavior.
