# Raw H5 Feature EDA Summary

Raw file: `11022188/raw/hdf/online-0/fl2user1/FLASH2_USER1_main_run58780_file1_20260603T100002.1.h5`

## Master Axis

- Gotthard index path: `/FL2/Support Infrastructure/Gotthard/images/index`
- Gotthard value path: `/FL2/Support Infrastructure/Gotthard/images/value`
- Gotthard index shape: `(4330,)`
- Gotthard value shape: `(4330, 110, 1280)`
- Gotthard train IDs: 2657120149 .. 2657124661

## GMD

- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value`: shape `(4513, 8, 30)`; VLS selected bunch range `10:40` maps to GMD raw bunch range `0:30`.
- `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel/value`: shape `(4513, 8, 30)`; VLS selected bunch range `10:40` maps to GMD raw bunch range `0:30`.

## Time-Resolution Classes

- sparse_train_indexed_monitor: 325
- train_resolved_array: 10
- train_resolved_matrix_small_axis: 6
- train_resolved_scalar: 5
- pulse_resolved_bunch_multi_channel: 2
- gotthard_master_image: 1

## Outputs

- `raw_param_inventory.csv`: all index/value parameter shapes, alignment coverage, and robust sampled ranges.
- `train_resolved_params.csv`: full-coverage Gotthard/GMD/train-resolved rows.
- `sparse_train_indexed_params.csv`: partial train-indexed monitor rows.
- `gmd_channel_stats.csv`: hall/tunnel GMD channel ranges after exact train alignment to Gotthard.
- `gmd_bunch_stats.csv`: hall/tunnel GMD channel ranges by bunch index after exact train alignment to Gotthard.
- `figures/`: quicklook plots.
