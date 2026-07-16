# Keys Extraction Audit

Raw file: `11022188/raw/hdf/online-0/fl2user1/FLASH2_USER1_main_run58780_file1_20260603T100002.1.h5`
Updated keys file: `.agents/skills/ml-feature-eda/references/keys`

## Counts

- H5 groups: 949
- H5 datasets: 700
- H5 parameter groups with both `index` and `value` datasets: 349
- `raw_param_inventory.csv` parameter rows: 349

## Coverage Check

- Inventory paths missing from H5 parameter groups: 0
- H5 parameter groups missing from inventory: 0
- Inventory paths missing `/index` dataset: 0
- Inventory paths missing `/value` dataset: 0

All 349 raw H5 `index/value` parameter groups are present in both `references/keys` and `raw_param_inventory.csv`.
