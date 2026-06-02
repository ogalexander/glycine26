## Environment

### Local (development)
- Machine: local workstation
- Data root: `glycine2026\11022188\`
- Raw HDF5: `glycine2026\11022188\raw\hdf5\online-0\fl2user1\`
- Processed local DAQ: `glycine2026\11022188\processed\local_DAQ\`
- Output H5 files: `glycine2026\11022188\processed\combined\`

### Remote (experiment, FLASH online cluster)
- Data root: TBD — placeholder paths in `analysis/scripts/config.py`,
  to be filled in once production paths on the FLASH cluster are confirmed
- Structure mirrors local layout

### Switching Environments
All file paths are resolved through `analysis/scripts/config.py`.
Set the environment variable `FLASH_ENV=remote` to switch to remote paths.
Default is `local`. `config.py` is the **only** place hardcoded paths live.

---

## Data Sources

### Config 1 test data (complete, aligned)
- Raw HDF5 (gmd, mpe, tID):
  `glycine2026\11022188\raw\hdf5\online-0\fl2user1\FLASH2_USER1_stream_2_run48346_*.h5`
- Local DAQ (tID, z, tofs_e, tofs_i):
  `glycine2026\11022188\processed\local_DAQ\delay_scan3\`

### Config 2 test data (synthetic — artificially combined)
- VLS source (different experiment, train IDs NOT aligned):
  `glycine2026\11022188\raw\hdf5\online-0\fl2user1\FLASH2_USER1_main_run54609_file20_20251104T015112.1.h5`
- All other fields (z, liq_tofs_e, mpe, gmd, tID) taken from config 1 data
  above, with `tofs_e` reused as `liq_tofs_e` (same detector type, different
  spectrometer — data is structurally identical for code development purposes)
- VLS data is aligned by index (not train ID) after truncating to the shorter
  of the two datasets
- Output: `glycine2026\11022188\processed\combined\test_config2.h5`

### Production H5 output (config 1 test)
- Output: `glycine2026\11022188\processed\combined\test_config1.h5`

---

## H5 File Schema
Combined H5 files written by `write_h5.py` (production CLI) and the
test-data builders `write_h5_test_config1.py` / `write_h5_test_config2.py`
share this schema. The first index is always the train index. Config 1
has `m = 400` bunches/train; the config 2 test file is truncated to
`m = 100` (the Gotthard VLS bunch count).

| Key                  | Shape        | Units  | Description                                |
|----------------------|--------------|--------|--------------------------------------------|
| `between_tdc_files`  | (n,)         | bool   | True = bad/incomplete train                |
| `gmd`                | (n, m)       | µJ     | Per-bunch pulse energy                     |
| `mpe`                | (n,)         | eV     | Per-train photon energy (upstream)         |
| `tID`                | (n,)         | —      | Train IDs                                  |
| `tofs_e`             | (n, m, 50)   | ns     | Electron TOF hits, zero-padded (cfg 1)     |
| `tofs_i`             | (n, m, 120)  | ns     | Ion TOF hits, zero-padded (cfg 1)          |
| `liq_tofs_e`         | (n, m, 50)   | ns     | Electron TOF hits, zero-padded (cfg 2)     |
| `vls`                | (n, m, 1280) | arb.   | X-ray spectrum from Gotthard (cfg 2)       |
| `z`                  | (n, m)       | arb.   | Delay stage position                       |
| `z_std`              | (n, m)       | arb.   | Std of delay stage position (per train)    |
| `hor_pos`            | (n,)         | arb.   | Per-train horizontal beam position         |
| `ver_pos`            | (n,)         | arb.   | Per-train vertical beam position           |
| `local_DAQ_running`  | (n,)         | bool   | True if local DAQ was active for the train |

The `liq_tofs_e` last dim (50) is inherited from `tofs_e` (synthetic test
data). When real config 2 data is available, confirm the actual maximum
hit count.

### Raw-H5 source paths (config 1)
The config 1 writer pulls these from each `FLASH2_USER1_stream_2_run48346_*.h5`:

| Project key | HDF5 path                                                                                       |
|-------------|--------------------------------------------------------------------------------------------------|
| `gmd`       | `/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value[:, 0, :]` (index 0 of the 8-channel axis = intensity per pulse) |
| `mpe`       | `/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/value`              |
| `hor_pos`   | `/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/value`               |
| `ver_pos`   | `/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/value`                 |

The 8-channel axis of `Pulse resolved energy` is *not* eight detector
channels — it is per-pulse metadata channels (intensity, aux intensity,
horizontal position, vertical position, sigmas of each, combined error
flag). Only index 0 (intensity) is used as `gmd`.

VLS source (config 2): `/FL2/Support Infrastructure/Gotthard/images/value`
in `FLASH2_USER1_main_run54609_file20_*.h5`, shape `(n_trains, 100, 1280)`.

### Optional `vls_moments` group
`save_vls_moments` (in `data_loading.py`) appends a `vls_moments/` group
to a combined H5 file holding per-shot VLS spectral moments:

| Dataset              | Shape   | Description                              |
|----------------------|---------|------------------------------------------|
| `vls_moments/sums`   | (n, m)  | Integrated VLS intensity per shot        |
| `vls_moments/coms`   | (n, m)  | Centre of mass (source-pixel units)      |
| `vls_moments/widths` | (n, m)  | sqrt of 2nd central moment (pixel units) |

Trains absent from the saved subset are NaN-filled. Group attrs
`crop_roi` and `background_roi` record the preprocessing applied before
the moments were computed. `load_data` restores this group when present.

---

## Combined H5 writers

### `write_h5.py` — production CLI
Aligns three streams by train ID — raw FLASH H5 (GMD, MPE, beam
position; +VLS for config 2), local-DAQ SDU `.txt` (z, z_std), local-DAQ
TDC `.lst` (TOFs) — and writes a combined H5 in the schema above. All
paths resolve through `config.py`.

```
python write_h5.py <measurement_name> <config> <run_no> [-o OUT.h5]
                   [--train-length N] [--chunk-size N]
                   [--max-ecounts N] [--max-icounts N]
                   [--n-vls-pixels N] [--folding-parameter F]
```

- `config=1`: TDC ch 1 → `tofs_e`, ch 2 → `tofs_i`.
- `config=2`: TDC ch 3 → `liq_tofs_e`; VLS read from
  `/FL2/Support Infrastructure/Gotthard/images/value` (paired with
  `.../index`) in the same raw H5 files.
- `extract_data_from_single_file` always decodes all three TDC channels;
  unused channels come back empty and never raise.
- `TDCIterator(fpaths, config=...)` defaults to `config=1` (back-compat
  with the test scripts).
- Defaults: `train_length` 400 (cfg 1) / 100 (cfg 2),
  `chunk_size` 1000 / 200, `max_ecounts` 50, `max_icounts` 120,
  `n_vls_pixels` 1280, `folding_parameter` 9969.225 ns.
- Output defaults to `COMBINED_DIR / "<measurement>.h5"`.

### `write_h5_test_config1.py`, `write_h5_test_config2.py`
Test-data builders for the bundled `test_config1.h5` and `test_config2.h5`.
The config 2 test file is synthetic — its VLS comes from a different run
and is index-aligned (not train-ID-aligned) with the config 1 streams.
Use `write_h5.py` for real measurements.

---

## Analysis Scripts API Summary

### `data_loading.py`
- `load_data(filepath, config, trim_start=2, trim_end=2, downsample_N=1) → ExperimentData`
  Loads H5, removes bad trains, trims edges. `downsample_N` keeps only
  1 train in N (testing). Restores a saved `vls_moments` group if present.
- `save_vls_moments(data, filepath, overwrite=False)`
  Scatters per-shot VLS moments back onto the file's full train axis
  (matched by `tID`) and writes a `vls_moments/` group.
- Re-exports `ExperimentData` for backwards-compatible imports.

### `experiment_data.py`
`ExperimentData` dataclass — container for one loaded combined-H5 dataset.
- Fields: `config`, `tID`, `gmd`, `mpe`, `z`, `between_tdc_files`;
  optional `tofs_e`, `tofs_i`, `liq_tofs_e`, `vls`; VLS metadata
  `vls_pixel_ax`, `vls_sums`, `vls_coms`, `vls_widths`, `vls_crop_roi`,
  `vls_background_roi`; `shot_mask`.
- Properties: `.n_trains`, `.n_bunches`, `.mpe_broadcast`, `.vls_pixels`,
  `.valid_shots`, `.n_valid_shots`.
- Transforms — each returns a **new** `ExperimentData` (never mutates):
  - `crop_vls(roi_min, roi_max)` — half-open VLS pixel ROI crop.
  - `subtract_background(background)` — subtract a 1D spectrum.
  - `auto_subtract_background(background_roi)` — subtract the mean
    spectrum over a half-open bunch range.
  - `compute_vls_moments()` — populate `vls_sums/coms/widths`.
  - `filter_shots(values, lo=None, hi=None, inside=True)` — mask shots
    in place; updates the cumulative `shot_mask` (the correct
    denominator for later averaging). Float arrays → NaN, TOF hit
    arrays → zeroed.
  - `filter_trains(values, lo=None, hi=None, inside=True)` — drop whole
    trains; shrinks every field along the train axis.
  - `trim_bunches(bunch_start, bunch_end)` — half-open slice of the
    bunch axis.

### `binning.py`
- `arb_bool_ar(bin_edges, bin_on) → (bin_cents, bool_ar)`
- `bin_bool_ar(start, stop, n_bins, bin_on) → (bin_cents, bool_ar)`
- `bin_bool_ar_pc(start, stop, n_bins, bin_on) → (bin_cents, bool_ar)`
- `bin_and_average(bool_ar, to_average) → (binned, binned_std, binned_n)`
- `bin_and_sum_ratio(bool_ar, numerator, denominator) → (ratio, binned_n)`

### `processing.py`
- `count_hits(tofs, tof_min, tof_max) → counts (n, m)`
- `hits_to_spectrum(tofs, tof_edges) → spectra (n, m, n_bins)`
- `average_spectrum(tofs, tof_edges, mask) → (tof_cents, mean, std)`
- `select_bunch_range(data, bunch_start, bunch_end) → flat dict`
- `crop_to_roi(spectra, roi_min, roi_max) → (cropped, pixel_ax)`
  Half-open crop of the last (pixel) axis; backs `crop_vls`.
- `spectral_moments(spectra, pixel_ax) → (sums, coms, widths)`
  Sum, centre of mass, and width along the last axis.

### `plotting.py`
- `plot_diagnostics(data, ...) → (fig, axes)`
- `plot_linearity(data, ...) → (fig, axes)`
- `plot_energy_binned_maps(data, mpe_edges, e_tof_edges, ...) → (fig, axes)`
- `plot_delay_dependence(data, mpe_range, z_edges, ...) → (fig, axes)`

### `compute_aggregates.py`
Streams a combined H5 in train chunks and writes binned matrix
aggregates for covariance analysis and ADMM ghost imaging — the full
per-shot matrices never live in memory.
- `compute_aggregates(input_h5, output_h5, *, gmd_edges, config,
  mode="spectral", tof_edges=None, ion_tof_edges=None, crop_roi=None,
  background_spec=None, z_edges=None, tof_roi=None, chunk_size=200,
  trim_start=0, trim_end=0, ...)`
  - `config=1`: D = electron TOF, C = ion TOF; computes `DtC`.
  - `config=2`: A = VLS, D = liquid-jet eTOF; computes `AtD`.
  - Both always compute the GMD aggregates `G`, `GtG`, `DtG`
    (+ `AtG`/`CtG`) so downstream code can form partial covariances
    against pulse energy.
  - `mode="time_resolved"` (config 2 only): bins by GMD × stage z; the
    eTOF collapses to a scalar count inside `tof_roi` (`n_tof = 1`).
- `load_aggregates(path) → AggregatesData` — config / mode inferred
  from which datasets are present.
- `AggregatesData` dataclass — all matrix fields are *per-bin means*
  (empty bins NaN). Properties: `.config`, `.mode`, `.n_gmd_bins`,
  `.n_z_bins`, `.n_pixels`, `.n_tof`, `.n_tof_i`, `.gmd_centres`,
  `.z_centres`, `.tof_centres`, `.ion_tof_centres`, `.var_G`
  (= `GtG − G²`).
- CLI: `python compute_aggregates.py CONFIG.py INPUT.h5 [-o OUT.h5]`.
  Output defaults to `<input>_aggregates.h5` (spectral) or
  `<input>_aggregates_tr.h5` (time-resolved).

### `admm_ghost.py`
ADMM solver for regularised spectral-domain ghost imaging. Per GMD bin
it finds the kernel `X` minimising
`½‖MX − B‖²_F + ½λ_p‖D_pX‖²_F + ½λ_t‖XD_tᵀ‖²_F + λ_s‖X‖_1`
with `M = Cov(A)`, `B = Cov(A, D)`.
- `solve_admm(M, B, *, lambda_smooth_pixel, lambda_smooth_tof,
  lambda_sparse, rho=1.0, diff_order=2, max_iter=500, tol_primal=1e-4,
  tol_dual=1e-4, X0=None, ...) → ADMMResult`
  x-step is a Sylvester equation diagonalised once via
  `scipy.linalg.eigh`; z-step is L1 soft-thresholding.
- `ADMMResult` dataclass: `.X`, `.n_iter`, `.converged`, `.history`
  (`primal`, `dual`, `objective` arrays).

### Notebooks (`analysis/notebooks/`)
- `diagnostics.ipynb` — GMD / beam-position / per-train diagnostics
  plus `plot_linearity` (GMD vs electrons and, for config 2, GMD vs
  integrated VLS).
- `tof_inspect.ipynb` — memory-efficient TOF histogramming (indexed-hit
  flattening); average / per-bunch / per-train spectra; GMD vs electron
  count correlation; average TOF spectrum per GMD bin.
- `vls_inspect.ipynb` — VLS crop / background / moments; average / per-
  bunch / per-train spectra; GMD-binned and COM-binned average spectra.
- `covariance_inspect.ipynb` — config 2: photon-photon `Cov(A,A)` and
  photon-electron `Cov(A,D)` per GMD bin.
- `covariance_inspect_cfg1.ipynb` — config 1: electron-ion `Cov(D,C)`
  and GMD-controlled partial covariance `Cov(D,C|G)`.
- `admm_solve.ipynb` — config 2 spectral ADMM inversion per GMD bin.
- `admm_solve_tr.ipynb` — time-resolved ADMM; kernel vs stage position.

### Analysis configs (`analysis/configs/`)
Python-module config files passed to `compute_aggregates.py`. Each
exposes module-level constants (`GMD_EDGES`, `CONFIG`, `MODE`,
`CHUNK_SIZE`, `TRIM_START/END`, plus mode-specific extras).
- `aggregates_example.py` — config 2, spectral (`CROP_ROI`,
  `BACKGROUND`, `TOF_EDGES`).
- `aggregates_example_cfg1.py` — config 1, spectral (`TOF_EDGES`,
  `ION_TOF_EDGES`).
- `aggregates_example_tr.py` — config 2, time-resolved (`Z_EDGES`,
  `TOF_ROI`).

---

## Aggregates H5 Schema
Written by `compute_aggregates.py`. All matrix datasets are **per-bin
means** over the shots in each bin (empty bins are NaN). `N_GMD` =
number of GMD bins; `N_Z` = number of stage-z bins (TR mode only).

### Config 2, spectral mode
| Dataset      | Shape                       | Description               |
|--------------|-----------------------------|---------------------------|
| `A`          | (N_GMD, n_pixels)           | Mean VLS spectrum         |
| `AtA`        | (N_GMD, n_pixels, n_pixels) | `<Aᵢ Aᵢᵀ>`                |
| `D`          | (N_GMD, n_tof)              | Mean eTOF spectrum        |
| `DtD`        | (N_GMD, n_tof, n_tof)       | `<Dᵢ Dᵢᵀ>`                |
| `AtD`        | (N_GMD, n_pixels, n_tof)    | `<Aᵢ Dᵢᵀ>`                |
| `G`, `GtG`   | (N_GMD,)                    | Mean GMD, mean GMD²       |
| `AtG`        | (N_GMD, n_pixels)           | `<Aᵢ Gᵢ>`                 |
| `DtG`        | (N_GMD, n_tof)              | `<Dᵢ Gᵢ>`                 |
| `n_per_bin`  | (N_GMD,)                    | Shot count per bin        |
| `gmd_edges`  | (N_GMD+1,)                  | GMD bin edges (µJ)        |
| `tof_edges`  | (n_tof+1,)                  | eTOF bin edges (ns)       |
| `vls_pixels` | (n_pixels,)                 | Source-pixel indices      |
| `background` | (n_pixels,)                 | Background subtracted     |

### Config 1, spectral mode
As above but with `C` / `CtC` / `DtC` / `CtG` (ion TOF) in place of the
VLS aggregates, `ion_tof_edges` in place of `vls_pixels` / `background`,
and no `A*` datasets. `D` is the electron TOF.

### Time-resolved mode (config 2)
An extra `N_Z` axis is inserted before the per-shot dimensions, `n_tof`
collapses to 1 (scalar eTOF count inside `tof_roi`), and `z_edges`
`(N_Z+1,)` is added.

`load_aggregates()` reads any of these layouts back into `AggregatesData`.

---

## Key Conventions
- **Zero-padded TOF arrays**: hits are non-zero values; zeros are padding.
  Always exclude zeros before histogramming or counting.
- **Normalisation**: energy-normalised averages use `sum(x)/sum(y)` per bin,
  NOT `mean(x/y)`. See `bin_and_sum_ratio`.
- **Bunch selection**: always pass `bunch_start` / `bunch_end` to analysis
  functions. The "good" bunch range must be determined per dataset from the
  diagnostic plots.
- **Config detection**: always passed explicitly as `config=1` or `config=2`.
  Never inferred automatically.
- **Units**: GMD in µJ, TOF in ns, photon energy in eV.
- **All plot functions return `(fig, axes)`** for downstream customisation.

---

## Current Status
- [x] Analysis scripts written (data_loading, experiment_data, binning,
      processing, plotting)
- [x] Config 1 test data available
- [x] VLS test data available (misaligned — combiner script written)
- [x] `write_h5.py` — production CLI; argparse-driven, takes
      `measurement_name`, `config` (1 or 2), `run_no`. Dispatches on
      `config` to write config-1 (e + i TOF) or config-2 (liq eTOF +
      VLS) combined H5 files. All paths via `config.py`.
- [x] `write_h5_test_config1.py` — written; runs end-to-end and produces
      `test_config1.h5` (73 634 trains × 400 bunches).
- [x] `write_h5_test_config2.py` — written; produces `test_config2.h5`
      (4 936 trains × 100 bunches, VLS-truncated).
- [x] `config.py` — written; switches paths via `FLASH_ENV`.
- [x] Writers tested end-to-end on local data (Step 6 validation).
- [x] `ExperimentData` fluent transforms — `crop_vls`,
      `subtract_background`, `auto_subtract_background`,
      `compute_vls_moments`, `filter_shots`, `filter_trains`,
      `trim_bunches`. VLS background subtraction + ROI cropping done.
- [x] `compute_aggregates.py` — written; runs end-to-end for config 1
      and config 2, spectral and time-resolved modes.
- [x] `admm_ghost.py` — ADMM ghost-imaging solver written and verified.
- [x] Notebooks: `diagnostics`, `tof_inspect`, `vls_inspect`,
      `covariance_inspect`, `covariance_inspect_cfg1`, `admm_solve`,
      `admm_solve_tr`.
- [x] `plot_linearity` integrated into `diagnostics.ipynb`; extended so
      config 2 shows integrated VLS as the secondary observable.
- [ ] Remaining plotting notebooks (energy maps, delay dependence) to be
      written.
- [ ] Remote profile paths in `config.py` to be filled in once the
      production location on the FLASH cluster is known.

---

## TODO / Open Questions
- Fill in placeholder remote paths in `analysis/scripts/config.py` once
  production paths are confirmed.
- Confirm VLS pixel axis calibration (pixel → eV) once available — the
  test data uses bare pixel index for the x-axis.
- Confirm `z` units and sign convention (delay stage → time delay in fs).
- Confirm `m` (bunches per train) for production data. Run 48346 has
  `m = 400` per the raw H5; production may differ.
- Confirm the max-hits dimension for real `liq_tofs_e` once a real
  config 2 acquisition is available (currently inherits the test
  `tofs_e` last-dim = 50).
- Write notebooks for `plot_energy_binned_maps` and
  `plot_delay_dependence`.
- Re-calibrate the ADMM regularisation weights (`LAMBDA_SMOOTH_PIXEL`,
  `LAMBDA_SMOOTH_TOF`, `LAMBDA_SPARSE`, `RHO`) on real config 2 data —
  the current defaults are tuned to the synthetic test covariances.