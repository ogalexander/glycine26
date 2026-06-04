# How A Is Built From VLS In `compute_xas_aggregates.py`

This note explains how the processed VLS data becomes the output `A`
matrix in `analysis/scripts/compute_xas_aggregates.py`.

The short version is:

1. load processed VLS and GMD arrays for the run,
2. crop the VLS pixel axis,
3. subtract a trainwise VLS background,
4. detect open shutter sections,
5. subtract a section background from the open section,
6. keep the signal bunches,
7. flatten the shots,
8. bin by GMD,
9. accumulate into `A`, then normalize to a per-bin mean.

## 1) Start From Processed VLS

The script first loads the processed run data and extracts the VLS and
GMD arrays. The VLS array has shape `(n_trains, m, n_pixels)`, where:

- `n_trains` is the number of trains kept after alignment,
- `m` is the number of bunches per train,
- `n_pixels` is the VLS pixel axis.

Relevant code:

- [analysis/scripts/compute_xas_aggregates.py](../analysis/scripts/compute_xas_aggregates.py)

## 2) Crop The VLS Pixel Axis

The VLS is cropped to the configured pixel ROI before any background
subtraction:

```python
data = data.crop_vls(roi_min, roi_max)
```

This keeps the pixel axis consistent for all later steps.

## 3) Subtract A Trainwise Background

Before section logic runs, the script subtracts a per-train background
from the VLS. That background is the mean spectrum over the non-signal
bunch range (`BG_BUNCH_RANGE`).

This step removes per-train DC offset and slow dark drift.

## 4) Detect Open Shutter Sections

The code reads the fast shutter signal, builds a per-train VLS score,
and splits the run into open and closed sections. Transitions are
trimmed so the edge trains are excluded.

The open sections are then assigned nominal photon energies by section
index.

## 5) Build The Section Background

For each open section, the script finds the immediately preceding closed
section and computes a 2D background:

- shape: `(m, n_pixels)`

This is stored in `bg2d`.

## 6) Form `A_block`

This is the key step.

For an open section, the script subtracts the section background from
the open-section VLS and keeps only the signal bunches:

```python
A_block = (
    vls[sec_open_idx][:, sig_b0:sig_b1, :]
    - bg2d[None, sig_b0:sig_b1, :]
)
```

So `A_block` has shape:

`(n_sec, n_sig, n_pixels)`

where:

- `n_sec` is the number of open trains in that section,
- `n_sig = sig_b1 - sig_b0`,
- `n_pixels` is the cropped pixel width.

This is the processed VLS data that later becomes `A`.

## 7) Flatten Shots

The block is reshaped from `(train, bunch, pixel)` into individual
shots:

```python
A_flat = A_block.reshape(-1, n_pixels)
```

Each row of `A_flat` is one background-subtracted VLS shot.

## 8) Filter Valid Shots

The corresponding GMD values are flattened and used to build a validity
mask:

- GMD must be finite,
- GMD must fall inside the configured GMD bins,
- the VLS row must be finite across all pixels.

Only valid shots contribute to the accumulators.

## 9) Bin By GMD And Accumulate Into `A`

For each GMD bin, the script selects the valid shots and adds their VLS
spectra into the running sum:

```python
A_sum[e_idx, b] += A_b.sum(axis=0)
```

So `A_sum` is a sum of spectra, not the final mean yet.

The same bin also accumulates:

- `AtA_sum` for second-order VLS moments,
- `AtG_sum` for VLS-GMD cross terms,
- `G_sum` and `GtG_sum` for GMD statistics.

## 10) Normalize To The Final `A`

After all sections are processed, the code divides by the number of
shots in each bin:

```python
A_mean = A_sum / n_per_bin
```

This final per-bin mean is what gets written to the output H5 dataset
`/A`.

## Final Meaning Of `A`

The output `A` is:

- a per-(nominal energy, GMD bin) mean VLS spectrum,
- already crop-processed and background-subtracted,
- built from valid signal-bunch shots only.

In other words, `A` is the processed VLS data that is ready for the
covariance / XAS analysis layers.
