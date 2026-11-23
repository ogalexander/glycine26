# AGENTS.md

## Project context

This project contains FLASH/DESY analysis for VLS spectra recorded with
the GOTTHARD detector: static photon-energy scans and pump-probe delay
scans measuring soft X-ray transmission and absorption changes.

The molecular target differs from the historical glycine experiment.
Do not assume historical energies, resonances, calibrations or selection
parameters apply. MBES is not part of this experiment.

Before non-trivial work, inspect the relevant project documentation,
configuration, source code and data schema. Treat historical documentation
as a reference that must be checked against the actual implementation.

## Protected sources and output locations

These locations are strictly read-only:

- Historical repository: `/home/mmanresa/glycine26`
- Historical data: `/asap3/flash/gpfs/fl24/2026/data/11022188`

Never modify, overwrite, repair, rename, move or delete anything there.
Do not change historical Git configuration, branches or working-tree files.

Open experimental HDF5 files only with `h5py.File(path, "r")`.
Never use write, append or update mode, including for processed historical
products or saved spectral moments.

Create project files and derived outputs only under:

`/home/mmanresa/FLASH_Gotthard`

Keep logs, caches, checkpoints and temporary outputs there too. Check
resolved output paths, including symlink targets, before writing. Use
appropriate permissions; never broaden permissions on protected sources.

Inspect legacy imports for side effects before using them. Prevent
automatic directory creation, bytecode, caches or output writes inside
protected locations.

Do not bypass sandbox restrictions or request elevated access to modify
protected locations. Report inaccessible or unreadable inputs separately
from missing files; never attempt in-place repairs.

## Historical reuse and traceability

- Prefer local `main` in the historical repository for GOTTHARD/VLS/XAS.
  Inspect it with read-only Git commands without switching branches.
- Record the source commit, exact script/function or notebook cell indices,
  and all adaptations for reused calculations. State the cell-index
  convention.
- Compile useful existing calculations before redesigning the pipeline.
  Keep changes minimal and distinguish bug fixes from methodological changes.
- Flag conflicting historical methods explicitly. Branch precedence does
  not establish scientific correctness.
- Keep source files untouched. Place adapted helpers in this project.
- Distinguish measured experimental data from synthetic test data.

## Configuration and reproducibility

- Centralize input roots, output roots and reusable settings in project
  configuration. Keep input and output locations separate.
- Display resolved input paths, run numbers and scientific parameters in
  notebook setup sections. Avoid hidden current-directory dependencies.
- Record dataset keys, units, selections, calibration provenance,
  preprocessing order and software dependencies.
- Fail clearly when required data or parameters are unavailable.
  Never silently substitute another run, reference, calibration or method.
- Do not invent missing calibration values, metadata or scientific results.

## HDF5 structure and alignment

- Inspect keys, shapes, dtypes and attributes before selecting data.
  Do not infer schema from filenames or historical detector conventions.
- State axis order explicitly. Distinguish raw `(train, frame, pixel)`
  spectra, GMD `(train, channel, pulse)` arrays, flattened shots and
  aggregate bin axes; verify these layouts for each input.
- Treat train-ID matching and within-train GOTTHARD/GMD pulse matching as
  separate operations. Equal array lengths do not prove synchronization.
- Validate missing/duplicate train IDs, frame counts, pulse counts,
  padding, validity flags and mask propagation.
- Make frame windows, GMD pulse windows, channel selection and VLS rolls
  explicit and run-dependent. Never apply a historical offset universally.
- Document cyclic-roll wraparound and exclude invalid wrapped samples
  where justified. Preserve source frame/pulse indices after selection.
- State whether ranges are half-open or inclusive. Keep spectra, GMD,
  delay and pump-state metadata aligned through filtering and reshaping.

## Scientific calculations

- Preserve existing calculations unless a change is explicitly requested
  or necessary; explain any change to assumptions, cuts, binning,
  subtraction, fitting, clipping or normalization.
- State each background source, averaging population, scaling and
  subtraction order. Distinguish detector dark, off-pulse baseline and
  closed-shutter background.
- Treat different normalization methods as distinct observables:
  detector intensity, GMD-normalized signal, inverse normalized signal,
  transmission, optical density and their changes.
- Preserve ratio-of-sums versus mean-of-ratios conventions explicitly.
  Use matching valid populations and report counts/exposure per bin.
- Do not automatically interpret GMD-normalized detector signals as OD.
  Define incident/reference intensity and the logarithm convention before
  calculating OD.
- Do not substitute integrated GMD for a spectrally resolved incident
  reference without documenting and validating the assumptions.
- Define pump/reference populations and delay baselines explicitly.
  Keep absolute differences, relative changes, asymmetries and ΔOD distinct.
- Do not transfer MBES/ion corrections, count-statistics assumptions or
  electron-history models to GOTTHARD data.
- Validate pixel-energy calibration, orientation, magnification, detector
  response and delay conversion/time zero for the applicable acquisition.
  Label historical or provisional parameters accordingly.
- Distinguish standard deviation, SEM and propagated uncertainty.
  Account for shared references/backgrounds and train correlations where
  relevant.
- Separate direct observations from interpretation. Statistical
  improvement alone is not evidence of physical significance.

## Notebooks, modules and plotting

- Use one notebook per meaningful analysis task, with short Markdown
  explanations of purpose, inputs, source provenance and assumptions.
- Keep parameter choices visible. Prefer reusable modules for shared or
  substantial logic and scripts for batch processing.
- Make notebooks runnable in order from a fresh kernel without hidden
  state, missing helper cells or undocumented dependencies.
- Preserve established plotting styles unless a change is requested.
  Label axes, units, normalization and uncertainties accurately.
- Prefer plotting functions that return `(fig, axes)` for customization.
- Avoid unrelated cell/output changes, broad reformatting and unnecessary
  restructuring. Make saving outputs explicit.

## Resources and Python environment

- Prefer chunked, selective or streamed reads. Inspect metadata before
  loading arrays; avoid full detector stacks and covariance matrices when
  unnecessary.
- Use small, explicit subsets for interactive validation. Record subset
  limits and do not present preview results as full-run conclusions.
- Prefer a configured project environment. Record dependencies in the
  project requirements/environment file; do not assume the historical
  environment is the new project's environment.
- Do not modify shared/system software or install into system Python.
  Explain required dependency changes before making them.

## Validation and editing

For non-trivial changes:

1. Inspect the relevant implementation, data schema and provenance.
2. Make the smallest focused change.
3. Validate on a bounded representative subset.
4. Compare against the historical calculation when practical.
5. Report numerical differences, checks performed and remaining limits.

Check alignment, masks, bin populations, empty bins and normalization
denominators where relevant. Do not run expensive analysis merely to
check syntax.

Preserve unrelated user changes and existing naming conventions.
Do not delete files unless explicitly requested.

## Git and communication

- Never push.
- Commit or create remote branches only when explicitly requested.
- Never run `git reset --hard` or discard user changes.
- Inspect Git status before edits when applicable; review diffs afterward.
- Summarize what changed, why, what was tested and what remains uncertain.
- Ask when a required scientific choice is unresolved rather than guessing.
- An inspection or planning request does not authorize notebook execution,
  compilation or file creation.
