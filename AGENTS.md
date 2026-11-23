# AGENTS.md

## Project context

This repository contains analysis code for FLASH/DESY x-ray experiment data.

Read `CLAUDE.md` before any non-trivial task involving:
- project structure
- data sources
- file paths
- HDF5 schema
- analysis logic
- experiment-specific assumptions

Do not assume that paths, schemas, or data conventions are obvious from filenames alone.

## Git safety

- Never push.
- Never commit unless explicitly asked.
- Never create remote branches unless explicitly asked.
- Never run `git reset --hard`.
- Never discard, restore, overwrite, or remove user changes unless explicitly asked.
- Preserve unrelated working-tree changes.
- Before editing files, inspect `git status` when relevant.
- After non-trivial edits, summarize which files were changed.
- Use `git diff` to inspect your own changes when useful.

## Maxwell / DESY safety

This repository may be used on DESY Maxwell.

- Do not run heavy analysis on login or display nodes.
- Use small validation subsets for interactive tests.
- Use Slurm for full-dataset, long-running, CPU-heavy, or memory-heavy jobs.
- Do not launch large multiprocessing jobs on display/login nodes.
- Do not modify shared system software.
- Do not install packages into system Python.
- Prefer the project virtual environment when available.
- Do not modify raw experimental data.

## Python environment

The project-local virtual environment is:

`/home/mmanresa/glycine26/.venv`

Prefer this environment for Python commands when working on Maxwell.

The main package list is in:

`requirements.txt`

Do not add or upgrade packages unless needed for the task.
If package changes are required, explain them before making them.

## Data handling

- Treat raw experiment data as read-only.
- Do not overwrite source HDF5 files.
- Write derived outputs only to appropriate processed/results locations.
- Be careful with large arrays and HDF5 datasets.
- Avoid loading complete large datasets into memory unless necessary.
- Prefer chunked, lazy, streamed, or selective loading where appropriate.
- Preserve train/bunch alignment and experiment indexing conventions.
- Do not silently change filtering, normalization, calibration, or binning conventions.

## Scientific analysis

- Preserve the existing analysis method unless the task explicitly asks for a change.
- When introducing a new method, keep the previous method available when practical.
- Clearly separate bug fixes from changes in scientific methodology.
- Do not change physical assumptions, calibration constants, cuts, fitting ranges, or normalization rules without making that change explicit.
- Compare new implementations against the previous implementation when practical.
- Report numerical differences when validating a changed method.
- Do not interpret statistical improvement as physical significance without evidence.

## Notebooks and modules

Prefer reusable Python modules for substantial analysis logic.

Use notebooks mainly for:
- exploration
- parameter choices
- plots
- diagnostics
- comparisons
- interpretation

When code becomes reusable, large, computationally heavy, or suitable for Slurm, prefer moving it into `.py` modules or scripts.

Do not make broad edits to notebooks unless explicitly requested.
When editing notebooks, avoid unnecessary changes to unrelated cells or outputs.

## Testing

For non-trivial code changes:

1. inspect the relevant code and context
2. make the smallest reasonable change
3. run a small or lightweight validation first
4. compare against previous behavior when applicable

Do not run expensive jobs just to verify syntax.



## File editing

- Make focused edits.
- Do not reformat unrelated code.
- Do not rename or reorganize files unless needed.
- Do not delete files unless explicitly asked.
- Do not modify generated data, logs, or checkpoint files unless relevant to the task.
- Keep existing naming conventions and project style where practical.

## Communication

For non-trivial changes, briefly state:
- what you changed
- why
- what was tested
- whether anything remains uncertain

If the task depends on missing information, ask rather than guessing.