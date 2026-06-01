# Dashboard + DB Quick Start

This project now includes two new folders:

- dashboard: Streamlit app for live monitoring, modular processing, results, and comparisons.
- db: SQLite metadata index for runs, jobs, artifacts, and compact summaries.

The existing analysis code under analysis/scripts remains the processing ground truth. The dashboard calls those existing functions and scripts instead of replacing them.

## Structure

### dashboard

- dashboard/app.py
  - Streamlit entry point.
  - Initializes DB schema and renders pages.
- dashboard/pages/
  - monitor_page.py: scans H5 files and checks readiness/stability.
  - process_page.py: module buttons (Core, Aggregates, optional Covariance/ADMM, Publish).
  - results_page.py: shows latest module outputs.
  - compare_page.py: initial run-to-run comparison scaffold.
- dashboard/services/
  - pipeline.py: adapters that call existing analysis modules.
  - monitor.py: file scanning and stability logic.
  - db_service.py: DB write/read bridge.
  - jobs.py: simple module execution wrapper.
  - signatures.py: compact signatures for settings compatibility.

### db

- db/schema.sql
  - DB tables and indexes.
- db/migrations.py
  - Applies schema.
- db/init_db.py
  - CLI helper to initialize DB.
- db/repository.py
  - High-level persistence API.
- db/connection.py
  - SQLite connection setup.

## How to Use (Brief)

1. Initialize the DB schema:

   python -m db.init_db

2. Start the dashboard:

   streamlit run dashboard/app.py

3. Use pages in order:

- Monitor:
  - Select a stable H5 file (ready state).
- Process:
  - Run Core first.
  - Run Aggregates next.
  - Run Covariance and ADMM only when needed.
  - Publish to save compact results metadata/summaries.
- Results:
  - Inspect outputs from the latest module runs.
- Compare:
  - Select two runs for basic comparison flow.

## Processing Philosophy

- Start simple: Core + Aggregates + Publish gives first scientific visibility fast.
- Keep heavy stages optional: Covariance and ADMM are deferred by default.
- Save selectively: DB stores metadata, summaries, and artifact pointers; heavy arrays stay in H5 outputs.

## Notes

- This is a skeleton for v1 and is intentionally lightweight.
- Existing analysis code is not modified.
- New additions are isolated in dashboard and db folders.
