# Static XAS Dashboard — Quick Start

Interactive Streamlit dashboard for exploring processed static XAS data:
load an H5 file, set GMD bin edges, and add/compare Plotly plot cards.

---

## 1 — Produce the input file

Run the static XAS processor (see `how_to_run_static_xas.md`):

```bash
python analysis/scripts/compute_static_xas.py \
    analysis/configs/xas_static/<config>.py
```

Default output: `11022188/processed/xas_static/run<N>_static_xas.h5`

---

## 2 — Start the dashboard

### Local workstation

```bash
source .venv/bin/activate
streamlit run dashboard/xas_static/app.py
```

Opens automatically at `http://localhost:8501`.

---

### Maxwell JupyterHub

#### Step 1 — Open a terminal on Maxwell

In JupyterHub: **File → New → Terminal**

#### Step 2 — Start Streamlit

```bash
cd /path/to/glycine26
source .venv/bin/activate

FLASH_ENV=remote streamlit run dashboard/xas_static/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
```

> Keep this terminal open (or use `tmux`/`screen`) — the server stops when
> the terminal closes.

#### Step 3 — Open in your browser

```
https://max-jhub.desy.de/user/<your_user>/proxy/8501/
```

For user `kaiyu`:

```
https://max-jhub.desy.de/user/kaiyu/proxy/8501/
```

If that returns 404, try:

```
https://max-jhub.desy.de/user/<your_user>/proxy/absolute/8501/
```

#### Notes
- `FLASH_ENV=remote` makes `config.py` resolve paths on the Maxwell
  filesystem instead of the local workstation.
- Port `8501` is the Streamlit default; change `--server.port` and the
  proxy URL together if it is already in use.
- Install dependencies once on Maxwell:
  ```bash
  pip install streamlit plotly h5py numpy
  ```

---

## 3 — Workflow

### Load data

1. Type the **run number** in the sidebar (e.g. `58793`).  
   The dashboard resolves the path to `xas_static/run58793_static_xas.h5`.
2. Click **Load data** (blue button).  
   A spinner appears while the file is read; results are cached.

### Set GMD bins

3. Edit the **Edges** field — enter a comma-separated list of µJ values,
   exactly like a Python list:  
   `0.0, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8`
4. Click **Apply binning**.  
   Binning is cached — re-clicking with the same edges is instant.

### Add plot cards

5. Choose a plot type from the **Add card** dropdown and click **＋ Add card**.
6. Repeat to build a grid of cards for comparison.
7. Use **✕** on a card to close it; **Clear all** removes every card.

---

## 4 — Plot types

| Card | What it shows | Per-card controls |
|---|---|---|
| **Correlation** | Pooled GMD vs mean VLS (hist2d + binned mean ± std) | Histogram bins slider |
| **Shot counts** | Shots per (energy, GMD bin) heatmap + marginal line | — |
| **Mean GMD** | Mean GMD per (energy, GMD bin) heatmap + per-bin lines | — |
| **VLS spectra** | Mean VLS heatmap (GMD bin × pixel) for one energy | Energy index slider |
| **XAS(E)** | XAS curve per GMD bin (`<GMD> / Σ_pix <VLS>`) | Legend click to toggle bins |

All plots are Plotly — zoom, pan, box-select, hover, and download PNG
from the toolbar in each card.

| **Shot counts** | Shots per (energy, GMD bin) heatmap + marginal line | — |
| **Mean GMD** | Mean GMD per (energy, GMD bin) heatmap + per-bin lines | — |
| **VLS spectra** | Mean VLS heatmap (GMD bin × pixel) for one energy | Energy index slider |
| **XAS(E)** | XAS curve per GMD bin (GMD / sum_pix VLS) | Legend click to hide/show bins |

All plots are Plotly — use the toolbar to zoom, pan, box-select, or download
a PNG. Click legend entries to toggle individual lines/bins.

### Adding multiple cards

You can add the same plot type more than once with different parameters
(e.g. two **VLS spectra** cards at different energy indices) for side-by-side
comparison. Cards are laid out in a 2-column grid.

Remove individual cards with the **✕** button in the card header.
