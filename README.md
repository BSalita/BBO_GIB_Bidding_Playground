# BBO_GIB_Bidding_Playground

Playground for experimenting with BBO GIB Bidding data analysis and rule learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the main API server that Streamlit talks to
# Loads the in-process core state and hot-reloadable plugins.
python bbo_bidding_queries_api.py

# Start Streamlit UI (in a separate terminal)
streamlit run bbo_bidding_queries_streamlit.py
```

Streamlit currently calls `http://127.0.0.1:8000` directly, so `bbo_bidding_queries_api.py`
must be running before the UI is usable.

## API Surfaces

The repo now has two API layers:

- `bbo_bidding_queries_api.py`: the main FastAPI app and current Streamlit backend
- Split surface wrappers for narrower use cases:
  - `bbo_bidding_queries_api_live.py`
  - `bbo_bidding_queries_api_oracle.py`
  - `bbo_bidding_queries_api_shared.py`

These split servers reuse the same underlying startup/init path and model registry,
so their startup logs look very similar. That is expected. They differ primarily in
which routes they expose.

### Which server should I run?

- For the current Streamlit app: run `python bbo_bidding_queries_api.py`
- For live-safe bidding path endpoints only: run `python bbo_bidding_queries_api_live.py`
- For analysis/oracle endpoints only: run `python bbo_bidding_queries_api_oracle.py`
- For shared data/query endpoints only: run `python bbo_bidding_queries_api_shared.py`

The split surfaces do not replace the main app for Streamlit yet.

## Swagger / OpenAPI

- Main app docs: `http://127.0.0.1:8000/docs`
- Static export: see `docs/OPENAPI.md` (generates `docs/openapi.json` and `docs/openapi.summary.md`)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                             │
│  (bbo_bidding_queries_streamlit.py)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST
┌─────────────────────▼───────────────────────────────────────┐
│              Main FastAPI Server (today)                    │
│  (bbo_bidding_queries_api.py on :8000)                      │
│  ├── G3Index (CSR traversal, sub-ms lookups)                │
│  ├── bt_seat1_df (461M rows, lightweight cols)              │
│  ├── deal_df (deals with criteria bitmaps)                  │
│  └── Hot-reloadable plugins (plugins/*.py)                  │
└─────────────────────────────────────────────────────────────┘
```

Optional split surfaces sit beside the main app and expose narrower route sets for
live-safe flows, oracle/analysis workflows, and shared query/data access.

## Key Features

### Auction Builder
Build auctions step-by-step with instant BT lookups. Uses Gemini-3.2 CSR index for O(1) traversal.

### New Rules Criteria
View learned criteria for any auction step, including:
- Base rules, Accepted/Rejected criteria
- Lift, pos_rate, neg_rate metrics
- Merged (deduped) rules

### Wrong Bid Analysis
Analyze criteria failures across deals to identify problematic rules.

## Learning per-step `New_Rules` from Actual Auctions (Ground Truth)

The rule learning pipeline is implemented via Jupyter notebooks:

- **`bbo_learn_new_rules_bulk.ipynb`** - Discovers high-correlation criteria for each auction step
- **`bbo_filter_new_rules.ipynb`** - Filters rules by quality thresholds and merges with base rules
- **`bbo_bt_compile_rules.py`** - Compiles learned rules back into the main bidding table

See `docs/Data_Pipeline.md` for the complete pipeline documentation.

## Performance Notes

- **API startup**: ~2 min (loads deals, BT, builds CSR index)
- **Auction traversal**: Sub-millisecond (O(1) CSR lookups)
- **Hot-reload**: Edit `plugins/*.py` without restarting server
- **Memory**: ~40GB RAM for full dataset