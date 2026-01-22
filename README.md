# BBO_GIB_Bidding_Playground

Playground for experimenting with BBO GIB Bidding data analysis and rule learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server (loads ~40GB of data, takes ~2 min)
python bbo_bidding_queries_api.py

# Start Streamlit UI (in a separate terminal)
streamlit run bbo_bidding_queries_streamlit.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                             │
│  (bbo_bidding_queries_streamlit.py)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Server                            │
│  (bbo_bidding_queries_api.py)                               │
│  ├── G3Index (CSR traversal, sub-ms lookups)                │
│  ├── bt_seat1_df (461M rows, lightweight cols)              │
│  ├── deal_df (deals with criteria bitmaps)                  │
│  └── Hot-reloadable plugins (plugins/*.py)                  │
└─────────────────────────────────────────────────────────────┘
```

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

See `Data_Pipeline.md` for the complete pipeline documentation.

## Performance Notes

- **API startup**: ~2 min (loads deals, BT, builds CSR index)
- **Auction traversal**: Sub-millisecond (O(1) CSR lookups)
- **Hot-reload**: Edit `plugins/*.py` without restarting server
- **Memory**: ~40GB RAM for full dataset