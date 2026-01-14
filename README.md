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

## Learning per-step `New_Rules` from Actual auctions (ground truth)

See `tools/learn_new_rules_per_step.py`.

- Learns per-step criteria that correlate with the *actual next bid* after a given auction prefix.
- Can emit `New_Rules` as `list[str]` per BT step (mapped to a specific `bt_index` + seat).

Example:

```bash
python -X utf8 tools/learn_new_rules_per_step.py ^
  --parent-bt-index 408062489 ^
  --max-deals 100000 ^
  --min-pos 300 ^
  --min-support 0.15 ^
  --top-k 6 ^
  --emit-bt-step ^
  --output-parquet data/new_rules_step_1s.parquet ^
  --progress print
```

## Performance Notes

- **API startup**: ~2 min (loads deals, BT, builds CSR index)
- **Auction traversal**: Sub-millisecond (O(1) CSR lookups)
- **Hot-reload**: Edit `plugins/*.py` without restarting server
- **Memory**: ~40GB RAM for full dataset