# BBO_GIB_Bidding_Playground
Playground for experimenting with BBO GIB Bidding.

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