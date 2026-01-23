# UI Functions (Sidebar “Function” Selectbox)

This document describes the Streamlit sidebar **Function** selectbox options and what each function renders in the UI.

- **Source**: `bbo_bidding_queries_streamlit.py`
- **Selector**: `func_choice = st.sidebar.selectbox("Function", [...])`
- **Dispatch**: `match func_choice: ... render_*()`

## Index

- [Deals by Auction Pattern](#deals-by-auction-pattern)
- [Analyze Actual Auctions](#analyze-actual-auctions)
- [Bidding Arena](#bidding-arena)
- [Auction Builder](#auction-builder)
- [Auction Criteria Debugger](#auction-criteria-debugger)
- [New Rules Metrics](#new-rules-metrics)
- [Wrong Bid Analysis](#wrong-bid-analysis)
- [Custom Criteria Editor](#custom-criteria-editor)
- [Rank Next Bids by EV](#rank-next-bids-by-ev)
- [Analyze Deal (PBN/LIN)](#analyze-deal-pbnlin)
- [Bidding Table Explorer](#bidding-table-explorer)
- [Find Auction Sequences](#find-auction-sequences)
- [PBN Database Lookup](#pbn-database-lookup)
- [Random Auction Samples](#random-auction-samples)
- [Opening Bids by Deal](#opening-bids-by-deal)
- [BT Seat Stats (On-the-fly)](#bt-seat-stats-on-the-fly)

## Function → renderer mapping

| UI Function (selectbox label) | Renderer |
| --- | --- |
| Deals by Auction Pattern | `render_deals_by_auction_pattern(pattern)` |
| Analyze Actual Auctions | `render_analyze_actual_auctions()` |
| Bidding Arena | `render_bidding_arena()` |
| Auction Builder | `render_auction_builder()` |
| Auction Criteria Debugger | `render_auction_criteria_debugger()` |
| New Rules Metrics | `render_new_rules_metrics()` |
| Wrong Bid Analysis | `render_wrong_bid_analysis()` |
| Custom Criteria Editor | `render_custom_criteria_editor()` |
| Rank Next Bids by EV | `render_rank_by_ev()` |
| Analyze Deal (PBN/LIN) | `render_analyze_deal()` |
| Bidding Table Explorer | `render_bidding_table_explorer()` |
| Find Auction Sequences | `render_find_auction_sequences(pattern, auction_sequence_indices)` |
| PBN Database Lookup | `render_pbn_database_lookup()` |
| Random Auction Samples | `render_random_auction_samples()` |
| Opening Bids by Deal | `render_opening_bids_by_deal()` |
| BT Seat Stats (On-the-fly) | `render_bt_seat_stats_tool()` |

## Deals by Auction Pattern

- **Short description**: Find deals matching an auction pattern’s criteria and compare Rules vs actual outcomes (DD/EV).
- **Key sidebar inputs**:
  - **Auction Regex** (auto-normalized; anchored when plain)
  - **Settings**: Auction Samples, Deal Samples per Auction, “Match all 4 dealer positions”, Random Seed
  - **Distribution Filter**: Filter Hand, Ordered Distribution (S-H-D-C), Sorted Shape
- **UI sections**:
  - **Auction samples** (per sampled BT auction)
    - Auction header per sample (e.g., “Auction 1: …”)
    - **Auction stats** (small summary grid)
    - **Contract Summary** (contract breakdown table)
    - **Matching Deals** (deals table)
  - **Rejected auctions due to custom criteria** (table of BT rows filtered out by CSV overlay)

## Analyze Actual Auctions

- **Short description**: Group deals by the *actual* auction sequence and analyze outcomes and (optionally) blocking criteria.
- **Key sidebar inputs**:
  - **Auction Regex** (with optional “Match all 4 dealer positions”)
  - **Settings**: Max Auction Groups, Deals per Group, “Show Blocking Criteria”, Random Seed
- **UI sections**:
  - **Auction groups**: per group header like “`<auction>` (N deals, M shown)”
  - **Bidding Table Criteria** (expander; lists `Agg_Expr_Seat_1..4` and `Expr` if available)
  - **Statistics** (direction summary table + score summary table)
  - **Deals** (sampled deals table per group)

## Bidding Arena

- **Short description**: Head-to-head comparison of bidding models (Rules/Actual/…): DD, EV, IMP deltas, and diagnostics.
- **Key sidebar inputs**:
  - **Pinned deal indexes (optional)**: force-include in sample comparisons
  - **Main controls**: Model A / Model B, Sample Size, Seed, optional Auction Pattern filter
- **UI sections** (major headings/subsections):
  - **🏟️ Bidding Arena** (main header)
  - **📊 Summary**
  - **🥊 Head-to-Head**
  - **📈 Contract Quality**
  - **📊 Segmentation Analysis**
  - **⚔️ Deal Comparison** (per selected deal / sample)
  - **🔍 Match Actual Auction** (BT matching / diagnostics)
  - **🔎 Selected Deal: BT Auction Sequence (Step-by-step)** (when drilling into a deal)
  - **Rules matches for this deal (all matching BT rows)** (when applicable)

## Auction Builder

- **Short description**: Build an auction step-by-step from BT continuations, inspect criteria, and validate against a pinned deal.
- **Key sidebar inputs** (high level):
  - **Pinned deal input** (deal index / PBN/LIN depending on configuration)
  - **Sampling controls**: Max Matching Deals, Max Best Auctions, Seed
  - **Path controls**: Apply / Pass Out / Undo / Clear (and direct-edit auction input)
- **UI sections**:
  - **📋 Current Deal** (pinned deal diagram + invariant single-row table)
  - **Bidding Sequence** (editable auction string + controls; includes bt_index in header when available)
  - **Bid suggestion tables** (3-column layout)
    - **Best Bids Ranked by Model**
    - **Best Bids Ranked by DD**
    - **Best Bids Ranked by EV**
  - **Best auctions (optional on demand)**
    - Button: “Show Best N Auctions Ranked by DD/EV”
    - **Best Auctions Ranked by DD**
    - **Best Auctions Ranked by EV**
    - Empty-case message: “Deal has no auction which will result in par score.”
  - **📋 Completed Auction Summary** (step-by-step table, per seat/bid)
  - **Current Auction** (summary grid; selectable)
  - **🎯 Matching Deals** (deal list for the selected step or completed auction)

## Auction Criteria Debugger

- **Short description**: Explain why a specific (target) auction is rejected as a Rules candidate by showing blocking criteria.
- **Key sidebar inputs / controls**:
  - Target Auction Pattern, Sample Size, Seed
- **UI sections** (explicit steps):
  - **1️⃣ BT Row for '<auction>'**
  - **2️⃣ Deals with Actual Auction Matching '<auction>'**
  - **3️⃣ Criteria Failures Analysis**
  - **🚫 Most Common Blocking Criteria** (when present)

## New Rules Metrics

- **Short description**: Inspect metrics for newly discovered criteria from the rule learning pipeline (`bbo_bt_new_rules.parquet`).
- **Key sidebar inputs / controls**:
  - Auction Step (prefix + next bid), optional BT Index
- **UI sections**:
  - Summary metrics (seat, pos/neg counts)
  - **📊 Criteria Details** (single table with membership/metrics across sets like base/accepted/rejected/merged)

## Wrong Bid Analysis

- **Short description**: Analyze wrong bids at scale: overall rates, common failed criteria, and “worst auctions” leaderboard.
- **Key sidebar inputs / controls**:
  - Tab-specific inputs (e.g., Top N criteria)
- **UI sections**:
  - Tabs:
    - **📊 Overall Stats**
      - **Wrong Bid Statistics**
      - **Per-Seat Breakdown**
    - **❌ Failed Criteria Summary**
      - **📊 Top Failed Criteria**
    - **🏆 Leaderboard**

## Custom Criteria Editor

- **Short description**: Manage the hot-reloadable CSV overlay (`bbo_custom_auction_criteria.csv`) used to modify/override criteria without restart.
- **Key sidebar inputs / controls**:
  - Reload from server, view active CSV path
  - Add/edit/delete rules in-session
- **UI sections**:
  - **📋 Current Rules** (editable list)
  - **➕ Add New Rule**
  - **💾 Save Changes**
  - **Rules by Seat** (preview/summary when applicable)

## Rank Next Bids by EV

- **Short description**: Given an auction prefix (or empty), rank all next bids by EV and provide follow-on DD/contract analysis.
- **Key sidebar inputs**:
  - Auction Prefix (empty ⇒ opening bids)
  - Max Deals, Vulnerability filter
  - Output options: Include Hands, Include DD Scores
  - Seed
- **UI sections** (major headings/subsections):
  - **🏆 Which candidate bids perform best?** (ranked bid table)
  - **🥇 Best scoring contracts for deals matching auction: ...** (contract/EV/score view)
  - **📄 Deals matching auction: ...** (deal table)
  - **🏆 Which final contracts score best?**
  - **📊 Par Contract Breakdown**

## Analyze Deal (PBN/LIN)

- **Short description**: Parse a PBN/LIN deal (string/path/URL), optionally compute par, and search for matching BT auctions.
- **Key sidebar inputs**:
  - Example selector + PBN/LIN text area
  - Match Auction Seat
  - Max Auctions to Show
  - Calculate Par Score + Vulnerability for par calc
- **UI sections**:
  - Deal parsing + validation feedback
  - Per-deal output: hands, derived stats, and auction matches (tables)

## Bidding Table Explorer

- **Short description**: Browse BT rows plus aggregate statistics for matching deals; optionally include bid category flags.
- **Key sidebar inputs**:
  - Auction Regex (+ “Match all 4 dealer positions”), Sample Size, Min Matching Deals
  - “Show bid category flags (Phase 4)”
  - Distribution Filter: seat, ordered distribution, sorted shape
- **UI sections**:
  - **Bidding Table Statistics Viewer** (main header)
  - **Data Summary** (metrics row)
  - **Bidding Table with Statistics** (main grid)
  - **Column Descriptions** (expander)
  - Distribution SQL query expander (when a distribution filter is active)

## Find Auction Sequences

- **Short description**: Search BT auction sequences by regex OR by explicit bt_index list; render per-sequence details.
- **Key sidebar inputs**:
  - Input mode: Auction Regex vs bt_index list (mutually exclusive)
  - Number of Samples, “Match all 4 dealer positions”, Seed
- **UI sections**:
  - Effective pattern caption (including 4-seat variants when enabled)
  - Per-sample: “Sample i: <auction> (Opener Seat k)” + sequence grid (with Seat + Agg_Expr_Seat)
  - **Rejected auctions due to custom criteria** (table)

## PBN Database Lookup

- **Short description**: Check whether a given PBN deal exists in the main deal database and show matching rows.
- **Key sidebar inputs**:
  - PBN Deal String (prefilled from `/pbn-sample` when available)
- **UI sections**:
  - Success/failure message (found vs not found)
  - Key info table
  - **Full Deal Data** (when present)
  - Matches table (when present)

## Random Auction Samples

- **Short description**: Randomly sample completed and/or partial auction sequences from BT for quick inspection.
- **Key sidebar inputs**:
  - Number of Samples
  - Auction Type: Completed Only / Partial Only / 50-50 Mix
  - Seed
- **UI sections**:
  - Per-sample “Sample i: <auction>” + sequence grid

## Opening Bids by Deal

- **Short description**: Sample deals and show which opening bids match the opening seat’s criteria.
- **Key sidebar inputs**:
  - Sample Deals
  - Seat filters (all vs subset)
  - Dealer direction filters (all vs subset)
  - Opening direction filters (all vs subset)
  - Seed
- **UI sections**:
  - Per-deal subheader “Dealer X – Deal Index Y”
  - Opening bids grid (ranked/sorted)
  - Deal info grid (hands + invariant auction/contract/par columns)

## BT Seat Stats (On-the-fly)

- **Short description**: Compute seat-level ranges/stats directly from deals for a given `bt_index` using criteria bitmaps.
- **Key sidebar inputs**:
  - bt_index (0 ⇒ sample random)
  - Seat (All / 1 / 2 / 3 / 4)
  - Max Deals (0 ⇒ all)
  - Seed (for random bt_index sampling)
- **UI sections**:
  - Instructional info block
  - **Seat Stats** (main output table)

