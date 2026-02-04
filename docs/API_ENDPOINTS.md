# API Endpoints (`bbo_bidding_queries_api.py`)

This document lists the FastAPI routes exposed by the BBO Bidding Queries API server.

For a Swagger/OpenAPI view:
- Live Swagger UI (server running): `http://127.0.0.1:8000/docs`
- Static export into repo: `docs/OPENAPI.md` (generates `docs/openapi.json` and `docs/openapi.summary.md`)

### Base URL

- Local dev: `http://127.0.0.1:8000`

### Conventions

- **Request/response**: JSON (unless noted).
- **Errors**: FastAPI-style `{"detail": ...}` with appropriate HTTP status codes.
- **Hot-reload metadata**: many endpoints attach `_reload_info` in the JSON response.
- **Stable detail/explanation schemas**: `/bid-details` and `/explain-bid` are intended to stay schema-stable for UI consumption.

### Endpoints

| Method | Path | Request model / params | Notes |
|---|---|---|---|
| POST | `/new-rules-lookup` | `NewRulesLookupRequest` | Look up detailed new-rules metrics for an auction/bt_index. |
| GET | `/status` | — | Service init / loading status. (`StatusResponse`) |
| POST | `/init` | — | Starts heavy initialization in background. (`InitResponse`) |
| GET | `/custom-criteria-info` | — | Info/stats about CSV overlay criteria. |
| GET | `/custom-criteria-rules` | — | Returns all overlay rules (CSV). |
| POST | `/custom-criteria-rules` | `CustomCriteriaSaveRequest` | Overwrite/save overlay rules (CSV). |
| POST | `/custom-criteria-validate` | `CustomCriteriaValidateRequest` | Validate a criterion expression string. |
| POST | `/custom-criteria-preview` | `CustomCriteriaPreviewRequest` | Preview overlay effects (server-side). |
| POST | `/custom-criteria-reload` | `CustomCriteriaReloadRequest` | Reload overlay CSV without restart. |
| POST | `/openings-by-deal-index` | `OpeningsByDealIndexRequest` | Opening bids by deal index (delegates to plugin handler). |
| POST | `/random-auction-sequences` | `RandomAuctionSequencesRequest` | Random auction sequence samples from BT. |
| POST | `/auction-sequences-matching` | `AuctionSequencesMatchingRequest` | Sample auctions matching a pattern (regex-style). |
| POST | `/auction-sequences-by-index` | `AuctionSequencesByIndexRequest` | Fetch auctions by `bt_index` list. |
| POST | `/deal-criteria-eval-batch` | `DealCriteriaEvalBatchRequest` | Bitmap criteria evaluation for one deal across many checks. |
| POST | `/deals-by-index` | `DealsByIndexRequest` | Fetch deal rows by user-facing `index` (monotonic fast-path). |
| POST | `/bt-categories-by-index` | `BTCategoriesByIndexRequest` | Categories-true lookup for `bt_index` list. |
| POST | `/resolve-auction-path` | `ResolveAuctionPathRequest` | Resolve a full auction string to a normalized per-step path. |
| POST | `/deals-matching-auction` | `DealsMatchingAuctionRequest` | Deals matching an auction pattern + criteria pipeline. |
| POST | `/sample-deals-by-auction-pattern` | `SampleDealsByAuctionPatternRequest` | Sample deals by actual-auction regex (no BT/rules). |
| POST | `/auction-pattern-counts` | `AuctionPatternCountsRequest` | Batch counts for actual-auction regex patterns (no BT/rules). |
| POST | `/bidding-table-statistics` | `BiddingTableStatisticsRequest` | Stats for BT rows / criteria distributions (plugin handler). |
| POST | `/process-pbn` | `ProcessPBNRequest` | Parse PBN/LIN and return computed deal info. |
| POST | `/find-matching-auctions` | `FindMatchingAuctionsRequest` | Find auctions matching hand features (plugin handler). |
| GET | `/pbn-sample` | — | Sample PBN from `deal_df` (for testing). |
| GET | `/pbn-random` | — | Random PBN from `deal_df` (YOLO mode). |
| POST | `/pbn-lookup` | `PBNLookupRequest` | Find a PBN deal in the dataset. |
| POST | `/group-by-bid` | `GroupByBidRequest` | Group deals by bid (plugin handler). |
| POST | `/bt-seat-stats` | `BTSeatStatsRequest` | On-the-fly stats for a BT row/seat. |
| POST | `/execute-sql` | `ExecuteSQLRequest` | Execute **SELECT-only** SQL against registered tables. |
| POST | `/wrong-bid-stats` | `WrongBidStatsRequest` | Aggregate wrong-bid statistics. |
| POST | `/failed-criteria-summary` | `FailedCriteriaSummaryRequest` | Which criteria fail most often. |
| POST | `/wrong-bid-leaderboard` | `WrongBidLeaderboardRequest` | Leaderboard of bids with highest error rates. |
| GET | `/bidding-models` | — | List available bidding models (`MODEL_REGISTRY`). |
| POST | `/bidding-arena` | `BiddingArenaRequest` | Head-to-head model comparison. |
| POST | `/auction-dd-analysis` | `AuctionDDAnalysisRequest` | DD columns for deals matching an auction’s criteria. |
| POST | `/list-next-bids` | `ListNextBidsRequest` | Next-bid options from BT (`next_bid_indices`). |
| POST | `/rank-bids-by-ev` | `RankBidsByEVRequest` | Rank next bids by EV across matching deals. |
| POST | `/contract-ev-deals` | `ContractEVDealsRequest` | Deals matching a chosen next-bid + contract EV row. |
| POST | `/bid-details` | `BidDetailsRequest` | Stable selected-bid details: top-K par contracts + entropy + Phase 2a auction-conditioned posteriors; server-side caching; pinned-deal exclusion; degraded modes. |
| POST | `/explain-bid` | `ExplainBidRequest` | EEO + templated explanation + optional counterfactual (“why not X?”), backed by computed evidence from `/bid-details` (no RAG yet). |
| POST | `/best-auctions-lookahead` | `BestAuctionsLookaheadRequest` | Server-side search for best completed auctions (DD/EV). |
| POST | `/best-auctions-lookahead/start` | `BestAuctionsLookaheadStartRequest` | Async version (returns `job_id`). |
| GET | `/best-auctions-lookahead/status/{job_id}` | path param `job_id` | Poll async job status/results. |
| POST | `/deal-matched-bt-sample` | `DealMatchedBTSampleRequest` | Sample matched BT rows for a pinned deal (GPU-verified index). |

### Notes: `/bid-details` and `/explain-bid`

- **Pinned-deal exclusion**: If `deal_index` is provided, `/bid-details` excludes that deal from aggregates when it’s present in the matched set, and returns:
  - `pinned_deal_excluded`
  - `matched_deals_total_excluding_pinned`
- **Phase 2a**: `/bid-details` returns `phase2a` with SELF/PARTNER/LHO/RHO rotated posteriors (dealer varies per deal).
- **Degraded modes**: `/bid-details` may set `degraded_mode` + `degraded_reasons` when match counts are sparse or phase2a cannot be computed.

