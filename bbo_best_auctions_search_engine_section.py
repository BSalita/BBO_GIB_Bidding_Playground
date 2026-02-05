    def get_valid_options(
        self,
        prefix_auc: str,
        deal_ctx: DealContext | None,
        timing: TimingStats,
        bt_index: int | None = None
    ) -> list[dict[str, Any]]:
        """Get valid next bids based on the auction prefix and deal context."""
        prefix_norm = "-".join(normalize_auction_tokens(prefix_auc))
        cache_key = (
            prefix_norm,
            deal_ctx.row_idx if deal_ctx else None,
            str(deal_ctx.dealer) if deal_ctx else None,
        )
        if cache_key in self.ui_valid_cache:
            return self.ui_valid_cache[cache_key]

        t0 = time.perf_counter()
        g3 = self.loader.g3_index
        if g3 is None:
            return []

        # 1. Get next child indices
        if bt_index is not None:
            start, end = g3.offsets[bt_index], g3.offsets[bt_index + 1]
            next_indices = g3.children[start:end]
        else:
            # Fallback for root or when bt_index not provided
            bt_prefix, _ = strip_leading_passes(prefix_norm)
            if not bt_prefix:
                next_indices = np.array(list(g3.openings.values()), dtype=np.uint32)
            else:
                curr = g3.walk(bt_prefix)
                if curr is None:
                    next_indices = np.array([], dtype=np.uint32)
                else:
                    start, end = g3.offsets[curr], g3.offsets[curr + 1]
                    next_indices = g3.children[start:end]

        if len(next_indices) == 0:
            self.ui_valid_cache[cache_key] = []
            return []

        # 2. Filter by criteria and dead-ends
        toks = normalize_auction_tokens(prefix_norm)
        dealer_rot = deal_ctx.dealer.rotate(len(toks)) if deal_ctx else Direction.N
        dir_str = str(dealer_rot)
        results = self.evaluator.pre_eval_results.get(dir_str)
        
        valid_opts = []
        for idx in next_indices:
            idx = int(idx)
            is_complete = bool(g3.is_complete[idx])
            has_children = (g3.offsets[idx + 1] > g3.offsets[idx])
            
            # Skip dead ends
            if not is_complete and not has_children:
                continue
                
            # Check criteria
            c_start, c_end = g3.crit_offsets[idx], g3.crit_offsets[idx + 1]
            if c_start < c_end and results is not None:
                passed = True
                for i in range(c_start, c_end):
                    if not results[g3.crit_ids[i]]:
                        passed = False
                        break
                if not passed:
                    continue
            
            # This index is valid!
            valid_opts.append({
                "bt_index": idx,
                "bid": g3.bid_str_map[g3.node_bidcodes[idx]],
                "is_completed_auction": is_complete,
            })

        timing.record(time.perf_counter() - t0)
        self.ui_valid_cache[cache_key] = valid_opts
        return valid_opts

    def _get_options_for_indices(self, next_indices: list[int]) -> list[dict[str, Any]]:
        """Deprecated: Options now built directly in get_valid_options."""
        return []

    def _filter_by_criteria_fast(self, candidates: list[dict[str, Any]], direction: Direction) -> list[dict[str, Any]]:
        """Deprecated: Logic moved to get_valid_options."""
        return []

    def _filter_by_criteria(self, *args, **kwargs):
        """Deprecated: use get_valid_options."""
        return []

    @staticmethod
    def _bt_seat_from_display_seat(display_seat: int, leading_passes: int) -> int:
        """Convert display seat to BT seat, accounting for leading passes."""
        return ((display_seat - 1 - leading_passes) % 4) + 1
