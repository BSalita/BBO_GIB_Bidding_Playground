"""
query_benchmarks.py

Benchmarks different modes of querying the same test set:

1) parquet_views: query Parquet directly via DuckDB views over read_parquet(...)
2) in_memory: read Parquet into memory (Polars -> Arrow) and query via DuckDB registered tables
3) build_duckdb_tables: create a persistent .duckdb database containing tables loaded from Parquet
4) duckdb_file: run the test set against an existing .duckdb database

Extra helpful modes included:
5) build_duckdb_views: create a .duckdb database containing views over Parquet paths
6) duckdb_file_views: run the test set against a .duckdb database that contains views over Parquet

The intent is repeatable apples-to-apples comparisons where only the data access mode changes,
not the SQL workload.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import duckdb  # type: ignore[reportMissingImports]
import polars as pl


@dataclass(frozen=True)
class BenchmarkQuery:
    name: str
    description: str
    sql: str


ModeName = Literal[
    "parquet_views",
    "in_memory",
    "duckdb_file",
    "duckdb_file_views",
    "build_duckdb_tables",
    "build_duckdb_views",
]


DEFAULT_DEALS_PARQUET = Path("E:/bridge/data/bbo/data") / "bbo_mldf_augmented.parquet"
DEFAULT_AUCTIONS_PARQUET = Path("E:/bridge/data/bbo/bidding") / "bbo_bt_seat1.parquet"
DEFAULT_DUCKDB_PATH = Path("E:/bridge/data/bbo/bidding") / "bench.duckdb"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _configure_duckdb_conn(
    conn: duckdb.DuckDBPyConnection,
    *,
    threads: int | None,
    memory_limit: str | None,
    enable_object_cache: bool,
    temp_directory: str | None = None,
) -> None:
    # Keep output clean and avoid interactive progress.
    conn.execute("PRAGMA enable_progress_bar=false;")

    if threads is not None:
        conn.execute(f"PRAGMA threads={int(threads)};")

    if memory_limit:
        # DuckDB accepts values like '1GB', '512MB', etc.
        conn.execute(f"PRAGMA memory_limit='{memory_limit}';")

    conn.execute(f"PRAGMA enable_object_cache={'true' if enable_object_cache else 'false'};")

    if temp_directory:
        # Spill-to-disk location for large operations; put this on a fast local SSD.
        conn.execute(f"PRAGMA temp_directory='{temp_directory}';")


def _install_imp_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    A tiny helper table so future benchmarks (or your own ad-hoc SQL) can compute IMPs by join
    rather than a giant CASE.
    """
    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE imp_table(lo INTEGER, hi INTEGER, imp INTEGER);
        INSERT INTO imp_table VALUES
          (0,19,0),
          (20,49,1),
          (50,89,2),
          (90,129,3),
          (130,169,4),
          (170,219,5),
          (220,269,6),
          (270,319,7),
          (320,369,8),
          (370,429,9),
          (430,499,10),
          (500,599,11),
          (600,749,12),
          (750,899,13),
          (900,1099,14),
          (1100,1299,15),
          (1300,1499,16),
          (1500,1749,17),
          (1750,1999,18),
          (2000,2249,19),
          (2250,2499,20),
          (2500,2999,21),
          (3000,3499,22),
          (3500,3999,23),
          (4000,2147483647,24);
        """
    )


def default_benchmarks() -> list[BenchmarkQuery]:
    """
    Keep benchmarks:
    - deterministic (ORDER BY stable columns, avoid RANDOM())
    - projection-friendly (use a limited set of columns)
    - representative (filters, regex, group-by, simple join-like CTE constraint)
    """
    return [
        BenchmarkQuery(
            name="deals_count_hcp_15_17",
            description="Count deals where North HCP in [15, 17].",
            sql="""
                SELECT COUNT(*)::BIGINT AS n
                FROM deals
                WHERE HCP_N BETWEEN 15 AND 17
            """.strip(),
        ),
        BenchmarkQuery(
            name="deals_group_by_bid_1nt_3nt",
            description="Group by actual auction string for 1NT-P-3NT..., compute counts and avg Score-ParScore.",
            sql="""
                WITH d AS (
                  SELECT
                    array_to_string(bid, '-') AS bid_str,
                    HCP_N,
                    Score,
                    ParScore
                  FROM deals
                )
                SELECT
                  bid_str,
                  COUNT(*)::BIGINT AS deal_count,
                  ROUND(AVG(Score - ParScore), 2) AS avg_score_minus_par,
                  ROUND(AVG(HCP_N), 2) AS avg_hcp_n
                FROM d
                WHERE REGEXP_MATCHES(bid_str, '^1N-p-3N', 'i')
                GROUP BY bid_str
                ORDER BY deal_count DESC, bid_str ASC
                LIMIT 25
            """.strip(),
        ),
        BenchmarkQuery(
            name="deals_contract_summary_1nt_3nt",
            description="Contract distribution for deals matching 1NT-P-3NT... auctions.",
            sql="""
                WITH d AS (
                  SELECT
                    array_to_string(bid, '-') AS bid_str,
                    Contract,
                    Score,
                    ParScore
                  FROM deals
                )
                SELECT
                  Contract,
                  COUNT(*)::BIGINT AS n,
                  ROUND(AVG(Score - ParScore), 2) AS avg_score_minus_par
                FROM d
                WHERE REGEXP_MATCHES(bid_str, '^1N-p-3N', 'i')
                GROUP BY Contract
                ORDER BY n DESC, Contract ASC
                LIMIT 30
            """.strip(),
        ),
        BenchmarkQuery(
            name="auctions_regex_completed_1nt_3nt",
            description="Find completed auctions matching exactly 1N-p-3N-p-p-p and show counts.",
            sql="""
                SELECT
                  Auction,
                  matching_deal_count
                FROM auctions
                WHERE is_completed_auction = true
                  AND REGEXP_MATCHES(Auction, '^1N-p-3N-p-p-p$', 'i')
                ORDER BY matching_deal_count DESC, Auction ASC
                LIMIT 50
            """.strip(),
        ),
        BenchmarkQuery(
            name="criteria_apply_single_auction_row",
            description="Pick one auction row deterministically and apply its HCP range to deals (CTE).",
            sql="""
                WITH a AS (
                  SELECT HCP_min_S1, HCP_max_S1
                  FROM auctions
                  WHERE is_completed_auction = true
                  ORDER BY matching_deal_count DESC, Auction ASC
                  LIMIT 1
                )
                SELECT COUNT(*)::BIGINT AS n
                FROM deals, a
                WHERE deals.HCP_N BETWEEN a.HCP_min_S1 AND a.HCP_max_S1
            """.strip(),
        ),
        BenchmarkQuery(
            name="deals_hash_sample_200",
            description="Deterministic sample using hash(index) predicate; returns 200 rows ordered by index.",
            sql="""
                SELECT
                  index,
                  Dealer,
                  Vul,
                  HCP_N,
                  Contract,
                  Score,
                  ParScore
                FROM deals
                WHERE (hash(index) % 1000) = 42
                ORDER BY index ASC
                LIMIT 200
            """.strip(),
        ),
        BenchmarkQuery(
            name="imp_join_micro",
            description="Compute Score-ParScore diffs and map to IMP by joining imp_table (small result).",
            sql="""
                WITH s AS (
                  SELECT
                    index,
                    ABS(Score - ParScore) AS abs_diff
                  FROM deals
                  WHERE (hash(index) % 5000) = 7
                  ORDER BY index ASC
                  LIMIT 1000
                )
                SELECT
                  COUNT(*)::BIGINT AS n,
                  ROUND(AVG(imp_table.imp), 3) AS avg_imp
                FROM s
                JOIN imp_table
                  ON s.abs_diff BETWEEN imp_table.lo AND imp_table.hi
            """.strip(),
        ),
    ]


def _normalize_sql(sql: str) -> str:
    # Best-effort normalization to make logs readable.
    return " ".join(sql.strip().split())


def _execute_fetchall(conn: duckdb.DuckDBPyConnection, sql: str, max_rows: int | None) -> list[tuple[Any, ...]]:
    if max_rows is None:
        return conn.execute(sql).fetchall()
    # Wrap to enforce a hard cap regardless of query text.
    wrapped = f"SELECT * FROM ({sql}) t LIMIT {int(max_rows)}"
    return conn.execute(wrapped).fetchall()


def _timed_runs(
    *,
    conn_factory: Callable[[], duckdb.DuckDBPyConnection],
    setup_fn: Callable[[duckdb.DuckDBPyConnection], None],
    sql: str,
    repeats: int,
    warmup: int,
    max_rows: int | None,
    new_connection_per_run: bool,
) -> tuple[list[float], list[int], list[tuple[Any, ...]] | None, float]:
    """
    Returns:
      - times (seconds) for measured runs
      - row_counts for measured runs (rows returned, after max_rows cap)
      - last_result (optional) for verification/inspection
      - warmup_seconds (seconds) total elapsed time spent in warmup runs
    """
    times: list[float] = []
    row_counts: list[int] = []
    last_result: list[tuple[Any, ...]] | None = None
    warmup_seconds = 0.0

    def _run_once(conn: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
        return _execute_fetchall(conn, sql, max_rows=max_rows)

    # Warmup
    warmup_n = max(0, warmup)
    if warmup_n > 0 and new_connection_per_run:
        t0_w = time.perf_counter()
        for _ in range(warmup_n):
            c = conn_factory()
            try:
                setup_fn(c)
                _run_once(c)
            finally:
                c.close()
        warmup_seconds = time.perf_counter() - t0_w

    if new_connection_per_run:
        for _ in range(repeats):
            c = conn_factory()
            try:
                setup_fn(c)
                t0 = time.perf_counter()
                last_result = _run_once(c)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                row_counts.append(len(last_result))
            finally:
                c.close()
        return times, row_counts, last_result, warmup_seconds

    # Reuse one connection (preferred for isolating scan/cache effects from connect overhead).
    c = conn_factory()
    try:
        setup_fn(c)
        # warmup runs on this same connection:
        if warmup_n > 0:
            t0_w = time.perf_counter()
            for _ in range(warmup_n):
                _run_once(c)
            warmup_seconds = time.perf_counter() - t0_w
        for _ in range(repeats):
            t0 = time.perf_counter()
            last_result = _run_once(c)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            row_counts.append(len(last_result))
    finally:
        c.close()
    return times, row_counts, last_result, warmup_seconds


def _print_table(rows: list[list[Any]]) -> None:
    if not rows:
        return
    cols = len(rows[0])
    widths = [0] * cols
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    for r in rows:
        parts = [str(v).ljust(widths[i]) for i, v in enumerate(r)]
        print("  " + "  ".join(parts))


def _summarize(times: list[float]) -> dict[str, float]:
    if not times:
        return {"min_ms": float("nan"), "p50_ms": float("nan"), "mean_ms": float("nan"), "max_ms": float("nan")}
    ms = [t * 1000.0 for t in times]
    return {
        "min_ms": min(ms),
        "p50_ms": statistics.median(ms),
        "mean_ms": statistics.mean(ms),
        "max_ms": max(ms),
    }


def _mode_requires_db(mode: ModeName) -> bool:
    return mode in {"duckdb_file", "duckdb_file_views"}


def _mode_is_build(mode: ModeName) -> bool:
    return mode in {"build_duckdb_tables", "build_duckdb_views"}


def _build_duckdb_db(
    *,
    db_path: Path,
    deals_path: Path,
    auctions_path: Path,
    kind: Literal["tables", "views"],
    overwrite: bool,
    analyze: bool,
    threads: int | None,
    memory_limit: str | None,
    enable_object_cache: bool,
    temp_directory: str | None,
) -> float:
    if overwrite and db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    conn = duckdb.connect(str(db_path))
    try:
        _configure_duckdb_conn(
            conn,
            threads=threads,
            memory_limit=memory_limit,
            enable_object_cache=enable_object_cache,
            temp_directory=temp_directory,
        )
        if kind == "tables":
            conn.execute("DROP TABLE IF EXISTS deals;")
            conn.execute("DROP TABLE IF EXISTS auctions;")
            t_deals0 = time.perf_counter()
            conn.execute(f"CREATE TABLE deals AS SELECT * FROM read_parquet('{deals_path.as_posix()}');")
            t_deals1 = time.perf_counter()
            print(f"Created table deals in {(t_deals1 - t_deals0) / 60.0:.2f} min")

            t_auc0 = time.perf_counter()
            conn.execute(f"CREATE TABLE auctions AS SELECT * FROM read_parquet('{auctions_path.as_posix()}');")
            t_auc1 = time.perf_counter()
            print(f"Created table auctions in {(t_auc1 - t_auc0) / 60.0:.2f} min")
        else:
            conn.execute("DROP VIEW IF EXISTS deals;")
            conn.execute("DROP VIEW IF EXISTS auctions;")
            conn.execute(f"CREATE VIEW deals AS SELECT * FROM read_parquet('{deals_path.as_posix()}');")
            conn.execute(f"CREATE VIEW auctions AS SELECT * FROM read_parquet('{auctions_path.as_posix()}');")

        # Optional: analyze to improve stats for table-backed DBs.
        if kind == "tables" and analyze:
            t_an0 = time.perf_counter()
            conn.execute("ANALYZE;")
            t_an1 = time.perf_counter()
            print(f"ANALYZE completed in {(t_an1 - t_an0) / 60.0:.2f} min")
    finally:
        conn.close()
    t1 = time.perf_counter()
    return t1 - t0


def _setup_mode(
    *,
    mode: ModeName,
    deals_path: Path,
    auctions_path: Path,
    db_path: Path,
    in_memory_deals_cols: list[str] | None,
    in_memory_auctions_cols: list[str] | None,
    threads: int | None,
    memory_limit: str | None,
    enable_object_cache: bool,
    max_rows: int | None,
    verify: bool,
    warmup: int,
    repeats: int,
    new_connection_per_run: bool,
    overwrite_db: bool,
) -> tuple[dict[str, Any], dict[str, list[tuple[Any, ...]]]]:
    """
    Returns:
      summary: dict with benchmark results and optional build timing
      results_by_benchmark: last result per benchmark (for verification/cross-mode compare)
    """
    benches = default_benchmarks()
    bench_map = {b.name: b for b in benches}

    run_results: dict[str, Any] = {"mode": mode, "benchmarks": {}}
    last_results: dict[str, list[tuple[Any, ...]]] = {}

    # Handle build modes up-front.
    if mode == "build_duckdb_tables":
        build_s = _build_duckdb_db(
            db_path=db_path,
            deals_path=deals_path,
            auctions_path=auctions_path,
            kind="tables",
            overwrite=overwrite_db,
            analyze=True,
            threads=threads,
            memory_limit=memory_limit,
            enable_object_cache=enable_object_cache,
            temp_directory=None,
        )
        run_results["build_seconds"] = build_s
        return run_results, last_results

    if mode == "build_duckdb_views":
        build_s = _build_duckdb_db(
            db_path=db_path,
            deals_path=deals_path,
            auctions_path=auctions_path,
            kind="views",
            overwrite=overwrite_db,
            analyze=False,
            threads=threads,
            memory_limit=memory_limit,
            enable_object_cache=enable_object_cache,
            temp_directory=None,
        )
        run_results["build_seconds"] = build_s
        return run_results, last_results

    def conn_factory() -> duckdb.DuckDBPyConnection:
        if mode in {"duckdb_file", "duckdb_file_views"}:
            return duckdb.connect(str(db_path), read_only=True)
        return duckdb.connect()

    def setup_fn(conn: duckdb.DuckDBPyConnection) -> None:
        _configure_duckdb_conn(conn, threads=threads, memory_limit=memory_limit, enable_object_cache=enable_object_cache)
        _install_imp_table(conn)

        if mode == "parquet_views":
            conn.execute(f"CREATE OR REPLACE TEMP VIEW deals AS SELECT * FROM read_parquet('{deals_path.as_posix()}');")
            conn.execute(
                f"CREATE OR REPLACE TEMP VIEW auctions AS SELECT * FROM read_parquet('{auctions_path.as_posix()}');"
            )
        elif mode == "in_memory":
            # Read into memory (projection-friendly).
            deals_df = pl.read_parquet(str(deals_path), columns=in_memory_deals_cols)
            auctions_df = pl.read_parquet(str(auctions_path), columns=in_memory_auctions_cols)
            conn.register("deals", deals_df.to_arrow())
            conn.register("auctions", auctions_df.to_arrow())
        elif mode in {"duckdb_file", "duckdb_file_views"}:
            # Uses deals/auctions tables or views already present in the file.
            # Validate existence early.
            try:
                conn.execute("SELECT 1 FROM deals LIMIT 1;").fetchone()
                conn.execute("SELECT 1 FROM auctions LIMIT 1;").fetchone()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"DB file does not expose required relations 'deals' and 'auctions': {db_path}"
                ) from e
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    for b in benches:
        times, row_counts, last, warmup_seconds = _timed_runs(
            conn_factory=conn_factory,
            setup_fn=setup_fn,
            sql=b.sql,
            repeats=repeats,
            warmup=warmup,
            max_rows=max_rows,
            new_connection_per_run=new_connection_per_run,
        )

        mean_rows = float(statistics.mean(row_counts)) if row_counts else 0.0
        rows_per_s_mean: float | None = None
        if row_counts and times and len(row_counts) == len(times):
            # Mean of per-run throughput to match "mean rows processed per second".
            throughputs = []
            for rc, t in zip(row_counts, times, strict=True):
                if t > 0:
                    throughputs.append(rc / t)
            rows_per_s_mean = float(statistics.mean(throughputs)) if throughputs else None

        run_results["benchmarks"][b.name] = {
            "description": b.description,
            "sql": _normalize_sql(b.sql),
            "repeats": repeats,
            "warmup": warmup,
            "warmup_total_ms": round(warmup_seconds * 1000.0, 3),
            "result_rows_mean": round(mean_rows, 3),
            "rows_per_s_mean": round(rows_per_s_mean, 3) if rows_per_s_mean is not None else None,
            **_summarize(times),
        }
        if verify and last is not None:
            last_results[b.name] = last

    return run_results, last_results


def _parse_cols_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def _default_in_memory_cols() -> tuple[list[str], list[str]]:
    """
    Defaults chosen to cover the built-in benchmarks while reducing RAM pressure.
    You can override via --in-memory-deals-cols / --in-memory-auctions-cols.
    """
    deals_cols = [
        "index",
        "Dealer",
        "Vul",
        "HCP_N",
        "Score",
        "ParScore",
        "bid",
        "Contract",
    ]
    auctions_cols = [
        "Auction",
        "is_completed_auction",
        "matching_deal_count",
        "HCP_min_S1",
        "HCP_max_S1",
    ]
    return deals_cols, auctions_cols


def cmd_list(args: argparse.Namespace) -> int:
    benches = default_benchmarks()
    print("Benchmarks:")
    for b in benches:
        print(f"- {b.name}: {b.description}")
    print("")
    print("Modes:")
    for m in [
        "parquet_views",
        "in_memory",
        "build_duckdb_tables",
        "duckdb_file",
        "build_duckdb_views",
        "duckdb_file_views",
    ]:
        print(f"- {m}")
    return 0


def cmd_build_db(args: argparse.Namespace) -> int:
    deals_path = Path(args.deals).resolve()
    auctions_path = Path(args.auctions).resolve()
    db_path = Path(args.db).resolve()

    _require_file(deals_path, "deals parquet")
    _require_file(auctions_path, "auctions parquet")

    kind = "tables" if args.kind == "tables" else "views"
    build_s = _build_duckdb_db(
        db_path=db_path,
        deals_path=deals_path,
        auctions_path=auctions_path,
        kind=kind,
        overwrite=args.overwrite,
        analyze=not args.no_analyze,
        threads=args.threads,
        memory_limit=args.memory_limit,
        enable_object_cache=not args.disable_object_cache,
        temp_directory=args.temp_dir,
    )
    print(f"Built {kind} DB at {db_path} in {build_s * 1000.0:.1f} ms")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    deals_path = Path(args.deals).resolve()
    auctions_path = Path(args.auctions).resolve()
    db_path = Path(args.db).resolve()

    if args.mode in {"parquet_views", "in_memory", "build_duckdb_tables", "build_duckdb_views"}:
        _require_file(deals_path, "deals parquet")
        _require_file(auctions_path, "auctions parquet")

    if _mode_requires_db(args.mode):
        _require_file(db_path, ".duckdb database file")

    in_mem_default_deals, in_mem_default_auctions = _default_in_memory_cols()
    in_memory_deals_cols = _parse_cols_arg(args.in_memory_deals_cols) or in_mem_default_deals
    in_memory_auctions_cols = _parse_cols_arg(args.in_memory_auctions_cols) or in_mem_default_auctions

    summary, last_results = _setup_mode(
        mode=args.mode,
        deals_path=deals_path,
        auctions_path=auctions_path,
        db_path=db_path,
        in_memory_deals_cols=in_memory_deals_cols,
        in_memory_auctions_cols=in_memory_auctions_cols,
        threads=args.threads,
        memory_limit=args.memory_limit,
        enable_object_cache=not args.disable_object_cache,
        max_rows=args.max_rows,
        verify=args.verify,
        warmup=args.warmup,
        repeats=args.repeats,
        new_connection_per_run=args.new_connection_per_run,
        overwrite_db=args.overwrite_db,
    )
    # Include run-level parameters in the output artifacts.
    summary["max_rows"] = args.max_rows

    # Human output
    if _mode_is_build(args.mode):
        print(json.dumps(summary, indent=2))
        return 0

    rows: list[list[Any]] = [
        ["benchmark", "p50_ms", "min_ms", "mean_ms", "max_ms", "rows/s (mean_ms/rows)"],
    ]
    total_warmup_ms = 0.0
    for bname, r in summary["benchmarks"].items():
        total_warmup_ms += float(r.get("warmup_total_ms", 0.0))
        mean_rows = float(r.get("result_rows_mean", 0.0) or 0.0)
        rows_per_s = r.get("rows_per_s_mean", None)
        if rows_per_s is None or mean_rows <= 0:
            rows_per_s_cell = ""
        else:
            # Show requested "calc" as mean_ms/rows to make the denominator explicit.
            # Example: 1481.93 (674.80/1000)
            rows_per_s_cell = f"{float(rows_per_s):.2f} ({float(r['mean_ms']):.2f}/{int(round(mean_rows))})"
        rows.append(
            [
                bname,
                f"{r['p50_ms']:.2f}",
                f"{r['min_ms']:.2f}",
                f"{r['mean_ms']:.2f}",
                f"{r['max_ms']:.2f}",
                rows_per_s_cell,
            ]
        )

    print(f"Mode: {summary['mode']}")
    print(f"Max rows: {summary.get('max_rows')}")
    if args.mode == "in_memory":
        print(f"in_memory.deals.columns = {in_memory_deals_cols}")
        print(f"in_memory.auctions.columns = {in_memory_auctions_cols}")
    if args.warmup and args.warmup > 0:
        print(f"Warmup total (all benchmarks): {total_warmup_ms:.3f} ms")
    print("")
    _print_table(rows)

    if args.warmup and args.warmup > 0:
        print("")
        print("Warmup totals by benchmark (ms):")
        for bname, r in summary["benchmarks"].items():
            print(f"  {bname}: {float(r.get('warmup_total_ms', 0.0)):.3f}")

    # Always write a timestamped results file by default (opt-out with --no-output-file).
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_output_file:
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"query_bench_{summary['mode']}_{timestamp}"

        json_path = out_dir / f"{base}.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        txt_path = out_dir / f"{base}.txt"
        txt_lines = []
        txt_lines.append(f"Mode: {summary['mode']}")
        txt_lines.append(f"Max rows: {summary.get('max_rows')}")
        if args.mode == "in_memory":
            txt_lines.append(f"in_memory.deals.columns = {in_memory_deals_cols}")
            txt_lines.append(f"in_memory.auctions.columns = {in_memory_auctions_cols}")
        txt_lines.append(f"Warmup total (all benchmarks): {total_warmup_ms:.3f} ms")
        txt_lines.append("")
        # Reconstruct the same table as stdout.
        txt_lines.append("  " + "  ".join(rows[0]))
        for r in rows[1:]:
            txt_lines.append("  " + "  ".join(str(x) for x in r))
        txt_lines.append("")
        txt_lines.append("Warmup totals by benchmark (ms):")
        for bname, r in summary["benchmarks"].items():
            txt_lines.append(f"  {bname}: {float(r.get('warmup_total_ms', 0.0)):.3f}")
        txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

        print("")
        print(f"Wrote results: {json_path}")
        print(f"Wrote results: {txt_path}")

    # Optional explicit JSON output path (kept for backwards compatibility / scripting).
    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_path}")

    if args.verify:
        # Print a tiny preview for quick manual inspection.
        # (Cross-mode comparison is easiest if you run with --output-json and compare externally.)
        print("")
        for bname, res in last_results.items():
            preview = res[: min(3, len(res))]
            print(f"verify preview: {bname}: rows={len(res)} first3={preview}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark query modes for BBO datasets (DuckDB/Parquet/in-memory/.duckdb).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List built-in benchmarks and modes.")
    p_list.set_defaults(func=cmd_list)

    p_build = sub.add_parser("build-db", help="Build a .duckdb database from Parquet (tables or views).")
    p_build.add_argument("--kind", choices=["tables", "views"], default="tables")
    p_build.add_argument("--deals", default=str(DEFAULT_DEALS_PARQUET))
    p_build.add_argument("--auctions", default=str(DEFAULT_AUCTIONS_PARQUET))
    p_build.add_argument("--db", default=str(DEFAULT_DUCKDB_PATH))
    p_build.add_argument("--overwrite", action="store_true", help="Overwrite the .duckdb file if it exists.")
    p_build.add_argument("--no-analyze", action="store_true", help="Skip ANALYZE step after loading tables.")
    p_build.add_argument(
        "--temp-dir",
        default=None,
        help="DuckDB temp_directory for spill-to-disk during build (put this on a fast local SSD).",
    )
    p_build.add_argument("--threads", type=int, default=None)
    p_build.add_argument("--memory-limit", default=None, help="DuckDB memory limit, e.g. '2GB'.")
    p_build.add_argument("--disable-object-cache", action="store_true")
    p_build.set_defaults(func=cmd_build_db)

    p_run = sub.add_parser("run", help="Run the benchmark suite for a specific mode.")
    p_run.add_argument("--mode", choices=list(ModeName.__args__), default="parquet_views")  # type: ignore[attr-defined]
    p_run.add_argument("--deals", default=str(DEFAULT_DEALS_PARQUET))
    p_run.add_argument("--auctions", default=str(DEFAULT_AUCTIONS_PARQUET))
    p_run.add_argument("--db", default=str(DEFAULT_DUCKDB_PATH))
    p_run.add_argument("--repeats", type=int, default=5)
    p_run.add_argument("--warmup", type=int, default=1)
    p_run.add_argument("--max-rows", type=int, default=None, help="Hard cap on returned rows per query.")
    p_run.add_argument("--threads", type=int, default=None)
    p_run.add_argument("--memory-limit", default=None, help="DuckDB memory limit, e.g. '2GB'.")
    p_run.add_argument("--disable-object-cache", action="store_true")
    p_run.add_argument(
        "--new-connection-per-run",
        action="store_true",
        help="Include connection + setup overhead in each measured run (otherwise reuse one conn).",
    )
    p_run.add_argument(
        "--in-memory-deals-cols",
        default=None,
        help="Comma-separated columns to load for in_memory mode (default is benchmark-minimal).",
    )
    p_run.add_argument(
        "--in-memory-auctions-cols",
        default=None,
        help="Comma-separated columns to load for in_memory mode (default is benchmark-minimal).",
    )
    p_run.add_argument("--verify", action="store_true", help="Fetch and print small result previews for sanity.")
    p_run.add_argument("--output-json", default=None, help="Write machine-readable results to JSON.")
    p_run.add_argument(
        "--output-dir",
        default="bench_results",
        help="Directory to write timestamped benchmark result files (JSON + TXT). Default: bench_results",
    )
    p_run.add_argument(
        "--no-output-file",
        action="store_true",
        help="Disable writing timestamped result files for the run subcommand.",
    )
    p_run.add_argument(
        "--overwrite-db",
        action="store_true",
        help="When mode is a build_* mode, overwrite DB if it exists.",
    )
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Make hashing stable across processes if caller uses Python's hash somewhere (we don't),
    # and keep runs a bit more reproducible.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


