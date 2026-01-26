"""Library-first core for BBO bidding queries.

This module hosts the long-lived in-process state and hot-reload plugin manager.
The FastAPI server becomes a thin adapter that delegates to this core.

Design goals:
- Keep Streamlit HTTP-only (no direct imports required by clients).
- Preserve plugin hot-reload for rapid debugging turnover.
- Keep the core free of FastAPI concepts (raise plain Exceptions; adapters map to HTTP errors).
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


def default_state() -> dict[str, Any]:
    """Default shared state structure (mirrors server expectations)."""
    return {
        "initialized": False,
        "initializing": False,
        "warming": False,  # True while pre-warming endpoints
        "prewarm_progress": None,  # Populated during endpoint pre-warm
        "error": None,
        "loading_step": None,  # Current loading step description
        "loaded_files": {},  # File name -> info string
        "deal_df": None,
        "bt_seat1_df": None,  # Pre-compiled BT table (bbo_bt_compiled.parquet)
        "bt_openings_df": None,  # Tiny opening-bid lookup table (built from bt_seat1_df)
        "g3_index": None,  # Gemini-3.2 CSR Traversal Index (built on startup)
        "deal_criteria_by_seat_dfs": None,
        "deal_criteria_by_direction_dfs": None,
        "results": None,
        # Criteria / aggregate statistics for completed auctions (seat-1 view).
        "bt_stats_df": None,
        # Completed auctions with Agg_Expr only (63MB, ~975K rows) for fast wrong-bid-stats
        "bt_completed_agg_df": None,
        # Optional: bid-category boolean flags from bbo_bt_categories.parquet (Phase 4).
        "bt_categories_df": None,
        "bt_category_cols": None,  # list[str]
        # Fast bt_index -> row position mapping for lightweight BT
        "bt_index_arr": None,  # numpy array of UInt32
        "bt_index_monotonic": False,
        "duckdb_conn": None,
        # Hot-reloadable overlay rules loaded from bbo_custom_auction_criteria.csv.
        "custom_criteria_overlay": [],
        # Loaded from bbo_bt_new_rules.parquet (optional - for detailed rule inspection)
        "new_rules_df": None,
        "custom_criteria_stats": {},
        # Set of available criterion names (from deal_criteria_by_direction_dfs)
        "available_criteria_names": None,
    }


class CoreNotReadyError(RuntimeError):
    """Raised when the core is not initialized (or failed)."""


class CoreService:
    """Owns shared state + plugin hot reload."""

    def __init__(self, *, plugins_dir: pathlib.Path) -> None:
        self.plugins_dir = pathlib.Path(plugins_dir)
        self.state_lock = threading.Lock()
        self.state: dict[str, Any] = default_state()

        # Hot-reload plugin tracking
        self._plugins_mtime: float = 0.0
        self._plugins_last_reload_epoch_s: float | None = None
        self._plugins_lock = threading.Lock()
        self.plugins: dict[str, Any] = {}

        # Init thread tracking
        self._init_thread: threading.Thread | None = None

        # Async job registry (generic)
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_max = 200
        self._jobs_ttl_s = 30 * 60  # 30 minutes
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="core-jobs")

    @property
    def store(self) -> "DataStore":
        return DataStore(self)

    @property
    def indexes(self) -> "Indexes":
        return Indexes(self)

    def snapshot_state(self) -> dict[str, Any]:
        """Thread-safe shallow snapshot suitable for plugin handlers."""
        with self.state_lock:
            return dict(self.state)

    def status_snapshot(self) -> dict[str, Any]:
        """Thread-safe status view for health endpoints."""
        with self.state_lock:
            return {
                "initialized": bool(self.state.get("initialized")),
                "initializing": bool(self.state.get("initializing")),
                "warming": bool(self.state.get("warming", False)),
                "prewarm_progress": self.state.get("prewarm_progress"),
                "error": self.state.get("error"),
                "loading_step": self.state.get("loading_step"),
                "loaded_files": self.state.get("loaded_files") or None,
            }

    def ensure_ready(self) -> None:
        with self.state_lock:
            if self.state.get("initialized"):
                return
            if self.state.get("error"):
                raise CoreNotReadyError(str(self.state.get("error")))
            if self.state.get("initializing"):
                raise CoreNotReadyError("Server is still initializing")
            raise CoreNotReadyError("Server not initialized")

    def get_plugin(self, name: str) -> Any:
        """Fetch a plugin module by short name (e.g. 'bbo_bidding_queries_api_handlers')."""
        mod = self.plugins.get(name)
        if mod is None:
            raise ImportError(f"Plugin '{name}' not found")
        return mod

    def reload_plugins(self) -> dict[str, object]:
        """Reload modules in plugins/ if mtime changed.

        Returns metadata dict: {"reloaded": bool, "mtime": float|None, "reloaded_at": float|None}
        """
        # Check if plugins directory exists
        if not self.plugins_dir.exists():
            return {"reloaded": False, "mtime": None, "reloaded_at": None}

        # Compute max mtime across dir + files
        try:
            current_mtime = self.plugins_dir.stat().st_mtime
            for p in self.plugins_dir.glob("*.py"):
                t = p.stat().st_mtime
                if t > current_mtime:
                    current_mtime = t
        except FileNotFoundError:
            return {"reloaded": False, "mtime": None, "reloaded_at": None}

        with self._plugins_lock:
            if current_mtime <= self._plugins_mtime:
                return {"reloaded": False, "mtime": self._plugins_mtime, "reloaded_at": self._plugins_last_reload_epoch_s}

            self._plugins_last_reload_epoch_s = time.time()
            self._plugins_mtime = current_mtime

            # Reload all .py files in plugins/
            for p in self.plugins_dir.glob("*.py"):
                if p.name == "__init__.py":
                    continue
                try:
                    module_name = p.stem
                    full_module_name = f"plugins.{module_name}"

                    # Reload if already imported; otherwise import.
                    if full_module_name in sys.modules:
                        mod = sys.modules[full_module_name]
                        if hasattr(mod, "__spec__") and mod.__spec__ is not None:
                            module = importlib.reload(mod)
                        else:
                            module = importlib.import_module(full_module_name)
                    else:
                        module = importlib.import_module(full_module_name)

                    # Store by short name for callers: plugins["bbo_bidding_queries_api_handlers"]
                    self.plugins[module_name] = module
                except Exception as e:
                    # Best-effort: keep old module if reload failed
                    self.plugins["_last_reload_error"] = str(e)

            return {"reloaded": True, "mtime": self._plugins_mtime, "reloaded_at": self._plugins_last_reload_epoch_s}

    def start_init_async(self, *, init_fn: Any, check_required_files_fn: Any) -> None:
        """Start initialization in a background thread (idempotent-ish)."""
        missing = list(check_required_files_fn())
        if missing:
            # Hard fail policy is enforced by adapter (os._exit in API script).
            raise FileNotFoundError("Missing required files:\n" + "\n".join(f"  - {m}" for m in missing))

        with self.state_lock:
            if self.state.get("initialized") or self.state.get("initializing"):
                return
            self.state["initializing"] = True
            self.state["error"] = None

        t = threading.Thread(target=init_fn, daemon=True)
        self._init_thread = t
        t.start()

    # ---------------------------------------------------------------------
    # Async job registry helpers (used for long-running handlers)
    # ---------------------------------------------------------------------

    def _jobs_gc(self, now_s: float | None = None) -> None:
        now = float(now_s if now_s is not None else time.time())
        with self._jobs_lock:
            to_del: list[str] = []
            for job_id, job in self._jobs.items():
                created_at = float(job.get("created_at_s") or now)
                if now - created_at > self._jobs_ttl_s:
                    to_del.append(str(job_id))
            for jid in to_del:
                self._jobs.pop(jid, None)

            if len(self._jobs) > self._jobs_max:
                items = sorted(self._jobs.items(), key=lambda kv: float(kv[1].get("created_at_s") or now))
                overflow = len(self._jobs) - self._jobs_max
                for i in range(max(0, overflow)):
                    self._jobs.pop(items[i][0], None)

    def start_job(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        run_fn: Callable[[], Any],
    ) -> None:
        """Start an async job in the core threadpool."""
        job = {
            "job_id": str(job_id),
            "status": "running",
            "created_at_s": time.time(),
            "finished_at_s": None,
            "request": dict(payload),
            "result": None,
            "error": None,
        }
        self._jobs_gc(job["created_at_s"])
        with self._jobs_lock:
            self._jobs[str(job_id)] = job

        def _runner() -> None:
            try:
                t_wall0 = time.perf_counter()
                t_cpu0 = time.process_time()
                resp = run_fn()
                t_wall1 = time.perf_counter()
                t_cpu1 = time.process_time()
                with self._jobs_lock:
                    j = self._jobs.get(str(job_id))
                    if j is not None:
                        j["status"] = "completed"
                        j["result"] = resp
                        j["wall_elapsed_s"] = round(t_wall1 - t_wall0, 3)
                        j["cpu_elapsed_s"] = round(t_cpu1 - t_cpu0, 3)
                        j["finished_at_s"] = time.time()
            except Exception as e:
                with self._jobs_lock:
                    j = self._jobs.get(str(job_id))
                    if j is not None:
                        j["status"] = "failed"
                        j["error"] = f"{e}"
                        j["finished_at_s"] = time.time()

        self._executor.submit(_runner)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._jobs_gc()
        with self._jobs_lock:
            job = self._jobs.get(str(job_id))
            return dict(job) if isinstance(job, dict) else None


@dataclass(frozen=True)
class DataStore:
    """Typed-ish accessors over CoreService.state."""

    core: CoreService

    def _get(self, key: str) -> Any:
        with self.core.state_lock:
            return self.core.state.get(key)

    @property
    def deal_df(self) -> Any:
        return self._get("deal_df")

    @property
    def bt_seat1_df(self) -> Any:
        return self._get("bt_seat1_df")

    @property
    def bt_openings_df(self) -> Any:
        return self._get("bt_openings_df")


@dataclass(frozen=True)
class Indexes:
    """Typed-ish accessors over CoreService.state."""

    core: CoreService

    def _get(self, key: str) -> Any:
        with self.core.state_lock:
            return self.core.state.get(key)

    @property
    def g3_index(self) -> Any:
        return self._get("g3_index")

