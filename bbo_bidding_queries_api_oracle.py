from __future__ import annotations

from bbo_api_surface_builder import ORACLE_PATHS, build_surface_app


app = build_surface_app(
    title="BBO Oracle Analysis API",
    allowed_paths=ORACLE_PATHS,
)
