from __future__ import annotations

from bbo_api_surface_builder import SHARED_PATHS, build_surface_app


app = build_surface_app(
    title="BBO Shared Data API",
    allowed_paths=SHARED_PATHS,
)
