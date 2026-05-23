"""Backward-compatible re-export for ``from murawa.services.pipeline import ...``."""

from murawa.services.analysis.pipeline import (
    analyze_frame,
    analyze_frame_run,
    analyze_match,
    analyze_match_run,
)

__all__ = [
    "analyze_frame",
    "analyze_frame_run",
    "analyze_match",
    "analyze_match_run",
]
