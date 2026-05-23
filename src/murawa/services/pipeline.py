"""Backward-compatible re-export. Prefer ``murawa.services.analysis.pipeline``."""

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
