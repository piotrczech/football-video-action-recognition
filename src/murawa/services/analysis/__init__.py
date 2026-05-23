from murawa.services.analysis import clip_teams

__all__ = [
    "analyze_frame",
    "analyze_frame_run",
    "analyze_match",
    "analyze_match_run",
    "clip_teams",
]

_PIPELINE_EXPORTS = {
    "analyze_frame",
    "analyze_frame_run",
    "analyze_match",
    "analyze_match_run",
}


def __getattr__(name: str):
    if name == "clip_teams":
        return clip_teams
    if name in _PIPELINE_EXPORTS:
        from murawa.services.analysis import pipeline as analysis_pipeline

        return getattr(analysis_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
