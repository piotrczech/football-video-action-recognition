from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATCH_VIDEO_SUFFIXES = {".mp4", ".webm"}


@dataclass(frozen=True)
class SavedMatchAnalysis:
    analysis_id: str
    video_path: Path
    modified_at_ns: int
    summary_path: Path | None
    summary: dict[str, Any] | None


def list_saved_match_analyses(project_root: Path) -> list[SavedMatchAnalysis]:
    videos_dir = project_root / "outputs" / "videos"
    predictions_dir = project_root / "outputs" / "predictions"
    if not videos_dir.exists() or not videos_dir.is_dir():
        return []

    saved_analyses: list[SavedMatchAnalysis] = []
    for video_path in videos_dir.iterdir():
        if not _is_match_video(video_path):
            continue

        try:
            modified_at_ns = video_path.stat().st_mtime_ns
        except OSError:
            continue

        summary_path = _find_summary_path(predictions_dir, analysis_id=video_path.stem)
        saved_analyses.append(
            SavedMatchAnalysis(
                analysis_id=video_path.stem,
                video_path=video_path,
                modified_at_ns=modified_at_ns,
                summary_path=summary_path,
                summary=_load_summary(summary_path),
            )
        )

    return sorted(
        saved_analyses,
        key=lambda analysis: (analysis.modified_at_ns, analysis.video_path.name),
        reverse=True,
    )


def _is_match_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MATCH_VIDEO_SUFFIXES


def _find_summary_path(predictions_dir: Path, *, analysis_id: str) -> Path | None:
    if not predictions_dir.exists() or not predictions_dir.is_dir():
        return None

    for run_dir in sorted(predictions_dir.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            continue
        candidate = run_dir / analysis_id / "prediction_summary.json"
        if candidate.is_file():
            return candidate
    return None


def _load_summary(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload
