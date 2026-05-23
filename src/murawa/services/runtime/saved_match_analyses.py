from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from murawa.settings import OUTPUTS_PREDICTIONS, OUTPUTS_VIDEOS

MATCH_VIDEO_SUFFIXES = {".mp4", ".webm"}


@dataclass(frozen=True)
class SavedMatchAnalysis:
    analysis_id: str
    preview_path: Path
    download_path: Path | None
    modified_at_ns: int
    summary_path: Path | None
    summary: dict[str, Any] | None

    @property
    def video_path(self) -> Path:
        """Backward-compatible alias for browser preview path."""
        return self.preview_path


def list_saved_match_analyses(project_root: Path) -> list[SavedMatchAnalysis]:
    videos_dir = project_root / OUTPUTS_VIDEOS
    predictions_dir = project_root / OUTPUTS_PREDICTIONS
    if not videos_dir.exists() or not videos_dir.is_dir():
        return []

    grouped_videos: dict[str, dict[str, Path]] = {}
    for video_path in videos_dir.iterdir():
        if not _is_match_video(video_path):
            continue
        grouped_videos.setdefault(video_path.stem, {})[video_path.suffix.lower()] = video_path

    saved_analyses: list[SavedMatchAnalysis] = []
    for analysis_id, paths in grouped_videos.items():
        preview_path = paths.get(".webm")
        if preview_path is None:
            continue

        download_path = paths.get(".mp4")
        try:
            modified_at_ns = preview_path.stat().st_mtime_ns
        except OSError:
            continue

        summary_path = _find_summary_path(predictions_dir, analysis_id=analysis_id)
        saved_analyses.append(
            SavedMatchAnalysis(
                analysis_id=analysis_id,
                preview_path=preview_path,
                download_path=download_path,
                modified_at_ns=modified_at_ns,
                summary_path=summary_path,
                summary=_load_summary(summary_path),
            )
        )

    return sorted(
        saved_analyses,
        key=lambda analysis: (analysis.modified_at_ns, analysis.preview_path.name),
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
