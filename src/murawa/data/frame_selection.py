from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from murawa.data.soccernet_frame_selection import (
    preprocess_soccernet_selected_dataset,
    select_soccernet_frames,
)


@dataclass(frozen=True)
class FrameSelectionConfig:
    raw_root: Path
    selected_root: Path
    frame_step: int = 30
    keep_original_resolution: bool = True


def select_n_frames(config: FrameSelectionConfig) -> Path:
    """Select every n-th frame from SoccerNet and write it to data/selected."""
    return select_soccernet_frames(config)


def preprocess_selected_frames(selected_root: Path, *, normalize: bool = False) -> Path:
    """Run lightweight preprocessing on already selected SoccerNet frames."""
    return preprocess_soccernet_selected_dataset(selected_root=selected_root, normalize=normalize)