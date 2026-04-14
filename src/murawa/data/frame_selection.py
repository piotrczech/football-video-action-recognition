from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameSelectionConfig:
    raw_root: Path
    selected_root: Path
    frame_step: int = 30
    keep_original_resolution: bool = True


def select_n_frames(config: FrameSelectionConfig) -> Path:
    """Issue #08: select every n-th frame from data/raw and write to data/selected."""
    raise NotImplementedError(
        "TODO(Issue #08): implement deterministic every-nth frame selection with configurable frame_step, "
        "input from data/raw and output to data/selected in common project format."
    )


def preprocess_selected_frames(selected_root: Path, *, normalize: bool = False) -> Path:
    """Issue #08: run basic preprocessing on selected frames before variant assembly (#09)."""
    raise NotImplementedError(
        "TODO(Issue #08): implement lightweight preprocessing for selected frames and validate on a small sample; "
        "this step should stay basic (no heavy augmentation policy here)."
    )
