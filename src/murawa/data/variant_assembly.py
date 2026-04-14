from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VariantAssemblyConfig:
    selected_root: Path
    final_root: Path
    variant_name: str
    include_ball_extra: bool = False
    enable_transforms: bool = False


def assemble_variant(config: VariantAssemblyConfig) -> Path:
    """Issue #09: assemble named experiment-ready variant in data/ready."""
    raise NotImplementedError(
        "TODO(Issue #09): assemble clearly named variants (base, ball-extended, transformed/preprocessed) "
        "in data/ready and ensure each variant is directly consumable by training scripts."
    )


def describe_variant(variant_dir: Path) -> dict[str, str]:
    """Issue #09: produce concise metadata describing what differentiates this variant."""
    raise NotImplementedError(
        "TODO(Issue #09): generate short variant notes (source composition + preprocessing flags) "
        "so run-to-variant traceability is explicit."
    )
