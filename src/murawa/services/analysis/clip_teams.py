from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from murawa.services.vision.team_assignment import PLAYER_CLASSES
from murawa.services.vision.team_assignment_helpers import cluster_two_groups, normalize_class_name


def stabilize_match_team_assignments_with_clip_prototypes(
    frame_batches: list[list[dict]],
) -> tuple[list[list[dict]], dict]:
    """Stabilize team labels using two jersey-color prototypes for the whole clip.

    Per-frame team assignment can swap team_a/team_b between sampled frames.
    This function uses all sampled frames to build two global team color prototypes
    and then relabels every player detection against these stable prototypes.
    """
    stabilized_batches = [[dict(detection) for detection in batch] for batch in frame_batches]
    observations: list[dict[str, object]] = []

    for batch_index, batch in enumerate(stabilized_batches):
        for detection_index, detection in enumerate(batch):
            class_name = normalize_class_name(detection.get("class"))
            if class_name not in PLAYER_CLASSES:
                continue

            color_bgr = read_jersey_color_array(detection.get("jersey_color_bgr"))
            if color_bgr is None:
                continue

            observations.append(
                {
                    "batch_index": batch_index,
                    "detection_index": detection_index,
                    "color_bgr": color_bgr,
                    "raw_team": str(detection.get("team", "unknown")),
                    "weight": max(
                        0.10,
                        min(
                            1.0,
                            read_numeric_confidence(
                                detection.get("team_confidence"),
                                fallback=detection.get("confidence"),
                            ),
                        ),
                    ),
                }
            )

    if len(observations) < 2:
        return stabilized_batches, {
            "enabled": False,
            "method": "clip_level_jersey_color_prototypes",
            "reason": "not_enough_jersey_color_observations",
            "observations": len(observations),
        }

    colors_bgr = np.stack(
        [observation["color_bgr"] for observation in observations]
    ).astype(np.float32)
    colors_lab = bgr_colors_to_lab(colors_bgr)

    centers_lab, groups = cluster_two_groups(colors_lab)
    group_counts = Counter(int(group) for group in groups.tolist())

    if len(group_counts) < 2:
        return stabilized_batches, {
            "enabled": False,
            "method": "clip_level_jersey_color_prototypes",
            "reason": "single_color_group",
            "observations": len(observations),
        }

    group_to_team = map_clip_groups_to_existing_team_labels(
        observations=observations,
        groups=groups,
    )

    team_colors_bgr = clip_team_colors_bgr(
        observations=observations,
        groups=groups,
        group_to_team=group_to_team,
    )
    relabeled_counts: Counter[str] = Counter()

    for observation_index, observation in enumerate(observations):
        batch_index = int(observation["batch_index"])
        detection_index = int(observation["detection_index"])
        group = int(groups[observation_index])
        team = group_to_team[group]

        detection = stabilized_batches[batch_index][detection_index]
        raw_team = detection.get("team")

        detection["team_raw_frame"] = raw_team
        detection["team"] = team
        detection["team_smoothing"] = "clip_level_color_prototype"
        detection["team_confidence"] = round(
            clip_assignment_confidence(
                point_lab=colors_lab[observation_index],
                centers_lab=centers_lab,
                group=group,
            ),
            4,
        )
        relabeled_counts[team] += 1

    return stabilized_batches, {
        "enabled": True,
        "method": "clip_level_jersey_color_prototypes",
        "scope": "whole_clip",
        "observations": len(observations),
        "group_counts": {
            str(group): int(count) for group, count in sorted(group_counts.items())
        },
        "team_counts_from_color_observations": dict(sorted(relabeled_counts.items())),
        "team_colors_bgr": team_colors_bgr,
        "notes": [
            "Two global jersey-color prototypes are estimated from all sampled frames.",
            "Player detections are relabeled against the same prototypes across the whole clip.",
            "Track smoothing is applied afterwards, so track_id voting uses stabilized labels.",
        ],
    }


def team_summary_with_clip_prototypes(*, summary: dict, clip_team_summary: dict) -> dict:
    enriched = dict(summary)

    if not clip_team_summary.get("enabled"):
        return enriched

    enriched["method"] = "clip_level_jersey_color_prototypes"
    enriched["scope"] = "whole_clip"
    enriched["clip_level"] = clip_team_summary

    team_colors = clip_team_summary.get("team_colors_bgr")
    if isinstance(team_colors, dict) and team_colors:
        enriched["team_colors_bgr"] = team_colors

    return enriched


def read_jersey_color_array(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None

    try:
        return np.asarray([float(channel) for channel in value], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def read_numeric_confidence(value: object, *, fallback: object = None) -> float:
    for candidate in (value, fallback):
        if isinstance(candidate, bool):
            continue
        try:
            return max(0.0, min(1.0, float(candidate)))
        except (TypeError, ValueError):
            continue
    return 0.5


def bgr_colors_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    colors = colors_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(colors, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def map_clip_groups_to_existing_team_labels(
    *,
    observations: list[dict[str, object]],
    groups: np.ndarray,
) -> dict[int, str]:
    """Orient global color groups to existing team_a/team_b labels.

    This keeps labels as close as possible to the previous frame-level assignment,
    but makes the mapping stable across the whole clip.
    """
    group_votes: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}

    for observation, group in zip(observations, groups.tolist(), strict=True):
        raw_team = str(observation.get("raw_team", ""))
        if raw_team not in {"team_a", "team_b"}:
            continue

        weight = float(observation.get("weight", 1.0))
        group_votes[int(group)][raw_team] += weight

    same_score = group_votes[0]["team_a"] + group_votes[1]["team_b"]
    swapped_score = group_votes[0]["team_b"] + group_votes[1]["team_a"]

    if same_score >= swapped_score:
        return {0: "team_a", 1: "team_b"}

    return {0: "team_b", 1: "team_a"}


def clip_team_colors_bgr(
    *,
    observations: list[dict[str, object]],
    groups: np.ndarray,
    group_to_team: dict[int, str],
) -> dict[str, list[int]]:
    colors_by_team: dict[str, list[np.ndarray]] = {"team_a": [], "team_b": []}

    for observation, group in zip(observations, groups.tolist(), strict=True):
        team = group_to_team[int(group)]
        color_bgr = observation.get("color_bgr")
        if isinstance(color_bgr, np.ndarray):
            colors_by_team[team].append(color_bgr.astype(np.float32))

    output: dict[str, list[int]] = {}
    for team, colors in colors_by_team.items():
        if not colors:
            continue

        median_color = np.median(np.stack(colors), axis=0)
        output[team] = [
            int(round(float(channel)))
            for channel in median_color.tolist()
        ]

    return output


def clip_assignment_confidence(
    *,
    point_lab: np.ndarray,
    centers_lab: np.ndarray,
    group: int,
) -> float:
    own_distance = float(np.linalg.norm(point_lab - centers_lab[group]))
    other_group = 1 - int(group)
    other_distance = float(np.linalg.norm(point_lab - centers_lab[other_group]))

    denominator = own_distance + other_distance
    if denominator <= 1e-6:
        return 0.5

    return max(0.0, min(1.0, other_distance / denominator))
