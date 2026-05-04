"""Label helpers for LLaMAC emotion prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

EMOTION_ID_TO_LABEL: dict[int, str] = {
    1: "neutral",
    2: "fun",
    3: "sadness",
    4: "anger",
    5: "fear",
}
EMOTION_LABELS: list[str] = [EMOTION_ID_TO_LABEL[i] for i in sorted(EMOTION_ID_TO_LABEL)]
EMOTION_IDS: list[int] = sorted(EMOTION_ID_TO_LABEL)

ANSWER_COLUMNS: list[str] = [
    "SubjectID",
    "Trial",
    "NoVideo",
    "Valence",
    "Arousal",
    "Dominance",
    "Liking",
    "EmotType",
    "EmotStr",
    "Seen",
]

TARGET_COLUMNS: list[str] = ["IntendedType", "ReportedType"]
SELF_REPORT_COLUMNS: list[str] = [
    "NoVideo",
    "Valence",
    "Arousal",
    "Dominance",
    "Liking",
    "EmotType",
    "EmotStr",
    "Seen",
    *TARGET_COLUMNS,
]

OFFICIAL_EXCLUDE_SUBJECTS: tuple[int, ...] = (32, 37, 40, 47, 54, 55, 56, 70, 99)
OFFICIAL_EXCLUDE_TRIALS: dict[int, tuple[int, ...]] = {
    7: (27, 28, 36),
    19: (18,),
    59: (35, 13),
    111: (1,),
    20: (26,),
    89: (13,),
    105: (38, 50),
    107: (38,),
}


@dataclass(frozen=True)
class TargetSpec:
    """Resolved target-column metadata."""

    name: str
    column: str
    labels: list[int]
    label_names: list[str]


def map_novideo_to_intended(value: object) -> int | None:
    """Map NoVideo stimulus id to intended emotion id {1..5}."""
    try:
        no_video = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 1 <= no_video <= 10:
        return 1
    if 11 <= no_video <= 20:
        return 2
    if 21 <= no_video <= 30:
        return 3
    if 31 <= no_video <= 40:
        return 4
    if 41 <= no_video <= 50:
        return 5
    return None


def add_target_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Add IntendedType and ReportedType columns using the public benchmark mapping."""
    if "NoVideo" not in frame.columns:
        raise ValueError("NoVideo column is required to derive IntendedType")
    if "EmotType" not in frame.columns:
        raise ValueError("EmotType column is required to derive ReportedType")

    intended_expr = (
        pl.when(pl.col("NoVideo").cast(pl.Int64, strict=False).is_between(1, 10))
        .then(1)
        .when(pl.col("NoVideo").cast(pl.Int64, strict=False).is_between(11, 20))
        .then(2)
        .when(pl.col("NoVideo").cast(pl.Int64, strict=False).is_between(21, 30))
        .then(3)
        .when(pl.col("NoVideo").cast(pl.Int64, strict=False).is_between(31, 40))
        .then(4)
        .when(pl.col("NoVideo").cast(pl.Int64, strict=False).is_between(41, 50))
        .then(5)
        .otherwise(None)
        .cast(pl.Int64)
        .alias("IntendedType")
    )
    return frame.with_columns(
        intended_expr,
        pl.col("EmotType").cast(pl.Int64, strict=False).alias("ReportedType"),
    )


def resolve_target(name: str) -> TargetSpec:
    """Resolve a CLI target name to a dataframe column."""
    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"reported", "reported_type", "self_report", "emottype", "emotion"}:
        return TargetSpec("reported", "ReportedType", EMOTION_IDS, EMOTION_LABELS)
    if normalized in {"intended", "intended_type", "targeted", "novideo"}:
        return TargetSpec("intended", "IntendedType", EMOTION_IDS, EMOTION_LABELS)
    raise ValueError(f"Unsupported target {name!r}; use 'reported' or 'intended'.")


def filter_official_valid_trials(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the official notebook's subject/trial exclusions."""
    required = {"SubjectID", "Trial"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for official filtering: {sorted(missing)}")

    out = frame.with_columns(
        pl.col("SubjectID").cast(pl.Int64, strict=False).alias("__subject_int"),
        pl.col("Trial").cast(pl.Int64, strict=False).alias("__trial_int"),
    )
    keep = ~pl.col("__subject_int").is_in(list(OFFICIAL_EXCLUDE_SUBJECTS))
    for sid, trials in OFFICIAL_EXCLUDE_TRIALS.items():
        keep = keep & ~((pl.col("__subject_int") == sid) & pl.col("__trial_int").is_in(list(trials)))
    return out.filter(keep).drop(["__subject_int", "__trial_int"])


def validate_emotion_ids(values: Iterable[int | float | None], *, allow_null: bool = False) -> None:
    """Raise if any emotion ids fall outside the expected five-class domain."""
    valid = set(EMOTION_IDS)
    bad: list[object] = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if not allow_null:
                bad.append(value)
            continue
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            bad.append(value)
            continue
        if as_int not in valid:
            bad.append(value)
    if bad:
        preview = bad[:10]
        raise ValueError(f"Emotion ids outside {sorted(valid)}: {preview}")
