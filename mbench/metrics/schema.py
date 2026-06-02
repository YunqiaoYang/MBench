# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Schema for MBench-T trigger-conditioned evaluation.

The primary scoring unit is the generated video/item: first verify whether a
memory-triggering event occurs, then score memory reliability only when the
trigger is satisfied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptContext:
    """Minimal prompt context passed to item-level VLM judges."""

    sample_id: str
    subset: str
    caption_segments: tuple[str, ...]
    object_card: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_segments(self) -> int:
        return len(self.caption_segments)

@dataclass(frozen=True)
class MBenchTItemScore:
    """Item-level trigger and memory reliability result."""

    item_id: str
    model_id: str | None
    metric: str
    triggered: bool
    trigger_score: float
    reliability_score: float | None = None
    trigger_reason: str | None = None
    consistency_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_reliability_score(self) -> bool:
        return self.triggered and self.reliability_score is not None


@dataclass(frozen=True)
class TriggerConditionedSummary:
    """Aggregate C_trig, S_rel, and M-Score for a metric group."""

    metric: str
    group: str
    total_items: int
    triggered_items: int
    scored_items: int
    trigger_coverage: float
    memory_reliability: float
    m_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
