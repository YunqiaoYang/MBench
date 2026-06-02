# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Trigger-conditioned scoring utilities for MBench-T."""

from __future__ import annotations

from collections.abc import Iterable

from mbench.metrics.metric_math import harmonic_mean
from mbench.metrics.schema import (
    MBenchTItemScore,
    TriggerConditionedSummary,
)


def summarize_trigger_conditioned(
    results: Iterable[MBenchTItemScore],
    *,
    metric: str,
    group: str = "all",
) -> TriggerConditionedSummary:
    """Aggregate per-item trigger-conditioned results.

    C_trig is computed over all items.  S_rel is computed only on triggered
    items that have a valid reliability score.
    """

    items = list(results)
    total = len(items)
    triggered = [item for item in items if item.triggered]
    scored_values = [
        float(item.reliability_score)
        for item in triggered
        if item.has_reliability_score()
    ]

    trigger_coverage = len(triggered) / total if total else 0.0
    memory_reliability = sum(scored_values) / len(scored_values) if scored_values else 0.0
    m_score = harmonic_mean(trigger_coverage, memory_reliability)

    return TriggerConditionedSummary(
        metric=metric,
        group=group,
        total_items=total,
        triggered_items=len(triggered),
        scored_items=len(scored_values),
        trigger_coverage=round(trigger_coverage, 6),
        memory_reliability=round(memory_reliability, 6),
        m_score=round(m_score, 6),
        metadata={
            "untriggered_items": total - len(triggered),
            "triggered_without_score": len(triggered) - len(scored_values),
        },
    )
