# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Judge interface for MBench-T trigger-conditioned evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mbench.metrics.schema import PromptContext


@dataclass(frozen=True)
class JudgeResult:
    """Structured output returned by a VLM or test judge."""

    score: float
    reason: str | None = None
    confidence: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriggerJudgeResult(JudgeResult):
    """Item-level trigger decision."""

    triggered: bool = False


class MBenchTJudge(Protocol):
    """Protocol implemented by concrete VLM judges or test doubles."""

    def judge_trigger(
        self,
        *,
        context: PromptContext,
        video_path: str,
        prompt: str,
    ) -> TriggerJudgeResult:
        """Return whether the generated video triggered the memory challenge."""

    def judge_consistency(
        self,
        *,
        context: PromptContext,
        video_path: str,
        prompt: str,
    ) -> JudgeResult:
        """Return metric-specific consistency scores after trigger success."""
