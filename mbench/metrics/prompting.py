# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Prompt helpers for MBench-T item-level VLM judges.

One binary trigger question per generated video, followed by a consistency score only when
the trigger is satisfied.  These helpers only format context for a VLM; they do
not rule-parse claims or make scoring decisions.
"""

from __future__ import annotations

import json
from typing import Any

from mbench.core.item import EvaluationItem
from mbench.metrics.schema import PromptContext


def prompt_context_from_sample(sample: dict[str, Any]) -> PromptContext:
    """Build the minimal prompt context from an MBench-T sample.json object."""

    segments = sample.get("caption_segments") or []
    if not isinstance(segments, list):
        raise ValueError("sample.json field 'caption_segments' must be a list")
    object_card = sample.get("object_card")
    return PromptContext(
        sample_id=str(sample.get("sample_id") or "unknown_sample"),
        subset=str(sample.get("subset") or "unknown_subset"),
        caption_segments=tuple(str(segment).strip() for segment in segments),
        object_card=object_card if isinstance(object_card, dict) else None,
        metadata={
            "has_object_card": isinstance(object_card, dict),
        },
    )


def prompt_context_from_item(item: EvaluationItem) -> PromptContext:
    """Build prompt context from the canonical EvaluationItem schema."""

    segments: list[str] = []
    if item.annotations.prompt_segments:
        segments = [segment.text for segment in item.annotations.prompt_segments]
    elif item.annotations.prompt:
        segments = [item.annotations.prompt]
    else:
        raw_segments = item.metadata.get("caption_segments")
        if isinstance(raw_segments, list):
            segments = [str(segment).strip() for segment in raw_segments]

    object_card = item.metadata.get("object_card")
    return PromptContext(
        sample_id=str(item.metadata.get("sample_id") or item.item_id),
        subset=str(item.metadata.get("subset") or "unknown_subset"),
        caption_segments=tuple(segment for segment in segments if segment),
        object_card=object_card if isinstance(object_card, dict) else None,
        metadata={
            "item_id": item.item_id,
            "model_id": item.model_id,
            "condition_id": item.metadata.get("condition_id"),
            "has_object_card": isinstance(object_card, dict),
        },
    )


def build_trigger_judge_prompt(context: PromptContext) -> str:
    """Format the item-level binary trigger question for a VLM judge."""

    return "\n".join(
        [
            "You are evaluating whether a generated video triggered the memory challenge described by the prompt.",
            "Answer the item-level question only: did the video create a valid memory challenge?",
            "Use a binary label: triggered=true or triggered=false, and provide a short reason.",
            "",
            _format_context(context),
            "",
            "Trigger question:",
            "Did the video execute the memory-triggering event implied by the prompt, such as entity exit/return, occlusion/reveal, camera departure/return, lighting degradation/recovery, or hidden event progress?",
            "",
            "Return JSON with keys: triggered, confidence, reason.",
        ]
    )


def build_consistency_judge_prompt(context: PromptContext) -> str:
    """Format the consistency question used only after trigger success."""

    return "\n".join(
        [
            "You are evaluating memory reliability for a generated video that already triggered the memory challenge.",
            "Score whether the generated video preserves the relevant identity, object attributes, environment layout, or causal outcome described by the prompt.",
            "",
            _format_context(context),
            "",
            "Consistency question:",
            "Given that the trigger occurred, how reliably does the video preserve the relevant memory content across the challenge?",
            "",
            "Return JSON with keys: reliability_score, confidence, reason. reliability_score must be in [0, 1].",
        ]
    )


def _format_context(context: PromptContext) -> str:
    lines = [
        f"sample_id: {context.sample_id}",
        f"subset: {context.subset}",
        "caption_segments:",
    ]
    for idx, segment in enumerate(context.caption_segments, start=1):
        lines.append(f"  {idx}. {segment}")
    if context.object_card:
        lines.extend(
            [
                "object_card:",
                json.dumps(context.object_card, ensure_ascii=False, indent=2),
            ]
        )
    return "\n".join(lines)
