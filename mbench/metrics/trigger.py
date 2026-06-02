# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Shared VLM trigger judging for MBench-T metrics.

Trigger gating policy (paper §4 trigger-conditioned scoring + §App):

- ``mbencha.*`` metrics never invoke the VLM trigger. Action-conditioned
  rollouts deterministically actuate the memory challenge, so a semantic
  trigger gate is unnecessary; this is hard-disabled here so an A-side metric
  can never accidentally make a VLM call.
- ``mbencht.*`` metrics invoke the trigger only when the caller explicitly
  enables it. Concretely: pass ``--vlm-judge openai-compatible`` on the CLI
  (which both supplies ``judge=`` and clears ``ignore_trigger``). Without that
  flag the metric short-circuits to a triggered=true / trigger_score=1.0
  result so downstream M-Score code keeps a uniform shape, but no API call
  is made.

This module is the single integration point — every metric that wants
trigger gating routes through ``judge_vlm_trigger_for_item`` so the policy
above is applied uniformly.
"""

from __future__ import annotations

from typing import Any

from mbench.core.item import EvaluationItem
from mbench.metrics.judge import TriggerJudgeResult
from mbench.metrics.prompting import PromptContext, prompt_context_from_item
from mbench.metrics.video_io import first_video_path


def _bypass_result(reason: str, mode: str) -> TriggerJudgeResult:
    """Return a synthetic triggered=true result without invoking the VLM."""
    return TriggerJudgeResult(
        triggered=True,
        score=1.0,
        confidence=None,
        reason=reason,
        metadata={"trigger_mode": mode},
    )


def judge_vlm_trigger_for_item(
    item: EvaluationItem,
    kwargs: dict[str, Any],
    *,
    context: PromptContext | None = None,
    metric_name: str | None = None,
    trigger_kind: str | None = None,
) -> TriggerJudgeResult:
    """Run the shared MBench-T trigger judge for one generated video.

    ``triggered=true`` means the video visibly enters a memory challenge
    (exit/entry, occlusion, scene departure-return, etc.), i.e. the sample is
    evaluable for memory reliability. It does not judge whether the model
    preserved memory correctly.

    Returns a synthetic ``triggered=true`` (no API call) when the trigger is
    gated off — see the module docstring for the full policy.
    """

    # MBench-A: hard-disabled. Action-conditioned rollouts get full coverage
    # by construction (paper §App).
    if metric_name and metric_name.startswith("mbencha."):
        return _bypass_result(
            "trigger disabled on MBench-A side (action-conditioned rollout, paper §App)",
            "a_side_disabled",
        )

    # Caller can still force the bypass (e.g. ablation, score-only debug runs).
    if bool(kwargs.get("ignore_trigger", False)):
        return _bypass_result("trigger ignored; reliability-only run", "ignored")

    judge = kwargs.get("judge")
    if judge is None:
        # MBench-T trigger is opt-in: when the user did not pass --vlm-judge
        # we report triggered=true uniformly so the downstream score schema
        # stays the same shape; no VLM call is made.
        return _bypass_result(
            "trigger gate skipped: --vlm-judge not provided; reporting raw consistency",
            "no_judge",
        )

    context = context or prompt_context_from_item(item)
    prompt = kwargs.get("trigger_prompt")
    if not prompt:
        prompt = build_vlm_trigger_prompt(
            context,
            metric_name=metric_name,
            trigger_kind=trigger_kind,
        )

    result = judge.judge_trigger(
        context=context,
        video_path=first_video_path(item),
        prompt=prompt,
    )
    metadata = dict(result.metadata or {})
    if metric_name:
        metadata.setdefault("metric_name", metric_name)
    if trigger_kind:
        metadata.setdefault("trigger_kind", trigger_kind)
    metadata.setdefault("trigger_mode", "vlm")
    return TriggerJudgeResult(
        triggered=result.triggered,
        score=result.score,
        confidence=result.confidence,
        reason=result.reason,
        metadata=metadata,
    )


def build_vlm_trigger_prompt(
    context: PromptContext,
    *,
    metric_name: str | None = None,
    trigger_kind: str | None = None,
) -> str:
    """Build the shared prompt for MBench-T trigger judging."""

    question = _challenge_trigger_question(context)
    return "\n".join(
        [
            "You are a strict video evaluator for MBench, a benchmark for long-video memory.",
            "triggered=true means the video visibly enters the intended memory challenge, regardless of whether it preserves memory correctly.",
            "Use only the provided video frames and prompt context. Do not infer success from the prompt alone.",
            "Return triggered=false if the relevant event is absent, too ambiguous, unrelated, or impossible to judge from the video.",
            "Return JSON only with keys: triggered, confidence, reason. Keep reason under 25 words.",
            "",
            f"metric_name: {metric_name or 'unknown'}",
            f"trigger_kind: {trigger_kind or context.subset}",
            f"sample_id: {context.sample_id}",
            f"subset: {context.subset}",
            "caption_segments:",
            *_format_segments(context),
            *_format_object_card(context),
            "",
            "Decision question:",
            question,
            "",
            "Additional decision rules:",
            "- Judge only whether the memory challenge is visibly entered.",
            "- Do not evaluate memory correctness in trigger mode.",
        ]
    )


def _challenge_trigger_question(context: PromptContext) -> str:
    subset = context.subset
    if subset == "human":
        return (
            "Does the video actually create a human memory challenge, such as a person appearing, "
            "leaving/being occluded across middle segments, and later reappearing?"
        )
    if subset == "object":
        return (
            "Does the video actually create an object memory challenge for the target object, "
            "such as appearing, leaving/being occluded/changing view, and later reappearing?"
        )
    if subset == "environment":
        return (
            "Does the video actually create an environment memory challenge, such as moving away "
            "from a scene/view/layout and later revisiting the same place or layout?"
        )
    if subset == "causal":
        return (
            "Does the video actually create a causal memory challenge, such as an event or physical "
            "process that should continue while partially hidden or across later segments?"
        )
    return "Does the video visibly contain the memory-triggering event implied by the prompt?"


def _format_segments(context: PromptContext) -> list[str]:
    return [f"  {idx}. {segment}" for idx, segment in enumerate(context.caption_segments, start=1)]


def _format_object_card(context: PromptContext) -> list[str]:
    if not context.object_card:
        return []
    return [
        "object_card:",
        str(context.object_card),
    ]
