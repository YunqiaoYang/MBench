# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""MBench-A / MBench-T causal consistency metric plugins.

Taxonomy:
- interaction: camera_interaction (A only), prompt_interaction (A + T)
- self_evolution: state_progress, progress_correctness (VLM-judge based)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from mbench.core.contract import InputContract
from mbench.core.item import EvaluationItem
from mbench.core.registry import Metric
from mbench.core.result import ItemResult, UnitResult
from mbench.metrics.causal_calculations import (
    CLIPSegmentScorer,
    T_CAUSAL_PROGRESS_CORRECTNESS_METRIC,
    T_CAUSAL_STATE_PROGRESS_METRIC,
    compute_caption_segment_clip_scores,
    evaluate_causal_progress_correctness_context,
    evaluate_causal_state_progress_context,
)
from mbench.metrics.environment_calculations import load_da3_data
from mbench.metrics.video_io import first_video_path

# =============================================================================
# Metric name constants
# =============================================================================

A_CAUSAL_CAMERA_INTERACTION_METRIC = "mbencha.causal.camera_interaction"
A_CAUSAL_PROMPT_INTERACTION_METRIC = "mbencha.causal.prompt_interaction"
A_CAUSAL_STATE_PROGRESS_METRIC = "mbencha.causal.state_progress"
A_CAUSAL_PROGRESS_CORRECTNESS_METRIC = "mbencha.causal.progress_correctness"

T_CAUSAL_PROMPT_INTERACTION_METRIC = "mbencht.causal.prompt_interaction"

T_CAUSAL_TRIGGER_SCORE_KEYS = ("triggered", "trigger_score")
T_CAUSAL_STATE_PROGRESS_SCORE_KEYS = T_CAUSAL_TRIGGER_SCORE_KEYS + (
    "state_evolution_score",
    "progress_gate",
    "reliability_score",
)
T_CAUSAL_PROGRESS_CORRECTNESS_SCORE_KEYS = T_CAUSAL_TRIGGER_SCORE_KEYS + (
    "evolution_correctness_score",
    "progress_correctness",
    "reliability_score",
)
T_CAUSAL_PROMPT_INTERACTION_SCORE_KEYS = (
    "triggered",
    "trigger_score",
    "clip_score",
    "num_frames",
)


def _prompt_context_from_item(item: EvaluationItem, *, fallback_caption: str | None = None) -> Any:
    from mbench.metrics.prompting import PromptContext, prompt_context_from_item

    ctx = prompt_context_from_item(item)
    if not ctx.caption_segments and fallback_caption:
        return PromptContext(
            sample_id=ctx.sample_id,
            subset=ctx.subset,
            caption_segments=(fallback_caption,),
            object_card=ctx.object_card,
            metadata=dict(ctx.metadata),
        )
    return ctx


def _prompts_from_item(item: EvaluationItem) -> list[str]:
    if item.annotations.prompt_segments:
        return [segment.text for segment in item.annotations.prompt_segments]
    raw_segments = item.metadata.get("caption_segments")
    if isinstance(raw_segments, list):
        segments = [str(segment).strip() for segment in raw_segments if str(segment).strip()]
        if segments:
            return segments
    if item.annotations.prompt:
        return [item.annotations.prompt]
    prompts = item.metadata.get("prompts")
    if isinstance(prompts, list):
        return [str(prompt) for prompt in prompts if str(prompt).strip()]
    return []


def _vlm_scores_from_item_score(score: Any) -> dict[str, float]:
    metadata = score.metadata or {}
    raw = metadata.get("causal_vlm_scores") or metadata.get("self_evolution_scores") or {}
    return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}


# =============================================================================
# MBench-A — interaction
# =============================================================================


class CameraInteractionConsistencyMetric(Metric):
    name = A_CAUSAL_CAMERA_INTERACTION_METRIC
    description = "Camera action-following consistency computed from DA3 pose motion."
    artifact_name = "da3"
    unit_type = "motion_segment"
    score_keys = ("action_instruction_score",)

    @property
    def contract(self) -> InputContract:  # type: ignore[override]
        return InputContract.from_field_paths(
            required=["media.videos", f"artifacts.{self.artifact_name}", "annotations.action"],
        )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        artifact = item.artifacts.get(self.artifact_name)
        action = item.annotations.action
        if artifact is None or artifact.path is None:
            return [self._invalid(item, "missing DA3 artifact")]
        if action is None:
            return [self._invalid(item, "missing action annotation")]

        try:
            data = load_da3_data(artifact.path)
            scores, per_frame_cosines = _score_camera_motion(data["extrinsics"], action.label)
            return [UnitResult(
                item_id=item.item_id,
                metric=self.name,
                unit_type=self.unit_type,
                unit_id=f"{item.item_id}__camera_motion",
                scores=scores,
                valid=bool(scores),
                metadata={
                    "action": action.label,
                    "per_frame_cosines": per_frame_cosines,
                },
            )]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__camera_motion",
            scores={},
            valid=False,
            error=error,
        )


def _pad_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    """Pad (N, 3, 4) extrinsics to (N, 4, 4) with [0,0,0,1] bottom row."""
    if extrinsics.shape[1] == 4:
        return extrinsics
    N = len(extrinsics)
    padded = np.tile(np.eye(4, dtype=extrinsics.dtype)[None], (N, 1, 1))
    padded[:, :3, :] = extrinsics
    return padded


def _rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle vector (right-hand rule)."""
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return np.zeros(3)

    if np.pi - theta < 1e-10:
        A = R + np.eye(3)
        k = A[:, int(np.argmax(np.linalg.norm(A, axis=0)))]
        return k / np.linalg.norm(k) * theta

    k = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return k * theta


def _compute_twists(extrinsics: np.ndarray) -> np.ndarray:
    """Compute 6-DoF twist [v; ω] for each frame transition.

    Args:
        extrinsics: (N, 3, 4) or (N, 4, 4) camera-to-world matrices.

    Returns:
        (N-1, 6) array, each row = [v_x, v_y, v_z, ω_x, ω_y, ω_z].
    """
    E = _pad_to_4x4(extrinsics)
    N = len(E)
    twists = np.zeros((N - 1, 6))
    for i in range(N - 1):
        t_i = E[i, :3, 3]
        t_next = E[i + 1, :3, 3]
        twists[i, :3] = t_next - t_i

        R_i = E[i, :3, :3]
        R_next = E[i + 1, :3, :3]
        dR = R_next @ R_i.T
        twists[i, 3:] = _rotation_to_axis_angle(dR)

    return twists


def _score_camera_motion(
    extrinsics: np.ndarray, action_label: str
) -> tuple[dict[str, float], list[float]]:
    """Score camera motion via 6-DoF twist cosine similarity.

    The Action Instruction Score is the mean of per-frame cos(ξ_i, ξ_gt)
    across all frame transitions, as defined in the paper.
    """
    twists = _compute_twists(extrinsics)
    if len(twists) < 2:
        return {}, []

    # Ground-truth twist directions (unit-norm, magnitude-free)
    if action_label == "left_then_right":
        gt_first = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        gt_second = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    elif action_label == "right_then_left":
        gt_first = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        gt_second = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    elif action_label in ("forward_then_backward", "forward_backward"):
        gt_first = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        gt_second = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    else:
        return {}, []

    mid = len(twists) // 2
    gt = np.zeros((len(twists), 6))
    gt[:mid] = gt_first
    gt[mid:] = gt_second

    per_frame_cosines: list[float] = []
    for i in range(len(twists)):
        denom = float(np.linalg.norm(twists[i]))
        cos_i = float(np.dot(twists[i], gt[i]) / denom) if denom > 1e-12 else 0.0
        per_frame_cosines.append(round(cos_i, 6))

    score = float(np.mean(per_frame_cosines))
    return {"action_instruction_score": round(score, 5)}, per_frame_cosines


class PromptInteractionConsistencyMetric(Metric):
    name = A_CAUSAL_PROMPT_INTERACTION_METRIC
    description = (
        "Prompt interaction consistency via OpenCLIP; video duration is split evenly "
        "across prompt segments (same rule as MBench-T)."
    )
    unit_type = "prompt_segment"
    score_keys = T_CAUSAL_PROMPT_INTERACTION_SCORE_KEYS

    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=[
            "annotations.prompt",
            "annotations.prompt_segments",
            "metadata.caption_segments",
        ],
    )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            prompts = _prompts_from_item(item)
            if not prompts:
                return [self._invalid(item, "missing prompt or caption_segments")]
            scorer = kwargs.get("clip_scorer") or CLIPSegmentScorer(
                device=str(kwargs.get("device", "cuda")),
            )
            segment_scores = compute_caption_segment_clip_scores(
                first_video_path(item),
                prompts,
                scorer=scorer,
                num_samples_per_segment=int(kwargs.get("num_samples_per_segment", 8)),
            )
            return [
                UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__segment_{record['segment_idx']}",
                    scores={
                        "clip_score": float(record["clip_score"]),
                        "num_frames": float(record["num_frames"]),
                    },
                    valid=bool(record["num_frames"] > 0),
                    metadata={
                        "prompt": record["prompt"],
                        "start_time": record["start_time"],
                        "end_time": record["end_time"],
                    },
                )
                for record in segment_scores
            ]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__prompt_interaction",
            scores={},
            valid=False,
            error=error,
        )


# =============================================================================
# MBench-A — self_evolution (VLM judge, no trigger gate)
# =============================================================================


A_CAUSAL_STATE_PROGRESS_SCORE_KEYS = (
    "state_evolution_score",
    "reliability_score",
)
A_CAUSAL_PROGRESS_CORRECTNESS_SCORE_KEYS = (
    "evolution_correctness_score",
    "progress_correctness",
    "reliability_score",
)


# =============================================================================
# MBench-T — interaction
# =============================================================================


@lru_cache(maxsize=4)
def _cached_clip_scorer(device: str) -> CLIPSegmentScorer:
    return CLIPSegmentScorer(device=device)


class TPromptInteractionConsistencyMetric(Metric):
    name = T_CAUSAL_PROMPT_INTERACTION_METRIC
    description = "MBench-T prompt interaction via CLIP over caption-aligned temporal segments."
    unit_type = "caption_segment"
    score_keys = T_CAUSAL_PROMPT_INTERACTION_SCORE_KEYS
    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=["annotations.prompt", "annotations.prompt_segments", "metadata.caption_segments"],
    )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            from mbench.metrics.trigger import judge_vlm_trigger_for_item

            context = _prompt_context_from_item(item)
            prompts = list(context.caption_segments)
            if not prompts:
                return [self._invalid(item, "missing prompt or caption_segments")]
            trigger = judge_vlm_trigger_for_item(
                item,
                kwargs,
                context=context,
                metric_name=self.name,
                trigger_kind="prompt_interaction",
            )
            scorer = kwargs.get("clip_scorer") or _cached_clip_scorer(str(kwargs.get("device", "cuda")))
            segment_scores = compute_caption_segment_clip_scores(
                first_video_path(item),
                prompts,
                scorer=scorer,
                num_samples_per_segment=int(kwargs.get("num_samples_per_segment", 8)),
            )
            return [UnitResult(
                item_id=item.item_id,
                metric=self.name,
                unit_type=self.unit_type,
                unit_id=f"{item.item_id}__caption_segment_{record['segment_idx']}",
                scores={
                    "triggered": 1.0 if trigger.triggered else 0.0,
                    "trigger_score": float(trigger.score),
                    "clip_score": float(record["clip_score"]),
                    "num_frames": float(record["num_frames"]),
                },
                valid=bool(record["num_frames"] > 0),
                metadata={
                    "model_id": item.model_id,
                    "sample_id": context.sample_id,
                    "condition_id": context.metadata.get("condition_id"),
                    "prompt": record["prompt"],
                    "segment_idx": record["segment_idx"],
                    "start_time": record["start_time"],
                    "end_time": record["end_time"],
                    "trigger_reason": trigger.reason,
                    "trigger_confidence": trigger.confidence,
                    "vlm_trigger_metadata": trigger.metadata,
                },
            ) for record in segment_scores]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        valid_units = [unit for unit in units if unit.valid and "clip_score" in unit.scores]
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no prompt interaction unit result",
            )
        if not valid_units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=len(units),
                valid=False,
                error=units[0].error,
            )
        clip_scores = [float(unit.scores["clip_score"]) for unit in valid_units]
        trigger_values = [float(unit.scores["trigger_score"]) for unit in units if "trigger_score" in unit.scores]
        triggered = max(float(unit.scores.get("triggered", 0.0)) for unit in units)
        return ItemResult(
            item_id=item.item_id,
            model_id=item.model_id,
            metric=self.name,
            scores={
                "triggered": triggered,
                "trigger_score": round(sum(trigger_values) / len(trigger_values), 6) if trigger_values else 0.0,
                "clip_score": round(sum(clip_scores) / len(clip_scores), 6),
                "num_segments": float(len(valid_units)),
            },
            num_units=len(units),
            valid=True,
        )

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__prompt_interaction",
            scores={},
            valid=False,
            error=error,
        )



# =============================================================================
# MBench-T — self_evolution (VLM)
# =============================================================================


class _VCausalSelfEvolutionMetric(Metric):
    unit_type = "causal_self_evolution"
    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=["annotations.prompt", "annotations.prompt_segments", "metadata.caption_segments"],
    )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        judge = kwargs.get("judge")
        a_side = self.name.startswith("mbencha.")
        # Paper §App: MBench-A side never invokes the VLM trigger — action-
        # conditioned rollouts deterministically actuate the memory challenge.
        # MBench-T side honors caller's choice (default off via cli flag).
        skip_trigger = True if a_side else bool(kwargs.get("ignore_trigger", False))
        if judge is None:
            return [self._invalid(item, "missing required keyword argument: judge")]
        context = _prompt_context_from_item(item)
        # A causal: single caption → wrap as single-segment caption_segments
        if not context.caption_segments:
            single_caption = (
                (item.annotations.prompt and str(item.annotations.prompt))
                or (item.metadata.get("caption") and str(item.metadata["caption"]))
            )
            if single_caption:
                context = _prompt_context_from_item(item, fallback_caption=single_caption)
        if not context.caption_segments:
            return [self._invalid(item, "missing prompt or caption_segments")]
        try:
            score = self._evaluate_context(
                item_id=item.item_id,
                model_id=item.model_id,
                context=context,
                video_path=first_video_path(item),
                judge=judge,
                metric=self.name,
                skip_trigger=skip_trigger,
            )
            if not skip_trigger and not score.triggered:
                return [UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__{self._unit_suffix}",
                    scores={
                        "triggered": 0.0,
                        "trigger_score": float(score.trigger_score),
                    },
                    valid=True,
                    metadata={
                        "model_id": item.model_id,
                        "trigger_reason": score.trigger_reason,
                        **score.metadata,
                    },
                )]
        except Exception as e:
            return [self._invalid(item, str(e))]

        raw_scores = _vlm_scores_from_item_score(score)
        unit_scores = {
            "triggered": 1.0 if score.triggered else 0.0,
            "trigger_score": float(score.trigger_score),
        }
        unit_scores.update(self._scores_from_vlm(raw_scores, score))
        return [UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__{self._unit_suffix}",
            scores=unit_scores,
            valid=True,
            metadata={
                "model_id": item.model_id,
                "trigger_reason": score.trigger_reason,
                "consistency_reason": score.consistency_reason,
                **score.metadata,
            },
        )]

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no causal self-evolution unit result",
            )
        unit = units[0]
        return ItemResult(
            item_id=item.item_id,
            model_id=item.model_id,
            metric=self.name,
            scores=dict(unit.scores),
            num_units=len(units),
            valid=unit.valid,
            error=unit.error,
        )

    def _evaluate_context(
        self,
        *,
        item_id: str,
        model_id: str | None,
        context: Any,
        video_path: str,
        judge: Any,
        metric: str,
        skip_trigger: bool,
    ) -> Any:
        raise NotImplementedError

    def _scores_from_vlm(self, raw_scores: dict[str, float], score: Any) -> dict[str, float]:
        raise NotImplementedError

    @property
    def _unit_suffix(self) -> str:
        raise NotImplementedError

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__{self._unit_suffix}",
            scores={},
            valid=False,
            error=error,
        )


# =============================================================================
# MBench-A — self-evolution (VLM-based, same logic as V)
# =============================================================================


class StateProgressConsistencyMetric(_VCausalSelfEvolutionMetric):
    """MBench-A state progress via VLM (same logic as V, action-conditioned context)."""

    name = A_CAUSAL_STATE_PROGRESS_METRIC
    description = "MBench-A causal state progress (VLM; action-conditioned rollout)."
    score_keys = A_CAUSAL_STATE_PROGRESS_SCORE_KEYS
    _unit_suffix = "state_progress"
    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=["annotations.prompt", "metadata.caption"],
    )

    def _evaluate_context(self, **kwargs: Any) -> Any:
        return evaluate_causal_state_progress_context(**kwargs)

    def _scores_from_vlm(self, raw_scores: dict[str, float], score: Any) -> dict[str, float]:
        reliability = float(raw_scores.get("progress_gate", 0.0))
        return {
            "state_evolution_score": float(raw_scores.get("progress_trigger_score", 0.0)),
            "reliability_score": reliability,
        }


class ProgressCorrectnessConsistencyMetric(_VCausalSelfEvolutionMetric):
    """MBench-A progress correctness via VLM (same logic as V, action-conditioned context)."""

    name = A_CAUSAL_PROGRESS_CORRECTNESS_METRIC
    description = "MBench-A causal progress correctness (VLM; action-conditioned rollout)."
    score_keys = A_CAUSAL_PROGRESS_CORRECTNESS_SCORE_KEYS
    _unit_suffix = "progress_correctness"
    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=["annotations.prompt", "metadata.caption"],
    )

    def _evaluate_context(self, **kwargs: Any) -> Any:
        return evaluate_causal_progress_correctness_context(**kwargs)

    def _scores_from_vlm(self, raw_scores: dict[str, float], score: Any) -> dict[str, float]:
        evolution = float(raw_scores.get("evolution_correctness_score", 0.0))
        reliability = float(raw_scores.get("progress_correctness", evolution / 5.0))
        return {
            "evolution_correctness_score": evolution,
            "progress_correctness": reliability,
            "reliability_score": reliability,
        }


# =============================================================================
# MBench-A registration
# =============================================================================


A_CAUSAL_INTERACTION_METRICS = (
    CameraInteractionConsistencyMetric,
    PromptInteractionConsistencyMetric,
)

A_CAUSAL_SELF_EVOLUTION_METRICS = (
    StateProgressConsistencyMetric,
    ProgressCorrectnessConsistencyMetric,
)

A_CAUSAL_METRICS = (
    *A_CAUSAL_INTERACTION_METRICS,
    *A_CAUSAL_SELF_EVOLUTION_METRICS,
)

CAUSAL_METRICS = A_CAUSAL_METRICS


# =============================================================================
# MBench-T — self-evolution
# =============================================================================


class TStateProgressConsistencyMetric(_VCausalSelfEvolutionMetric):
    name = T_CAUSAL_STATE_PROGRESS_METRIC
    description = "MBench-T causal state progress (dedicated VLM call; optional trigger gate)."
    score_keys = T_CAUSAL_STATE_PROGRESS_SCORE_KEYS
    _unit_suffix = "state_progress"

    def _evaluate_context(self, **kwargs: Any) -> Any:
        return evaluate_causal_state_progress_context(**kwargs)

    def _scores_from_vlm(self, raw_scores: dict[str, float], score: Any) -> dict[str, float]:
        reliability = float(raw_scores.get("progress_gate", 0.0))
        return {
            "state_evolution_score": float(raw_scores.get("progress_trigger_score", 0.0)),
            "progress_gate": reliability,
            "reliability_score": reliability,
        }


class TProgressCorrectnessConsistencyMetric(_VCausalSelfEvolutionMetric):
    name = T_CAUSAL_PROGRESS_CORRECTNESS_METRIC
    description = "MBench-T causal progress correctness (separate VLM call; optional trigger gate)."
    score_keys = T_CAUSAL_PROGRESS_CORRECTNESS_SCORE_KEYS
    _unit_suffix = "progress_correctness"

    def _evaluate_context(self, **kwargs: Any) -> Any:
        return evaluate_causal_progress_correctness_context(**kwargs)

    def _scores_from_vlm(self, raw_scores: dict[str, float], score: Any) -> dict[str, float]:
        evolution = float(raw_scores.get("evolution_correctness_score", 0.0))
        reliability = float(raw_scores.get("progress_correctness", evolution / 5.0))
        return {
            "evolution_correctness_score": evolution,
            "progress_correctness": reliability,
            "reliability_score": reliability,
        }


# =============================================================================
# MBench-T registration
# =============================================================================


T_CAUSAL_INTERACTION_METRICS = (TPromptInteractionConsistencyMetric,)

T_CAUSAL_SELF_EVOLUTION_METRICS = (
    TStateProgressConsistencyMetric,
    TProgressCorrectnessConsistencyMetric,
)

T_CAUSAL_METRICS = (
    *T_CAUSAL_INTERACTION_METRICS,
    *T_CAUSAL_SELF_EVOLUTION_METRICS,
)
