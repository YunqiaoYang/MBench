# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Causal calculations shared by MBench-A and MBench-T.

- Interaction: OpenCLIP text-video segment scoring (proportional temporal windows).
- Self-evolution (V): separate VLM calls for state progress vs progress correctness.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mbench.metrics.video_io import extract_segment_pil_frames


class CLIPSegmentScorer:
    def __init__(
        self,
        *,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = "cpu",
    ):
        import open_clip

        self.device = device
        self.model, _train, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score(self, text: str, images: list[Any]) -> float:
        import torch

        if not images:
            return float("nan")
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = []
            for image in images:
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                scores.append(max(0.0, float((text_features @ image_features.T).item()) * 100.0))
        return float(np.mean(scores))


def compute_segment_clip_scores(
    video_path: str,
    prompts: list[str],
    *,
    scorer: CLIPSegmentScorer,
    segment_duration: float = 5.0,
    num_samples_per_segment: int = 8,
) -> list[dict[str, Any]]:
    """Deprecated fixed 5s windows; prefer compute_caption_segment_clip_scores."""
    results = []
    for idx, prompt in enumerate(prompts):
        frames, start_time, end_time = extract_segment_pil_frames(
            video_path,
            segment_idx=idx,
            segment_duration=segment_duration,
            num_samples=num_samples_per_segment,
        )
        clip_score = scorer.score(prompt, frames)
        results.append({
            "segment_idx": idx,
            "start_time": start_time,
            "end_time": end_time,
            "prompt": prompt,
            "clip_score": round(clip_score, 4) if np.isfinite(clip_score) else clip_score,
            "num_frames": len(frames),
        })
    return results


def compute_caption_segment_clip_scores(
    video_path: str,
    prompts: list[str],
    *,
    scorer: CLIPSegmentScorer,
    num_samples_per_segment: int = 8,
) -> list[dict[str, Any]]:
    """CLIP scores aligned to prompt count (video duration split evenly per segment)."""
    import cv2
    from PIL import Image

    if not prompts:
        return []

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"cannot read video for CLIP scoring: {video_path}")

    num_segments = len(prompts)
    results: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        start = int(round(total * idx / num_segments))
        end = max(int(round(total * (idx + 1) / num_segments)), start + 1)
        count = end - start
        if count <= num_samples_per_segment:
            indices = list(range(start, end))
        else:
            indices = np.linspace(start, end - 1, num_samples_per_segment, dtype=int).tolist()

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        start_time = start / fps
        end_time = end / fps
        clip_score = scorer.score(prompt, frames)
        results.append({
            "segment_idx": idx,
            "start_time": round(start_time, 4),
            "end_time": round(end_time, 4),
            "prompt": prompt,
            "clip_score": round(clip_score, 4) if np.isfinite(clip_score) else clip_score,
            "num_frames": len(frames),
        })
    cap.release()
    return results


#========================#
# MBench-T self-evolution (VLM) calculations
#========================#

T_CAUSAL_STATE_PROGRESS_METRIC = "mbencht.causal.state_progress"
T_CAUSAL_PROGRESS_CORRECTNESS_METRIC = "mbencht.causal.progress_correctness"

_CAUSAL_PROGRESS_POSITIVE_EXAMPLE = (
    "Caption segments describe a red-hot metal piece, then an apron fully hides it, "
    "then the apron moves away and the same metal reappears bent into a forked shape. "
    "The video shows the hide/reveal and the bent metal after reveal."
)
_CAUSAL_PROGRESS_NEGATIVE_EXAMPLE = (
    "Caption segments say the metal becomes bent after reveal, but the video shows "
    "the same straight metal unchanged after the apron leaves."
)
_CAUSAL_CORRECTNESS_POSITIVE_EXAMPLE = (
    "After the apron moves away, the metal shape changes in a physically plausible way "
    "consistent with forging (gradual reveal, bent form matches the narrative)."
)
_CAUSAL_CORRECTNESS_NEGATIVE_EXAMPLE = (
    "The metal instantly teleports to a different shape without plausible intermediate "
    "motion, or changes material/color in a causally impossible way."
)


def _format_causal_context_lines(context: Any) -> list[str]:
    lines = [
        f"sample_id: {context.sample_id}",
        f"subset: {context.subset}",
        "caption_segments:",
    ]
    for idx, segment in enumerate(context.caption_segments, start=1):
        lines.append(f"  {idx}. {segment}")
    return lines


def build_state_progress_judge_prompt(context: Any) -> str:
    return "\n".join([
        "You are evaluating CAUSAL STATE PROGRESS in a generated video (MBench causal subset).",
        "Score whether the expected state change in the caption actually happens in the video.",
        "Ignore camera motion, lighting, and style unless they block seeing the state change.",
        "",
        *_format_causal_context_lines(context),
        "",
        "Scoring scale for progress_trigger_score in [1, 5]:",
        "  1 = the expected state evolution did not happen",
        "  5 = the expected state evolution fully happened",
        "",
        "Positive example (score 5):",
        f"  {_CAUSAL_PROGRESS_POSITIVE_EXAMPLE}",
        "",
        "Negative example (score 1):",
        f"  {_CAUSAL_PROGRESS_NEGATIVE_EXAMPLE}",
        "",
        "Return JSON with keys: progress_trigger_score, confidence (optional), reason.",
    ])


def build_progress_correctness_judge_prompt(context: Any) -> str:
    return "\n".join([
        "You are evaluating CAUSAL PROGRESS CORRECTNESS in a generated video (MBench causal subset).",
        "Score whether the state transition is physically and causally plausible.",
        "You may score this even when the change is subtle; focus on plausibility, not aesthetics.",
        "",
        *_format_causal_context_lines(context),
        "",
        "Scoring scale for evolution_correctness_score in [1, 5]:",
        "  1 = physically/causally invalid evolution",
        "  5 = fully plausible and correct evolution",
        "",
        "Positive example (score 5):",
        f"  {_CAUSAL_CORRECTNESS_POSITIVE_EXAMPLE}",
        "",
        "Negative example (score 1):",
        f"  {_CAUSAL_CORRECTNESS_NEGATIVE_EXAMPLE}",
        "",
        "Return JSON with keys: evolution_correctness_score, confidence (optional), reason.",
    ])


def _run_causal_vlm_evaluation(
    *,
    item_id: str,
    model_id: str | None,
    context: Any,
    video_path: str,
    judge: Any,
    metric: str,
    consistency_prompt_builder: Callable[[Any], str],
    consistency_score_keys: tuple[str, ...],
    parse_consistency_scores: Callable[[dict[str, float]], dict[str, float]],
    reliability_from_scores: Callable[[dict[str, float]], float],
    skip_trigger: bool = False,
) -> Any:
    from mbench.metrics.trigger import build_vlm_trigger_prompt
    from mbench.metrics.schema import MBenchTItemScore

    metadata: dict[str, Any] = {
        "sample_id": context.sample_id,
        "subset": context.subset,
        "video_path": video_path,
    }
    triggered = True
    trigger_score = 1.0
    trigger_reason = "trigger skipped" if skip_trigger else None

    if not skip_trigger:
        trigger_prompt = build_vlm_trigger_prompt(
            context,
            metric_name=metric,
            trigger_kind="causal",
        )
        trigger = judge.judge_trigger(
            context=context,
            video_path=video_path,
            prompt=trigger_prompt,
        )
        metadata.update({
            "trigger_confidence": trigger.confidence,
            "trigger_metadata": trigger.metadata,
        })
        triggered = bool(trigger.triggered)
        trigger_score = 1.0 if triggered else 0.0
        trigger_reason = trigger.reason
        if not triggered:
            return MBenchTItemScore(
                item_id=item_id,
                model_id=model_id,
                metric=metric,
                triggered=False,
                trigger_score=0.0,
                reliability_score=None,
                trigger_reason=trigger_reason,
                consistency_reason=None,
                metadata=metadata,
            )

    consistency_prompt = consistency_prompt_builder(context)
    consistency = judge.judge_consistency(
        context=context,
        video_path=video_path,
        prompt=consistency_prompt,
        score_keys=consistency_score_keys,
    )
    raw_scores = parse_consistency_scores(consistency.scores)
    metadata.update({
        "consistency_confidence": consistency.confidence,
        "causal_vlm_scores": raw_scores,
        "consistency_metadata": consistency.metadata,
        "skip_trigger": skip_trigger,
    })
    return MBenchTItemScore(
        item_id=item_id,
        model_id=model_id,
        metric=metric,
        triggered=True,
        trigger_score=trigger_score,
        reliability_score=reliability_from_scores(raw_scores),
        trigger_reason=trigger_reason,
        consistency_reason=consistency.reason,
        metadata=metadata,
    )


def evaluate_causal_state_progress(
    *,
    item_id: str,
    model_id: str | None,
    sample: dict[str, Any],
    video_path: str,
    judge: Any,
    metric: str = T_CAUSAL_STATE_PROGRESS_METRIC,
    skip_trigger: bool = False,
) -> Any:
    from mbench.metrics.prompting import prompt_context_from_sample

    context = prompt_context_from_sample(sample)
    if context.subset != "causal":
        raise ValueError(
            f"{metric} expects subset='causal', got {context.subset!r} for {context.sample_id}"
        )
    return evaluate_causal_state_progress_context(
        item_id=item_id,
        model_id=model_id,
        context=context,
        video_path=video_path,
        judge=judge,
        metric=metric,
        skip_trigger=skip_trigger,
    )


def evaluate_causal_state_progress_context(
    *,
    item_id: str,
    model_id: str | None,
    context: Any,
    video_path: str,
    judge: Any,
    metric: str = T_CAUSAL_STATE_PROGRESS_METRIC,
    skip_trigger: bool = False,
) -> Any:
    return _run_causal_vlm_evaluation(
        item_id=item_id,
        model_id=model_id,
        context=context,
        video_path=video_path,
        judge=judge,
        metric=metric,
        consistency_prompt_builder=build_state_progress_judge_prompt,
        consistency_score_keys=("progress_trigger_score",),
        parse_consistency_scores=progress_vlm_scores_from_judge_scores,
        reliability_from_scores=lambda s: float(s["progress_gate"]),
        skip_trigger=skip_trigger,
    )


def evaluate_causal_progress_correctness(
    *,
    item_id: str,
    model_id: str | None,
    sample: dict[str, Any],
    video_path: str,
    judge: Any,
    metric: str = T_CAUSAL_PROGRESS_CORRECTNESS_METRIC,
    skip_trigger: bool = False,
) -> Any:
    from mbench.metrics.prompting import prompt_context_from_sample

    context = prompt_context_from_sample(sample)
    if context.subset != "causal":
        raise ValueError(
            f"{metric} expects subset='causal', got {context.subset!r} for {context.sample_id}"
        )
    return evaluate_causal_progress_correctness_context(
        item_id=item_id,
        model_id=model_id,
        context=context,
        video_path=video_path,
        judge=judge,
        metric=metric,
        skip_trigger=skip_trigger,
    )


def evaluate_causal_progress_correctness_context(
    *,
    item_id: str,
    model_id: str | None,
    context: Any,
    video_path: str,
    judge: Any,
    metric: str = T_CAUSAL_PROGRESS_CORRECTNESS_METRIC,
    skip_trigger: bool = False,
) -> Any:
    return _run_causal_vlm_evaluation(
        item_id=item_id,
        model_id=model_id,
        context=context,
        video_path=video_path,
        judge=judge,
        metric=metric,
        consistency_prompt_builder=build_progress_correctness_judge_prompt,
        consistency_score_keys=("evolution_correctness_score",),
        parse_consistency_scores=correctness_vlm_scores_from_judge_scores,
        reliability_from_scores=lambda s: float(s["progress_correctness"]),
        skip_trigger=skip_trigger,
    )


def progress_vlm_scores_from_judge_scores(scores: dict[str, float]) -> dict[str, float]:
    if "progress_trigger_score" not in scores:
        raise ValueError("causal progress judge result missing 'progress_trigger_score'")
    progress_trigger_score = _clamp(float(scores["progress_trigger_score"]), 1.0, 5.0)
    progress_gate = (progress_trigger_score - 1.0) / 4.0
    return {
        "progress_trigger_score": round(progress_trigger_score, 6),
        "progress_gate": round(progress_gate, 6),
    }


def correctness_vlm_scores_from_judge_scores(scores: dict[str, float]) -> dict[str, float]:
    if "evolution_correctness_score" not in scores:
        raise ValueError(
            "causal correctness judge result missing 'evolution_correctness_score'"
        )
    evolution_correctness_score = _clamp(float(scores["evolution_correctness_score"]), 1.0, 5.0)
    progress_correctness = evolution_correctness_score / 5.0
    return {
        "evolution_correctness_score": round(evolution_correctness_score, 6),
        "progress_correctness": round(progress_correctness, 6),
    }


def causal_vlm_scores_from_judge_scores(scores: dict[str, float]) -> dict[str, float]:
    """Parse a combined judge payload (backward compatibility)."""
    progress = progress_vlm_scores_from_judge_scores(scores)
    correctness = correctness_vlm_scores_from_judge_scores(scores)
    combined_score = progress["progress_gate"] * correctness["evolution_correctness_score"]
    return {
        **progress,
        **correctness,
        "combined_score": round(combined_score, 6),
        "self_evolution_score": round(combined_score, 6),
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))
