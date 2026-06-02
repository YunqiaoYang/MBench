# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License.

"""Unified canonical scoring schema for every MBench metric.

Every metric — regardless of its internal calculation — emits the same shape
of ``scores`` dict at the unit, item, and summary layers:

    A-side metric (no trigger gating):
        {"score": float in [0, 1], "score_raw"?: float}

    T-side metric (trigger-conditioned):
        {"triggered": 0.0 | 1.0,
         "trigger_score": float in [0, 1],
         "score": float in [0, 1],
         "score_raw"?: float}

``score`` is the canonical reliability value reported in the paper Table 1
(multiplied by 100). ``score_raw`` is the metric's native raw signal (a
distance, a 1–5 VLM rating, an unnormalized similarity) — present only when
it differs from ``score``.

Debug fields produced internally by metric implementations
(``pair_selection_score``, ``return_similarity``, ``middle_similarity``,
``mean_*``, ``std_*``, ``num_*``, alternative-method scores like ``lpips`` /
``ssim`` / ``dino_sim``) are intentionally *not* projected here. They survive
on ``UnitResult.metadata`` for transparent debugging but are excluded from the
``scores`` dict so downstream aggregation, leaderboard, and paper code can
treat every metric uniformly.

The mapping is the single source of truth — to repoint a paper column at a
different sub-score, edit the relevant row here, no metric class touched.
"""

from __future__ import annotations

import math
from typing import Any, Callable

# A spec is a function ``raw_scores: dict -> dict`` that returns the canonical
# subset (with derived ``score`` / ``score_raw``). Returning ``None`` for a key
# omits it. ``triggered`` and ``trigger_score`` are passed through verbatim
# when present in the raw dict (handled centrally below).

ScoreFn = Callable[[dict[str, Any]], dict[str, Any]]


def _safe(scores: dict[str, Any], key: str) -> float | None:
    v = scores.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-metric canonical projections.
#
# ``raw == score`` cases (departure_return_sim already in [0,1], action cosine
# in [0,1], reprojection_iou in [0,1], etc.) emit only ``score``.
# Distance / VLM-1-5 cases emit both ``score`` (normalized) and ``score_raw``
# (native unit) so debugging and per-pair re-normalization stay possible.
# ---------------------------------------------------------------------------


def _identity(scores: dict[str, Any]) -> dict[str, Any]:
    s = _safe(scores, "departure_return_sim")
    return {"score": s} if s is not None else {}


def _object_geometry(scores: dict[str, Any]) -> dict[str, Any]:
    s = _safe(scores, "reprojection_iou")
    return {"score": s} if s is not None else {}


def _epipolar(scores: dict[str, Any]) -> dict[str, Any]:
    raw = _safe(scores, "median_epi_dist")
    if raw is None:
        return {}
    return {"score": math.exp(-raw / 5.0), "score_raw": raw}


def _reprojection(scores: dict[str, Any]) -> dict[str, Any]:
    raw = _safe(scores, "median_reproj")
    if raw is None:
        return {}
    return {"score": math.exp(-raw / 12.0), "score_raw": raw}


def _style(scores: dict[str, Any]) -> dict[str, Any]:
    raw = _safe(scores, "gram_dist")
    if raw is None:
        return {}
    return {"score": max(0.0, min(1.0, 1.0 - raw)), "score_raw": raw}


def _lighting(scores: dict[str, Any]) -> dict[str, Any]:
    s = _safe(scores, "lighting_score")
    if s is None:
        return {}
    out: dict[str, Any] = {"score": s}
    raw = _safe(scores, "lighting_distance")
    if raw is not None:
        out["score_raw"] = raw
    return out


def _action_interaction(scores: dict[str, Any]) -> dict[str, Any]:
    s = _safe(scores, "action_instruction_score")
    return {"score": s} if s is not None else {}


def _prompt_interaction(scores: dict[str, Any]) -> dict[str, Any]:
    s = _safe(scores, "clip_score")
    return {"score": s} if s is not None else {}


def _state_progress(scores: dict[str, Any]) -> dict[str, Any]:
    raw = _safe(scores, "state_evolution_score")
    if raw is None:
        return {}
    return {"score": (raw - 1.0) / 4.0, "score_raw": raw}


def _progress_correctness(scores: dict[str, Any]) -> dict[str, Any]:
    raw = _safe(scores, "evolution_correctness_score")
    if raw is None:
        return {}
    return {"score": raw / 5.0, "score_raw": raw}


# Placeholder metrics (e.g. ``mbencha.environment.global_layout``) emit no
# score keys; canonicalization simply returns an empty dict.
def _NO_SCORES(_: dict[str, Any]) -> dict[str, Any]:
    return {}


CANONICAL_PROJECTIONS: dict[str, ScoreFn] = {
    # MBench-A entity
    "mbencha.entity.human_identity_consistency": _identity,
    "mbencha.entity.human_appearance_consistency": _identity,
    "mbencha.entity.object_texture_consistency": _identity,
    "mbencha.entity.object_geometry_consistency": _object_geometry,
    # MBench-A environment
    "mbencha.environment.spatial_epipolar": _epipolar,
    "mbencha.environment.spatial_reprojection": _reprojection,
    "mbencha.environment.rendering_style": _style,
    "mbencha.environment.rendering_lighting": _lighting,
    "mbencha.environment.global_layout": _NO_SCORES,
    # MBench-A causal
    "mbencha.causal.camera_interaction": _action_interaction,
    "mbencha.causal.prompt_interaction": _prompt_interaction,
    "mbencha.causal.state_progress": _state_progress,
    "mbencha.causal.progress_correctness": _progress_correctness,
    # MBench-T entity
    "mbencht.entity.human_identity_consistency": _identity,
    "mbencht.entity.human_appearance_consistency": _identity,
    "mbencht.entity.object_texture_consistency": _identity,
    "mbencht.entity.object_geometry_consistency": _object_geometry,
    # MBench-T environment
    "mbencht.environment.spatial_epipolar": _epipolar,
    "mbencht.environment.spatial_reprojection": _reprojection,
    "mbencht.environment.rendering_style": _style,
    "mbencht.environment.rendering_lighting": _lighting,
    # MBench-T causal
    "mbencht.causal.prompt_interaction": _prompt_interaction,
    "mbencht.causal.state_progress": _state_progress,
    "mbencht.causal.progress_correctness": _progress_correctness,
}


# ``mbencha.*`` metrics never go through trigger gating per paper §App
# (action-conditioned rollouts deterministically actuate the memory challenge).
# Trigger keys produced anywhere on the A side are dropped at canonicalization.
A_SIDE_PREFIX = "mbencha."


def canonicalize_scores(metric_name: str, raw_scores: dict[str, Any]) -> dict[str, Any]:
    """Project a metric's internal ``scores`` dict onto the canonical schema.

    - On the T side, ``triggered`` / ``trigger_score`` from the VLM judge are
      preserved verbatim.
    - On the A side they are dropped (paper §App: A omits the VLM trigger).
    - The metric's primary signal is rewritten to ``score`` (and ``score_raw``
      when the raw differs from the [0,1] reliability).
    - All other internal keys (debug pairs, mean/std/num counts, alternative
      similarity metrics) are dropped from ``scores`` — callers should keep
      them on ``UnitResult.metadata`` if they want them.
    """

    out: dict[str, Any] = {}

    if not metric_name.startswith(A_SIDE_PREFIX):
        triggered = raw_scores.get("triggered")
        if triggered is not None:
            out["triggered"] = float(triggered)
        trigger_score = raw_scores.get("trigger_score")
        if trigger_score is not None:
            out["trigger_score"] = float(trigger_score)

    projection = CANONICAL_PROJECTIONS.get(metric_name)
    if projection is None:
        # Unknown metric: be conservative, pass through ``score``/``score_raw``
        # if the producer happened to use the canonical names already.
        for key in ("score", "score_raw"):
            v = _safe(raw_scores, key)
            if v is not None:
                out[key] = v
        return out

    out.update(projection(raw_scores))
    return out


CANONICAL_UNIT_SCORE_KEYS: tuple[str, ...] = (
    "triggered",
    "trigger_score",
    "score",
    "score_raw",
)
"""Stable advertised key set for the canonical scores dict.

Not every metric emits all four (``triggered`` / ``trigger_score`` only on T
side; ``score_raw`` only when ≠ ``score``); this tuple is the *superset* and
the order in which keys appear in summary tables.
"""

__all__ = [
    "CANONICAL_PROJECTIONS",
    "CANONICAL_UNIT_SCORE_KEYS",
    "canonicalize_scores",
]
