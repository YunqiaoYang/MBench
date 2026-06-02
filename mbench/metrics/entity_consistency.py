# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Entity consistency metric plugins for MBench-A and MBench-T.

- MBench-A metrics use whole-video departure-return track scoring.
- MBench-T entity-human metrics use caption-segment trigger-conditioned scoring.
- Heavy vision work lives in entity_calculations.py; this file owns metric wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable

from mbench.core.contract import InputContract
from mbench.core.item import EvaluationItem
from mbench.core.registry import Metric
from mbench.core.result import ItemResult, UnitResult
from mbench.metrics.entity_calculations import (
    compute_object_geometry_track,
    compute_object_texture_track,
    compute_v_object_geometry_track,
    compute_v_object_texture_track,
    object_anchor_from_item,
    object_text_from_card,
    compute_human_appearance_tracks,
    compute_entity_trigger_conditioned_score,
    detect_and_track_faces,
    entity_memory_context_from_item,
    init_insightface,
    sample_bgr_frames_by_caption_segment,
    track_mean_embeddings,
    write_human_debug_artifacts,
)
from mbench.metrics.environment_calculations import load_da3_data
from mbench.metrics.metric_math import departure_return_similarity, finite_mean
from mbench.metrics.video_io import first_video_path, sample_bgr_frames


A_ENTITY_HUMAN_IDENTITY_METRIC = "mbencha.entity.human_identity_consistency"
A_ENTITY_HUMAN_APPEARANCE_METRIC = "mbencha.entity.human_appearance_consistency"
A_ENTITY_OBJECT_TEXTURE_METRIC = "mbencha.entity.object_texture_consistency"
A_ENTITY_OBJECT_GEOMETRY_METRIC = "mbencha.entity.object_geometry_consistency"
A_ENTITY_DR_SCORE_KEYS = (
    "departure_return_sim",
    "first_seg_internal_sim",
    "last_seg_internal_sim",
)

T_ENTITY_HUMAN_IDENTITY_METRIC = "mbencht.entity.human_identity_consistency"
T_ENTITY_HUMAN_APPEARANCE_METRIC = "mbencht.entity.human_appearance_consistency"
T_ENTITY_OBJECT_TEXTURE_METRIC = "mbencht.entity.object_texture_consistency"
T_ENTITY_OBJECT_GEOMETRY_METRIC = "mbencht.entity.object_geometry_consistency"
T_ENTITY_HUMAN_SCORE_KEYS = (
    "triggered",
    "trigger_score",
    "initial_presence_score",
    "middle_absence_score",
    "final_presence_score",
    "departure_return_sim",
    "reliability_score",
)
T_ENTITY_OBJECT_DR_SCORE_KEYS = (
    "triggered",
    "trigger_score",
    *A_ENTITY_DR_SCORE_KEYS,
    "reliability_score",
)
T_ENTITY_OBJECT_GEOMETRY_SCORE_KEYS = (
    "triggered",
    "trigger_score",
    "reliability_score",
    "hu_moment_distance",
    "aspect_ratio_std",
    "depth_change_rate",
    "reprojection_iou",
    "center_reproj_error",
)
SegmentEvidenceExtractor = Callable[[EvaluationItem, list[list[Any]], dict[str, Any]], list[list[Any]]]
# =============================================================================
# Shared A Entity Departure-Return Base
# =============================================================================


class _TrackDRMetric(Metric):
    embedding_key = "embedding"
    unit_type = "track"
    n_frames = 32
    score_keys: tuple[str, ...] = A_ENTITY_DR_SCORE_KEYS

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            tracks = self._compute_tracks(item, **kwargs)
            return [self._track_result(item, track, idx) for idx, track in enumerate(tracks)]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def _compute_tracks(self, item: EvaluationItem, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _track_result(self, item: EvaluationItem, track: dict[str, Any], idx: int) -> UnitResult:
        records = _extract_records(track)
        raw_scores = departure_return_similarity(records, embedding_key=self.embedding_key)
        scores = {
            key: float(value)
            for key, value in raw_scores.items()
            if key in self.score_keys and isinstance(value, (int, float))
        }
        unit_id = str(track.get("track_id") or track.get("object_id") or f"track_{idx}")
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=unit_id,
            scores=scores,
            valid=bool(scores.get("departure_return_sim") is not None),
            metadata={
                key: value
                for key, value in raw_scores.items()
                if key not in self.score_keys
            },
        )

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__{self.name}",
            scores={},
            valid=False,
            error=error,
        )


def _extract_units(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [p for p in value if isinstance(p, dict)]
        return [payload]
    return []


def _extract_records(track: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("features", "detections", "frames", "appearance"):
        value = track.get(key)
        if isinstance(value, list):
            records = []
            for idx, record in enumerate(value):
                if isinstance(record, dict):
                    if "frame_idx" not in record:
                        record = {**record, "frame_idx": idx}
                    records.append(record)
            return records
    return []


# =============================================================================
# MBench-A Entity Metrics
# =============================================================================


class HumanIdentityConsistencyMetric(_TrackDRMetric):
    name = A_ENTITY_HUMAN_IDENTITY_METRIC
    description = "Human face identity departure-return consistency computed from ArcFace tracks."
    embedding_key = "embedding"
    unit_type = "person_track"
    score_keys = A_ENTITY_DR_SCORE_KEYS

    contract = InputContract.from_field_paths(required=["media.videos"])

    def _compute_tracks(self, item: EvaluationItem, **kwargs: Any) -> list[dict[str, Any]]:
        video_path = first_video_path(item)
        frames, _indices = sample_bgr_frames(video_path, int(kwargs.get("n_frames", self.n_frames)))
        app = kwargs.get("insightface_app") or init_insightface(int(kwargs.get("device_id", 0)))
        return detect_and_track_faces(app, frames)


class HumanAppearanceConsistencyMetric(_TrackDRMetric):
    name = A_ENTITY_HUMAN_APPEARANCE_METRIC
    description = "Human full-body appearance departure-return consistency computed from DINOv2 track features."
    embedding_key = "embedding"
    unit_type = "person_track"
    score_keys = A_ENTITY_DR_SCORE_KEYS

    contract = InputContract.from_field_paths(required=["media.videos"])

    def _compute_tracks(self, item: EvaluationItem, **kwargs: Any) -> list[dict[str, Any]]:
        video_path = first_video_path(item)
        frames, _indices = sample_bgr_frames(video_path, int(kwargs.get("n_frames", self.n_frames)))
        app = kwargs.get("insightface_app") or init_insightface(int(kwargs.get("device_id", 0)))
        face_tracks = detect_and_track_faces(app, frames)
        return compute_human_appearance_tracks(
            face_tracks,
            frames,
            device=str(kwargs.get("device", "cuda")),
        )


class ObjectTextureConsistencyMetric(_TrackDRMetric):
    name = A_ENTITY_OBJECT_TEXTURE_METRIC
    description = (
        "Object texture departure-return consistency for the annotated target object: "
        "SAM2Video tracking from GT anchor center and DINOv2 embeddings."
    )
    embedding_key = "embedding"
    unit_type = "target_object"
    score_keys = A_ENTITY_DR_SCORE_KEYS

    contract = InputContract.from_field_paths(
        required=["media.videos", "metadata.object_anchor"],
    )

    def _compute_tracks(self, item: EvaluationItem, **kwargs: Any) -> list[dict[str, Any]]:
        anchor = object_anchor_from_item(item)
        video_path = first_video_path(item)
        frames, _indices = sample_bgr_frames(video_path, int(kwargs.get("n_frames", self.n_frames)))
        track = compute_object_texture_track(
            frames,
            center_x=anchor["center_x"],
            center_y=anchor["center_y"],
            anchor_width=anchor["width"],
            anchor_height=anchor["height"],
            device=str(kwargs.get("device", "cuda")),
            sam2=kwargs.get("sam2"),
            sam2_video=kwargs.get("sam2_video"),
            dinov2=kwargs.get("dinov2"),
        )
        return [track] if track else []


class ObjectGeometryConsistencyMetric(Metric):
    name = A_ENTITY_OBJECT_GEOMETRY_METRIC
    description = "Object geometry consistency computed from mask shape, depth, and reprojection geometry artifacts."
    artifact_name = "da3"
    unit_type = "target_object"
    score_keys = (
        "hu_moment_distance",
        "aspect_ratio_std",
        "depth_change_rate",
        "reprojection_iou",
        "center_reproj_error",
    )

    @property
    def contract(self) -> InputContract:  # type: ignore[override]
        return InputContract.from_field_paths(
            required=["media.videos", "artifacts.da3", "metadata.object_anchor"],
        )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        artifact = item.artifacts.get(self.artifact_name)
        if artifact is None or artifact.path is None:
            return [self._invalid_geometry(item, "missing object geometry artifact")]

        try:
            anchor = object_anchor_from_item(item)
        except ValueError as e:
            return [self._invalid_geometry(item, str(e))]

        try:
            video_path = first_video_path(item)
            frames, frame_indices = sample_bgr_frames(
                video_path,
                int(kwargs.get("n_frames", 32)),
            )
            da3_data = load_da3_data(artifact.path)
            obj = compute_object_geometry_track(
                da3_data,
                frames,
                frame_indices,
                center_x=anchor["center_x"],
                center_y=anchor["center_y"],
                anchor_width=anchor["width"],
                anchor_height=anchor["height"],
                device=str(kwargs.get("device", "cuda")),
                sam2=kwargs.get("sam2"),
                sam2_video=kwargs.get("sam2_video"),
            )
            if obj is None:
                return [self._invalid_geometry(item, "no object geometry track from anchor")]
            return [self._object_result(item, obj, 0)]
        except Exception as e:
            return [self._invalid_geometry(item, str(e))]

    def _invalid_geometry(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__object_geometry",
            valid=False,
            error=error,
        )

    def _object_result(self, item: EvaluationItem, obj: dict[str, Any], idx: int) -> UnitResult:
        frames = _extract_records(obj)
        scores = {
            "hu_moment_distance": _mean_delta(frames, "hu_moments"),
            "aspect_ratio_std": _std_value(frames, "aspect_ratio"),
            "depth_change_rate": _mean_relative_delta(frames, "mean_depth"),
            "reprojection_iou": finite_mean([
                float(p["reprojection_iou"])
                for p in obj.get("pairs", [])
                if isinstance(p, dict) and isinstance(p.get("reprojection_iou"), (int, float))
            ]),
            "center_reproj_error": finite_mean([
                float(p["center_reproj_error"])
                for p in obj.get("pairs", [])
                if isinstance(p, dict) and isinstance(p.get("center_reproj_error"), (int, float))
            ]),
        }
        clean_scores = {k: float(v) for k, v in scores.items() if v is not None}
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=str(obj.get("object_id") or obj.get("track_id") or f"object_{idx}"),
            scores=clean_scores,
            valid=bool(clean_scores),
        )


# =============================================================================
# MBench-A Object Geometry Helpers
# =============================================================================


def _mean_delta(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in records if key in r]
    if len(values) < 2:
        return None
    import numpy as np

    return finite_mean([
        float(np.linalg.norm(np.asarray(values[i], dtype=np.float64) - np.asarray(values[i - 1], dtype=np.float64)))
        for i in range(1, len(values))
    ])


def _std_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    if not values:
        return None
    import numpy as np

    return float(np.std(values))


def _mean_relative_delta(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    if len(values) < 2:
        return None
    deltas = []
    for i in range(1, len(values)):
        denom = abs(values[i - 1])
        if denom > 1e-8:
            deltas.append(abs(values[i] - values[i - 1]) / denom)
    return finite_mean(deltas)


# =============================================================================
# Shared V Entity Trigger-Conditioned Base
# =============================================================================


class _VEntityHumanTriggerConditionedMetric(Metric):
    unit_type = "entity_human_trigger"
    score_keys = T_ENTITY_HUMAN_SCORE_KEYS
    embedding_kind = "identity"
    default_absence_similarity_threshold = 0.55
    contract = InputContract.from_field_paths(
        required=["media.videos"],
        optional=["annotations.prompt", "annotations.prompt_segments", "metadata.caption_segments"],
    )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            from mbench.metrics.trigger import judge_vlm_trigger_for_item

            context = entity_memory_context_from_item(item)
            caption_segments = context["caption_segments"]
            if not caption_segments:
                return [self._invalid(item, "missing prompt or caption_segments")]
            trigger = judge_vlm_trigger_for_item(
                item,
                kwargs,
                metric_name=self.name,
                trigger_kind="entity_human",
            )
            segment_frames = sample_bgr_frames_by_caption_segment(
                first_video_path(item),
                num_segments=len(caption_segments),
                num_samples_per_segment=int(kwargs.get("num_samples_per_segment", 4)),
            )
            extractor: SegmentEvidenceExtractor = kwargs.get("segment_evidence_extractor") or self._extract_segment_evidence
            segment_embeddings = extractor(item, segment_frames, kwargs)
            score = compute_entity_trigger_conditioned_score(
                segment_embeddings,
                absence_similarity_threshold=float(kwargs.get("absence_similarity_threshold", self.default_absence_similarity_threshold)),
            )
            algorithm_triggered = score.get("triggered", 0.0)
            algorithm_trigger_score = score.get("trigger_score", 0.0)
            score["triggered"] = 1.0 if trigger.triggered else 0.0
            score["trigger_score"] = float(trigger.score)
            if not trigger.triggered:
                score.pop("reliability_score", None)
            metadata: dict[str, Any] = {
                "model_id": item.model_id,
                "sample_id": context["sample_id"],
                "condition_id": context["metadata"].get("condition_id"),
                "num_caption_segments": len(caption_segments),
                "segment_presence_counts": [len(e) for e in segment_embeddings],
                "embedding_kind": self.embedding_kind,
                "algorithm_triggered": algorithm_triggered,
                "algorithm_trigger_score": algorithm_trigger_score,
                "trigger_reason": trigger.reason,
                "trigger_confidence": trigger.confidence,
                "vlm_trigger_metadata": trigger.metadata,
            }
            debug_dir = kwargs.get("debug_dir")
            if debug_dir:
                metadata.update(
                    write_human_debug_artifacts(
                        item=item,
                        metric_name=self.name,
                        debug_dir=Path(debug_dir),
                        segment_frames=segment_frames,
                        caption_segments=list(caption_segments),
                        scores=score,
                    )
                )
            return [
                UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__entity_human_trigger",
                    scores=score,
                    valid=True,
                    metadata=metadata,
                )
            ]
        except Exception as e:
            return [self._invalid(item, str(e))]

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no entity human trigger unit result",
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

    def _extract_segment_evidence(self, item: EvaluationItem, segment_frames: list[list[Any]], kwargs: dict[str, Any]) -> list[list[Any]]:
        raise NotImplementedError

    def _invalid(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__entity_human_trigger",
            scores={},
            valid=False,
            error=error,
        )


# =============================================================================
# MBench-T Entity-Human Metrics
# =============================================================================


class TEntityHumanIdentityConsistencyMetric(_VEntityHumanTriggerConditionedMetric):
    name = T_ENTITY_HUMAN_IDENTITY_METRIC
    description = "MBench-T entity/human identity consistency with trigger-conditioned caption-segment scoring."
    embedding_kind = "face_identity"

    def _extract_segment_evidence(self, item: EvaluationItem, segment_frames: list[list[Any]], kwargs: dict[str, Any]) -> list[list[Any]]:
        app = kwargs.get("insightface_app") or init_insightface(int(kwargs.get("device_id", 0)))
        return [track_mean_embeddings(detect_and_track_faces(app, frames)) for frames in segment_frames]


class TEntityHumanAppearanceConsistencyMetric(_VEntityHumanTriggerConditionedMetric):
    name = T_ENTITY_HUMAN_APPEARANCE_METRIC
    description = "MBench-T entity/human appearance consistency with trigger-conditioned caption-segment scoring."
    embedding_kind = "body_appearance"

    def _extract_segment_evidence(self, item: EvaluationItem, segment_frames: list[list[Any]], kwargs: dict[str, Any]) -> list[list[Any]]:
        app = kwargs.get("insightface_app") or init_insightface(int(kwargs.get("device_id", 0)))
        device = str(kwargs.get("device", "cuda"))
        segment_embeddings: list[list[Any]] = []
        for frames in segment_frames:
            face_tracks = detect_and_track_faces(app, frames)
            appearance_tracks = compute_human_appearance_tracks(face_tracks, frames, device=device)
            segment_embeddings.append(track_mean_embeddings(appearance_tracks))
        return segment_embeddings


# =============================================================================
# MBench-T Entity — Object
# =============================================================================


class TEntityObjectTextureConsistencyMetric(_TrackDRMetric):
    name = T_ENTITY_OBJECT_TEXTURE_METRIC
    description = (
        "MBench-T entity/object texture consistency: first Grounding DINO detection, "
        "SAM2Video mask track, DINOv2 embeddings, departure-return scoring (aligned with MBench-A)."
    )
    embedding_key = "embedding"
    unit_type = "target_object"
    score_keys = T_ENTITY_OBJECT_DR_SCORE_KEYS

    contract = InputContract.from_field_paths(
        required=["media.videos", "metadata.object_card"],
        optional=["annotations.prompt", "annotations.prompt_segments", "metadata.caption_segments"],
    )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        try:
            from mbench.metrics.trigger import judge_vlm_trigger_for_item

            trigger = judge_vlm_trigger_for_item(
                item,
                kwargs,
                metric_name=self.name,
                trigger_kind="entity_object",
            )
            units = super().evaluate(item, **kwargs)
            for unit in units:
                unit.scores["triggered"] = 1.0 if trigger.triggered else 0.0
                unit.scores["trigger_score"] = float(trigger.score)
                if unit.valid and trigger.triggered and "departure_return_sim" in unit.scores:
                    unit.scores["reliability_score"] = float(unit.scores["departure_return_sim"])
                else:
                    unit.scores.pop("reliability_score", None)
                unit.metadata.update({
                    "trigger_reason": trigger.reason,
                    "trigger_confidence": trigger.confidence,
                    "vlm_trigger_metadata": trigger.metadata,
                })
            return units
        except Exception as e:
            return [self._invalid(item, str(e))]

    def _compute_tracks(self, item: EvaluationItem, **kwargs: Any) -> list[dict[str, Any]]:
        object_text = object_text_from_card(item.metadata.get("object_card"))
        if not object_text:
            raise ValueError("missing object_card object_name")
        video_path = first_video_path(item)
        frames, _indices = sample_bgr_frames(video_path, int(kwargs.get("n_frames", self.n_frames)))
        track = compute_v_object_texture_track(
            frames,
            object_text,
            device=str(kwargs.get("device", "cuda")),
            min_mask_area=float(kwargs.get("min_mask_area", 0.005)),
            grounding=kwargs.get("grounding_model"),
            sam2=kwargs.get("sam2"),
            sam2_video=kwargs.get("sam2_video"),
            dinov2=kwargs.get("dinov2"),
            box_threshold=float(kwargs.get("box_threshold", 0.25)),
            text_threshold=float(kwargs.get("text_threshold", 0.20)),
        )
        return [track] if track else []


class TEntityObjectGeometryConsistencyMetric(Metric):
    name = T_ENTITY_OBJECT_GEOMETRY_METRIC
    description = (
        "MBench-T entity/object geometry consistency from DA3 depth and "
        "per-frame Grounding DINO + SAM2 point masks."
    )
    artifact_name = "da3"
    unit_type = "target_object"
    score_keys = T_ENTITY_OBJECT_GEOMETRY_SCORE_KEYS

    @property
    def contract(self) -> InputContract:  # type: ignore[override]
        return InputContract.from_field_paths(
            required=["media.videos", "artifacts.da3", "metadata.object_card"],
            optional=["annotations.prompt", "annotations.prompt_segments", "metadata.caption_segments"],
        )

    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        artifact = item.artifacts.get(self.artifact_name)
        if artifact is None or artifact.path is None:
            return [self._invalid_geometry(item, "missing object geometry artifact")]

        context = entity_memory_context_from_item(item)
        object_text = object_text_from_card(context["object_card"])
        if not object_text:
            return [self._invalid_geometry(item, "missing object_card object_name")]

        try:
            from mbench.metrics.trigger import judge_vlm_trigger_for_item

            trigger = judge_vlm_trigger_for_item(
                item,
                kwargs,
                metric_name=self.name,
                trigger_kind="entity_object",
            )
            if not trigger.triggered:
                return [UnitResult(
                    item_id=item.item_id,
                    metric=self.name,
                    unit_type=self.unit_type,
                    unit_id=f"{item.item_id}__object_geometry",
                    scores={
                        "triggered": 0.0,
                        "trigger_score": float(trigger.score),
                    },
                    valid=True,
                    metadata={
                        "model_id": item.model_id,
                        "trigger_reason": trigger.reason,
                        "trigger_confidence": trigger.confidence,
                        "vlm_trigger_metadata": trigger.metadata,
                    },
                )]
            video_path = first_video_path(item)
            frames, frame_indices = sample_bgr_frames(
                video_path,
                int(kwargs.get("n_frames", 32)),
            )
            da3_data = load_da3_data(artifact.path)
            obj = compute_v_object_geometry_track(
                da3_data,
                frames,
                frame_indices,
                object_text,
                device=str(kwargs.get("device", "cuda")),
                min_mask_area=float(kwargs.get("min_mask_area", 0.005)),
                grounding=kwargs.get("grounding_model"),
                sam2=kwargs.get("sam2"),
                sam2_video=kwargs.get("sam2_video"),
                box_threshold=float(kwargs.get("box_threshold", 0.25)),
                text_threshold=float(kwargs.get("text_threshold", 0.20)),
            )
            if obj is None:
                return [self._invalid_geometry(item, "no object geometry track from grounding")]
            unit = self._object_result(item, obj, 0)
            unit.scores["triggered"] = 1.0
            unit.scores["trigger_score"] = float(trigger.score)
            if "reprojection_iou" in unit.scores:
                unit.scores["reliability_score"] = float(unit.scores["reprojection_iou"])
            unit.metadata.update({
                "trigger_reason": trigger.reason,
                "trigger_confidence": trigger.confidence,
                "vlm_trigger_metadata": trigger.metadata,
            })
            return [unit]
        except Exception as e:
            return [self._invalid_geometry(item, str(e))]

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        if not units:
            return ItemResult(
                item_id=item.item_id,
                model_id=item.model_id,
                metric=self.name,
                scores={},
                num_units=0,
                valid=False,
                error="no entity object geometry unit result",
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

    def _invalid_geometry(self, item: EvaluationItem, error: str) -> UnitResult:
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=f"{item.item_id}__object_geometry",
            valid=False,
            error=error,
        )

    def _object_result(self, item: EvaluationItem, obj: dict[str, Any], idx: int) -> UnitResult:
        frames = _extract_records(obj)
        scores = {
            "hu_moment_distance": _mean_delta(frames, "hu_moments"),
            "aspect_ratio_std": _std_value(frames, "aspect_ratio"),
            "depth_change_rate": _mean_relative_delta(frames, "mean_depth"),
            "reprojection_iou": finite_mean([
                float(p["reprojection_iou"])
                for p in obj.get("pairs", [])
                if isinstance(p, dict) and isinstance(p.get("reprojection_iou"), (int, float))
            ]),
            "center_reproj_error": finite_mean([
                float(p["center_reproj_error"])
                for p in obj.get("pairs", [])
                if isinstance(p, dict) and isinstance(p.get("center_reproj_error"), (int, float))
            ]),
        }
        clean_scores = {k: float(v) for k, v in scores.items() if v is not None}
        return UnitResult(
            item_id=item.item_id,
            metric=self.name,
            unit_type=self.unit_type,
            unit_id=str(obj.get("object_id") or obj.get("track_id") or f"object_{idx}"),
            scores=clean_scores,
            valid=bool(clean_scores),
        )


# =============================================================================
# Registration Groups
# =============================================================================


A_ENTITY_METRICS = (
    HumanIdentityConsistencyMetric,
    HumanAppearanceConsistencyMetric,
    ObjectTextureConsistencyMetric,
    ObjectGeometryConsistencyMetric,
)

T_ENTITY_HUMAN_METRICS = (
    TEntityHumanIdentityConsistencyMetric,
    TEntityHumanAppearanceConsistencyMetric,
)
T_ENTITY_OBJECT_METRICS = (
    TEntityObjectTextureConsistencyMetric,
    TEntityObjectGeometryConsistencyMetric,
)
T_ENTITY_METRICS = (
    *T_ENTITY_HUMAN_METRICS,
    *T_ENTITY_OBJECT_METRICS,
)
