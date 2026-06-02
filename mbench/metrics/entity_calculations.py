# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Entity consistency calculations shared by MBench-A and MBench-T.

- Vision calculations: InsightFace, SAM2, DINOv2, object tracking, DA3 geometry.
- V entity-human scoring helpers: caption segments, trigger, and reliability.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mbench.metrics.metric_math import cosine_similarity, finite_mean


MIN_MASK_RATIO = 0.01
MAX_MASK_RATIO = 0.60
DINO_INPUT_SIZE = 518


# =============================================================================
# Face Tracking
# =============================================================================


@dataclass
class EntityTrack:
    track_id: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    _emb_sum: np.ndarray | None = None
    _emb_count: int = 0

    def add(self, frame_idx: int, embedding: Any, bbox: Any | None = None) -> None:
        emb = np.asarray(embedding, dtype=np.float32)
        record = {"frame_idx": frame_idx, "embedding": emb.tolist()}
        if bbox is not None:
            record["bbox"] = np.asarray(bbox).astype(int).tolist()
        self.detections.append(record)
        if self._emb_sum is None:
            self._emb_sum = emb.astype(np.float64)
            self._emb_count = 1
        else:
            self._emb_sum += emb.astype(np.float64)
            self._emb_count += 1

    @property
    def avg_embedding(self) -> np.ndarray:
        assert self._emb_sum is not None
        return (self._emb_sum / max(self._emb_count, 1)).astype(np.float32)

    @property
    def last_embedding(self) -> np.ndarray:
        return np.asarray(self.detections[-1]["embedding"], dtype=np.float32)

    def match_score(self, embedding: Any) -> float:
        emb = np.asarray(embedding, dtype=np.float32)
        return max(cosine_similarity(self.last_embedding, emb), cosine_similarity(self.avg_embedding, emb))


@lru_cache(maxsize=8)
def init_insightface(device_id: int = 0):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=device_id, det_size=(640, 640))
    return app


def detect_and_track_faces(app: Any, frames: list[Any], match_threshold: float = 0.15) -> list[dict[str, Any]]:
    tracks: list[EntityTrack] = []
    next_track_id = 0
    for frame_idx, frame in enumerate(frames):
        faces = app.get(frame)
        face_infos = [
            {"bbox": face.bbox.astype(int), "embedding": face.embedding}
            for face in faces
        ]
        if not tracks:
            for face in face_infos:
                track = EntityTrack(next_track_id)
                track.add(frame_idx, face["embedding"], face["bbox"])
                tracks.append(track)
                next_track_id += 1
            continue

        if not face_infos:
            continue

        sim_matrix = np.array([
            [track.match_score(face["embedding"]) for face in face_infos]
            for track in tracks
        ])
        matched_tracks: set[int] = set()
        matched_faces: set[int] = set()
        while sim_matrix.size > 0:
            track_idx, face_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            if sim_matrix[track_idx, face_idx] < match_threshold:
                break
            if track_idx not in matched_tracks and face_idx not in matched_faces:
                tracks[track_idx].add(
                    frame_idx,
                    face_infos[face_idx]["embedding"],
                    face_infos[face_idx]["bbox"],
                )
                matched_tracks.add(track_idx)
                matched_faces.add(face_idx)
            sim_matrix[track_idx, :] = -1
            sim_matrix[:, face_idx] = -1

        for face_idx, face in enumerate(face_infos):
            if face_idx not in matched_faces:
                track = EntityTrack(next_track_id)
                track.add(frame_idx, face["embedding"], face["bbox"])
                tracks.append(track)
                next_track_id += 1

    return [
        {"track_id": track.track_id, "detections": track.detections}
        for track in tracks
        if len(track.detections) >= 3
    ]


# =============================================================================
# SAM2 / DINO Feature Extraction
# =============================================================================


@lru_cache(maxsize=8)
def init_sam2(device: str = "cuda"):
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device).eval()
    return processor, model


@lru_cache(maxsize=8)
def init_dinov2(device: str = "cuda"):
    import torchvision.transforms as T
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def run_sam2_point(processor: Any, model: Any, frame_rgb: Any, cx: int, cy: int, device: str):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    inputs = processor(
        images=Image.fromarray(frame_rgb),
        input_points=[[[[cx, cy]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    mask_logits = outputs.pred_masks.squeeze(0).squeeze(0)
    scores = outputs.iou_scores.squeeze(0).squeeze(0)
    best_mask = None
    best_score = -1.0
    for idx in range(mask_logits.shape[0]):
        logit = mask_logits[idx:idx + 1].unsqueeze(0).float()
        mask = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
        mask = (mask.squeeze() > 0).cpu().numpy()
        ratio = float(mask.sum()) / float(h * w)
        if MIN_MASK_RATIO < ratio < MAX_MASK_RATIO and float(scores[idx]) > best_score:
            best_mask = mask
            best_score = float(scores[idx])
    return best_mask


def extract_dino_features(model: Any, transform: Any, frame_rgb: Any, mask: Any, device: str):
    import torch

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = frame_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    return outputs.last_hidden_state[:, 0][0].cpu().numpy().tolist()


# =============================================================================
# Human Appearance Tracks
# =============================================================================


def compute_human_appearance_tracks(
    face_tracks: list[dict[str, Any]],
    frames_bgr: list[Any],
    *,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    import cv2
    import torch

    actual_device = device if torch.cuda.is_available() else "cpu"
    sam_processor, sam_model = init_sam2(actual_device)
    dino_model, dino_transform = init_dinov2(actual_device)
    results = []
    for track in face_tracks:
        records = []
        for det in track["detections"]:
            frame_idx = int(det["frame_idx"])
            if frame_idx >= len(frames_bgr):
                continue
            frame_rgb = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2RGB)
            bbox = det.get("bbox")
            if not bbox:
                continue
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            mask = run_sam2_point(sam_processor, sam_model, frame_rgb, cx, cy, actual_device)
            if mask is None:
                continue
            emb = extract_dino_features(dino_model, dino_transform, frame_rgb, mask, actual_device)
            if emb is not None:
                records.append({"frame_idx": frame_idx, "embedding": emb})
        results.append({"track_id": track["track_id"], "detections": records})
    return results


# =============================================================================
# V Entity Caption-Segment Trigger Scoring
# =============================================================================


def entity_memory_context_from_item(item: Any) -> dict[str, Any]:
    """Extract entity memory caption context without depending on MBench-T modules."""
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
    return {
        "sample_id": str(item.metadata.get("sample_id") or item.item_id),
        "subset": str(item.metadata.get("subset") or "unknown_subset"),
        "caption_segments": tuple(segment for segment in segments if segment),
        "object_card": object_card if isinstance(object_card, dict) else None,
        "metadata": {
            "item_id": item.item_id,
            "model_id": item.model_id,
            "condition_id": item.metadata.get("condition_id"),
            "has_object_card": isinstance(object_card, dict),
        },
    }


def compute_entity_trigger_conditioned_score(
    segment_embeddings: list[list[Any]],
    *,
    absence_similarity_threshold: float = 0.55,
) -> dict[str, float]:
    if len(segment_embeddings) < 2:
        raise ValueError("human trigger scoring requires at least two caption segments")
    initial = segment_embeddings[0]
    final = segment_embeddings[-1]
    middle = segment_embeddings[1:-1]
    initial_presence_score = 1.0 if initial else 0.0
    final_presence_score = 1.0 if final else 0.0
    departure_return_sim = _mean_pairwise_similarity(initial, final)
    middle_absence_score = _middle_target_absence_score(initial, middle, absence_similarity_threshold=absence_similarity_threshold)
    triggered = initial_presence_score > 0.0 and final_presence_score > 0.0 and middle_absence_score >= 1.0
    trigger_score = 1.0 if triggered else 0.0
    reliability_score = _clean_score(departure_return_sim)
    return {
        "triggered": trigger_score,
        "trigger_score": trigger_score,
        "initial_presence_score": initial_presence_score,
        "middle_absence_score": round(middle_absence_score, 6),
        "final_presence_score": final_presence_score,
        "departure_return_sim": round(reliability_score, 6),
        "reliability_score": round(reliability_score, 6),
    }


def sample_bgr_frames_by_caption_segment(video_path: str, *, num_segments: int, num_samples_per_segment: int = 4) -> list[list[Any]]:
    import cv2
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"cannot read video frames: {video_path}")
    segment_frames: list[list[Any]] = []
    for segment_idx in range(num_segments):
        start = int(round(total * segment_idx / num_segments))
        end = max(int(round(total * (segment_idx + 1) / num_segments)), start + 1)
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
                frames.append(frame)
        segment_frames.append(frames)
    cap.release()
    return segment_frames


def write_human_debug_artifacts(*, item: Any, metric_name: str, debug_dir: Path, segment_frames: list[list[Any]], caption_segments: list[str], scores: dict[str, float]) -> dict[str, str]:
    import cv2
    from PIL import Image, ImageDraw
    safe_item_id = _safe_name(item.item_id)
    out_dir = debug_dir / metric_name / safe_item_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "segments.png"
    json_path = out_dir / "scores.json"
    thumbs: list[Image.Image] = []
    for idx, frames in enumerate(segment_frames):
        if frames:
            frame = frames[len(frames) // 2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(rgb).resize((224, 126))
        else:
            thumb = Image.new("RGB", (224, 126), color=(30, 30, 30))
        canvas = Image.new("RGB", (224, 196), color=(245, 245, 245))
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 132), f"seg {idx + 1}", fill=(0, 0, 0))
        draw.text((6, 150), _truncate(caption_segments[idx] if idx < len(caption_segments) else "", 34), fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (224 * len(thumbs), 246), color=(255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 224, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 204), " ".join([
        f"trigger={scores.get('triggered', 0.0):.0f}",
        f"middle_abs={scores.get('middle_absence_score', 0.0):.3f}",
        f"reliability={scores.get('reliability_score', 0.0):.3f}",
    ]), fill=(0, 0, 0))
    sheet.save(image_path)
    json_path.write_text(json.dumps({"item_id": item.item_id, "model_id": item.model_id, "metric": metric_name, "scores": scores, "caption_segments": caption_segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"debug_image": str(image_path), "debug_json": str(json_path)}


def track_mean_embeddings(tracks: list[dict[str, Any]]) -> list[Any]:
    embeddings: list[Any] = []
    for track in tracks:
        records = track.get("detections") or track.get("features") or []
        vectors = [np.asarray(record["embedding"], dtype=np.float64) for record in records if isinstance(record, dict) and "embedding" in record]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0).tolist())
    return embeddings


def _middle_target_absence_score(initial_embeddings: list[Any], middle_embeddings: list[list[Any]], *, absence_similarity_threshold: float) -> float:
    if not initial_embeddings or not middle_embeddings:
        return 0.0
    absent_anywhere = []
    for embeddings in middle_embeddings:
        if not embeddings:
            absent_anywhere.append(1.0)
            continue
        max_sim = _max_pairwise_similarity(initial_embeddings, embeddings)
        absent_anywhere.append(1.0 if max_sim < absence_similarity_threshold else 0.0)
    return float(max(absent_anywhere)) if absent_anywhere else 0.0


def _max_pairwise_similarity(left: list[Any], right: list[Any]) -> float:
    sims = [cosine_similarity(a, b) for a in left for b in right]
    value = finite_mean([max(sims)]) if sims else None
    return float(value) if value is not None else float("nan")


def _mean_pairwise_similarity(left: list[Any], right: list[Any]) -> float:
    sims = [cosine_similarity(a, b) for a in left for b in right]
    value = finite_mean(sims) if sims else None
    return float(value) if value is not None else float("nan")


def _clean_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


# =============================================================================
# MBench-A Object Anchor Tracking (SAM2Video)
# =============================================================================

GT_OBJECT_FRAME_WIDTH = 1280
GT_OBJECT_FRAME_HEIGHT = 720


def object_anchor_from_item(item: Any) -> dict[str, Any]:
    anchor = item.metadata.get("object_anchor")
    if not isinstance(anchor, dict):
        raise ValueError("missing metadata.object_anchor")
    cx = anchor.get("center_x")
    cy = anchor.get("center_y")
    if cx is None or cy is None:
        raise ValueError("object_anchor missing center_x/center_y")
    width = int(anchor.get("width") or GT_OBJECT_FRAME_WIDTH)
    height = int(anchor.get("height") or GT_OBJECT_FRAME_HEIGHT)
    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "width": width,
        "height": height,
    }


def scale_anchor_to_frame(
    center_x: float,
    center_y: float,
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[float, float]:
    return (
        center_x * dst_width / float(src_width),
        center_y * dst_height / float(src_height),
    )


def get_anchor_multipoints(
    processor: Any,
    model: Any,
    frame_rgb: Any,
    center_point: tuple[float, float] | list[float],
    *,
    device: str = "cuda",
    n_points: int = 5,
    erosion_ratio: float = 0.15,
) -> tuple[list[tuple[float, float]], Any]:
    """Frame-0 SAM2 single-image prompt → multipoint seeds for SAM2Video (eval_spatial parity)."""
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    pil_image = Image.fromarray(frame_rgb)
    cx, cy = float(center_point[0]), float(center_point[1])

    inputs = processor(
        images=pil_image,
        input_points=[[[[cx, cy]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    mask_logits = outputs.pred_masks.squeeze(0).squeeze(0)
    scores = outputs.iou_scores.squeeze(0).squeeze(0)
    best_idx = int(scores.argmax().item()) if scores is not None and len(scores) else 0
    logit = mask_logits[best_idx:best_idx + 1].unsqueeze(0).float()
    mask = (
        F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
        .squeeze()
        > 0
    ).cpu().numpy()

    if mask.sum() < 100:
        return [(cx, cy)], mask

    ys, xs = np.where(mask)
    bbox_diag = np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
    erode_px = max(int(bbox_diag * erosion_ratio), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if eroded.sum() < 50:
        eroded = mask

    ey, ex = np.where(eroded)
    candidates = np.stack([ex, ey], axis=1).astype(float)
    centroid = candidates.mean(axis=0)
    dists = np.sqrt(((candidates - centroid) ** 2).sum(axis=1))
    selected = [candidates[dists.argmin()]]
    for _ in range(n_points - 1):
        if len(selected) >= len(candidates):
            break
        sel_arr = np.array(selected)
        min_dists = np.full(len(candidates), np.inf)
        for point in sel_arr:
            d = np.sqrt(((candidates - point) ** 2).sum(axis=1))
            min_dists = np.minimum(min_dists, d)
        selected.append(candidates[int(min_dists.argmax())])
    return [(float(p[0]), float(p[1])) for p in selected], mask


@lru_cache(maxsize=8)
def init_sam2_video(device: str = "cuda"):
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2-hiera-large")
    model = Sam2VideoModel.from_pretrained("facebook/sam2-hiera-large").to(device).eval()
    return processor, model


def segment_frame_grid(
    processor: Any,
    model: Any,
    frame_rgb: Any,
    *,
    device: str,
    grid_size: int = 4,
    nms_iou_thresh: float = 0.5,
    min_area: float = 0.01,
    max_area: float = 0.40,
    max_objects: int = 5,
) -> list[dict[str, Any]]:
    h, w = frame_rgb.shape[:2]
    masks = []
    for y in np.linspace(h * 0.05, h * 0.95, grid_size):
        for x in np.linspace(w * 0.05, w * 0.95, grid_size):
            mask = run_sam2_point(processor, model, frame_rgb, int(x), int(y), device)
            if mask is None:
                continue
            area = float(mask.sum()) / float(h * w)
            if min_area < area < max_area:
                masks.append((mask, area, (int(x), int(y))))

    masks.sort(key=lambda m: m[1], reverse=True)
    kept = []
    for mask, area, point in masks:
        duplicate = False
        for old_mask, _old_area, _old_point in kept:
            union = (mask | old_mask).sum()
            if union > 0 and (mask & old_mask).sum() / union > nms_iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append((mask, area, point))
        if len(kept) >= max_objects:
            break

    return [
        {"object_id": f"object_{idx}", "mask": mask, "area": area, "point": point}
        for idx, (mask, area, point) in enumerate(kept)
    ]


def track_objects_in_video(
    processor: Any,
    model: Any,
    frames_rgb: list[Any],
    anchor_points: list[Any],
    obj_ids: list[int],
    *,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    init_frame_idx: int = 0,
) -> dict[int, list[tuple[int, Any]]]:
    """Track objects with SAM2Video (propagate_in_video_iterator), not per-frame SAM2."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if not frames_rgb:
        return {}
    h, w = frames_rgb[0].shape[:2]
    frame_area = float(h * w)
    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    results: dict[int, list[tuple[int, Any]]] = {}

    for point_set, oid in zip(anchor_points, obj_ids):
        session = processor.init_video_session(
            video=pil_frames,
            inference_device=device,
            max_vision_features_cache_size=len(pil_frames),
        )
        if (
            isinstance(point_set, (list, tuple))
            and len(point_set) > 0
            and isinstance(point_set[0], (list, tuple))
        ):
            pts_list = [[float(p[0]), float(p[1])] for p in point_set]
            input_points = [[pts_list]]
            input_labels = [[[1] * len(pts_list)]]
        else:
            cx, cy = point_set
            input_points = [[[[float(cx), float(cy)]]]]
            input_labels = [[[1]]]

        processor.add_inputs_to_inference_session(
            session,
            frame_idx=init_frame_idx,
            obj_ids=oid,
            input_points=input_points,
            input_labels=input_labels,
        )
        object_records: list[tuple[int, Any]] = []
        with torch.no_grad():
            for output in model.propagate_in_video_iterator(session, start_frame_idx=init_frame_idx):
                if output.pred_masks is None:
                    continue
                logit = output.pred_masks[0:1].float()
                mask = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
                mask = (mask.squeeze() > 0).cpu().numpy()
                if float(mask.sum()) / frame_area > min_mask_area:
                    object_records.append((int(output.frame_idx), mask))
        results[oid] = object_records
    return results


def track_single_object_at_anchor(
    frames_rgb: list[Any],
    *,
    center_x: float,
    center_y: float,
    anchor_width: int = GT_OBJECT_FRAME_WIDTH,
    anchor_height: int = GT_OBJECT_FRAME_HEIGHT,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
) -> list[tuple[int, Any]]:
    """SAM2Video track for one annotated object; frame-0 multipoint init only."""
    import torch

    if not frames_rgb:
        return []
    actual_device = device if torch.cuda.is_available() else "cpu"
    h, w = frames_rgb[0].shape[:2]
    scaled_cx, scaled_cy = scale_anchor_to_frame(
        center_x,
        center_y,
        src_width=anchor_width,
        src_height=anchor_height,
        dst_width=w,
        dst_height=h,
    )

    sam_processor, sam_model = sam2 or init_sam2(actual_device)
    multipoints, _ref_mask = get_anchor_multipoints(
        sam_processor,
        sam_model,
        frames_rgb[0],
        (scaled_cx, scaled_cy),
        device=actual_device,
    )

    video_processor, video_model = sam2_video or init_sam2_video(actual_device)
    tracks = track_objects_in_video(
        video_processor,
        video_model,
        frames_rgb,
        [multipoints],
        [0],
        device=actual_device,
        min_mask_area=min_mask_area,
    )
    return tracks.get(0, [])


def compute_object_texture_track(
    frames_bgr: list[Any],
    *,
    center_x: float,
    center_y: float,
    anchor_width: int = GT_OBJECT_FRAME_WIDTH,
    anchor_height: int = GT_OBJECT_FRAME_HEIGHT,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
    dinov2: tuple[Any, Any] | None = None,
) -> dict[str, Any] | None:
    import cv2
    import torch

    actual_device = device if torch.cuda.is_available() else "cpu"
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr if frame is not None]
    if not frames_rgb:
        return None

    detections = track_single_object_at_anchor(
        frames_rgb,
        center_x=center_x,
        center_y=center_y,
        anchor_width=anchor_width,
        anchor_height=anchor_height,
        device=actual_device,
        min_mask_area=min_mask_area,
        sam2=sam2,
        sam2_video=sam2_video,
    )
    if not detections:
        return None

    dino_model, dino_transform = dinov2 or init_dinov2(actual_device)
    records = []
    for frame_idx, mask in detections:
        emb = extract_dino_features(
            dino_model,
            dino_transform,
            frames_rgb[frame_idx],
            mask,
            actual_device,
        )
        if emb is not None:
            records.append({"frame_idx": frame_idx, "embedding": emb})
    if not records:
        return None
    return {"object_id": "target_object", "detections": records}


# =============================================================================
# Object Geometry Tracks
# =============================================================================


def map_sam_detections_to_global_indices(
    detections: list[tuple[int, Any]],
    frame_indices: list[int],
) -> list[tuple[int, Any]]:
    """Map SAM2Video local frame indices to global video / DA3 depth indices."""
    mapped: list[tuple[int, Any]] = []
    for local_idx, mask in detections:
        if local_idx < 0 or local_idx >= len(frame_indices):
            continue
        mapped.append((int(frame_indices[local_idx]), mask))
    return mapped


def compute_object_geometry_track(
    da3_data: dict[str, Any],
    frames_bgr: list[Any],
    frame_indices: list[int],
    *,
    center_x: float,
    center_y: float,
    anchor_width: int = GT_OBJECT_FRAME_WIDTH,
    anchor_height: int = GT_OBJECT_FRAME_HEIGHT,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
) -> dict[str, Any] | None:
    """Geometry on DA3 depth using the same SAM2Video masks as object texture (sampled frames)."""
    import cv2
    import torch

    actual_device = device if torch.cuda.is_available() else "cpu"
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr if frame is not None]
    if not frames_rgb:
        return None

    detections = track_single_object_at_anchor(
        frames_rgb,
        center_x=center_x,
        center_y=center_y,
        anchor_width=anchor_width,
        anchor_height=anchor_height,
        device=actual_device,
        min_mask_area=min_mask_area,
        sam2=sam2,
        sam2_video=sam2_video,
    )
    if not detections:
        return None

    global_detections = map_sam_detections_to_global_indices(detections, frame_indices)
    return _geometry_track_from_detections(da3_data, global_detections)


def _hu_moments(mask: Any) -> np.ndarray:
    import cv2

    moments = cv2.moments(mask.astype("uint8") * 255)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def _aspect_ratio(mask: Any) -> float:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return float("nan")
    return float(xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)


def _mean_depth(mask: Any, depth: Any) -> float:
    values = depth[mask]
    values = values[values > 1e-4]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _reproject_mask(mask_src, depth_src, k_src, ext_src, k_dst, ext_dst, dst_shape):
    ys, xs = np.where(mask_src)
    if len(ys) == 0:
        return np.zeros(dst_shape, dtype=bool)
    z = depth_src[ys, xs].astype(np.float64)
    valid = z > 1e-4
    if valid.sum() < 10:
        return np.zeros(dst_shape, dtype=bool)
    pts_2d = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(ys))], axis=0)
    pts_cam = np.linalg.inv(k_src.astype(np.float64)) @ pts_2d * z[None, :]
    r_src = ext_src[:3, :3].astype(np.float64)
    t_src = ext_src[:3, 3:4].astype(np.float64)
    r_dst = ext_dst[:3, :3].astype(np.float64)
    t_dst = ext_dst[:3, 3:4].astype(np.float64)
    pts_world = r_src @ pts_cam + t_src
    pts_dst = r_dst.T @ (pts_world - t_dst)
    proj = k_dst.astype(np.float64) @ pts_dst
    valid = valid & (proj[2, :] > 1e-6)
    u = np.zeros(len(ys))
    v = np.zeros(len(ys))
    u[valid] = proj[0, valid] / proj[2, valid]
    v[valid] = proj[1, valid] / proj[2, valid]
    h, w = dst_shape
    valid = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.zeros(dst_shape, dtype=bool)
    out[np.round(v[valid]).astype(int), np.round(u[valid]).astype(int)] = True
    return out


def _mask_iou(first: Any, second: Any) -> float:
    inter = (first & second).sum()
    union = (first | second).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _center_distance(first: Any, second: Any) -> float:
    first_y, first_x = np.where(first)
    second_y, second_x = np.where(second)
    if len(first_y) == 0 or len(second_y) == 0:
        return float("nan")
    diag = float(np.sqrt(first.shape[0] ** 2 + first.shape[1] ** 2))
    return float(np.linalg.norm([
        first_x.mean() - second_x.mean(),
        first_y.mean() - second_y.mean(),
    ]) / max(diag, 1e-8))


# =============================================================================
# MBench-T Object (Grounding DINO + SAM2 point)
# =============================================================================


def object_text_from_card(object_card: dict[str, Any] | None) -> str:
    if not isinstance(object_card, dict):
        return ""
    name = str(object_card.get("object_name") or object_card.get("name") or "").strip()
    attrs = object_card.get("core_attributes")
    attr_text = " ".join(str(attr).strip() for attr in attrs if str(attr).strip()) if isinstance(attrs, list) else ""
    return " ".join(part for part in (attr_text, name) if part).strip()


@lru_cache(maxsize=4)
def _init_grounding_dino(device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model, device


def _post_process_grounding_dino(
    processor: Any,
    model: Any,
    outputs: Any,
    input_ids: Any,
    target_sizes: Any,
    *,
    box_threshold: float,
    text_threshold: float,
) -> dict[str, Any]:
    """Call the grounded-detection post-processor across transformers API variants."""
    target_sizes_arg = target_sizes
    if hasattr(target_sizes, "tolist"):
        target_sizes_arg = target_sizes.tolist()

    kwargs_variants = [
        {
            "outputs": outputs,
            "input_ids": input_ids,
            "threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes_arg,
        },
        {
            "outputs": outputs,
            "input_ids": input_ids,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes_arg,
        },
        {"outputs": outputs, "input_ids": input_ids, "threshold": box_threshold, "target_sizes": target_sizes_arg},
        {"outputs": outputs, "threshold": box_threshold, "target_sizes": target_sizes_arg},
    ]

    candidates: list[Any] = []
    if hasattr(model, "post_process_grounded_object_detection"):
        candidates.append(model.post_process_grounded_object_detection)
    if hasattr(processor, "post_process_grounded_object_detection"):
        candidates.append(processor.post_process_grounded_object_detection)
    if hasattr(processor, "post_process_object_detection"):
        candidates.append(processor.post_process_object_detection)

    last_error: Exception | None = None
    for fn in candidates:
        for kwargs in kwargs_variants:
            try:
                result = fn(**kwargs)
                if isinstance(result, list):
                    return result[0]
                return result
            except TypeError as exc:
                last_error = exc
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("no Grounding DINO post-processing method available")


def ground_object_bbox(
    frame_rgb: Any,
    object_text: str,
    *,
    grounding: Any,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> list[float] | None:
    import torch
    from PIL import Image

    processor, model, device = grounding
    image = Image.fromarray(frame_rgb)
    text = object_text if object_text.endswith(".") else object_text + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    detections = _post_process_grounding_dino(
        processor,
        model,
        outputs,
        inputs.input_ids,
        target_sizes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    boxes = detections.get("boxes")
    scores = detections.get("scores")
    if boxes is None or len(boxes) == 0:
        return None
    best_idx = int(torch.argmax(scores).item()) if scores is not None and len(scores) else 0
    return [float(v) for v in boxes[best_idx].detach().cpu().tolist()]


def sam2_mask_from_bbox(sam_processor: Any, sam_model: Any, frame_rgb: Any, bbox: list[float], device: str):
    x1, y1, x2, y2 = bbox
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    return run_sam2_point(sam_processor, sam_model, frame_rgb, cx, cy, device)


def _first_grounded_bbox(
    frames_rgb: list[Any],
    object_text: str,
    *,
    start_idx: int,
    grounding: Any,
    box_threshold: float,
    text_threshold: float,
) -> tuple[int | None, list[float] | None]:
    for idx in range(start_idx, len(frames_rgb)):
        bbox = ground_object_bbox(
            frames_rgb[idx],
            object_text,
            grounding=grounding,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if bbox is not None:
            return idx, bbox
    return None, None


def _sam2video_propagate_from_bbox(
    frames_rgb: list[Any],
    init_idx: int,
    init_bbox: list[float],
    *,
    device: str,
    min_mask_area: float,
    sam2: tuple[Any, Any],
    sam2_video: tuple[Any, Any],
) -> list[tuple[int, Any]]:
    x1, y1, x2, y2 = init_bbox
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    sam_processor, sam_model = sam2
    multipoints, _ref_mask = get_anchor_multipoints(
        sam_processor,
        sam_model,
        frames_rgb[init_idx],
        center,
        device=device,
    )
    video_processor, video_model = sam2_video
    tracks = track_objects_in_video(
        video_processor,
        video_model,
        frames_rgb,
        [multipoints],
        [0],
        device=device,
        min_mask_area=min_mask_area,
        init_frame_idx=init_idx,
    )
    return tracks.get(0, [])


def track_v_object_grounded_sam2video(
    frames_rgb: list[Any],
    object_text: str,
    *,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    grounding: Any | None = None,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    reinit_box_threshold: float = 0.40,
    reinit_text_threshold: float | None = None,
    enable_reinit: bool = False,
) -> list[tuple[int, Any]]:
    """Grounding DINO + SAM2Video for V object (texture / geometry).

    First hit uses ``box_threshold`` (looser). Optional tail re-init: after the last
    masked frame only, run stricter DINO + SAM2Video; skip masks covering >35% of the
    frame (typical false positives when the object is absent).
    """
    import torch

    if not frames_rgb or not object_text.strip():
        return []
    actual_device = device if torch.cuda.is_available() else "cpu"
    grounding_model = grounding or _init_grounding_dino(actual_device)
    reinit_text = text_threshold if reinit_text_threshold is None else reinit_text_threshold

    init_idx, init_bbox = _first_grounded_bbox(
        frames_rgb,
        object_text,
        start_idx=0,
        grounding=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if init_idx is None or init_bbox is None:
        return []

    sam_processor, sam_model = sam2 or init_sam2(actual_device)
    video_processor, video_model = sam2_video or init_sam2_video(actual_device)
    sam_bundle = (sam_processor, sam_model)
    video_bundle = (video_processor, video_model)

    merged: dict[int, Any] = {}

    def _mask_area_ratio(mask: Any) -> float:
        return float(mask.sum()) / float(mask.shape[0] * mask.shape[1])

    def _merge_segment(
        init_i: int,
        bbox: list[float],
        *,
        min_local_idx: int | None = None,
        max_mask_area: float | None = None,
    ) -> int:
        segment = _sam2video_propagate_from_bbox(
            frames_rgb,
            init_i,
            bbox,
            device=actual_device,
            min_mask_area=min_mask_area,
            sam2=sam_bundle,
            sam2_video=video_bundle,
        )
        added = 0
        for local_idx, mask in segment:
            li = int(local_idx)
            if min_local_idx is not None and li < min_local_idx:
                continue
            if max_mask_area is not None and _mask_area_ratio(mask) > max_mask_area:
                continue
            if li not in merged:
                merged[li] = mask
                added += 1
        return added

    _merge_segment(init_idx, init_bbox)

    # Re-init only on the tail after the last masked frame. Never fill internal gaps
    # (object absent in the middle); filling gaps bridges DR segments and causes DINO
    # false positives where nothing is visible.
    if enable_reinit and merged:
        min_trailing_gap = 2
        while True:
            last_masked = max(merged)
            tail_start = last_masked + 1
            if tail_start >= len(frames_rgb):
                break
            if (len(frames_rgb) - tail_start) < min_trailing_gap:
                break
            hit_idx, hit_bbox = _first_grounded_bbox(
                frames_rgb,
                object_text,
                start_idx=tail_start,
                grounding=grounding_model,
                box_threshold=reinit_box_threshold,
                text_threshold=reinit_text,
            )
            if hit_idx is None or hit_bbox is None:
                break
            if _merge_segment(
                hit_idx,
                hit_bbox,
                min_local_idx=hit_idx,
                max_mask_area=0.35,
            ) == 0:
                break

    return sorted(merged.items(), key=lambda item: item[0])


def _geometry_track_from_detections(
    da3_data: dict[str, Any],
    detections: list[tuple[int, Any]],
) -> dict[str, Any] | None:
    """Build geometry frame/pair records from a SAM2Video mask track on DA3 frames."""
    import cv2

    if len(detections) < 2:
        return None

    frame_records = []
    for frame_idx, mask in detections:
        if frame_idx >= len(da3_data["depth"]):
            continue
        depth = da3_data["depth"][frame_idx]
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask.astype("uint8"),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        frame_records.append({
            "frame_idx": frame_idx,
            "hu_moments": _hu_moments(mask).tolist(),
            "aspect_ratio": _aspect_ratio(mask),
            "mean_depth": _mean_depth(mask, depth),
        })

    mask_by_frame = {frame_idx: mask for frame_idx, mask in detections}
    pair_records = []
    sorted_frames = sorted(mask_by_frame)
    for left, right in zip(sorted_frames, sorted_frames[1:]):
        if right >= len(da3_data["depth"]):
            continue
        src_mask = mask_by_frame[left]
        dst_mask = mask_by_frame[right]
        depth_src = da3_data["depth"][left]
        if src_mask.shape != depth_src.shape:
            src_mask = cv2.resize(
                src_mask.astype("uint8"),
                (depth_src.shape[1], depth_src.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if dst_mask.shape != da3_data["depth"][right].shape:
            dst_mask = cv2.resize(
                dst_mask.astype("uint8"),
                (da3_data["depth"][right].shape[1], da3_data["depth"][right].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        projected = _reproject_mask(
            src_mask,
            depth_src,
            da3_data["intrinsics"][left],
            da3_data["extrinsics"][left],
            da3_data["intrinsics"][right],
            da3_data["extrinsics"][right],
            dst_mask.shape,
        )
        pair_records.append({
            "frame_i": left,
            "frame_j": right,
            "reprojection_iou": _mask_iou(projected, dst_mask),
            "center_reproj_error": _center_distance(projected, dst_mask),
        })

    if not frame_records:
        return None
    return {
        "object_id": "target_object",
        "frames": frame_records,
        "pairs": pair_records,
    }


def compute_v_object_texture_track(
    frames_bgr: list[Any],
    object_text: str,
    *,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    grounding: Any | None = None,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
    dinov2: tuple[Any, Any] | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> dict[str, Any] | None:
    import cv2
    import torch

    actual_device = device if torch.cuda.is_available() else "cpu"
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr if frame is not None]
    if not frames_rgb:
        return None

    detections = track_v_object_grounded_sam2video(
        frames_rgb,
        object_text,
        device=actual_device,
        min_mask_area=min_mask_area,
        grounding=grounding,
        sam2=sam2,
        sam2_video=sam2_video,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not detections:
        return None

    dino_model, dino_transform = dinov2 or init_dinov2(actual_device)
    records = []
    for frame_idx, mask in detections:
        emb = extract_dino_features(
            dino_model,
            dino_transform,
            frames_rgb[frame_idx],
            mask,
            actual_device,
        )
        if emb is not None:
            records.append({"frame_idx": frame_idx, "embedding": emb})
    if not records:
        return None
    return {"object_id": "target_object", "detections": records}


def compute_v_object_geometry_track(
    da3_data: dict[str, Any],
    frames_bgr: list[Any],
    frame_indices: list[int],
    object_text: str,
    *,
    device: str = "cuda",
    min_mask_area: float = 0.005,
    grounding: Any | None = None,
    sam2: tuple[Any, Any] | None = None,
    sam2_video: tuple[Any, Any] | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> dict[str, Any] | None:
    """Geometry on DA3 depth using the same Grounding DINO + SAM2Video masks as V object texture."""
    import cv2

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr if frame is not None]
    if not frames_rgb:
        return None

    detections = track_v_object_grounded_sam2video(
        frames_rgb,
        object_text,
        device=device,
        min_mask_area=min_mask_area,
        grounding=grounding,
        sam2=sam2,
        sam2_video=sam2_video,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not detections:
        return None
    global_detections = map_sam_detections_to_global_indices(detections, frame_indices)
    return _geometry_track_from_detections(da3_data, global_detections)


def write_object_debug_artifacts(
    *,
    item: Any,
    metric_name: str,
    debug_dir: Path,
    segment_frames: list[list[Any]],
    caption_segments: list[str],
    evidence: list[list[dict[str, Any]]],
    scores: dict[str, float],
) -> dict[str, str]:
    import cv2
    from PIL import Image, ImageDraw

    safe_item_id = _object_debug_safe_name(item.item_id)
    out_dir = debug_dir / metric_name / safe_item_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "segments.png"
    json_path = out_dir / "scores.json"
    thumbs: list[Image.Image] = []
    for idx, frames in enumerate(segment_frames):
        if frames:
            frame = frames[len(frames) // 2].copy()
            records = evidence[idx] if idx < len(evidence) else []
            if records:
                record = records[0]
                bbox = record.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                mask = record.get("mask")
                if mask is not None:
                    overlay = frame.copy()
                    overlay[np.asarray(mask, dtype=bool)] = (0, 160, 255)
                    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(rgb).resize((224, 126))
        else:
            thumb = Image.new("RGB", (224, 126), color=(30, 30, 30))
        canvas = Image.new("RGB", (224, 198), color=(245, 245, 245))
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 132), f"seg {idx + 1} objects={len(evidence[idx]) if idx < len(evidence) else 0}", fill=(0, 0, 0))
        draw.text((6, 152), _object_debug_truncate(caption_segments[idx] if idx < len(caption_segments) else "", 34), fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (224 * len(thumbs), 248), color=(255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 224, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (6, 206),
        " ".join([
            f"dr={scores.get('departure_return_sim', 0.0):.3f}",
            f"first={scores.get('first_seg_internal_sim', 0.0):.3f}",
            f"last={scores.get('last_seg_internal_sim', 0.0):.3f}",
        ]),
        fill=(0, 0, 0),
    )
    sheet.save(image_path)
    serializable_evidence = [
        [{key: value for key, value in record.items() if key not in {"embedding", "mask"}} for record in segment]
        for segment in evidence
    ]
    json_path.write_text(
        json.dumps(
            {
                "item_id": item.item_id,
                "model_id": item.model_id,
                "metric": metric_name,
                "scores": scores,
                "caption_segments": caption_segments,
                "evidence": serializable_evidence,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"debug_image": str(image_path), "debug_json": str(json_path)}


def _object_debug_safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _object_debug_truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 3] + "..."

