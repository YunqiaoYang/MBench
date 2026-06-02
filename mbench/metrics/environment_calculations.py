# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Environment geometry and rendering calculations.

- Migrates epipolar and reprojection formulas from env_consistency.
- Loads DA3 depth, intrinsics, extrinsics, and extracted frames.
- Computes fold-pair frame matches for round-trip camera motions.
- Provides style metrics using the existing memory-consistency formulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

ImageSimilarityFn = Callable[[Any, Any], float]


def load_da3_data(path: str) -> dict[str, Any]:
    """Load DA3 output from either a video artifact directory or results.npz path."""
    root = Path(path)
    if root.is_file():
        npz_path = root
        video_dir = root.parents[2] if root.parent.name == "mini_npz" else root.parent
    else:
        video_dir = root
        npz_path = None
        for candidate in (
            video_dir / "results.npz",
            video_dir / "exports" / "mini_npz" / "results.npz",
        ):
            if candidate.is_file():
                npz_path = candidate
                break
        if npz_path is None:
            npz_path = video_dir / "results.npz"
    img_dir = video_dir / "input_images"

    if not npz_path.is_file():
        raise FileNotFoundError(f"DA3 results not found under {video_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"DA3 input_images directory not found: {img_dir}")

    data = np.load(str(npz_path))
    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No extracted DA3 frames found under: {img_dir}")

    depth = data["depth"]
    return {
        "extrinsics": data["extrinsics"],
        "intrinsics": data["intrinsics"],
        "depth": depth,
        "image_paths": image_paths,
        "depth_shape": (depth.shape[1], depth.shape[2]),
    }


def read_image_at_depth_res(image_path: Path, depth_shape: tuple[int, int]) -> Any:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not readable: {image_path}")
    h_d, w_d = depth_shape
    if img.shape[0] != h_d or img.shape[1] != w_d:
        img = cv2.resize(img, (w_d, h_d), interpolation=cv2.INTER_LINEAR)
    return img


def rotation_distance_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    r_diff = r1 @ r2.T
    trace = np.clip(np.trace(r_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def find_fold_pairs(extrinsics: np.ndarray, num_pairs: int = 20) -> list[dict[str, Any]]:
    """Match outbound frames to closest-pose inbound frames."""
    turn = _find_turnaround_frame(extrinsics)
    outbound = list(range(0, turn))
    inbound = list(range(turn, len(extrinsics)))
    if len(outbound) < 5 or len(inbound) < 5:
        return _find_fixed_pairs(len(extrinsics), num_pairs)

    pairs: list[dict[str, Any]] = []
    for i in outbound:
        best_j: int | None = None
        best_combined = float("inf")
        best_t, best_r = 0.0, 0.0
        for j in inbound:
            t_dist = float(np.linalg.norm(extrinsics[i, :, 3] - extrinsics[j, :, 3]))
            r_dist = rotation_distance_deg(extrinsics[i, :, :3], extrinsics[j, :, :3])
            combined = t_dist + r_dist / 180.0
            if combined < best_combined:
                best_j, best_combined = j, combined
                best_t, best_r = t_dist, r_dist
        if best_j is not None:
            pairs.append({
                "frame_out": i,
                "frame_in": best_j,
                "temporal_span": abs(best_j - i),
                "translation_dist": round(best_t, 5),
                "rotation_dist_deg": round(best_r, 2),
                "pair_mode": "pose",
            })

    pairs.sort(key=lambda p: -int(p["temporal_span"]))
    if len(pairs) > num_pairs:
        indices = np.linspace(0, len(pairs) - 1, num_pairs, dtype=int)
        pairs = [pairs[int(i)] for i in indices]
    return pairs


def _find_turnaround_frame(extrinsics: np.ndarray) -> int:
    t0 = extrinsics[0, :, 3]
    r0 = extrinsics[0, :, :3]
    combined = []
    for ext in extrinsics:
        t_dist = np.linalg.norm(ext[:, 3] - t0)
        r_dist = rotation_distance_deg(ext[:, :3], r0)
        combined.append(t_dist + r_dist / 180.0)
    return int(np.argmax(np.asarray(combined)))


def _find_fixed_pairs(n_frames: int, num_pairs: int) -> list[dict[str, Any]]:
    turn = n_frames // 2
    pairs = []
    for i in range(turn):
        j = n_frames - 1 - i
        if j > i:
            pairs.append({
                "frame_out": i,
                "frame_in": j,
                "temporal_span": j - i,
                "translation_dist": None,
                "rotation_dist_deg": None,
                "pair_mode": "fixed",
            })
    pairs.sort(key=lambda p: -int(p["temporal_span"]))
    if len(pairs) > num_pairs:
        indices = np.linspace(0, len(pairs) - 1, num_pairs, dtype=int)
        pairs = [pairs[int(i)] for i in indices]
    return pairs


def compute_fundamental_from_poses(k_i, ext_i, k_j, ext_j):
    r_i = ext_i[:3, :3].astype(np.float64)
    t_i = ext_i[:3, 3].astype(np.float64)
    r_j = ext_j[:3, :3].astype(np.float64)
    t_j = ext_j[:3, 3].astype(np.float64)
    r_rel = r_j.T @ r_i
    t_rel = r_j.T @ (t_i - t_j)
    e = _skew(t_rel) @ r_rel
    f = np.linalg.inv(k_j.astype(np.float64)).T @ e @ np.linalg.inv(k_i.astype(np.float64))
    norm = np.linalg.norm(f)
    if norm > 1e-12:
        f = f / norm
    return f


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array(
        [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]],
        dtype=np.float64,
    )


def extract_and_match_sift(gray_i, gray_j, ratio_thresh: float = 0.75):
    import cv2

    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(gray_i, None)
    kp2, des2 = sift.detectAndCompute(gray_j, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([]), np.array([])

    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good) < 5:
        return np.array([]), np.array([])
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def compute_epipolar_error(img_i, img_j, k_i, ext_i, k_j, ext_j) -> dict[str, float]:
    import cv2

    gray_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    gray_j = cv2.cvtColor(img_j, cv2.COLOR_BGR2GRAY)
    pts1, pts2 = extract_and_match_sift(gray_i, gray_j)
    if len(pts1) < 5:
        return {"num_matches": 0}

    f = compute_fundamental_from_poses(k_i, ext_i, k_j, ext_j)
    distances = _symmetric_epipolar_distance(pts1, pts2, f)
    return {
        "median_epi_dist": round(float(np.median(distances)), 4),
        "mean_epi_dist": round(float(np.mean(distances)), 4),
        "std_epi_dist": round(float(np.std(distances)), 4),
        "num_matches": float(len(pts1)),
    }


def _symmetric_epipolar_distance(pts1, pts2, f):
    n = len(pts1)
    ones = np.ones((n, 1))
    p1_h = np.hstack([pts1, ones])
    p2_h = np.hstack([pts2, ones])
    fp1 = (f @ p1_h.T).T
    ftp2 = (f.T @ p2_h.T).T
    num = np.abs(np.sum(p2_h * fp1, axis=1))
    d1 = num / np.sqrt(fp1[:, 0] ** 2 + fp1[:, 1] ** 2 + 1e-12)
    d2 = num / np.sqrt(ftp2[:, 0] ** 2 + ftp2[:, 1] ** 2 + 1e-12)
    return 0.5 * (d1 + d2)


def compute_reprojection_error(img_i, img_j, depth_i, depth_j, k_i, ext_i, k_j, ext_j):
    import cv2

    gray_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    gray_j = cv2.cvtColor(img_j, cv2.COLOR_BGR2GRAY)
    pts1, pts2 = extract_and_match_sift(gray_i, gray_j)
    if len(pts1) < 5:
        return {"num_matches": 0}

    p2_proj, valid_fwd = _reproject_points(pts1, depth_i, k_i, ext_i, k_j, ext_j)
    p1_proj, valid_bwd = _reproject_points(pts2, depth_j, k_j, ext_j, k_i, ext_i)
    fwd_errors = np.linalg.norm(pts2 - p2_proj, axis=1)[valid_fwd]
    bwd_errors = np.linalg.norm(pts1 - p1_proj, axis=1)[valid_bwd]
    all_errors = np.concatenate([fwd_errors, bwd_errors])
    if len(all_errors) < 3:
        return {
            "num_matches": float(len(pts1)),
            "num_valid_fwd": float(valid_fwd.sum()),
            "num_valid_bwd": float(valid_bwd.sum()),
        }

    return {
        "median_reproj": round(float(np.median(all_errors)), 4),
        "mean_reproj": round(float(np.mean(all_errors)), 4),
        "std_reproj": round(float(np.std(all_errors)), 4),
        "num_matches": float(len(pts1)),
        "num_valid_fwd": float(valid_fwd.sum()),
        "num_valid_bwd": float(valid_bwd.sum()),
    }


def _reproject_points(pts_src, depth_src, k_src, ext_src, k_dst, ext_dst):
    n = len(pts_src)
    h, w = depth_src.shape
    u = np.clip(np.round(pts_src[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(pts_src[:, 1]).astype(int), 0, h - 1)
    z = depth_src[v, u].astype(np.float64)
    valid = z > 1e-4
    pixels = np.stack([pts_src[:, 0], pts_src[:, 1], np.ones(n)], axis=0)
    pts_cam = np.linalg.inv(k_src.astype(np.float64)) @ pixels * z[None, :]
    r_src = ext_src[:3, :3].astype(np.float64)
    t_src = ext_src[:3, 3].astype(np.float64)
    r_dst = ext_dst[:3, :3].astype(np.float64)
    t_dst = ext_dst[:3, 3].astype(np.float64)
    pts_world = r_src @ pts_cam + t_src[:, None]
    pts_cam_dst = r_dst.T @ (pts_world - t_dst[:, None])
    pts_proj_h = k_dst.astype(np.float64) @ pts_cam_dst
    z_proj = pts_proj_h[2, :]
    valid = valid & (z_proj > 1e-6)
    pts_proj = np.zeros((n, 2))
    pts_proj[valid, 0] = pts_proj_h[0, valid] / z_proj[valid]
    pts_proj[valid, 1] = pts_proj_h[1, valid] / z_proj[valid]
    valid = valid & (pts_proj[:, 0] >= 0) & (pts_proj[:, 0] < w) & \
        (pts_proj[:, 1] >= 0) & (pts_proj[:, 1] < h)
    return pts_proj, valid


#========================#
# MBench-T environment calculations
#========================#


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance_to_score(value: Any, *, tau: float) -> float:
    try:
        dist = float(value)
    except (TypeError, ValueError):
        return 0.0
    return clamp01(float(np.exp(-dist / max(tau, 1e-8))))


def pose_return_pair_score(
    translation_dist: float,
    rotation_dist_deg: float,
    *,
    tau_translation: float = 0.1,
    tau_rotation_deg: float = 10.0,
) -> float:
    return clamp01(float(np.exp(
        -float(translation_dist) / max(tau_translation, 1e-8)
        -float(rotation_dist_deg) / max(tau_rotation_deg, 1e-8)
    )))


def segment_frame_indices(
    n_frames: int,
    num_segments: int,
    segment_ids: list[int],
    *,
    samples_per_segment: int = 1,
) -> list[int]:
    out: list[int] = []
    for seg_idx in sorted(set(segment_ids)):
        start = int(round(n_frames * seg_idx / num_segments))
        end = max(int(round(n_frames * (seg_idx + 1) / num_segments)), start + 1)
        if samples_per_segment <= 1 or end - start <= 1:
            indices = [(start + end - 1) // 2]
        else:
            indices = np.linspace(start, end - 1, min(samples_per_segment, end - start), dtype=int).tolist()
        out.extend(min(n_frames - 1, max(0, int(idx))) for idx in indices)
    return out


def hist_image_similarity(img_a: Any, img_b: Any) -> float:
    import cv2

    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return clamp01(float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))


def find_caption_out_in_pairs(
    data: dict[str, Any],
    *,
    num_segments: int,
    num_pairs: int = 5,
    image_similarity_fn: ImageSimilarityFn | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    """Select early-late frame pairs by DA3 pose return score (histogram is diagnostic only)."""
    n_frames = len(data["image_paths"])
    if n_frames < 2 or num_segments < 2:
        return []
    sim_fn = image_similarity_fn or hist_image_similarity
    early = segment_frame_indices(
        n_frames, num_segments, [0, min(1, num_segments - 1)], samples_per_segment=3,
    )
    middle = segment_frame_indices(n_frames, num_segments, [num_segments // 2], samples_per_segment=3)
    late = segment_frame_indices(
        n_frames,
        num_segments,
        [max(0, num_segments - 2), num_segments - 1],
        samples_per_segment=3,
    )
    if not early or not late:
        return []
    scored_pairs: list[dict[str, Any]] = []
    for i in early:
        img_i = read_image_at_depth_res(data["image_paths"][i], data["depth_shape"])
        best_middle = max(
            (
                sim_fn(img_i, read_image_at_depth_res(data["image_paths"][m], data["depth_shape"]))
                for m in middle
            ),
            default=0.0,
        )
        for j in late:
            if j <= i:
                continue
            return_similarity = sim_fn(
                img_i, read_image_at_depth_res(data["image_paths"][j], data["depth_shape"]),
            )
            translation_dist = float(np.linalg.norm(data["extrinsics"][i, :, 3] - data["extrinsics"][j, :, 3]))
            rotation_dist = rotation_distance_deg(data["extrinsics"][i, :, :3], data["extrinsics"][j, :, :3])
            pose_score = pose_return_pair_score(translation_dist, rotation_dist)
            scored_pairs.append({
                "frame_out": i,
                "frame_in": j,
                "temporal_span": j - i,
                "translation_dist": round(translation_dist, 5),
                "rotation_dist_deg": round(rotation_dist, 2),
                "pose_return_score": round(pose_score, 6),
                "return_similarity": round(float(return_similarity), 6),
                "middle_similarity": round(float(best_middle), 6),
                "pair_mode": "caption_pose_return",
            })
    scored_pairs.sort(
        key=lambda p: (
            float(p["pose_return_score"]),
            int(p["temporal_span"]),
        ),
        reverse=True,
    )
    return scored_pairs[:num_pairs]


def find_global_pose_pairs(
    data: dict[str, Any],
    *,
    num_pairs: int = 5,
    min_gap_frac: float = 0.2,
    max_trans: float = 0.5,
    max_rot_deg: float = 30.0,
    **_: Any,
) -> list[dict[str, Any]]:
    """Caption-free global pose-filtered pair search.

    Selection rule (used by MBench-T spatial reprojection — text-conditioned
    rollouts do not have caption-aligned departure / return segments):

    - departure pool = first half of frames
    - return pool    = second half of frames, with at least ``min_gap_frac``
      of the video length between i and j
    - keep candidates with ``translation_dist < max_trans`` and
      ``rotation_dist_deg < max_rot_deg``
    - rank by ``pose_return_pair_score`` (then by temporal span, longer first)
    - return top ``num_pairs``

    Image / SIFT computation is deferred to the metric body — this function
    only does pose pre-filtering.
    """
    extr = data["extrinsics"]
    n = len(extr)
    if n < 10:
        return []

    half = n // 2
    min_gap = int(n * min_gap_frac)

    candidates: list[dict[str, Any]] = []
    for i in range(half):
        ti = extr[i, :, 3]
        ri = extr[i, :, :3]
        start_j = max(i + min_gap, half)
        if start_j >= n:
            continue
        td_vec = np.linalg.norm(extr[start_j:, :, 3] - ti, axis=1)
        good = np.where(td_vec < max_trans)[0]
        for offset in good:
            j = start_j + int(offset)
            rd = rotation_distance_deg(ri, extr[j, :, :3])
            if rd >= max_rot_deg:
                continue
            ps = pose_return_pair_score(float(td_vec[offset]), rd)
            candidates.append({
                "frame_out": i,
                "frame_in": j,
                "temporal_span": j - i,
                "translation_dist": round(float(td_vec[offset]), 5),
                "rotation_dist_deg": round(rd, 2),
                "pose_return_score": round(ps, 6),
                "return_similarity": 0.0,
                "middle_similarity": 0.0,
                "pair_mode": "global_pose_return",
            })

    candidates.sort(
        key=lambda p: (
            float(p["pose_return_score"]),
            int(p["temporal_span"]),
        ),
        reverse=True,
    )
    return candidates[:num_pairs]
