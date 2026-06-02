# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Video loading utilities for metric plugins.

- Samples BGR frames uniformly from a video file.
- Extracts PIL frames from fixed temporal segments.
- Keeps cv2 video handling reusable across entity and causal metrics.
- Avoids binding metric classes to historical directory layouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def first_video_path(item: Any) -> str:
    if not item.media.videos:
        raise ValueError("item has no video media")
    return item.media.videos[0].path


def sample_bgr_frames(video_path: str, n_frames: int = 32) -> tuple[list[Any], list[int]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"cannot read video frames: {video_path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames, indices.tolist()


def extract_segment_pil_frames(
    video_path: str,
    *,
    segment_idx: int,
    segment_duration: float = 5.0,
    num_samples: int = 8,
) -> tuple[list[Any], float, float]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise ValueError(f"cannot read video for CLIP scoring: {video_path}")

    duration = total / fps
    start_time = segment_idx * segment_duration
    end_time = min(start_time + segment_duration, duration)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    if end_frame <= start_frame:
        cap.release()
        return [], start_time, end_time

    count = end_frame - start_frame
    if count <= num_samples:
        indices = list(range(start_frame, end_frame))
    else:
        indices = np.linspace(start_frame, end_frame - 1, num_samples, dtype=int).tolist()

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames, start_time, end_time
