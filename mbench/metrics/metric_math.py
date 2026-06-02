# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Numeric helpers shared by metric implementations.

- Computes cosine similarity for embedding-based consistency metrics.
- Groups visible frame detections into continuous segments.
- Provides NaN-safe mean helpers for metric aggregation.
- Keeps small math utilities out of individual metric classes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean used for trigger-conditioned M-scores."""

    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def finite_mean(values: list[float]) -> float | None:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    if not valid:
        return None
    return float(np.mean(valid))


def cosine_similarity(a: Any, b: Any) -> float:
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(va, vb) / denom)


def visible_segments(
    records: list[dict[str, Any]],
    *,
    frame_key: str = "frame_idx",
    max_gap: int = 1,
) -> list[list[dict[str, Any]]]:
    """Split sorted frame records into contiguous visible segments."""
    if not records:
        return []

    sorted_records = sorted(records, key=lambda r: int(r[frame_key]))
    segments: list[list[dict[str, Any]]] = [[sorted_records[0]]]
    for record in sorted_records[1:]:
        prev = segments[-1][-1]
        if int(record[frame_key]) - int(prev[frame_key]) <= max_gap:
            segments[-1].append(record)
        else:
            segments.append([record])
    return segments


def departure_return_similarity(
    records: list[dict[str, Any]],
    *,
    embedding_key: str = "embedding",
    frame_key: str = "frame_idx",
) -> dict[str, float | int | bool | None]:
    """Compute departure-return similarity from per-frame embeddings."""
    segments = visible_segments(records, frame_key=frame_key)
    if len(segments) < 2:
        return {
            "has_departure_return": False,
            "departure_return_sim": None,
            "first_seg_internal_sim": _segment_internal_similarity(segments[0], embedding_key)
            if segments else None,
            "last_seg_internal_sim": None,
            "gap_frames": 0,
            "n_segments": len(segments),
        }

    first = segments[0]
    last = segments[-1]
    cross = [
        cosine_similarity(a[embedding_key], b[embedding_key])
        for a in first
        for b in last
        if embedding_key in a and embedding_key in b
    ]
    return {
        "has_departure_return": True,
        "departure_return_sim": finite_mean(cross),
        "first_seg_internal_sim": _segment_internal_similarity(first, embedding_key),
        "last_seg_internal_sim": _segment_internal_similarity(last, embedding_key),
        "gap_frames": int(last[0][frame_key]) - int(first[-1][frame_key]) - 1,
        "n_segments": len(segments),
    }


def _segment_internal_similarity(
    segment: list[dict[str, Any]],
    embedding_key: str,
) -> float | None:
    if len(segment) < 2:
        return None
    sims: list[float] = []
    for i, first in enumerate(segment):
        for second in segment[i + 1:]:
            if embedding_key in first and embedding_key in second:
                sims.append(cosine_similarity(first[embedding_key], second[embedding_key]))
    return finite_mean(sims)
