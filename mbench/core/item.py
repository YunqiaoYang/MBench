# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Core schema for normalized evaluation samples.

- Defines EvaluationItem as the stable item envelope used by the pipeline.
- Separates media, annotations, artifacts, and metadata by responsibility.
- Provides field-path lookup for InputContract validation.
- Converts only canonical core dictionaries; adapter-specific parsing stays outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoRef:
    path: str
    role: str = "generated"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRef:
    path: str
    role: str = "frame"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSegment:
    start: float
    end: float
    text: str


@dataclass
class ActionSpec:
    type: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRef:
    name: str
    path: str | None = None
    format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        if self.path is None:
            return False
        return Path(self.path).exists()


@dataclass
class MaskRef:
    path: str
    object_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Media:
    videos: list[VideoRef] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)


@dataclass
class Annotations:
    prompt: str | None = None
    prompt_segments: list[PromptSegment] | None = None
    action: ActionSpec | None = None
    masks: list[MaskRef] | None = None


@dataclass
class EvaluationItem:
    item_id: str
    media: Media = field(default_factory=Media)
    model_id: str | None = None
    annotations: Annotations = field(default_factory=Annotations)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, path: str) -> Any:
        """Resolve a dot-separated field path like 'media.videos' or 'artifacts.da3'."""
        parts = path.split(".")
        obj: Any = self
        for part in parts:
            if isinstance(obj, dict):
                if part not in obj:
                    return None
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def has_field(self, path: str) -> bool:
        val = self.get_field(path)
        if val is None:
            return False
        if isinstance(val, (list, dict)) and len(val) == 0:
            return False
        return True

    @classmethod
    def from_canonical_dict(cls, data: dict[str, Any]) -> EvaluationItem:
        """Build an item from the normalized core schema.

        Dataset adapters own external-format parsing. This constructor only
        accepts canonical media / annotations / artifacts / metadata fields.
        """
        media_raw = data.get("media", {})
        media = Media(
            videos=[
                VideoRef(**v) if isinstance(v, dict) else VideoRef(path=v)
                for v in media_raw.get("videos", [])
            ],
            images=[
                ImageRef(**v) if isinstance(v, dict) else ImageRef(path=v)
                for v in media_raw.get("images", [])
            ],
        )

        ann_raw = data.get("annotations", {})
        prompt_segs = None
        if "prompt_segments" in ann_raw and ann_raw["prompt_segments"] is not None:
            prompt_segs = [PromptSegment(**s) for s in ann_raw["prompt_segments"]]
        action = None
        if "action" in ann_raw and ann_raw["action"] is not None:
            action = ActionSpec(**ann_raw["action"])
        masks = None
        if "masks" in ann_raw and ann_raw["masks"] is not None:
            masks = [MaskRef(**m) for m in ann_raw["masks"]]
        annotations = Annotations(
            prompt=ann_raw.get("prompt"),
            prompt_segments=prompt_segs,
            action=action,
            masks=masks,
        )

        artifacts: dict[str, ArtifactRef] = {}
        for name, ref in data.get("artifacts", {}).items():
            if isinstance(ref, dict):
                artifacts[name] = ArtifactRef(name=name, **ref)
            elif isinstance(ref, str):
                artifacts[name] = ArtifactRef(name=name, path=ref)
            else:
                artifacts[name] = ArtifactRef(name=name)

        return cls(
            item_id=data["item_id"],
            model_id=data.get("model_id"),
            media=media,
            annotations=annotations,
            artifacts=artifacts,
            metadata=data.get("metadata", {}),
        )
