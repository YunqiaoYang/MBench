# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""MBench-A / MBench-T model-centric directory adapter.

- Reads ``dataset.yaml`` to detect ``dataset_id`` and the ``condition_id``
  pattern (A: ``{action}_{length}``, V: ``text``).
- Iterates each model package (``models/{model_id}/samples.jsonl``) and joins
  in the dataset-level ``samples/{subset}/{sample_id}/sample.json``.
- Resolves all media / artifact paths through a ``SourceLocator`` so the
  adapter works on both local directories and ``hf://datasets/...`` sources.
- Emits ``list[EvaluationItem]`` matching the canonical core schema.
"""

from __future__ import annotations

import json
from typing import Any

from mbench.core.item import EvaluationItem
from mbench.core.registry import DatasetAdapter
from mbench.io import SourceLocator, locator_for


_DATASET_A = "mbencha"
_DATASET_V = "mbencht"

_KNOWN_LENGTHS = {"10s", "25s"}


class MBenchDirAdapter(DatasetAdapter):
    name = "mbench-dir"
    description = (
        "Reads model-centric MBench-A / MBench-T dataset directories. "
        "Source can be a local path or hf://datasets/{repo}/{subpath}."
    )

    def load(
        self,
        source: str,
        models: list[str] | None = None,
        subsets: list[str] | None = None,
        conditions: list[str] | None = None,
        sample_ids: list[str] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[EvaluationItem]:
        """Return all model items in the dataset rooted at ``source``.

        Parameters
        ----------
        source:
            Local directory path, or ``hf://datasets/{repo}/{subpath}``.
        models:
            Optional whitelist of model_ids to load (defaults to all).
        subsets:
            Optional whitelist of subset names, e.g. ``["causal"]``.
        conditions:
            Optional whitelist of full condition_ids or decoded action labels.
            For MBench-A, ``left_then_right`` matches both
            ``left_then_right_10s`` and ``left_then_right_25s``.
        sample_ids:
            Optional whitelist of sample_ids.
        limit:
            If set, stop after this many items in total. Useful for HF
            sources, where each item triggers a small ``sample.json`` fetch
            and you want to probe the full pipeline cheaply first.
        """
        locator = locator_for(source)
        # For HF sources, prefetch all small metadata files in one snapshot
        # download. This turns hundreds of serial RTTs into one parallel fetch.
        if hasattr(locator, "prefetch_metadata"):
            locator.prefetch_metadata()

        dataset_yaml = self._load_dataset_yaml(locator)
        dataset_id = dataset_yaml.get("dataset_id")
        if dataset_id not in (_DATASET_A, _DATASET_V):
            raise ValueError(
                f"Unknown dataset_id in dataset.yaml: {dataset_id!r}; "
                f"expected one of {_DATASET_A!r}, {_DATASET_V!r}"
            )

        model_ids = locator.list_dir("models")
        if models:
            wanted = set(models)
            model_ids = [m for m in model_ids if m in wanted]

        items: list[EvaluationItem] = []
        sample_cache: dict[tuple[str, str], dict[str, Any]] = {}
        subset_filter = set(subsets or [])
        condition_filter = set(conditions or [])
        sample_filter = set(sample_ids or [])

        for model_id in model_ids:
            if limit is not None and len(items) >= limit:
                break
            samples_relpath = f"models/{model_id}/samples.jsonl"
            if not locator.exists(samples_relpath):
                continue
            content = locator.read_text(samples_relpath)
            for lineno, line in enumerate(content.splitlines(), start=1):
                if limit is not None and len(items) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {samples_relpath}:{lineno}: {e}"
                    ) from e

                if not self._row_matches_filters(
                    raw=raw,
                    dataset_id=dataset_id,
                    subsets=subset_filter,
                    conditions=condition_filter,
                    sample_ids=sample_filter,
                ):
                    continue

                canonical = self._row_to_canonical(
                    raw=raw,
                    dataset_id=dataset_id,
                    model_id=model_id,
                    locator=locator,
                    sample_cache=sample_cache,
                )
                items.append(EvaluationItem.from_canonical_dict(canonical))

        return items

    # --- helpers ---

    def _row_matches_filters(
        self,
        *,
        raw: dict[str, Any],
        dataset_id: str,
        subsets: set[str],
        conditions: set[str],
        sample_ids: set[str],
    ) -> bool:
        subset = raw.get("subset")
        sample_id = raw.get("sample_id")
        condition_id = raw.get("condition_id")

        if subsets and subset not in subsets:
            return False
        if sample_ids and sample_id not in sample_ids:
            return False
        if conditions:
            action, _length = self._parse_condition_id(str(condition_id), dataset_id)
            if condition_id not in conditions and action not in conditions:
                return False
        return True

    def _load_dataset_yaml(self, locator: SourceLocator) -> dict[str, Any]:
        if not locator.exists("dataset.yaml"):
            raise FileNotFoundError(
                "dataset.yaml not found at source root; "
                "MBench-A/V dataset directories must contain dataset.yaml at the top level."
            )
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Reading MBench dataset directories requires the 'pyyaml' package. "
                "Install it via: pip install pyyaml"
            ) from e
        return yaml.safe_load(locator.read_text("dataset.yaml")) or {}

    def _row_to_canonical(
        self,
        *,
        raw: dict[str, Any],
        dataset_id: str,
        model_id: str,
        locator: SourceLocator,
        sample_cache: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, Any]:
        if "item_id" not in raw:
            raise ValueError(
                f"samples.jsonl row in models/{model_id}/ missing required 'item_id'"
            )

        # Authoritative model_id is the directory we found this row in.
        if raw.get("model_id") not in (None, model_id):
            raise ValueError(
                f"model_id mismatch in models/{model_id}/samples.jsonl: "
                f"row reports {raw.get('model_id')!r}"
            )

        subset = raw.get("subset")
        sample_id = raw.get("sample_id")
        condition_id = raw.get("condition_id")
        if subset is None or sample_id is None or condition_id is None:
            raise ValueError(
                f"samples.jsonl row in models/{model_id}/ for item {raw['item_id']!r} "
                "is missing one of subset/sample_id/condition_id"
            )

        # Resolve media + artifact paths through the locator.
        media = self._resolve_media(raw.get("media", {}), model_id, locator)
        artifacts = self._resolve_artifacts(raw.get("artifacts", {}), model_id, locator)
        if "da3" not in artifacts:
            da3_relpath = f"artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz"
            if locator.exists(f"models/{model_id}/{da3_relpath}"):
                artifacts["da3"] = {
                    "path": locator.resolve(f"models/{model_id}/{da3_relpath}"),
                    "format": "npz",
                }

        # Build annotations from sample.json + condition_id.
        annotations: dict[str, Any] = dict(raw.get("annotations") or {})
        sample_meta = self._read_sample_meta(
            subset=subset,
            sample_id=sample_id,
            locator=locator,
            cache=sample_cache,
        )
        self._merge_sample_annotations(annotations, sample_meta, dataset_id)

        # condition_id parsing -> overrides annotations.action / sets metadata.length.
        # Per PROMPT.md §7 step 4: a sample can be rendered with multiple
        # condition_ids, so condition-derived action must win over sample.json
        # action (which may only describe one canonical variant).
        metadata: dict[str, Any] = dict(raw.get("metadata") or {})
        action_from_condition, length_from_condition = self._parse_condition_id(
            condition_id, dataset_id
        )
        if action_from_condition:
            annotations["action"] = {
                "type": "camera_motion",
                "label": action_from_condition,
            }
        if length_from_condition and "length" not in metadata:
            metadata["length"] = length_from_condition

        # Stable metadata fields used by aggregators / reporting.
        metadata.setdefault("dataset_id", dataset_id)
        metadata.setdefault("subset", subset)
        metadata.setdefault("sample_id", sample_id)
        metadata.setdefault("condition_id", condition_id)

        # Carry V-only + Object-only fields that don't fit Annotations dataclass.
        if "object_card" in (sample_meta or {}):
            metadata["object_card"] = sample_meta["object_card"]
        if "object_anchor" in (sample_meta or {}):
            metadata["object_anchor"] = sample_meta["object_anchor"]
        # caption_segments: list[str] in PROMPT.md, but core Annotations.prompt_segments
        # expects timed segments. Stash the plain list under metadata.
        plain_segments = (sample_meta or {}).get("caption_segments")
        if plain_segments is not None and "prompt_segments" not in annotations:
            metadata["caption_segments"] = plain_segments

        return {
            "item_id": raw["item_id"],
            "model_id": model_id,
            "media": media,
            "annotations": annotations,
            "artifacts": artifacts,
            "metadata": metadata,
        }

    def _resolve_media(
        self,
        media: dict[str, Any],
        model_id: str,
        locator: SourceLocator,
    ) -> dict[str, Any]:
        out: dict[str, list[dict[str, Any]]] = {}
        for key in ("videos", "images"):
            refs = media.get(key) or []
            resolved: list[dict[str, Any]] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                copy = dict(ref)
                raw_path = copy.get("path")
                if raw_path:
                    copy["path"] = locator.resolve(f"models/{model_id}/{raw_path}")
                resolved.append(copy)
            if resolved:
                out[key] = resolved
        return out

    def _resolve_artifacts(
        self,
        artifacts: dict[str, Any],
        model_id: str,
        locator: SourceLocator,
    ) -> dict[str, Any]:
        out: dict[str, dict[str, Any]] = {}
        for name, ref in (artifacts or {}).items():
            if not isinstance(ref, dict):
                continue
            copy = dict(ref)
            raw_path = copy.get("path")
            if raw_path:
                copy["path"] = locator.resolve(f"models/{model_id}/{raw_path}")
            out[name] = copy
        return out

    def _read_sample_meta(
        self,
        *,
        subset: str,
        sample_id: str,
        locator: SourceLocator,
        cache: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, Any]:
        key = (subset, sample_id)
        if key in cache:
            return cache[key]
        relpath = f"samples/{subset}/{sample_id}/sample.json"
        if not locator.exists(relpath):
            cache[key] = {}
            return cache[key]
        try:
            data = json.loads(locator.read_text(relpath))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid sample.json at {relpath}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"sample.json at {relpath} is not a JSON object")
        cache[key] = data
        return data

    def _merge_sample_annotations(
        self,
        annotations: dict[str, Any],
        sample_meta: dict[str, Any],
        dataset_id: str,
    ) -> None:
        if not sample_meta:
            return
        if "prompt" not in annotations and sample_meta.get("caption"):
            annotations["prompt"] = sample_meta["caption"]
        if "action" not in annotations and sample_meta.get("action"):
            annotations["action"] = sample_meta["action"]

    def _parse_condition_id(
        self,
        condition_id: str,
        dataset_id: str,
    ) -> tuple[str | None, str | None]:
        if dataset_id == _DATASET_V:
            # MBench-T uses a fixed sentinel.
            return None, None

        # MBench-A: {action}_{length}, where length ∈ {10s, 25s}. action may
        # itself contain underscores, e.g. "left_then_right_10s".
        for length in _KNOWN_LENGTHS:
            suffix = f"_{length}"
            if condition_id.endswith(suffix):
                action = condition_id[: -len(suffix)]
                if action:
                    return action, length
        return None, None
