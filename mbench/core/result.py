# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Result schemas and JSON output writer.

- Defines UnitResult, ItemResult, and SummaryResult as the standard output model.
- Keeps metric scores in plain dictionaries for plugin flexibility.
- Writes JSONL/JSON files under the stable run directory layout.
- Avoids benchmark-report logic; this layer only persists raw structured results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class UnitResult:
    item_id: str
    metric: str
    unit_type: str
    unit_id: str
    scores: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemResult:
    item_id: str
    model_id: str | None
    metric: str
    scores: dict[str, float] = field(default_factory=dict)
    num_units: int = 0
    valid: bool = True
    error: str | None = None


@dataclass
class SummaryResult:
    metric: str
    group_by: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _serialize(obj: Any) -> Any:
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


class ResultWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_items(self, items: list[dict[str, Any]]) -> Path:
        path = self.run_dir / "items.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, default=_serialize) + "\n")
        return path

    def write_metric_results(
        self,
        metric_name: str,
        units: list[UnitResult],
        items: list[ItemResult],
        summary: SummaryResult,
    ) -> Path:
        metric_dir = self.run_dir / "metrics" / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        units_path = metric_dir / "units.jsonl"
        with open(units_path, "w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(asdict(u), ensure_ascii=False, default=_serialize) + "\n")

        items_path = metric_dir / "items.jsonl"
        with open(items_path, "w", encoding="utf-8") as f:
            for i in items:
                f.write(json.dumps(asdict(i), ensure_ascii=False, default=_serialize) + "\n")

        summary_path = metric_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2, default=_serialize)

        return metric_dir
