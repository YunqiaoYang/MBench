# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Mean score aggregator for item-level metric results.

- Groups ItemResult records by a configurable attribute.
- Averages each score key within every group.
- Produces a SummaryResult compatible with ResultWriter.
- Serves as the small built-in aggregator for tests and default CLI use.
"""

from __future__ import annotations

from typing import Any

from mbench.core.registry import Aggregator
from mbench.core.result import ItemResult, SummaryResult


class MeanAggregator(Aggregator):
    name = "mean"
    description = "Averages metric scores per group."

    def aggregate(
        self,
        metric_name: str,
        item_results: list[ItemResult],
        group_by: str = "model_id",
    ) -> SummaryResult:
        groups: dict[str, list[ItemResult]] = {}
        for r in item_results:
            key = getattr(r, group_by, None) or "all"
            groups.setdefault(key, []).append(r)

        summary: dict[str, dict[str, Any]] = {}
        for group_key, results in groups.items():
            score_lists: dict[str, list[float]] = {}
            for r in results:
                for k, v in r.scores.items():
                    score_lists.setdefault(k, []).append(v)
            summary[group_key] = {
                "count": len(results),
                "scores": {k: sum(v) / len(v) for k, v in score_lists.items() if v},
            }

        return SummaryResult(
            metric=metric_name,
            group_by=group_by,
            results=summary,
        )
