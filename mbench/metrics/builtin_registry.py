# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License.

"""Aggregated metric class tuples for the MBench-A and MBench-T registries.

The CLI ``_bootstrap()`` walks these tuples to instantiate every built-in
metric. To extend the benchmark, add the new class to the per-axis tuple in
``causal_consistency``, ``entity_consistency``, or ``environment_consistency``
and it will appear here automatically.
"""

from __future__ import annotations

from typing import Type

from mbench.core.registry import Metric
from mbench.metrics.causal_consistency import (
    CAUSAL_METRICS,
    T_CAUSAL_METRICS,
)
from mbench.metrics.entity_consistency import A_ENTITY_METRICS, T_ENTITY_METRICS
from mbench.metrics.environment_consistency import (
    A_ENVIRONMENT_METRICS,
    T_ENVIRONMENT_METRICS,
)

MBENCH_A_METRICS: tuple[Type[Metric], ...] = (
    *A_ENTITY_METRICS,
    *A_ENVIRONMENT_METRICS,
    *CAUSAL_METRICS,
)

MBENCH_T_METRICS: tuple[Type[Metric], ...] = (
    *T_ENTITY_METRICS,
    *T_ENVIRONMENT_METRICS,
    *T_CAUSAL_METRICS,
)

__all__ = ["MBENCH_A_METRICS", "MBENCH_T_METRICS"]
