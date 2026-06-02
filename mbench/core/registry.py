# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Plugin interfaces and name-based registries.

- Defines Metric, DatasetAdapter, and Aggregator interfaces.
- Provides lightweight registries for core plugin categories.
- Protects plugin names from duplicate registration.
- Lets pipeline code resolve plugins without depending on concrete implementations.
"""

from __future__ import annotations

from typing import TypeVar, Generic

from .exceptions import DuplicateRegistration, PluginNotFound

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._plugins: dict[str, T] = {}

    def register(self, plugin: T) -> None:
        name = getattr(plugin, "name", None)
        if name is None:
            raise ValueError(f"{self._kind} plugin must have a 'name' attribute.")
        if name in self._plugins:
            raise DuplicateRegistration(self._kind, name)
        self._plugins[name] = plugin

    def get(self, name: str) -> T:
        if name not in self._plugins:
            raise PluginNotFound(self._kind, name, list(self._plugins.keys()))
        return self._plugins[name]

    def list_names(self) -> list[str]:
        return sorted(self._plugins.keys())

    def list_all(self) -> list[T]:
        return list(self._plugins.values())

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .contract import InputContract
from .item import EvaluationItem
from .result import UnitResult, ItemResult, SummaryResult


def mean_item_result(
    metric_name: str,
    item: EvaluationItem,
    units: list[UnitResult],
) -> ItemResult:
    """Fallback item aggregation: unweighted mean over each score key."""
    scores: dict[str, list[float]] = {}
    for u in units:
        for k, v in u.scores.items():
            scores.setdefault(k, []).append(v)
    avg_scores = {k: sum(v) / len(v) for k, v in scores.items() if v}
    return ItemResult(
        item_id=item.item_id,
        model_id=item.model_id,
        metric=metric_name,
        scores=avg_scores,
        num_units=len(units),
        valid=all(u.valid for u in units),
    )


class Metric(ABC):
    name: str
    description: str = ""
    contract: InputContract = InputContract()
    unit_type: str = "item"
    score_keys: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, item: EvaluationItem, **kwargs: Any) -> list[UnitResult]:
        ...

    def aggregate_item(self, item: EvaluationItem, units: list[UnitResult]) -> ItemResult:
        """Aggregate unit-level results into one item-level result.

        Metric authors should override this when item-level semantics are not a
        plain unweighted mean, e.g. duration-weighted segments, worst-case
        identity, or all-pairs causal checks.
        """
        return mean_item_result(self.name, item, units)


class DatasetAdapter(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def load(self, source: str, **kwargs: Any) -> list[EvaluationItem]:
        ...


class Aggregator(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def aggregate(
        self,
        metric_name: str,
        item_results: list[ItemResult],
        group_by: str = "model_id",
    ) -> SummaryResult:
        ...


# Singleton registries
metric_registry: Registry[Metric] = Registry("Metric")
adapter_registry: Registry[DatasetAdapter] = Registry("DatasetAdapter")

from .artifact import ArtifactProducer
artifact_producer_registry: Registry[ArtifactProducer] = Registry("ArtifactProducer")
aggregator_registry: Registry[Aggregator] = Registry("Aggregator")
