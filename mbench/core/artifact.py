# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Artifact production interfaces for optional prepare workflows.

- Defines ArtifactBundle and ProduceContext for producer outputs.
- Defines ArtifactProducer as the plugin interface for generated artifacts.
- Keeps artifact creation explicit through prepare commands.
- Does not participate in the default evaluation path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .contract import InputContract
from .item import ArtifactRef, EvaluationItem


@dataclass
class ArtifactBundle:
    refs: dict[str, ArtifactRef] = field(default_factory=dict)


@dataclass
class ProduceContext:
    output_dir: str = "artifacts"
    extra: dict[str, Any] = field(default_factory=dict)


class ArtifactProducer(ABC):
    name: str
    description: str = ""
    contract: InputContract = InputContract()
    produces: list[str] = []

    @abstractmethod
    def produce(self, item: EvaluationItem, ctx: ProduceContext) -> ArtifactBundle:
        ...
