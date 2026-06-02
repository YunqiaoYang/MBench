# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Exception types for user-facing core failures.

- Defines the base MBenchError for framework-level exceptions.
- Represents contract violations with item id, metric name, and missing fields.
- Reports duplicate plugin registration clearly.
- Reports missing plugins with available alternatives.
"""

from __future__ import annotations


class MBenchError(Exception):
    pass


class ContractViolation(MBenchError):
    def __init__(
        self,
        item_id: str,
        metric_name: str,
        missing_fields: list[str],
        artifact_producers: dict[str, list[str]] | None = None,
    ):
        self.item_id = item_id
        self.metric_name = metric_name
        self.missing_fields = missing_fields
        self.artifact_producers = artifact_producers or {}
        parts = [
            f"Item '{item_id}' does not satisfy contract for metric '{metric_name}'.",
            f"Missing required fields: {', '.join(missing_fields)}.",
        ]
        if self.artifact_producers:
            producer_hints = [
                f"{field} -> {', '.join(producers)}"
                for field, producers in self.artifact_producers.items()
            ]
            parts.append("Available artifact producers: " + " / ".join(producer_hints))
        super().__init__(" ".join(parts))


class DuplicateRegistration(MBenchError):
    def __init__(self, kind: str, name: str):
        super().__init__(f"Duplicate {kind} registration: '{name}' is already registered.")


class PluginNotFound(MBenchError):
    def __init__(self, kind: str, name: str, available: list[str]):
        avail = ", ".join(available) if available else "(none)"
        super().__init__(f"{kind} '{name}' not found. Available: {avail}")
