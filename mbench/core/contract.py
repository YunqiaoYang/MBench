# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Input contract declarations and validation helpers.

- Defines field-path requirements for metrics and artifact producers.
- Distinguishes required fields from optional fields.
- Validates EvaluationItem objects without knowing plugin internals.
- Returns structured missing-field information for user-facing errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .item import EvaluationItem


@dataclass
class ContractField:
    path: str
    description: str = ""


@dataclass
class InputContract:
    required: list[ContractField] = field(default_factory=list)
    optional: list[ContractField] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_field_paths(
        cls,
        required: list[str] | None = None,
        optional: list[str] | None = None,
        description: str = "",
    ) -> InputContract:
        return cls(
            required=[ContractField(path=p) for p in (required or [])],
            optional=[ContractField(path=p) for p in (optional or [])],
            description=description,
        )


@dataclass
class ValidationResult:
    satisfied: bool
    missing: list[str] = field(default_factory=list)
    available_optional: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)


def validate_contract(item: EvaluationItem, contract: InputContract) -> ValidationResult:
    missing: list[str] = []
    for f in contract.required:
        if not item.has_field(f.path):
            missing.append(f.path)

    available_opt: list[str] = []
    missing_opt: list[str] = []
    for f in contract.optional:
        if item.has_field(f.path):
            available_opt.append(f.path)
        else:
            missing_opt.append(f.path)

    return ValidationResult(
        satisfied=len(missing) == 0,
        missing=missing,
        available_optional=available_opt,
        missing_optional=missing_opt,
    )
