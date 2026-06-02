# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Timestamped evaluation progress: tqdm bar on stderr and detailed eval.log."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class EvalLogger:
    """Write progress to stderr and/or an append-only log file."""

    def __init__(
        self,
        *,
        console: bool = True,
        log_path: str | Path | None = None,
    ) -> None:
        self.console = console
        self.log_path = Path(log_path) if log_path is not None else None
        self._file: TextIO | None = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")
            self._write_file(f"=== eval log opened at {_now_iso()} ===")

    def log(self, message: str, *, to_console: bool | None = None, to_file: bool = True) -> None:
        line = f"[{_now_iso()}] {message}"
        show_console = self.console if to_console is None else to_console
        if show_console:
            print(line, file=sys.stderr, flush=True)
        if to_file:
            self._write_file(line)

    def close(self) -> None:
        if self._file is not None:
            self._write_file(f"=== eval log closed at {_now_iso()} ===")
            self._file.close()
            self._file = None

    def _write_file(self, line: str) -> None:
        if self._file is None:
            return
        self._file.write(line + "\n")
        self._file.flush()


def format_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "{}"
    parts: list[str] = []
    for key in sorted(scores):
        value = scores[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return "{" + ", ".join(parts) + "}"


def format_item_outcome(
    *,
    metric_name: str,
    item_idx: int,
    total_items: int,
    item_id: str,
    model_id: str | None,
    elapsed_s: float,
    valid: bool,
    scores: dict[str, float],
    num_units: int,
    valid_units: int,
    error: str | None = None,
    skipped: bool = False,
    missing_fields: list[str] | None = None,
) -> str:
    model_part = f" model={model_id}" if model_id else ""
    if skipped:
        missing = ",".join(missing_fields or [])
        return (
            f"[mbench] {metric_name} item {item_idx}/{total_items} skipped in {elapsed_s:.2f}s: "
            f"item_id={item_id}{model_part} missing={missing}"
        )
    err_suffix = f" error={error}" if error and not valid else ""
    return (
        f"[mbench] {metric_name} item {item_idx}/{total_items} done in {elapsed_s:.2f}s: "
        f"item_id={item_id}{model_part} valid={valid} scores={format_scores(scores)} "
        f"units={valid_units}/{num_units}{err_suffix}"
    )


def item_progress_bar(
    *,
    enabled: bool,
    total: int,
    desc: str,
    position: int = 0,
) -> Any:
    """Return a tqdm instance or a no-op stand-in."""
    if not enabled or total <= 0:
        return _NullProgressBar(total=total)

    from tqdm import tqdm

    return tqdm(
        total=total,
        desc=desc,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        position=position,
        leave=True,
    )


class _NullProgressBar:
    def __init__(self, *, total: int) -> None:
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix_str(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return

    def __enter__(self) -> _NullProgressBar:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
