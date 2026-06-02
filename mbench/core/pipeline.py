# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Evaluation and artifact-preparation orchestration.

- Validates items against selected metric contracts before evaluation.
- Runs metrics, aggregates item results, and writes standard outputs.
- Skips items missing metric-specific required fields during eval.
- Keeps artifact production in the separate prepare path, never default eval.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import sys
import threading
import time
from typing import Any

from .contract import validate_contract
from .eval_log import EvalLogger, format_item_outcome, format_scores, item_progress_bar
from .exceptions import ContractViolation
from .item import EvaluationItem
from .registry import (
    Metric,
    Aggregator,
    artifact_producer_registry,
)
from .result import ItemResult, ResultWriter, UnitResult


def _project_unit_scores(metric_name: str, unit: UnitResult) -> UnitResult:
    """Replace ``unit.scores`` with the canonical projection for ``metric_name``.

    Internal compute keys (debug pairs, mean/std/num counts, alternative
    similarity proxies) are preserved on ``unit.metadata['raw_scores']`` so
    debugging stays possible.
    """
    from mbench.metrics.canonical import canonicalize_scores

    if unit.unit_type == "skipped_contract":
        return unit
    raw = dict(unit.scores or {})
    canonical = canonicalize_scores(metric_name, raw)
    if canonical == raw:
        return unit
    new_metadata = dict(unit.metadata or {})
    # Only stash a copy when there were extra/non-canonical fields.
    extras = {k: v for k, v in raw.items() if k not in canonical}
    if extras:
        new_metadata.setdefault("raw_scores", extras)
    unit.scores = canonical
    unit.metadata = new_metadata
    return unit


def _project_item_scores(metric_name: str, item_result: ItemResult | None) -> ItemResult | None:
    if item_result is None:
        return item_result
    from mbench.metrics.canonical import canonicalize_scores

    raw = dict(item_result.scores or {})
    item_result.scores = canonicalize_scores(metric_name, raw)
    return item_result


def _suggest_artifact_producers(
    missing_fields: list[str],
) -> dict[str, list[str]]:
    producer_hints: dict[str, list[str]] = {}
    for field_path in missing_fields:
        if field_path.startswith("artifacts."):
            artifact_name = field_path.split(".", 1)[1]
            for producer in artifact_producer_registry.list_all():
                if artifact_name in producer.produces:
                    producer_hints.setdefault(field_path, []).append(producer.name)
    return producer_hints


def validate_items(
    items: list[EvaluationItem],
    metrics: list[Metric],
    fail_fast: bool = False,
) -> list[ContractViolation]:
    errors: list[ContractViolation] = []
    for item in items:
        for metric in metrics:
            result = validate_contract(item, metric.contract)
            if not result.satisfied:
                artifact_producers = _suggest_artifact_producers(result.missing)
                err = ContractViolation(
                    item_id=item.item_id,
                    metric_name=metric.name,
                    missing_fields=result.missing,
                    artifact_producers=artifact_producers,
                )
                errors.append(err)
                if fail_fast:
                    return errors
    return errors


def run_eval(
    items: list[EvaluationItem],
    metrics: list[Metric],
    aggregator: Aggregator,
    output_dir: str,
    skip_validation: bool = False,
    metric_kwargs: dict[str, Any] | None = None,
    progress: bool = False,
    progress_bar: bool | None = None,
    progress_every: int = 1,
    log_file: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run evaluation. Default: never auto-produces artifacts.

    Eval is statistical: if a metric needs an optional artifact such as DA3 and
    a particular item lacks it, skip that item for that metric instead of
    aborting the whole run. The standalone ``validate`` command remains strict.

    When ``progress`` is enabled, stderr shows a tqdm bar (unless disabled) and
    ``log_file`` records every item outcome with scores. Use ``progress_every`` to
    throttle duplicate console lines when the bar is off.
    """

    out_path = Path(output_dir)
    writer = ResultWriter(out_path)
    writer.write_items([{"item_id": it.item_id, "model_id": it.model_id} for it in items])

    run_summary: dict[str, Any] = {}
    total_items = len(items)
    total_metrics = len(metrics)
    progress_every = max(1, int(progress_every))
    show_bar = progress and (progress_bar if progress_bar is not None else True)
    if progress and progress_bar is None and not sys.stderr.isatty():
        # Piped batch jobs: tqdm still emits one line per step (good for tee logs).
        show_bar = True

    logger = EvalLogger(console=progress, log_path=log_file)
    if log_file is not None:
        logger.log(f"[mbench] log file: {log_file}")

    def log(message: str, *, to_console: bool | None = None, to_file: bool = True) -> None:
        logger.log(message, to_console=to_console, to_file=to_file)

    try:
        log(f"[mbench] starting eval: {total_items} items, {total_metrics} metrics -> {output_dir}")
        log(f"[mbench] metrics: {', '.join(metric.name for metric in metrics)}")
        for metric_idx, metric in enumerate(metrics, start=1):
            metric_start = time.monotonic()
            log(f"[mbench] metric {metric_idx}/{total_metrics} start: {metric.name}")
            all_units: list[UnitResult] = []
            all_items: list[ItemResult] = []
            skipped_items = 0

            bar_desc = f"[{metric_idx}/{total_metrics}] {metric.name}"
            pbar = item_progress_bar(enabled=show_bar, total=total_items, desc=bar_desc)

            def _evaluate_one(item: EvaluationItem, item_idx: int) -> tuple[int, list[UnitResult], ItemResult | None, bool]:
                """Evaluate one item for this metric. Returns (idx, units, item_result, skipped)."""
                units = metric.evaluate(item, **(metric_kwargs or {}))
                item_result = metric.aggregate_item(item, units)
                units = [_project_unit_scores(metric.name, u) for u in units]
                item_result = _project_item_scores(metric.name, item_result)
                return (item_idx, units, item_result, False)

            def _evaluate_with_contract(item: EvaluationItem, item_idx: int) -> tuple[int, list[UnitResult], ItemResult | None, bool]:
                """Evaluate with contract validation. Returns (idx, units, item_result, skipped)."""
                if not skip_validation:
                    if not validate_contract(item, metric.contract).satisfied:
                        skipped_unit = UnitResult(
                            item_id=item.item_id, metric=metric.name,
                            unit_type="skipped_contract",
                            unit_id=f"{item.item_id}__skipped_contract",
                            scores={}, valid=False,
                            error="Missing required fields",
                            metadata={"model_id": item.model_id, "missing_fields": []},
                        )
                        return (item_idx, [skipped_unit], None, True)
                units = metric.evaluate(item, **(metric_kwargs or {}))
                item_result = metric.aggregate_item(item, units)
                units = [_project_unit_scores(metric.name, u) for u in units]
                item_result = _project_item_scores(metric.name, item_result)
                return (item_idx, units, item_result, False)

            if workers > 1:
                # Parallel: submit all items, collect results
                results: dict[int, tuple[list[UnitResult], ItemResult | None, bool]] = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_evaluate_with_contract, item, i): i
                        for i, item in enumerate(items, start=1)
                    }
                    for future in as_completed(futures):
                        idx, units, item_result, skipped = future.result()
                        results[idx] = (units, item_result, skipped)
                        pbar.set_postfix_str(
                            f"{len(results)}/{total_items}", refresh=False)
                        pbar.update(1)

                # Collect in order
                for item_idx in range(1, total_items + 1):
                    item = items[item_idx - 1]
                    units, item_result, skipped = results[item_idx]
                    if skipped:
                        all_units.extend(units)
                        skipped_items += 1
                    else:
                        all_units.extend(units)
                        all_items.append(item_result)
                    elapsed = 0.0  # not tracked per-item in parallel mode
                    outcome = format_item_outcome(
                        metric_name=metric.name, item_idx=item_idx,
                        total_items=total_items,
                        item_id=item.item_id, model_id=item.model_id,
                        elapsed_s=elapsed,
                        valid=(not skipped and item_result.valid) if item_result else False,
                        scores=item_result.scores if item_result else {},
                        num_units=len(units),
                        valid_units=sum(1 for u in units if u.valid) if not skipped else 0,
                        error=item_result.error if item_result else "skipped",
                    )
                    log(outcome, to_console=(item_idx % progress_every == 0), to_file=True)
            else:
                for item_idx, item in enumerate(items, start=1):
                    should_log_console = (
                        not show_bar
                        and (item_idx == 1 or item_idx == total_items or item_idx % progress_every == 0)
                    )
                    item_start = time.monotonic()
                    log(
                        f"[mbench] {metric.name} item {item_idx}/{total_items} start: "
                        f"item_id={item.item_id} model={item.model_id}",
                        to_console=should_log_console or item_idx == 1,
                        to_file=True,
                    )
                    if not skip_validation:
                        contract_result = validate_contract(item, metric.contract)
                        if not contract_result.satisfied:
                            skipped = UnitResult(
                                item_id=item.item_id, metric=metric.name,
                                unit_type="skipped_contract",
                                unit_id=f"{item.item_id}__skipped_contract",
                                scores={}, valid=False,
                                error="Missing required fields: " + ",".join(contract_result.missing),
                                metadata={"model_id": item.model_id, "missing_fields": contract_result.missing},
                            )
                            all_units.append(skipped)
                            skipped_items += 1
                            elapsed = time.monotonic() - item_start
                            outcome = format_item_outcome(
                                metric_name=metric.name, item_idx=item_idx,
                                total_items=total_items,
                                item_id=item.item_id, model_id=item.model_id,
                                elapsed_s=elapsed, valid=False, scores={},
                                num_units=0, valid_units=0, skipped=True,
                                missing_fields=contract_result.missing,
                            )
                            log(outcome, to_console=should_log_console, to_file=True)
                            pbar.set_postfix_str(f"skip {item.item_id[:24]}", refresh=False)
                            pbar.update(1)
                            continue
                    units = metric.evaluate(item, **(metric_kwargs or {}))
                    item_result = metric.aggregate_item(item, units)
                    units = [_project_unit_scores(metric.name, u) for u in units]
                    item_result = _project_item_scores(metric.name, item_result)
                    all_units.extend(units)
                    all_items.append(item_result)
                    elapsed = time.monotonic() - item_start
                    valid_units = sum(1 for unit in units if unit.valid)
                    outcome = format_item_outcome(
                        metric_name=metric.name, item_idx=item_idx,
                        total_items=total_items,
                        item_id=item.item_id, model_id=item.model_id,
                        elapsed_s=elapsed, valid=item_result.valid,
                        scores=item_result.scores, num_units=len(units),
                        valid_units=valid_units, error=item_result.error,
                    )
                    log(outcome, to_console=should_log_console, to_file=True)
                    postfix = "ok" if item_result.valid else "err"
                    pbar.set_postfix_str(f"{postfix} {item.item_id[:20]}", refresh=False)
                    pbar.update(1)

            pbar.close()

            summary = aggregator.aggregate(metric.name, all_items)
            writer.write_metric_results(metric.name, all_units, all_items, summary)
            run_summary[metric.name] = asdict(summary)
            metric_elapsed = time.monotonic() - metric_start
            valid_items = sum(1 for item_result in all_items if item_result.valid)
            log(
                f"[mbench] metric {metric_idx}/{total_metrics} done: {metric.name} in {metric_elapsed:.2f}s "
                f"(items_valid={valid_items}/{len(all_items)} skipped_contract={skipped_items})"
            )
            for group, data in summary.results.items():
                scores = data.get("scores", {})
                if scores:
                    log(
                        f"[mbench] {metric.name} aggregate group={group} scores={format_scores(scores)}",
                        to_console=True,
                        to_file=True,
                    )

        log(f"[mbench] eval finished: {total_items} items, {total_metrics} metrics")
        return run_summary
    finally:
        logger.close()


def run_prepare(
    items: list[EvaluationItem],
    producer_name: str,
    output_dir: str,
) -> dict[str, list[str]]:
    """Run artifact production (sidetrack, never called during eval)."""
    from .artifact import ProduceContext

    producer = artifact_producer_registry.get(producer_name)

    ctx = ProduceContext(output_dir=output_dir)
    produced: dict[str, list[str]] = {}
    for item in items:
        result = validate_contract(item, producer.contract)
        if not result.satisfied:
            continue
        bundle = producer.produce(item, ctx)
        produced[item.item_id] = list(bundle.refs.keys())
    return produced
