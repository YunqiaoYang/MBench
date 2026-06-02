# Copyright 2026 MBench community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Zhang Zhang / Tsinghua University] in [2026].

"""Command-line entry point for the MBench core.

- Registers built-in adapters, metrics, artifact producers, and aggregators.
- Exposes list-metrics, validate, eval, and prepare commands.
- Keeps CLI handlers thin by delegating core behavior to pipeline functions.
- Surfaces contract and plugin errors through user-readable command output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mbench.core.registry import (
    metric_registry,
    adapter_registry,
    artifact_producer_registry,
    aggregator_registry,
)


def _print_recovery_commands(errors: list[Exception], source: str) -> None:
    """Render CLI recovery commands from structured core errors."""
    seen: set[str] = set()
    for err in errors:
        artifact_producers = getattr(err, "artifact_producers", {})
        for producers in artifact_producers.values():
            for producer in producers:
                cmd = f"mbench prepare {source} --producer {producer} --output artifacts/"
                if cmd not in seen:
                    print(f"  Suggested command: {cmd}", file=sys.stderr)
                    seen.add(cmd)


def _bootstrap() -> None:
    """Register all built-in plugins."""
    from mbench.metrics.builtin_registry import MBENCH_A_METRICS, MBENCH_T_METRICS
    from mbench.aggregators.mean import MeanAggregator
    from mbench.adapters.mbench_dir import MBenchDirAdapter

    for plugin, registry in [
        (MeanAggregator(), aggregator_registry),
        (MBenchDirAdapter(), adapter_registry),
    ]:
        if plugin.name not in registry:
            registry.register(plugin)

    for metric_cls in (*MBENCH_A_METRICS, *MBENCH_T_METRICS):
        metric = metric_cls()
        if metric.name not in metric_registry:
            metric_registry.register(metric)


def cmd_list_metrics(args: argparse.Namespace) -> None:
    print("Registered metrics:")
    for m in metric_registry.list_all():
        print(f"  {m.name:30s}  {m.description}")


def _adapter_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Pick up optional adapter parameters present on argparse namespace."""
    kw: dict[str, Any] = {}
    if getattr(args, "models", None):
        kw["models"] = _csv_arg(args.models)
    if getattr(args, "subsets", None):
        kw["subsets"] = _csv_arg(args.subsets)
    if getattr(args, "conditions", None):
        kw["conditions"] = _csv_arg(args.conditions)
    if getattr(args, "sample_ids", None):
        kw["sample_ids"] = _csv_arg(args.sample_ids)
    if getattr(args, "limit", None) is not None:
        kw["limit"] = args.limit
    return kw


def _csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _add_adapter_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", help="Comma-separated model_id whitelist (default: all)")
    parser.add_argument("--subsets", help="Comma-separated subset whitelist, e.g. causal,human")
    parser.add_argument(
        "--conditions",
        help=(
            "Comma-separated condition_id or action whitelist, e.g. "
            "left_then_right,left_then_right_10s"
        ),
    )
    parser.add_argument("--sample-ids", help="Comma-separated sample_id whitelist")
    parser.add_argument("--limit", type=int, help="Stop loading after N filtered items")


def cmd_validate(args: argparse.Namespace) -> None:
    from mbench.core.pipeline import validate_items

    adapter = adapter_registry.get(args.adapter)
    items = adapter.load(args.source, **_adapter_load_kwargs(args))

    metric_names = [n.strip() for n in args.metrics.split(",")]
    metrics = [metric_registry.get(n) for n in metric_names]

    errors = validate_items(items, metrics)
    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        _print_recovery_commands(errors, args.source)
        sys.exit(1)
    else:
        print(f"All {len(items)} items satisfy the contracts for: {', '.join(metric_names)}")


def cmd_eval(args: argparse.Namespace) -> None:
    from mbench.core.pipeline import run_eval
    from mbench.core.exceptions import ContractViolation

    adapter = adapter_registry.get(args.adapter)
    if not args.quiet:
        print(f"Loading items from {args.source}...", file=sys.stderr, flush=True)
    items = adapter.load(args.source, **_adapter_load_kwargs(args))
    if not args.quiet:
        print(f"Loaded {len(items)} items.", file=sys.stderr, flush=True)

    metric_names = [n.strip() for n in args.metrics.split(",")]
    metrics = [metric_registry.get(n) for n in metric_names]
    aggregator = aggregator_registry.get(args.aggregator)
    metric_kwargs: dict[str, Any] = {}
    if getattr(args, "vlm_judge", None):
        if args.vlm_judge != "openai-compatible":
            raise ValueError(f"Unsupported VLM judge: {args.vlm_judge}")
        from mbench.metrics.vlm_openai import OpenAICompatMBenchTJudge

        judge_kwargs: dict[str, Any] = {"n_frames": args.vlm_n_frames}
        if args.vlm_base_url:
            judge_kwargs["base_url"] = args.vlm_base_url
        if args.vlm_api_key:
            judge_kwargs["api_key"] = args.vlm_api_key
        if args.vlm_model:
            judge_kwargs["model"] = args.vlm_model
        if args.vlm_endpoint_path:
            judge_kwargs["endpoint_path"] = args.vlm_endpoint_path
        if args.vlm_api_style:
            judge_kwargs["api_style"] = args.vlm_api_style
        if args.vlm_max_tokens:
            judge_kwargs["max_tokens"] = args.vlm_max_tokens
        if getattr(args, "vlm_max_retries", None):
            judge_kwargs["max_retries"] = args.vlm_max_retries
        metric_kwargs["judge"] = OpenAICompatMBenchTJudge(**judge_kwargs)
    if getattr(args, "ignore_trigger", False):
        metric_kwargs["ignore_trigger"] = True

    log_file: str | None = None
    if not getattr(args, "no_log_file", False):
        log_file = args.log_file or str(Path(args.output) / "eval.log")

    try:
        summary = run_eval(
            items,
            metrics,
            aggregator,
            args.output,
            skip_validation=args.skip_validation,
            metric_kwargs=metric_kwargs,
            progress=not args.quiet,
            progress_bar=not getattr(args, "no_progress_bar", False),
            progress_every=args.progress_every,
            log_file=log_file,
            workers=getattr(args, "workers", 1),
        )
    except ContractViolation as e:
        print("Contract validation failed:", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        _print_recovery_commands([e], args.source)
        sys.exit(1)
    print(f"Evaluation complete. Results written to {args.output}/")
    if log_file:
        print(f"Progress log: {log_file}")
    for metric_name, s in summary.items():
        print(f"  {metric_name}:")
        for group, data in s.get("results", {}).items():
            scores = data.get("scores", {})
            score_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"    {group}: {score_str}")


def cmd_prepare(args: argparse.Namespace) -> None:
    from mbench.core.pipeline import run_prepare

    adapter = adapter_registry.get(args.adapter)
    items = adapter.load(args.source, **_adapter_load_kwargs(args))
    print(f"Running artifact producer '{args.producer}' on {len(items)} items...")
    produced = run_prepare(items, args.producer, args.output)
    for item_id, artifact_names in produced.items():
        print(f"  {item_id}: produced {', '.join(artifact_names)}")
    print(f"Artifacts written to {args.output}/")


def main(argv: list[str] | None = None) -> None:
    _bootstrap()

    parser = argparse.ArgumentParser(
        prog="mbench",
        description="MBench: contract-driven evaluation pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-metrics", help="List all registered metrics")

    p_validate = sub.add_parser("validate", help="Validate input source against metric contracts")
    p_validate.add_argument("source", help="MBench-A/V dataset root (local path or hf://datasets/...)")
    p_validate.add_argument("--adapter", default="mbench-dir", help="Dataset adapter name (default: mbench-dir)")
    p_validate.add_argument("--metrics", required=True, help="Comma-separated metric names")
    _add_adapter_filter_args(p_validate)

    p_eval = sub.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("source", help="MBench-A/V dataset root (local path or hf://datasets/...)")
    p_eval.add_argument("--adapter", default="mbench-dir", help="Dataset adapter name (default: mbench-dir)")
    p_eval.add_argument("--metrics", required=True, help="Comma-separated metric names")
    p_eval.add_argument("--output", required=True, help="Output directory for results")
    p_eval.add_argument("--aggregator", default="mean", help="Aggregator name (default: mean)")
    p_eval.add_argument("--skip-validation", action="store_true", help="Skip contract validation (unsafe, use only if you know what you're doing)")
    p_eval.add_argument(
        "--quiet",
        action="store_true",
        help="Disable stderr progress bar and console lines (eval.log still written unless --no-log-file)",
    )
    p_eval.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm bar; use line-based console logs every --progress-every items instead",
    )
    p_eval.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="When --no-progress-bar, print a console line every N items (eval.log always logs every item)",
    )
    p_eval.add_argument(
        "--log-file",
        help="Append timestamped progress to this file (default: <output>/eval.log)",
    )
    p_eval.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write eval.log under the output directory",
    )
    p_eval.add_argument(
        "--vlm-judge",
        choices=["openai-compatible"],
        help="Instantiate a VLM judge and pass it to metrics that require judge=.",
    )
    p_eval.add_argument("--vlm-base-url", help="Override MBENCH_VLM_BASE_URL for --vlm-judge")
    p_eval.add_argument("--vlm-api-key", help="Override MBENCH_VLM_API_KEY for --vlm-judge; prefer env vars for secrets")
    p_eval.add_argument("--vlm-model", help="Override MBENCH_VLM_MODEL for --vlm-judge")
    p_eval.add_argument("--vlm-endpoint-path", help="Append this endpoint path to --vlm-base-url, e.g. /v1/chat/completions")
    p_eval.add_argument(
        "--vlm-api-style",
        choices=["chat_completions", "responses"],
        default="chat_completions",
        help="OpenAI-compatible API style used by --vlm-judge",
    )
    p_eval.add_argument("--workers", type=int, default=1, help="Number of parallel eval workers (use with VLM metrics)")
    p_eval.add_argument("--vlm-max-retries", type=int, default=3, help="Max VLM API retries on rate-limit (429) / server errors")
    p_eval.add_argument("--vlm-n-frames", type=int, default=8, help="Uniform video frames sent to VLM judge")
    p_eval.add_argument("--vlm-max-tokens", type=int, help="Max output tokens for VLM judge JSON responses")
    p_eval.add_argument(
        "--ignore-trigger",
        action="store_true",
        help="Compute reliability scores without a trigger judge; intended for score-only/debug runs.",
    )
    _add_adapter_filter_args(p_eval)

    p_prepare = sub.add_parser("prepare", help="Run artifact producer (sidetrack)")
    p_prepare.add_argument("source", help="MBench-A/V dataset root (local path or hf://datasets/...)")
    p_prepare.add_argument("--adapter", default="mbench-dir", help="Dataset adapter name (default: mbench-dir)")
    p_prepare.add_argument("--producer", required=True, help="Artifact producer name")
    p_prepare.add_argument("--output", required=True, help="Output directory for artifacts")
    _add_adapter_filter_args(p_prepare)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "list-metrics": cmd_list_metrics,
        "validate": cmd_validate,
        "eval": cmd_eval,
        "prepare": cmd_prepare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
