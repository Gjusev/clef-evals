#!/usr/bin/env python3
"""Reproducible benchmark runner for clef-evals.

Runs the committed datasets against the hosted Clef API and writes a JSON
report with accuracy, calibration (ECE/Brier), latency percentiles, token
usage, and cost per 1k calls.

Usage:
    # full benchmark (requires CLEF_ACCOUNT_ID + CLEF_API_TOKEN):
    python evals/run_eval.py

    # one dataset, flash model, higher concurrency:
    python evals/run_eval.py --dataset evals/data/support_routing.jsonl \
        --model @cf/cloudflare/clef-flash --concurrency 8

    # results land in evals/results/<dataset>-<selector>-<timestamp>.json
    # and are printed as a table at the end.

The committed reference numbers from Cloudflare's published Decision Index
live in evals/results/published_reference.json; compare your live run
against them with the regression gate (see .github/actions/regression-gate).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clef_evals import AsyncClefJudge, ClefConfig, ClefJudge, load_eval_set  # noqa: E402

DEFAULT_DATASETS = sorted((REPO_ROOT / "evals" / "data").glob("*.jsonl"))
RESULTS_DIR = REPO_ROOT / "evals" / "results"


def run_dataset(
    dataset_path: Path,
    *,
    model: str,
    concurrency: int,
    use_async: bool,
) -> dict[str, Any]:
    """Evaluate one dataset and return a JSON-serializable report entry."""
    eval_set = load_eval_set(dataset_path)
    config = ClefConfig.from_env().with_overrides(model=model)
    if use_async:
        judge = AsyncClefJudge(config)
        result = asyncio.run(judge.evaluate(eval_set, concurrency=concurrency))
    else:
        judge = ClefJudge(config)
        result = judge.evaluate(eval_set)
    judge.close()
    selector = config.model_selector
    return {
        "dataset": dataset_path.name,
        "model": model,
        "selector": selector,
        "n_items": len(eval_set),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in result.to_dict().items() if key != "model"},
    }


def print_table(entries: list[dict[str, Any]]) -> None:
    """Print a compact comparison table of all runs."""
    header = f"{'dataset':<28}{'model':<12}{'n':>4}{'acc':>8}{'ece':>8}{'brier':>8}{'p50ms':>8}{'p95ms':>8}{'$/1k':>9}"
    print(header)
    print("-" * len(header))
    for entry in entries:
        print(
            f"{entry['dataset']:<28}{entry['selector']:<12}{entry['n_samples']:>4}"
            f"{entry['accuracy']:>8.4f}{entry['ece']:>8.4f}{entry['brier']:>8.4f}"
            f"{entry['latency_p50']:>8.1f}{entry['latency_p95']:>8.1f}"
            f"{entry['cost_usd_per_1k_calls']:>9.6f}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset path (repeatable). Defaults to every evals/data/*.jsonl.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        choices=["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"],
        help="Model to benchmark (repeatable). Defaults to both.",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel API calls (default 5)")
    parser.add_argument("--sync", action="store_true", help="Use the sequential sync client")
    args = parser.parse_args(argv)

    dataset_paths = [Path(p) for p in args.dataset] if args.dataset else list(DEFAULT_DATASETS)
    models = args.model or ["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"]

    try:
        entries = [
            run_dataset(
                dataset,
                model=model,
                concurrency=args.concurrency,
                use_async=not args.sync,
            )
            for dataset in dataset_paths
            for model in models
        ]
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_DIR / f"run-{stamp}.json"
    output.write_text(json.dumps({"runs": entries}, indent=2), encoding="utf-8")
    print_table(entries)
    print(f"\nresults written to {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
