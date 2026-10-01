"""Command-line interface for clef-evals.

Exit codes: 0 = gate passed, 1 = gate failed or dataset errors, 2 = usage/config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ClefConfig
from .exceptions import ClefError, ConfigurationError
from .judge import AsyncClefJudge, ClefJudge, load_eval_set


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="clef-eval",
        description="Evaluate Cloudflare Clef as a judge with calibration gating.",
    )
    parser.add_argument("--version", action="version", version=f"clef-evals {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an eval set and optionally gate on thresholds")
    run.add_argument("dataset", help="Path to a .json array or .jsonl eval set")
    run.add_argument(
        "--model",
        choices=["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"],
        default=None,
        help="Hosted Clef model (default: CLEF_MODEL or clef)",
    )
    run.add_argument("--concurrency", type=int, default=5, help="Max parallel API calls (default: 5)")
    run.add_argument("--min-accuracy", type=float, default=None, help="Gate: fail if accuracy is lower")
    run.add_argument("--max-ece", type=float, default=None, help="Gate: fail if ECE is higher")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    run.add_argument("--output", type=Path, default=None, help="Also write the full result JSON here")
    run.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    run.add_argument("--async", dest="use_async", action="store_true", help="Use the async client")

    return parser


def _run_command(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand and return the process exit code."""
    if args.quiet:
        logging.getLogger("clef_evals").setLevel(logging.ERROR)
    try:
        eval_set = load_eval_set(args.dataset)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        if args.use_async:
            import asyncio

            judge = AsyncClefJudge(config=ClefConfig.from_env().with_overrides(model=args.model))
            result = asyncio.run(judge.evaluate(eval_set, concurrency=args.concurrency))
        else:
            judge = ClefJudge(config=ClefConfig.from_env().with_overrides(model=args.model))
            result = judge.evaluate(eval_set)
    except ConfigurationError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except ClefError as error:
        print(f"clef error: {error}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    gate_problems: list[str] = []
    if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
        gate_problems.append(
            f"accuracy {result.accuracy:.4f} < required {args.min_accuracy:.4f}"
        )
    if args.max_ece is not None and result.ece > args.max_ece:
        gate_problems.append(f"ece {result.ece:.4f} > allowed {args.max_ece:.4f}")
    if result.failures:
        gate_problems.append(f"{result.failures} item(s) failed with API errors")

    if args.json:
        print(json.dumps({**result.to_dict(), "gate_passed": not gate_problems}, indent=2))
    else:
        print(result.summary())
        if gate_problems:
            print("gate: FAILED")
            for problem in gate_problems:
                print(f"  - {problem}")
        else:
            print("gate: PASSED")

    return 0 if not gate_problems else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``clef-eval`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
