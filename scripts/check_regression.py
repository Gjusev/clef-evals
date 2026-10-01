#!/usr/bin/env python3
"""Regression gate: compare a fresh measurement JSON against a committed baseline.

Exit codes: 0 = no regression, 1 = regression detected, 2 = usage error.

Metric specs follow ``path:goal[:tolerance]`` where ``path`` is a dotted path
into the JSON document, ``goal`` is ``max`` (metric may grow by at most
``tolerance``, e.g. ECE) or ``min`` (metric may shrink by at most
``tolerance``, e.g. accuracy), and ``tolerance`` defaults to 0.02.

Example:
    python scripts/check_regression.py \\
        --current results/run-20260101T000000Z.json \\
        --baseline results/baselines/support-routing.json \\
        --metric accuracy:min:0.03 --metric ece:max
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def dig(document: dict[str, Any] | list[Any], dotted_path: str) -> Any:
    """Resolve a dotted path like 'runs.0.accuracy' through dicts and lists."""
    value: Any = document
    for part in dotted_path.split("."):
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                raise KeyError(f"path {dotted_path!r} not found (bad index {part!r})") from None
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"path {dotted_path!r} not found (missing {part!r})")
    return value


def check_metric(
    current: dict[str, Any],
    baseline: dict[str, Any],
    spec: str,
) -> str | None:
    """Check one ``path:goal[:tolerance]`` spec; return a problem string or None."""
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[1] not in ("max", "min"):
        raise ValueError(f"metric spec {spec!r} must be path:max|min[:tolerance]")
    metric_path, goal = parts[0], parts[1]
    tolerance = float(parts[2]) if len(parts) == 3 else 0.02

    current_value = float(dig(current, metric_path))
    baseline_value = float(dig(baseline, metric_path))

    if goal == "min":
        # accuracy-like: may not drop more than tolerance
        floor = baseline_value - tolerance
        if current_value < floor:
            return (
                f"regression: {metric_path} {current_value:.4f} < "
                f"{floor:.4f} (baseline {baseline_value:.4f} - tolerance {tolerance})"
            )
    else:
        # ECE-like: may not grow by more than tolerance
        ceiling = baseline_value + tolerance
        if current_value > ceiling:
            return (
                f"regression: {metric_path} {current_value:.4f} > "
                f"{ceiling:.4f} (baseline {baseline_value:.4f} + tolerance {tolerance})"
            )
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, help="Fresh measurement JSON")
    parser.add_argument("--baseline", required=True, help="Committed baseline JSON")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Spec path:max|min[:tolerance] (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        current = json.loads(Path(args.current).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"error reading inputs: {error}", file=sys.stderr)
        return 2

    problems: list[str] = []
    try:
        for spec in args.metric:
            problem = check_metric(current, baseline, spec)
            if problem:
                problems.append(problem)
    except (ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if problems:
        print("regression gate: FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("regression gate: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
