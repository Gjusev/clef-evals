"""Calibration and performance metrics for Clef evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    Partitions predictions into ``n_bins`` bins on [0, 1] and returns the
    confidence-weighted mean absolute gap between bin accuracy and bin mean
    confidence. 0.0 means perfectly calibrated.

    Args:
        confidences: Model confidence in the predicted answer, in [0, 1].
        correct: Whether each prediction was factually correct.
        n_bins: Number of equal-width bins (must be >= 1).

    Returns:
        ECE in [0, 1]. Returns 0.0 for empty inputs.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    bin_totals = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_confidence_sum = [0.0] * n_bins
    for confidence, was_correct in zip(confidences, correct, strict=False):
        clipped = min(1.0, max(0.0, float(confidence)))
        index = min(n_bins - 1, int(clipped * n_bins))
        bin_totals[index] += 1
        bin_correct[index] += 1 if was_correct else 0
        bin_confidence_sum[index] += clipped
    total = len(confidences)
    score = 0.0
    for count, n_correct, confidence_sum in zip(
        bin_totals, bin_correct, bin_confidence_sum, strict=False
    ):
        if count == 0:
            continue
        bin_accuracy = n_correct / count
        bin_mean_confidence = confidence_sum / count
        score += (count / total) * abs(bin_accuracy - bin_mean_confidence)
    return score


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Binary Brier score: mean squared error of confidence vs outcome.

    Args:
        confidences: Model confidence in the predicted answer, in [0, 1].
        correct: Whether each prediction was factually correct.

    Returns:
        Brier score in [0, 1]; lower is better. 0.0 for empty inputs.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    total = 0.0
    for confidence, was_correct in zip(confidences, correct, strict=False):
        target = 1.0 if was_correct else 0.0
        total += (float(confidence) - target) ** 2
    return total / len(confidences)


def brier_multiclass(
    probabilities: Sequence[Mapping[str, float]],
    golds: Sequence[str],
) -> float:
    """Multiclass Brier score over full probability distributions.

    For each item, sums ``(p_c - y_c)^2`` over every class observed across the
    dataset (missing classes count as probability 0). Lower is better; 0 is a
    perfect one-hot prediction set.

    Args:
        probabilities: Per-item distribution (must sum to ~1).
        golds: Gold class label per item.

    Returns:
        Mean multiclass Brier score in [0, 2]. 0.0 for empty inputs.
    """
    if len(probabilities) != len(golds):
        raise ValueError("probabilities and golds must have the same length")
    if not probabilities:
        return 0.0
    classes: set[str] = set()
    for distribution in probabilities:
        classes.update(distribution)
    classes.update(golds)
    total = 0.0
    for distribution, gold in zip(probabilities, golds, strict=False):
        item_error = 0.0
        for class_name in classes:
            predicted = float(distribution.get(class_name, 0.0))
            target = 1.0 if class_name == gold else 0.0
            item_error += (predicted - target) ** 2
        total += item_error
    return total / len(probabilities)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values`` for ``q`` in [0, 100].

    Returns 0.0 for empty input; returns the single value for one-element
    input regardless of ``q``.
    """
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute p50/p95/p99 latency in milliseconds."""
    return {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
    }


@dataclass
class EvalResult:
    """Outcome of evaluating a dataset with Clef as judge.

    Attributes:
        accuracy: Fraction of correct predictions.
        ece: Expected Calibration Error of the confidences.
        brier: Binary Brier score of the confidences.
        brier_multiclass: Multiclass Brier score over full distributions.
        n_samples: Number of evaluated items.
        correct: Per-item correctness flags.
        confidences: Per-item confidence in the predicted answer.
        predictions: Per-item predicted label (bool for binary items).
        golds: Per-item gold label.
        latencies_ms: Per-call wall-clock latency in milliseconds.
        latency_p50: Median latency in milliseconds.
        latency_p95: 95th percentile latency in milliseconds.
        latency_p99: 99th percentile latency in milliseconds.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens produced.
        input_cost_usd: Total input-token cost in USD at the published price.
        cost_usd_per_1k_calls: Mean input cost of 1,000 judge calls.
        model: Model id used for the evaluation.
        failures: Number of items that failed with an API error.
    """

    accuracy: float
    ece: float
    brier: float
    brier_multiclass: float
    n_samples: int
    correct: list[bool] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    predictions: list[Any] = field(default_factory=list)
    golds: list[Any] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    cost_usd_per_1k_calls: float = 0.0
    model: str = ""
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for reports and CI artifacts."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        return (
            f"samples={self.n_samples} failures={self.failures}\n"
            f"accuracy={self.accuracy:.4f}\n"
            f"ece={self.ece:.4f}\n"
            f"brier={self.brier:.4f} brier_multiclass={self.brier_multiclass:.4f}\n"
            f"latency_ms p50={self.latency_p50:.1f} p95={self.latency_p95:.1f} p99={self.latency_p99:.1f}\n"
            f"input_tokens={self.input_tokens} output_tokens={self.output_tokens}\n"
            f"cost: ${self.input_cost_usd:.6f} total | ${self.cost_usd_per_1k_calls:.6f} per 1k calls"
        )


#: Backwards-compatible alias for the v0.1.0 name.
ClefEvalResult = EvalResult
