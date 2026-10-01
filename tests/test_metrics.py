"""Tests for calibration and latency metrics."""

from __future__ import annotations

import pytest

from clef_evals.metrics import (
    brier_multiclass,
    brier_score,
    ece,
    latency_percentiles,
    percentile,
)


class TestECE:
    def test_perfectly_calibrated_bins(self) -> None:
        # Bin [0.5, 0.6): two items, 1 of 2 correct -> |0.5 - 0.55| = 0.05.
        confidences = [0.51, 0.59, 0.9]
        correct = [False, True, True]
        value = ece(confidences, correct, n_bins=10)
        assert 0 <= value <= 1
        # Hand-computed: bin5 has 2 items acc 0.5 conf 0.55 gap 0.05 (weight 2/3),
        # bin9 has 1 item acc 1.0 conf 0.9 gap 0.1 (weight 1/3).
        assert value == pytest.approx((2 / 3) * 0.05 + (1 / 3) * 0.1)

    def test_confidence_1_all_correct_is_zero(self) -> None:
        assert ece([1.0, 1.0], [True, True], n_bins=1) == pytest.approx(0.0)

    def test_confidence_1_all_wrong_is_max_gap(self) -> None:
        assert ece([1.0, 1.0], [False, False], n_bins=1) == pytest.approx(1.0)

    def test_empty_inputs_return_zero(self) -> None:
        assert ece([], []) == 0.0

    def test_values_clipped_into_unit_interval(self) -> None:
        # 1.5 lands in the top bin, -0.2 in the bottom bin.
        value = ece([1.5, -0.2], [True, False], n_bins=2)
        assert 0 <= value <= 1

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            ece([0.5], [])

    def test_invalid_bin_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            ece([0.5], [True], n_bins=0)

    def test_single_bin_is_absolute_gap(self) -> None:
        assert ece([0.8, 0.8], [True, False], n_bins=1) == pytest.approx(0.3)


class TestBrierScore:
    def test_perfect_confident_correct(self) -> None:
        assert brier_score([1.0], [True]) == pytest.approx(0.0)

    def test_perfect_confident_wrong(self) -> None:
        assert brier_score([1.0], [False]) == pytest.approx(1.0)

    def test_zero_confidence_balanced(self) -> None:
        assert brier_score([0.5], [True]) == pytest.approx(0.25)

    def test_mean_over_items(self) -> None:
        assert brier_score([1.0, 1.0], [True, False]) == pytest.approx(0.5)

    def test_empty_inputs_return_zero(self) -> None:
        assert brier_score([], []) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            brier_score([1.0, 0.5], [True])


class TestBrierMulticlass:
    def test_perfect_one_hot_distributions(self) -> None:
        probabilities = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
        assert brier_multiclass(probabilities, ["a", "b"]) == pytest.approx(0.0)

    def test_confidently_wrong_scores_high(self) -> None:
        assert brier_multiclass([{"a": 1.0, "b": 0.0}], ["b"]) == pytest.approx(2.0)

    def test_missing_class_counts_as_zero_probability(self) -> None:
        # Gold class absent from the distribution costs the full (0-1)^2.
        assert brier_multiclass([{"a": 1.0}], ["b"]) == pytest.approx(2.0)

    def test_uniform_distribution(self) -> None:
        value = brier_multiclass([{"a": 0.5, "b": 0.5}], ["a"])
        assert value == pytest.approx((0.5 - 1.0) ** 2 + (0.5 - 0.0) ** 2)

    def test_empty_inputs_return_zero(self) -> None:
        assert brier_multiclass([], []) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            brier_multiclass([{"a": 1.0}], [])


class TestPercentile:
    def test_percentile_of_single_value(self) -> None:
        assert percentile([7.0], 99) == 7.0

    def test_median_interpolates(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_extremes(self) -> None:
        values = [10.0, 20.0, 30.0]
        assert percentile(values, 0) == pytest.approx(10.0)
        assert percentile(values, 100) == pytest.approx(30.0)

    def test_invalid_q_raises(self) -> None:
        with pytest.raises(ValueError, match="within"):
            percentile([1.0], 150)

    def test_empty_returns_zero(self) -> None:
        assert percentile([], 50) == 0.0


class TestLatencyPercentiles:
    def test_returns_all_three_buckets(self) -> None:
        values = list(range(1, 101))  # 1..100 ms
        result = latency_percentiles(values)
        assert set(result) == {"p50", "p95", "p99"}
        assert result["p50"] == pytest.approx(50.5)
        assert result["p95"] == pytest.approx(95.05)
        assert result["p99"] == pytest.approx(99.01)

    def test_empty_latencies(self) -> None:
        assert latency_percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
