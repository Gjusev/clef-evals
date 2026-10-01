"""Tests for the standalone regression gate script."""

from __future__ import annotations

import pytest

from scripts.check_regression import check_metric, dig, main

CURRENT = {"accuracy": 0.91, "ece": 0.08, "runs": [{"accuracy": 0.93}]}
BASELINE = {"accuracy": 0.90, "ece": 0.07, "runs": [{"accuracy": 0.92}]}


class TestDig:
    def test_dict_path(self) -> None:
        assert dig(CURRENT, "ece") == 0.08

    def test_list_index_path(self) -> None:
        assert dig(CURRENT, "runs.0.accuracy") == 0.93

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="latency"):
            dig(CURRENT, "latency.p95")

    def test_bad_index_raises(self) -> None:
        with pytest.raises(KeyError, match="bad index"):
            dig(CURRENT, "runs.5.accuracy")


class TestCheckMetric:
    def test_min_passes_within_tolerance(self) -> None:
        assert check_metric(CURRENT, BASELINE, "accuracy:min:0.03") is None

    def test_min_fails_beyond_tolerance(self) -> None:
        assert "regression" in check_metric({"accuracy": 0.80}, BASELINE, "accuracy:min:0.03")

    def test_max_passes_within_tolerance(self) -> None:
        assert check_metric(CURRENT, BASELINE, "ece:max") is None

    def test_max_fails_beyond_tolerance(self) -> None:
        assert "regression" in check_metric({"ece": 0.5}, BASELINE, "ece:max")

    def test_invalid_goal_raises(self) -> None:
        with pytest.raises(ValueError, match="path:max|min"):
            check_metric(CURRENT, BASELINE, "accuracy:up")

    def test_invalid_spec_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="path:max|min"):
            check_metric(CURRENT, BASELINE, "accuracy")


class TestMain:
    def test_passing_gate_exits_zero(self, tmp_path: object) -> None:
        import pathlib

        cur = pathlib.Path(str(tmp_path)) / "current.json"
        base = pathlib.Path(str(tmp_path)) / "baseline.json"
        cur.write_text('{"accuracy": 0.91}', encoding="utf-8")
        base.write_text('{"accuracy": 0.90}', encoding="utf-8")
        assert main(["--current", str(cur), "--baseline", str(base), "--metric", "accuracy:min"]) == 0

    def test_failing_gate_exits_one(self, tmp_path: object) -> None:
        import pathlib

        cur = pathlib.Path(str(tmp_path)) / "current.json"
        base = pathlib.Path(str(tmp_path)) / "baseline.json"
        cur.write_text('{"accuracy": 0.10}', encoding="utf-8")
        base.write_text('{"accuracy": 0.90}', encoding="utf-8")
        assert main(["--current", str(cur), "--baseline", str(base), "--metric", "accuracy:min"]) == 1

    def test_missing_file_exits_two(self, tmp_path: object) -> None:
        import pathlib

        base = pathlib.Path(str(tmp_path)) / "baseline.json"
        base.write_text("{}", encoding="utf-8")
        assert main(["--current", "nope.json", "--baseline", str(base), "--metric", "accuracy:min"]) == 2
