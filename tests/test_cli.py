"""Tests for the CLI: gating, exit codes, JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import clef_evals.cli as cli
from clef_evals.exceptions import ClefError, ConfigurationError
from clef_evals.metrics import EvalResult


def make_result(**overrides: Any) -> EvalResult:
    defaults = dict(
        accuracy=0.9,
        ece=0.05,
        brier=0.08,
        brier_multiclass=0.2,
        n_samples=10,
        failures=0,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


class FakeJudge:
    instances: list[FakeJudge] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.evaluated_with: list[Any] = []
        FakeJudge.instances.append(self)

    def evaluate(self, eval_set: Any) -> EvalResult:
        self.evaluated_with.append(eval_set)
        return make_result()


class FailingJudge(FakeJudge):
    def evaluate(self, eval_set: Any) -> EvalResult:
        raise ClefError("boom")


@pytest.fixture(autouse=True)
def clef_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI runs need valid env config even though the judge is mocked."""
    monkeypatch.setenv("CLEF_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLEF_API_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def reset_instances() -> Any:
    FakeJudge.instances = []
    yield
    FakeJudge.instances = []


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "state": "refund please",
                    "instructions": "Which team?",
                    "choices": ["billing", "technical"],
                    "gold": "billing",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_run_passes_without_thresholds(
    dataset: Path, capsys: pytest.CaptureFixture
) -> None:
    import unittest.mock

    with unittest.mock.patch("clef_evals.cli.ClefJudge", FakeJudge):
        code = cli.main(["run", str(dataset)])
    assert code == 0
    out = capsys.readouterr().out
    assert "gate: PASSED" in out
    assert "accuracy=0.9000" in out


def test_run_gates_on_min_accuracy(dataset: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["run", str(dataset), "--min-accuracy", "0.95"])
    assert code == 1
    assert "gate: FAILED" in capsys.readouterr().out


def test_run_gates_on_max_ece(dataset: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["run", str(dataset), "--max-ece", "0.01"])
    assert code == 1


def test_run_gates_on_failures(dataset: Path) -> None:
    import unittest.mock

    class ZeroJudge(FakeJudge):
        def evaluate(self, eval_set: Any) -> EvalResult:
            return make_result(failures=2, n_samples=8)

    with unittest.mock.patch("clef_evals.cli.ClefJudge", ZeroJudge):
        code = cli.main(["run", str(dataset)])
    assert code == 1


def test_run_json_output_includes_gate(
    dataset: Path, capsys: pytest.CaptureFixture
) -> None:
    import unittest.mock

    with unittest.mock.patch("clef_evals.cli.ClefJudge", FakeJudge):
        code = cli.main(["run", str(dataset), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_passed"] is True
    assert payload["accuracy"] == 0.9


def test_run_writes_output_file(dataset: Path, tmp_path: Path) -> None:
    import unittest.mock

    output = tmp_path / "out.json"
    with unittest.mock.patch("clef_evals.cli.ClefJudge", FakeJudge):
        code = cli.main(["run", str(dataset), "--output", str(output)])
    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["accuracy"] == 0.9


def test_missing_dataset_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["run", str(tmp_path / "nope.json")])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_invalid_dataset_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert cli.main(["run", str(path)]) == 2


def test_configuration_error_exits_2(dataset: Path, capsys: pytest.CaptureFixture) -> None:
    import unittest.mock

    with unittest.mock.patch.object(
        cli.ClefConfig, "from_env", side_effect=ConfigurationError("missing", problems=["x"])
    ):
        code = cli.main(["run", str(dataset)])
    assert code == 2
    assert "configuration error" in capsys.readouterr().err


def test_clef_error_exits_2(dataset: Path, capsys: pytest.CaptureFixture) -> None:
    import unittest.mock

    with unittest.mock.patch("clef_evals.cli.ClefJudge", FailingJudge):
        code = cli.main(["run", str(dataset)])
    assert code == 2
    assert "clef error" in capsys.readouterr().err


def test_async_flag_uses_async_judge(dataset: Path) -> None:
    import unittest.mock

    captured: dict[str, type] = {}

    class FakeAsync(FakeJudge):
        async def evaluate(self, eval_set: Any, concurrency: int = 5) -> EvalResult:
            captured["type"] = type(self)
            return make_result()

    with unittest.mock.patch("clef_evals.cli.AsyncClefJudge", FakeAsync):
        code = cli.main(["run", str(dataset), "--async"])
    assert code == 0
    assert captured["type"] is FakeAsync


def test_version_flag(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "clef-evals 0.2.0" in capsys.readouterr().out


def test_quiet_suppresses_progress_logging(dataset: Path) -> None:
    import logging

    cli.main(["run", str(dataset), "--quiet"])
    assert logging.getLogger("clef_evals").level == logging.ERROR
