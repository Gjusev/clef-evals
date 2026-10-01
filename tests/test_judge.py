"""Tests for the judge layer: single calls, evaluate(), datasets, async."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import envelope, make_client, typed_handler

from clef_evals.client import ClefClient
from clef_evals.config import ClefConfig
from clef_evals.exceptions import ClefError, ConfigurationError
from clef_evals.judge import AsyncClefJudge, ClefJudge, load_eval_set

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
EVAL_SET_PATH = FIXTURE_DIR / "eval_set.json"

EVAL_SET = load_eval_set(EVAL_SET_PATH)


@pytest.fixture()
def always_billing(config: ClefConfig) -> Callable[..., ClefJudge]:
    """Judge whose API always answers 'billing' for choices and 0.87 for noul."""
    handler = typed_handler(choice="billing", noul_probability=0.87)
    client = ClefClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return ClefJudge(config, client=client)


class TestSingleCalls:
    def test_judge_choice_returns_decision(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = typed_handler(choice="billing")
        judge = ClefJudge(config, client=mock_client(handler))
        decision = judge.judge_choice("state", "Which team?", {"billing": "b", "tech": "t"})
        assert decision.choice == "billing"
        assert decision.probabilities["billing"] == pytest.approx(0.93)
        assert decision.usage.input_tokens == 100
        assert decision.latency_ms >= 0.0

    def test_judge_binary_returns_probability(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        judge = ClefJudge(config, client=mock_client(typed_handler(noul_probability=0.87)))
        assert judge.judge_binary("state", "Is this urgent?") == pytest.approx(0.87)

    def test_wrong_answer_type_raises(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:

        # Responds with a noul answer under the choice question id.
        mismatch = envelope(
            {
                "model": "@cf/cloudflare/clef",
                "answers": {"decision": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )
        judge = ClefJudge(
            config, client=mock_client(lambda request: httpx.Response(200, json=mismatch))
        )
        with pytest.raises(ClefError, match="choice answer"):
            judge.judge_choice("s", "Which?", {"a": "1", "b": "2"})

    def test_run_questions_returns_full_response(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        from clef_evals.models import Choice, Noul

        judge = ClefJudge(config, client=mock_client(typed_handler()))
        response = judge.run_questions(
            "state",
            {
                "decision": Choice(instructions="team?", criteria={"a": "1", "b": "2"}),
                "extra": Noul(instructions="yes?"),
            },
        )
        assert set(response.answers) == {"decision", "extra"}


class TestEvaluate:
    def test_mixed_dataset_metrics(self, always_billing: ClefJudge) -> None:
        result = always_billing.evaluate(EVAL_SET)
        # billing/technical/sales choices + urgent(True)/not-urgent(False) binary.
        assert result.n_samples == 5
        assert result.failures == 0
        assert result.accuracy == pytest.approx(2 / 5)
        assert result.correct == [True, False, False, True, False]
        assert result.predictions == ["billing", "billing", "billing", True, True]
        assert result.input_tokens == 5 * 100  # uniform mock usage per call
        assert result.model == "@cf/cloudflare/clef"
        # Multiclass Brier only spans the three choice items.
        assert result.brier_multiclass >= 0.0
        assert result.latency_p99 >= result.latency_p95 >= result.latency_p50

    def test_cost_per_1k_calls_uses_published_price(self, always_billing: ClefJudge) -> None:
        result = always_billing.evaluate(EVAL_SET)
        mean_input = result.input_tokens / result.n_samples
        assert result.cost_usd_per_1k_calls == pytest.approx(mean_input * 1000 * 0.24 / 1e6)

    def test_failed_items_are_counted_not_fatal(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        def failing_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if "Android app has been crashing" in json.dumps(body.get("state", "")):
                # Deterministic 500: even retries fail for this one item.
                return httpx.Response(500, json={"errors": []})
            answers = {}
            for question_id, question in body["questions"].items():
                if question["type"] == "choice":
                    answers[question_id] = {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.9, "technical": 0.06, "sales": 0.04},
                        "confidence": 0.9,
                    }
                else:
                    answers[question_id] = {"type": "noul", "noul": 0.87}
            return httpx.Response(
                200,
                json={
                    "result": {
                        "model": "@cf/cloudflare/clef",
                        "answers": answers,
                        "usage": {"input_tokens": 100, "output_tokens": 8},
                    },
                    "success": True,
                    "errors": [],
                    "messages": [],
                },
            )

        judge = ClefJudge(config, client=mock_client(failing_handler))
        result = judge.evaluate(EVAL_SET)
        assert result.failures == 1
        assert result.n_samples == 4

    def test_all_items_failing_yields_empty_result(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        judge = ClefJudge(
            config,
            client=mock_client(lambda request: httpx.Response(500, json={"errors": []})),
        )
        result = judge.evaluate(EVAL_SET)
        assert result.n_samples == 0
        assert result.accuracy == 0.0
        assert result.failures == 5

    def test_invalid_items_raise_before_any_api_call(
        self, always_billing: ClefJudge
    ) -> None:
        with pytest.raises(ValueError, match="missing keys"):
            always_billing.evaluate([{"state": "s", "instructions": "i"}])
        with pytest.raises(ValueError, match="criteria"):
            always_billing.evaluate(
                [{"state": "s", "instructions": "i", "gold": "billing"}]
            )

    def test_to_dict_round_trips(self, always_billing: ClefJudge) -> None:
        result = always_billing.evaluate(EVAL_SET)
        payload = result.to_dict()
        assert payload["accuracy"] == result.accuracy
        assert len(payload["confidences"]) == result.n_samples
        assert "gate" not in payload  # report data only


class TestConstruction:
    def test_kwargs_override_env_config(
        self, config: ClefConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("CLEF_ACCOUNT_ID", "CLEF_API_TOKEN", "CLEF_MODEL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CLEF_ACCOUNT_ID", "env-acc")
        monkeypatch.setenv("CLEF_API_TOKEN", "env-tok")
        judge = ClefJudge(model="@cf/cloudflare/clef-flash")
        assert judge.config.account_id == "env-acc"
        assert judge.config.model == "@cf/cloudflare/clef-flash"

    def test_missing_env_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("CLEF_ACCOUNT_ID", "CLEF_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(ConfigurationError):
            ClefJudge()

    def test_close_on_injected_client_is_noop(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        client = mock_client(typed_handler())
        judge = ClefJudge(config, client=client)
        judge.close()  # must not raise; injected client left usable
        client.run("s", {"q": __import__("clef_evals").Noul(instructions="y?")})

    def test_context_manager(self, config: ClefConfig) -> None:
        with ClefJudge(config) as judge:
            assert judge.client is not None
        # Owned client closed: internal handle reset.


class TestLoadEvalSet:
    def test_loads_json_fixture(self) -> None:
        items = load_eval_set(EVAL_SET_PATH)
        assert len(items) == 5
        assert items[0].is_binary is False
        assert items[3].is_binary is True

    def test_loads_jsonl(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "set.jsonl"
        path.write_text(
            json.dumps({"state": "s", "instructions": "i", "gold": True}) + "\n\n"
            + json.dumps({"state": "s2", "instructions": "i2", "choices": ["a", "b"], "gold": "a"}),
            encoding="utf-8",
        )
        items = load_eval_set(path)
        assert len(items) == 2
        assert items[1].criteria() == {"a": "", "b": ""}

    def test_non_array_json_raises(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "bad.json"
        path.write_text('{"state": "x"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_eval_set(path)

    def test_missing_required_keys_raise(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "set.json"
        path.write_text(json.dumps([{"state": "s"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys"):
            load_eval_set(path)


class TestAsyncEvaluate:
    def test_async_matches_sync_results(self, config: ClefConfig) -> None:
        sync_judge = ClefJudge(
            config, client=make_client(config, typed_handler())
        )
        async_judge = AsyncClefJudge(
            config, client=make_client(config, typed_handler())
        )
        sync_result = sync_judge.evaluate(EVAL_SET)
        async_result = asyncio.run(async_judge.evaluate(EVAL_SET, concurrency=4))
        assert async_result.correct == sync_result.correct
        assert async_result.confidences == sync_result.confidences
        assert async_result.input_tokens == sync_result.input_tokens

    def test_async_counts_failures_without_raising(self, config: ClefConfig) -> None:
        async_judge = AsyncClefJudge(
            config,
            client=make_client(config, lambda request: httpx.Response(500, json={"errors": []})),
        )
        result = asyncio.run(async_judge.evaluate(EVAL_SET))
        assert result.failures == 5
        assert result.n_samples == 0

    def test_async_single_question_helpers(self, config: ClefConfig) -> None:

        judge = AsyncClefJudge(
            config, client=make_client(config, typed_handler())
        )

        async def scenario() -> None:
            probability = await judge.judge_binary_async("s", "yes?")
            assert probability == pytest.approx(0.87)
            decision = await judge.judge_choice_async("s", "which?", {"a": "1", "b": "2"})
            assert decision.choice == "billing"

        asyncio.run(scenario())
