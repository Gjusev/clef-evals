"""Shared fixtures: a mock Clef transport wired to real API response shapes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from clef_evals.client import ClefClient
from clef_evals.config import ClefConfig

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture by file name."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def envelope(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Clef result payload in the Cloudflare success envelope."""
    return {"result": result, "success": True, "errors": [], "messages": []}


def choice_result(
    choice: str = "billing",
    probabilities: dict[str, float] | None = None,
    tokens: tuple[int, int] = (120, 8),
) -> dict[str, Any]:
    """Build a Clef result with one choice answer under question id 'decision'."""
    if probabilities is None:
        probabilities = {"billing": 0.93, "technical": 0.05, "sales": 0.02}
    return {
        "model": "@cf/cloudflare/clef",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": probabilities.get(choice, 0.5),
            }
        },
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    }


def noul_result(probability: float = 0.87, tokens: tuple[int, int] = (90, 4)) -> dict[str, Any]:
    """Build a Clef result with one noul answer under question id 'is_true'."""
    return {
        "model": "@cf/cloudflare/clef",
        "answers": {"is_true": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    }


def typed_handler(
    choice: str = "billing",
    noul_probability: float = 0.87,
    status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Handler that answers each question with the right type based on the request."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers: dict[str, dict[str, object]] = {}
        for question_id, question in body["questions"].items():
            if question["type"] == "choice":
                result = choice_result(choice=choice)
                answers[question_id] = result["answers"]["decision"]
            else:
                result = noul_result(probability=noul_probability)
                answers[question_id] = result["answers"]["is_true"]
        return httpx.Response(
            status,
            json={
                "result": {
                    "model": "@cf/cloudflare/clef",
                    "answers": answers,
                    "usage": {"input_tokens": 100, "output_tokens": 8},
                },
                "success": status == 200,
                "errors": [],
                "messages": [],
            },
        )

    return handler


@pytest.fixture()
def config() -> ClefConfig:
    """Fast-retry config so failure-path tests stay quick."""
    return ClefConfig(
        account_id="test-account",
        api_token="test-token",
        backoff_initial=0.001,
        backoff_max=0.002,
    )


def build_client(config: ClefConfig, handler: Callable[[httpx.Request], httpx.Response]) -> ClefClient:
    """Build a ClefClient whose sync and async paths both use the mock handler."""
    transport = httpx.MockTransport(handler)
    return ClefClient(
        config,
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
    )


make_client = build_client


@pytest.fixture()
def mock_client(config: ClefConfig) -> Callable[..., ClefClient]:
    """Factory fixture building a client over httpx.MockTransport (sync and async)."""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> ClefClient:
        transport = httpx.MockTransport(handler)
        return ClefClient(
            config,
            http_client=httpx.Client(transport=transport),
            async_http_client=httpx.AsyncClient(transport=transport),
        )

    return factory


@pytest.fixture()
def judge(
    config: ClefConfig,
    mock_client: Callable[..., ClefClient],
) -> Callable[..., ClefJudge]:
    """Factory fixture building a ClefJudge over a mock transport."""

    from clef_evals.judge import ClefJudge

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> ClefJudge:
        return ClefJudge(config, client=mock_client(handler))

    return factory


from clef_evals.judge import ClefJudge  # noqa: E402  (used in type hints above)
