"""Tests for the Clef HTTP client: retries, error translation, async paths."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from conftest import load_fixture

from clef_evals.client import ClefClient
from clef_evals.config import ClefConfig
from clef_evals.exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
)
from clef_evals.models import Noul

QUESTIONS = {"q": Noul(instructions="Is this urgent?")}


class CountingHandler:
    """Handler that serves scripted responses and records every request."""

    def __init__(self, responses: list[Callable[[int], httpx.Response]]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index](len(self.requests))


def success_with_fixture(request_index: int) -> httpx.Response:
    return httpx.Response(200, json=load_fixture("choice_response.json"))


class TestSuccessfulRequest:
    def test_parses_fixture_response(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([success_with_fixture])
        client = mock_client(handler)
        response = client.run("a state", QUESTIONS)
        assert response.model == "@cf/cloudflare/clef"
        assert "decision" in response.answers
        assert response.usage.input_tokens == 120

    def test_request_carries_auth_and_payload(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([success_with_fixture])
        mock_client(handler).run("the state", QUESTIONS)
        request = handler.requests[0]
        assert request.headers["Authorization"] == "Bearer test-token"
        body = json.loads(request.content)
        assert body["model"] == "clef"
        assert body["state"] == "the state"
        assert body["questions"]["q"]["type"] == "noul"
        assert "/accounts/test-account/ai/run/@cf/cloudflare/clef" in str(request.url)

    def test_latency_is_measured(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        response = mock_client(CountingHandler([success_with_fixture])).run("s", QUESTIONS)
        assert response.latency_ms >= 0.0


class TestErrorTranslation:
    def test_401_raises_auth_error_without_retry(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler(
            [lambda _: httpx.Response(401, json=load_fixture("error_envelope.json"))]
        )
        with pytest.raises(ClefAuthError) as excinfo:
            mock_client(handler).run("s", QUESTIONS)
        assert excinfo.value.status_code == 401
        assert "Unauthorized" in excinfo.value.message
        assert len(handler.requests) == 1

    def test_403_maps_to_auth_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(403, json={"errors": []})])
        with pytest.raises(ClefAuthError):
            mock_client(handler).run("s", QUESTIONS)

    def test_404_is_non_retryable_api_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler(
            [lambda _: httpx.Response(404, json={"errors": [{"code": 7000, "message": "no route"}]})]
        )
        with pytest.raises(ClefAPIError) as excinfo:
            mock_client(handler).run("s", QUESTIONS)
        assert excinfo.value.error_code == 7000
        assert excinfo.value.retryable is False
        assert len(handler.requests) == 1

    def test_500_raises_server_error_after_retries(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(500, json={"errors": []})])
        with pytest.raises(ClefServerError) as excinfo:
            mock_client(handler).run("s", QUESTIONS)
        assert excinfo.value.retryable is True
        assert len(handler.requests) == config.max_retries + 1

    def test_500_then_success_recovers(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler(
            [lambda _: httpx.Response(500, json={"errors": []}), success_with_fixture]
        )
        response = mock_client(handler).run("s", QUESTIONS)
        assert "decision" in response.answers
        assert len(handler.requests) == 2

    def test_429_honors_retry_after_header(
        self,
        config: ClefConfig,
        mock_client: Callable[..., ClefClient],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("clef_evals.client.time.sleep", sleeps.append)
        handler = CountingHandler(
            [
                lambda _: httpx.Response(
                    429, json={"errors": []}, headers={"Retry-After": "1.5"}
                ),
                success_with_fixture,
            ]
        )
        response = mock_client(handler).run("s", QUESTIONS)
        assert "decision" in response.answers
        assert sleeps == [1.5]

    def test_429_exhausting_retries_raises_rate_limit(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(429, json={"errors": []})])
        with pytest.raises(ClefRateLimitError):
            mock_client(handler).run("s", QUESTIONS)
        assert len(handler.requests) == config.max_retries + 1

    def test_non_json_error_body_still_raises_server_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(502, text="bad gateway")])
        with pytest.raises(ClefServerError):
            mock_client(handler).run("s", QUESTIONS)

    def test_success_false_envelope_raises_api_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        body = {"result": None, "success": False, "errors": [{"code": 1, "message": "bad"}]}
        handler = CountingHandler([lambda _: httpx.Response(200, json=body)])
        with pytest.raises(ClefAPIError, match="bad"):
            mock_client(handler).run("s", QUESTIONS)

    def test_malformed_json_on_success_raises_response_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(200, text="not json")])
        with pytest.raises(ClefResponseError, match="not JSON"):
            mock_client(handler).run("s", QUESTIONS)

    def test_timeout_maps_to_timeout_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        def raise_timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("too slow", request=request)

        handler = CountingHandler([raise_timeout])
        with pytest.raises(ClefTimeoutError):
            mock_client(handler).run("s", QUESTIONS)
        assert len(handler.requests) == config.max_retries + 1

    def test_network_error_maps_to_network_error(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        def raise_connect_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        handler = CountingHandler([raise_connect_error])
        with pytest.raises(ClefNetworkError):
            mock_client(handler).run("s", QUESTIONS)

    def test_timeout_then_success_recovers(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        def raise_timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        handler = CountingHandler([raise_timeout, success_with_fixture])
        response = mock_client(handler).run("s", QUESTIONS)
        assert "decision" in response.answers


class TestLifecycle:
    def test_close_does_not_close_injected_client(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([success_with_fixture])
        injected = httpx.Client(transport=httpx.MockTransport(handler))
        client = ClefClient(config, http_client=injected)
        client.close()
        assert not injected.is_closed

    def test_close_closes_owned_client(self, config: ClefConfig) -> None:
        client = ClefClient(config)
        owned = client._sync_client()
        client.close()
        assert owned.is_closed

    def test_sync_context_manager(self, config: ClefConfig) -> None:
        with ClefClient(config) as client:
            assert client._sync_client() is not None
        assert client._client is None  # reset after close

    def test_async_context_manager(self, config: ClefConfig) -> None:
        async def scenario() -> None:
            async with ClefClient(config) as client:
                client._get_async_client()
            assert client._async_client is None

        asyncio.run(scenario())

    def test_aclose_closes_async_client(self, config: ClefConfig) -> None:
        async def scenario() -> None:
            client = ClefClient(config)
            client._get_async_client()
            await client.aclose()
            assert client._async_client is None

        asyncio.run(scenario())


class TestAsyncRun:
    def test_arun_success(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([success_with_fixture])
        client = mock_client(handler)

        async def scenario() -> None:
            response = await client.arun("s", QUESTIONS)
            assert "decision" in response.answers

        asyncio.run(scenario())
        assert handler.requests[0].headers["Authorization"] == "Bearer test-token"

    def test_arun_retries_server_errors(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler(
            [lambda _: httpx.Response(503, json={"errors": []}), success_with_fixture]
        )
        client = mock_client(handler)

        async def scenario() -> None:
            response = await client.arun("s", QUESTIONS)
            assert "decision" in response.answers

        asyncio.run(scenario())
        assert len(handler.requests) == 2

    def test_arun_raises_after_exhausting_rate_limit(
        self, config: ClefConfig, mock_client: Callable[..., ClefClient]
    ) -> None:
        handler = CountingHandler([lambda _: httpx.Response(429, json={"errors": []})])
        client = mock_client(handler)

        async def scenario() -> None:
            await client.arun("s", QUESTIONS)

        with pytest.raises(ClefRateLimitError):
            asyncio.run(scenario())


class TestBackoffMath:
    def test_backoff_grows_and_caps(self) -> None:
        from clef_evals.client import _backoff_delay

        for attempt, expected_max in [(0, 0.5), (1, 1.0), (2, 2.0), (10, 8.0)]:
            delay = _backoff_delay(attempt, initial=0.5, maximum=8.0)
            assert 0.25 * expected_max <= delay <= expected_max

    def test_retry_after_parsing(self) -> None:
        from clef_evals.client import _parse_retry_after

        headers = httpx.Headers({"Retry-After": "2"})
        assert _parse_retry_after(headers) == 2.0
        assert _parse_retry_after(httpx.Headers({})) is None
        assert _parse_retry_after(httpx.Headers({"Retry-After": "bogus"})) is None
        assert _parse_retry_after(httpx.Headers({"Retry-After": "999"})) == 30.0
