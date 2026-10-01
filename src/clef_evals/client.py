"""HTTP client for the Clef API with retries, backoff, and structured errors."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Mapping
from typing import Any

import httpx

from .config import ClefConfig
from .exceptions import (
    ClefAPIError,
    ClefAuthError,
    ClefNetworkError,
    ClefRateLimitError,
    ClefResponseError,
    ClefServerError,
    ClefTimeoutError,
)
from .models import ClefResponse, Question, build_payload, parse_clef_response

logger = logging.getLogger("clef_evals.client")

#: HTTP statuses that are always worth retrying with backoff.
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 522, 524})
#: Maximum seconds a server-provided Retry-After may delay a retry.
_MAX_RETRY_AFTER = 30.0


def _backoff_delay(attempt: int, initial: float, maximum: float) -> float:
    """Exponential backoff with full jitter for the given zero-based attempt."""
    delay = min(maximum, initial * (2**attempt))
    return delay * (0.5 + random.random() / 2)


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """Extract a clamped Retry-After delay in seconds, if the header is sane."""
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, min(_MAX_RETRY_AFTER, float(raw)))
    except ValueError:
        return None


def _raise_for_error_envelope(status_code: int, body: Any, response_headers: httpx.Headers) -> None:
    """Translate a Cloudflare error envelope into a typed exception."""
    message = "Clef API request failed"
    error_code: int | None = None
    if isinstance(body, Mapping):
        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, Mapping):
                message = str(first.get("message", message))
                raw_code = first.get("code")
                if isinstance(raw_code, int):
                    error_code = raw_code
    if status_code in (401, 403):
        raise ClefAuthError(message, status_code=status_code, error_code=error_code)
    if status_code == 429:
        raise ClefRateLimitError(
            message,
            status_code=status_code,
            error_code=error_code,
            retry_after=_parse_retry_after(response_headers),
        )
    if status_code >= 500:
        raise ClefServerError(message, status_code=status_code, error_code=error_code, retryable=True)
    raise ClefAPIError(
        message,
        status_code=status_code,
        error_code=error_code,
        retryable=status_code in _RETRYABLE_STATUSES,
    )


class ClefClient:
    """Synchronous and asynchronous client for the Clef decision API.

    The client owns retry policy, timeouts, latency measurement, and error
    translation; it never prints and never logs the API token.

    Args:
        config: Validated :class:`clef_evals.config.ClefConfig`.
        http_client: Optional pre-built ``httpx.Client`` (tests inject a
            ``httpx.MockTransport``-backed client here). Created when omitted.
        async_http_client: Optional pre-built ``httpx.AsyncClient`` for
            :meth:`arun`; created lazily when omitted.
    """

    def __init__(
        self,
        config: ClefConfig,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = http_client
        self._async_client = async_http_client
        self._owns_sync = http_client is None
        self._owns_async = async_http_client is None

    # -- lifecycle -------------------------------------------------------------

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.timeout)
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._async_client

    def close(self) -> None:
        """Close clients owned by this instance (injected clients are left open)."""
        if self._owns_sync and self._client is not None:
            self._client.close()
            self._client = None
        if self._async_client is not None:
            # Closing a loop-bound client outside its loop is unsafe; close it
            # only when no event loop claims it (the sync-close path).
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._async_client.aclose())
                self._async_client = None

    async def aclose(self) -> None:
        """Async counterpart of :meth:`close` for :meth:`arun` users."""
        if self._owns_sync and self._client is not None:
            self._client.close()
            self._client = None
        if self._owns_async and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> ClefClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> ClefClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- public API ------------------------------------------------------------

    def run(
        self,
        state: Any,
        questions: Mapping[str, Question],
        images: list[dict[str, str]] | None = None,
    ) -> ClefResponse:
        """Send one decision request and return the parsed response.

        Retries retryable failures (rate limits, server errors, network and
        timeout errors) with exponential backoff + jitter, honoring
        ``Retry-After`` for rate limits.

        Args:
            state: Content to evaluate (string or structured JSON data).
            questions: Typed questions keyed by question id.
            images: Optional embedded images (see API docs for constraints).

        Returns:
            The parsed :class:`clef_evals.models.ClefResponse`.

        Raises:
            ClefAuthError: Credentials were rejected (not retried).
            ClefAPIError: Non-retryable API error after all retries.
            ClefRateLimitError: Still rate limited after all retries.
            ClefServerError: Server error persisted after all retries.
            ClefTimeoutError: Request timed out after all retries.
            ClefNetworkError: Network failure persisted after all retries.
            ClefResponseError: Response body did not match the Clef schema.
        """
        payload = build_payload(self.config.model_selector, state, questions, images)
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._run_once(payload)
            except (ClefRateLimitError, ClefServerError, ClefTimeoutError, ClefNetworkError) as error:
                last_error = error
                if attempt == attempts - 1:
                    break
                self._sleep_before_retry(attempt, error)
        assert last_error is not None  # loop runs at least once
        raise last_error

    async def arun(
        self,
        state: Any,
        questions: Mapping[str, Question],
        images: list[dict[str, str]] | None = None,
    ) -> ClefResponse:
        """Async counterpart of :meth:`run` with identical retry semantics."""
        payload = build_payload(self.config.model_selector, state, questions, images)
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._arun_once(payload)
            except (ClefRateLimitError, ClefServerError, ClefTimeoutError, ClefNetworkError) as error:
                last_error = error
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(self._delay_for(attempt, error))
        assert last_error is not None
        raise last_error

    # -- internals -------------------------------------------------------------

    def _delay_for(self, attempt: int, error: Exception) -> float:
        if isinstance(error, ClefRateLimitError) and error.retry_after is not None:
            return error.retry_after
        return _backoff_delay(attempt, self.config.backoff_initial, self.config.backoff_max)

    def _sleep_before_retry(self, attempt: int, error: Exception) -> None:
        delay = self._delay_for(attempt, error)
        logger.warning(
            "Clef request failed (attempt %d, retrying in %.2fs): %s",
            attempt + 1,
            delay,
            error,
        )
        time.sleep(delay)

    def _run_once(self, payload: dict[str, Any]) -> ClefResponse:
        started = time.perf_counter()
        try:
            response = self._sync_client().post(
                self.config.run_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_token}"},
            )
        except httpx.TimeoutException as error:
            raise ClefTimeoutError(f"request timed out after {self.config.timeout}s: {error}") from error
        except httpx.HTTPError as error:
            raise ClefNetworkError(f"network error: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._interpret(response, latency_ms)

    async def _arun_once(self, payload: dict[str, Any]) -> ClefResponse:
        started = time.perf_counter()
        try:
            response = await self._get_async_client().post(
                self.config.run_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_token}"},
            )
        except httpx.TimeoutException as error:
            raise ClefTimeoutError(f"request timed out after {self.config.timeout}s: {error}") from error
        except httpx.HTTPError as error:
            raise ClefNetworkError(f"network error: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._interpret(response, latency_ms)

    def _interpret(self, response: httpx.Response, latency_ms: float) -> ClefResponse:
        logger.debug("Clef API responded status=%d latency_ms=%.1f", response.status_code, latency_ms)
        if response.status_code != 200:
            try:
                body: Any = response.json()
            except ValueError:
                body = None
            _raise_for_error_envelope(response.status_code, body, response.headers)
        try:
            envelope = response.json()
        except ValueError as error:
            raise ClefResponseError(f"response body is not JSON: {error}") from error
        if not isinstance(envelope, Mapping) or envelope.get("success") is not True:
            body = envelope if isinstance(envelope, Mapping) else {}
            _raise_for_error_envelope(response.status_code, body, response.headers)
        result = envelope.get("result")
        return parse_clef_response(result if isinstance(result, Mapping) else {}, latency_ms)
