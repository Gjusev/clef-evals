"""Structured exception hierarchy for clef-evals.

Every error raised intentionally by this library derives from :class:`ClefError`
so callers can catch one base class. Errors carry structured ``details`` that
are safe to log (they never contain the API token).
"""

from __future__ import annotations

from typing import Any


class ClefError(Exception):
    """Base class for every error raised by clef-evals.

    Attributes:
        message: Human-readable description of the failure.
        details: Structured context safe to log; never contains secrets.
    """

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{key}={value!r}" for key, value in self.details.items())
        return f"{self.message} ({rendered})"


class ConfigurationError(ClefError):
    """Raised when configuration or environment variables are missing/invalid.

    ``details["problems"]`` lists every detected problem at once so users can
    fix their environment in a single pass.
    """


class ClefAPIError(ClefError):
    """Raised when the Cloudflare API returns an error response.

    Attributes:
        status_code: HTTP status code of the response, if any.
        error_code: Cloudflare error code from the ``errors[]`` envelope.
        retryable: Whether retrying the same request may succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: int | None = None,
        retryable: bool = False,
        **details: Any,
    ) -> None:
        super().__init__(message, **details)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable


class ClefAuthError(ClefAPIError):
    """Authentication or authorization failed (HTTP 401/403). Not retryable."""


class ClefRateLimitError(ClefAPIError):
    """The API rate limited the request (HTTP 429). Retryable.

    Attributes:
        retry_after: Seconds the API asked the client to wait, if provided.
    """

    def __init__(self, message: str, *, retry_after: float | None = None, **details: Any) -> None:
        super().__init__(message, retryable=True, **details)
        self.retry_after = retry_after


class ClefServerError(ClefAPIError):
    """Cloudflare returned a transient server error (HTTP 5xx). Retryable."""


class ClefResponseError(ClefError):
    """The API replied with a body that does not match the Clef schema."""


class ClefTimeoutError(ClefError):
    """A request timed out (optionally after exhausting retries). Retryable."""


class ClefNetworkError(ClefError):
    """A network-level failure occurred (DNS, connection refused, reset). Retryable."""
