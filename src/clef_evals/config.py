"""Configuration handling with environment validation for clef-evals."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .exceptions import ConfigurationError

#: Default Workers AI base URL (Cloudflare API v4).
DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
#: Default hosted Clef model identifier.
DEFAULT_MODEL = "@cf/cloudflare/clef"


def _parse_float(name: str, raw: str, problems: list[str]) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{name} must be a number, got {raw!r}")
        return None
    if value <= 0:
        problems.append(f"{name} must be positive, got {value}")
        return None
    return value


def _parse_int(name: str, raw: str, problems: list[str]) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{name} must be an integer, got {raw!r}")
        return None
    if value < 0:
        problems.append(f"{name} must be >= 0, got {value}")
        return None
    return value


@dataclass(frozen=True)
class ClefConfig:
    """Immutable configuration for talking to the Clef API.

    Prefer :meth:`from_env` in applications; construct directly in tests.

    Attributes:
        account_id: Cloudflare account identifier.
        api_token: Cloudflare API token with Workers AI permissions.
        model: Hosted model id, ``@cf/cloudflare/clef`` or ``@cf/cloudflare/clef-flash``.
        base_url: Cloudflare API v4 root URL.
        timeout: Per-request timeout in seconds.
        max_retries: Retries per request for retryable failures (0 disables).
        backoff_initial: First backoff delay in seconds; doubles each retry.
        backoff_max: Upper bound for a single backoff delay in seconds.
        log_level: Default log level applied when :meth:`apply_logging` is called.
    """

    account_id: str
    api_token: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    max_retries: int = 2
    backoff_initial: float = 0.5
    backoff_max: float = 8.0
    log_level: str = "WARNING"
    extra: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClefConfig:
        """Build a config from environment variables, validating everything.

        Required variables are ``CLEF_ACCOUNT_ID`` and ``CLEF_API_TOKEN`` (the
        ``CLOUDFLARE_ACCOUNT_ID`` / ``CLOUDFLARE_API_TOKEN`` pair is accepted
        as a fallback). Raises :class:`ConfigurationError` listing *all*
        problems at once so misconfigurations are fixed in a single pass.

        Args:
            env: Environment mapping to read; defaults to ``os.environ``.

        Returns:
            A validated :class:`ClefConfig`.
        """
        source = os.environ if env is None else env
        problems: list[str] = []

        account_id = source.get("CLEF_ACCOUNT_ID") or source.get("CLOUDFLARE_ACCOUNT_ID") or ""
        api_token = source.get("CLEF_API_TOKEN") or source.get("CLOUDFLARE_API_TOKEN") or ""
        if not account_id:
            problems.append("CLEF_ACCOUNT_ID (or CLOUDFLARE_ACCOUNT_ID) is required")
        if not api_token:
            problems.append("CLEF_API_TOKEN (or CLOUDFLARE_API_TOKEN) is required")

        model = source.get("CLEF_MODEL", DEFAULT_MODEL)
        if model not in ("@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"):
            problems.append(
                f"CLEF_MODEL must be '@cf/cloudflare/clef' or '@cf/cloudflare/clef-flash', got {model!r}"
            )

        base_url = source.get("CLEF_BASE_URL", DEFAULT_BASE_URL)
        if not base_url.startswith(("http://", "https://")):
            problems.append(f"CLEF_BASE_URL must be an http(s) URL, got {base_url!r}")

        timeout_raw = source.get("CLEF_TIMEOUT")
        timeout = _parse_float("CLEF_TIMEOUT", timeout_raw, problems) if timeout_raw else 30.0
        retries_raw = source.get("CLEF_MAX_RETRIES")
        max_retries = _parse_int("CLEF_MAX_RETRIES", retries_raw, problems) if retries_raw else 2

        log_level = source.get("CLEF_LOG_LEVEL", "WARNING").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            problems.append(f"CLEF_LOG_LEVEL must be a standard level name, got {log_level!r}")

        if problems:
            raise ConfigurationError(
                "Invalid Clef configuration", problems=problems
            )
        return cls(
            account_id=account_id,
            api_token=api_token,
            model=model,
            base_url=base_url.rstrip("/"),
            timeout=timeout if timeout is not None else 30.0,
            max_retries=max_retries if max_retries is not None else 2,
            log_level=log_level,
        )

    def with_overrides(self, **changes: object) -> ClefConfig:
        """Return a copy with the given fields replaced (None values ignored).

        Useful for explicit keyword arguments to take precedence over env
        defaults: ``config.with_overrides(model="...") if model else config``.
        """
        effective = {key: value for key, value in changes.items() if value is not None}
        return replace(self, **effective)  # type: ignore[arg-type]

    @property
    def model_selector(self) -> str:
        """Body-level model selector required by the API: ``clef`` or ``clef-flash``."""
        return self.model.rsplit("/", 1)[-1]

    @property
    def run_url(self) -> str:
        """Full Workers AI run endpoint for this account and model."""
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"

    def apply_logging(self, logger: logging.Logger | None = None) -> None:
        """Route clef_evals log records at the configured level.

        Only touches the ``clef_evals`` logger hierarchy, never the root
        logger, so host applications keep full control.
        """
        target = logger or logging.getLogger("clef_evals")
        target.setLevel(getattr(logging, self.log_level, logging.WARNING))
