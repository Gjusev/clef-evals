"""Tests for ClefConfig: env parsing, validation, and derived properties."""

from __future__ import annotations

import logging

import pytest

from clef_evals.config import ClefConfig
from clef_evals.exceptions import ConfigurationError

BASE_ENV = {
    "CLEF_ACCOUNT_ID": "acc-123",
    "CLEF_API_TOKEN": "tok-456",
}


def test_from_env_with_required_vars() -> None:
    config = ClefConfig.from_env(BASE_ENV)
    assert config.account_id == "acc-123"
    assert config.api_token == "tok-456"
    assert config.model == "@cf/cloudflare/clef"
    assert config.timeout == 30.0
    assert config.max_retries == 2


def test_from_env_accepts_cloudflare_fallbacks() -> None:
    config = ClefConfig.from_env(
        {"CLOUDFLARE_ACCOUNT_ID": "cf-acc", "CLOUDFLARE_API_TOKEN": "cf-tok"}
    )
    assert config.account_id == "cf-acc"
    assert config.api_token == "cf-tok"


def test_from_env_missing_everything_reports_all_problems() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        ClefConfig.from_env({})
    problems = excinfo.value.details["problems"]
    assert len(problems) == 2
    assert "CLEF_ACCOUNT_ID" in problems[0]
    assert "CLEF_API_TOKEN" in problems[1]


def test_from_env_collects_multiple_problems_at_once() -> None:
    env = {
        "CLEF_ACCOUNT_ID": "acc",
        "CLEF_API_TOKEN": "tok",
        "CLEF_MODEL": "gpt-4",
        "CLEF_TIMEOUT": "soon",
        "CLEF_MAX_RETRIES": "-3",
        "CLEF_LOG_LEVEL": "LOUD",
    }
    with pytest.raises(ConfigurationError) as excinfo:
        ClefConfig.from_env(env)
    problems = excinfo.value.details["problems"]
    assert len(problems) == 4


def test_from_env_parses_numeric_overrides() -> None:
    env = {**BASE_ENV, "CLEF_TIMEOUT": "5.5", "CLEF_MAX_RETRIES": "4", "CLEF_LOG_LEVEL": "debug"}
    config = ClefConfig.from_env(env)
    assert config.timeout == 5.5
    assert config.max_retries == 4
    assert config.log_level == "DEBUG"


def test_from_env_rejects_non_positive_timeout() -> None:
    with pytest.raises(ConfigurationError):
        ClefConfig.from_env({**BASE_ENV, "CLEF_TIMEOUT": "0"})


def test_with_overrides_ignores_none_values() -> None:
    config = ClefConfig(account_id="a", api_token="t")
    overridden = config.with_overrides(model="@cf/cloudflare/clef-flash", timeout=None)
    assert overridden.model == "@cf/cloudflare/clef-flash"
    assert overridden.timeout == config.timeout


def test_with_overrides_keeps_original_untouched() -> None:
    config = ClefConfig(account_id="a", api_token="t")
    config.with_overrides(max_retries=9)
    assert config.max_retries == 2


def test_model_selector_derives_body_selector() -> None:
    assert ClefConfig(account_id="a", api_token="t").model_selector == "clef"
    flash = ClefConfig(account_id="a", api_token="t", model="@cf/cloudflare/clef-flash")
    assert flash.model_selector == "clef-flash"


def test_run_url_contains_account_and_model() -> None:
    config = ClefConfig(account_id="acc", api_token="t")
    assert config.run_url == (
        "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef"
    )


def test_apply_logging_sets_level_only_for_library_logger() -> None:
    logger = logging.getLogger("clef_evals")
    original = logger.level
    try:
        config = ClefConfig(account_id="a", api_token="t", log_level="INFO")
        config.apply_logging()
        assert logger.level == logging.INFO
        assert logging.getLogger().level == logging.WARNING
    finally:
        logger.setLevel(original)
