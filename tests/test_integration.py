"""Integration tests against the real Cloudflare Clef API.

These call the real API and consume tokens. They are excluded by default:
run them explicitly with ``pytest -m integration`` after exporting
CLEF_ACCOUNT_ID and CLEF_API_TOKEN (or CLOUDFLARE_ACCOUNT_ID /
CLOUDFLARE_API_TOKEN).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("CLEF_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        )
        or not (
            os.environ.get("CLEF_API_TOKEN") or os.environ.get("CLOUDFLARE_API_TOKEN")
        ),
        reason="CLEF_ACCOUNT_ID / CLEF_API_TOKEN not configured",
    ),
]


def test_real_choice_decision() -> None:
    from clef_evals import ClefJudge

    with ClefJudge() as judge:
        decision = judge.judge_choice(
            "Email: I need a refund for my double-charged invoice.",
            "Which team should handle this?",
            {"billing": "Payments, invoices, refunds", "technical": "Bugs and outages"},
        )
    assert decision.choice in {"billing", "technical"}
    assert 0.0 <= decision.probabilities[decision.choice] <= 1.0


def test_real_evaluate_small_set() -> None:
    from clef_evals import ClefJudge

    items = [
        {
            "state": "Email: I need a refund",
            "instructions": "Which team?",
            "choices": ["billing", "technical"],
            "gold": "billing",
        },
        {
            "state": "Email: the app crashes on launch",
            "instructions": "Which team?",
            "choices": ["billing", "technical"],
            "gold": "technical",
        },
    ]
    with ClefJudge() as judge:
        result = judge.evaluate(items)
    assert result.n_samples == 2
    assert result.failures == 0
    assert 0.0 <= result.accuracy <= 1.0
