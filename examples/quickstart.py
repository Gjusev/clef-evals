"""Copy-paste quickstart for clef-evals.

Requires CLEF_ACCOUNT_ID and CLEF_API_TOKEN in the environment:
    python examples/quickstart.py
"""

from clef_evals import ClefJudge, brier_score, ece

EVAL_SET = [
    {
        "state": "Email: I was charged twice for my monthly subscription.",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments, invoices, refunds",
            "technical": "Outages, bugs, configuration",
            "sales": "Plans, pricing, upgrades",
        },
        "gold": "billing",
    },
    {
        "state": "Email: The Android app crashes on launch since the 2.4 update.",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments, invoices, refunds",
            "technical": "Outages, bugs, configuration",
            "sales": "Plans, pricing, upgrades",
        },
        "gold": "technical",
    },
    {
        "state": "Checkout has been failing for every customer for the last hour.",
        "instructions": "Is this support request urgent?",
        "gold": True,
    },
]


def main() -> None:
    with ClefJudge() as judge:
        result = judge.evaluate(EVAL_SET)
        print(result.summary())
        print(f"\nstandalone metrics: ece={ece(result.confidences, result.correct):.4f} "
              f"brier={brier_score(result.confidences, result.correct):.4f}")


if __name__ == "__main__":
    main()
