# clef-evals

Calibration-first evaluation toolkit for Cloudflare's [Clef](https://blog.cloudflare.com/clef-decision-models/) decision models.

## What it does

- **Judge**: run Clef as an LLM-as-judge over eval sets with typed questions
- **Calibrate**: compute Expected Calibration Error and Brier score
- **Gate**: fail CI builds when accuracy or calibration regresses

## Quick start

```bash
pip install clef-evals
export CLEF_ACCOUNT_ID=your_id
export CLEF_API_TOKEN=your_token
```

```python
from clef_evals import ClefJudge, ece, brier_score

judge = ClefJudge()
result = judge.evaluate([
    {"state": "Email: I need a refund", "question": "category",
     "choices": ["billing", "technical", "sales"], "gold": "billing"},
    # ... more items
])
print(f"accuracy: {result.accuracy:.4f}")
print(f"ece:      {result.ece:.4f}")
print(f"brier:    {result.brier:.4f}")
```

## License

Apache 2.0
