# clef-evals

[![CI](https://github.com/Gjusev/clef-evals/actions/workflows/test.yml/badge.svg)](https://github.com/Gjusev/clef-evals/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/clef-evals)](https://pypi.org/project/clef-evals/)
[![Python](https://img.shields.io/pypi/pyversions/clef-evals)](https://pypi.org/project/clef-evals/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/Gjusev/clef-evals/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)

**Calibration-first evaluation toolkit for [Cloudflare Clef](https://developers.cloudflare.com/workers-ai/models/clef/) decision models.**
Judge cheap, audit confidence.

Most eval harnesses stop at accuracy. clef-evals also asks whether Clef's
probabilities mean what they say. It computes Expected Calibration Error and
Brier score over your own datasets, then turns both into a CI gate. A model
that is right but overconfident fails your build before it fails your users.

## Install

```bash
pip install clef-evals
export CLEF_ACCOUNT_ID=your_account_id
export CLEF_API_TOKEN=your_api_token
```

Requires Python 3.10+. The only runtime dependency is `httpx`.

## Quick start

```python
from clef_evals import ClefJudge

judge = ClefJudge()  # config from environment

result = judge.evaluate([
    {"state": "Email: I need a refund", "instructions": "Which team?",
     "criteria": {"billing": "Payments, invoices, refunds",
                  "technical": "Bugs and outages",
                  "sales": "Plans and upgrades"},
     "gold": "billing"},
    {"state": "Checkout is down for everyone", "instructions": "Is this urgent?",
     "gold": True},  # binary items use a boolean gold
])
print(result.summary())
```

```
samples=2 failures=0
accuracy=1.0000
ece=0.0700
brier=0.0049 brier_multiclass=0.0082
latency_ms p50=210.1 p95=238.6 p99=238.6
input_tokens=240 output_tokens=16
cost: $0.000058 total | $0.028800 per 1k calls
```

Async fan-out with a bounded semaphore:

```python
import asyncio
from clef_evals import AsyncClefJudge

judge = AsyncClefJudge()          # httpx.AsyncClient under the hood
result = asyncio.run(judge.evaluate(eval_set, concurrency=8))
```

Single decisions with the full probability distribution:

```python
decision = judge.judge_choice(
    "Email: charged twice", "Which team?",
    {"billing": "Payments, invoices, refunds", "technical": "Bugs and outages"},
)
print(decision.choice, decision.probabilities)   # billing {'billing': 0.93, ...}
p_yes = judge.judge_binary("Checkout is down", "Is this urgent?")  # 0.97
```

## Architecture

Animated version: [docs/pipeline.svg](docs/pipeline.svg) ·
Showcase video: [brag-output/brag.mp4](brag-output/brag.mp4) (rendered by
[brag-output/render_video.py](brag-output/render_video.py), no stock assets)
· Walkthrough notebook: [research/clef_calibration_walkthrough.ipynb](research/clef_calibration_walkthrough.ipynb)

```
        ┌─────────────────────────────────────────────────────────┐
        │                    your CI / your code                  │
        └──────────┬─────────────────────────────────┬────────────┘
                   │                                 │
           ┌───────▼────────┐              ┌─────────▼─────────┐
           │   ClefJudge    │              │  clef-eval CLI    │
           │ sync + Async   │              │  run / gate       │
           └───────┬────────┘              └─────────┬─────────┘
                   │                                 │
           ┌───────▼─────────────────────────────────▼─────────┐
           │ ClefClient                                        │
           │  retries · exponential backoff + jitter           │
           │  timeouts · Retry-After · structured errors       │
           └───────┬───────────────────────────────────────────┘
                   │ HTTPS POST /accounts/{id}/ai/run/@cf/cloudflare/clef
           ┌───────▼───────────────────────────────────────────┐
           │ Cloudflare Workers AI (clef 27B / clef-flash 9B)  │
           │ state + typed questions -> probabilities          │
           └───────┬───────────────────────────────────────────┘
                   │ per-option probabilities + usage
           ┌───────▼───────────────────────────────────────────┐
           │ metrics                                           │
           │  accuracy · ECE · Brier · Brier-multiclass        │
           │  latency p50/p95/p99 · cost per 1k calls          │
           └───────┬───────────────────────────────────────────┘
                   │ EvalResult JSON
           ┌───────▼───────────────────────────────────────────┐
           │ regression-gate GitHub Action                     │
           │  fresh run  vs  committed baseline                │
           └───────────────────────────────────────────────────┘
```

## CLI

```bash
# evaluate a dataset (JSON array or JSONL), human summary
clef-eval run evals/data/support_routing.jsonl

# machine-readable, save artifact
clef-eval run evals/data/support_routing.jsonl --json --output results/run.json

# CI gate: fail the build when quality or calibration regress
clef-eval run evals/data/support_routing.jsonl \
    --min-accuracy 0.90 --max-ece 0.15
```

Exit codes: `0` gate passed · `1` gate failed · `2` config or dataset error.

## Benchmarks

### Published reference (Cloudflare's Decision Index 0.2.1)

Numbers below are **Cloudflare's published measurements** on their
infrastructure ([model card](https://huggingface.co/Cloudflare/clef),
[blog](https://blog.cloudflare.com/clef-decision-models/)), not measurements
made with this toolkit. Full table committed at
`evals/results/published_reference.json`.

| Benchmark | Clef | Clef-flash | Jev | Laya |
|---|---:|---:|---:|---:|
| BFCL · case exact | 98.5 | **98.8** | 95.8 | 38.1 |
| BANKING77 · macro-F1 | **94.2** | 90.9 | 79.7 | 14.3 |
| CLINC150+OOS · macro-F1 | **97.4** | 66.8 | 89.3 | 3.2 |
| When2Call · accuracy | 72.4 | 65.6 | **81.0** | 11.9 |
| ForecastBench · Brier (↓) | 13.9 | **10.6** | 17.4 | 41.1 |
| Median latency · ms | 209.3 | 38.8 | 524.1 | **5.8** |
| p95 latency · ms | 238.6 | **122.4** | 536.0 | 222.5 |

### Our runs

| Dataset | Model | n | accuracy | ECE | Brier | p50 / p95 / p99 (ms) | $/1k calls |
|---|---|---:|---:|---:|---:|---|---:|
| support_routing | clef | *pending first live run* | | | | | |
| support_routing | clef-flash | *pending first live run* | | | | | |

Reproduce and add your numbers (needs `CLEF_ACCOUNT_ID`/`CLEF_API_TOKEN`):

```bash
make eval                                        # both models, all datasets
python evals/run_eval.py --model @cf/cloudflare/clef-flash --concurrency 8
make test-integration                            # pytest against the real API
```

Cost model: published price is **$0.24 per M input tokens**
(e.g. ~120 input-token calls ≈ **$0.029 per 1k calls**). Output-token pricing
is not published by Cloudflare; `output_tokens` is reported but not priced.

### clef vs laya

| | Clef (Workers AI) | Clef-flash | Laya |
|---|---|---|---|
| Type | 27B decision model (hosted) | 9B decision model (hosted) | decision model (open weights) |
| Context window | 65,536 tokens | 65,536 | 32,768 |
| Vision / images | yes | yes | no |
| Median latency | 209.3 ms | 38.8 ms | **5.8 ms** |
| p95 latency | 238.6 ms | **122.4 ms** | 222.5 ms |
| Quality (BFCL / BANKING77 / CLINC150) | 98.5 / **94.2** / **97.4** | **98.8** / 90.9 / 66.8 | 38.1 / 14.3 / 3.2 |
| Calibration (ForecastBench Brier, ↓) | 13.9 | **10.6** | 41.1 |
| Cost | $0.24 / M input tokens (hosted) | $0.24 / M input tokens | self-hosted (your GPUs) |

**Reading:** Laya wins raw latency. Clef wins quality and calibration by
large margins, with clef-flash as the fast middle ground. For routing and
gating workloads, miscalibrated confidence is what breaks automation. That is
exactly what this toolkit measures on your data.

## CI gate (reusable GitHub Action)

Commit a baseline JSON (any `EvalResult.to_dict()` output), then gate PRs:

```yaml
- uses: jorgealizola/clef-evals/.github/actions/regression-gate@main
  with:
    current: results/run.json
    baseline: results/baselines/support-routing-clef.json
    metrics: |
      accuracy:min:0.03
      ece:max
      latency_p95:max:50
```

`accuracy:min` = may not drop more than tolerance; `ece:max` = may not grow.
Pure Python at gate time: no credentials, no network.

## Live benchmark automation

`evals.yml` runs the benchmark on demand (Actions -> Evals -> Run workflow) as
soon as repo secrets `CLEF_ACCOUNT_ID` / `CLEF_API_TOKEN` exist, uploads the
measurement JSON as an artifact, and optionally gates it against the committed
baseline with the regression-gate action. Without secrets the workflow exits
cleanly and says so.

## Kaggle kernel

Reproduce the benchmark on Kaggle's free CPU runtime (verified push/status/output loop):

```bash
# add secrets CLEF_ACCOUNT_ID / CLEF_API_TOKEN on kaggle.com first
KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel
KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/clef-evals-benchmark
KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/clef-evals-benchmark -p out/
```

## Error handling

Every failure is a typed exception under `ClefError`:

```
ClefError
├── ConfigurationError      missing/invalid env (reports ALL problems at once)
├── ClefAPIError            API refused the request
│   ├── ClefAuthError       401/403 (not retried)
│   ├── ClefRateLimitError  429 (retried, honors Retry-After)
│   └── ClefServerError     5xx (retried)
├── ClefResponseError       body does not match the Clef schema
├── ClefTimeoutError        retried
└── ClefNetworkError        DNS / connection (retried)
```

Retries default to `max_retries=2` with exponential backoff + jitter; every
error carries `message` and log-safe `details`. The library logs to the
`clef_evals` logger. It never prints and never logs your token.

## Limitations (honest section)

- **Calibration metrics audit, they don't fix.** ECE/Brier tell you how much
  to trust Clef's probabilities on *your* distribution; they don't recalibrate
  them. Use the reported confidence accordingly (or calibrate downstream).
- **Cost model covers input tokens only.** Cloudflare publishes $0.24/M input
  tokens but no output-token price for Clef at the time of writing. `output_tokens`
  is reported so you can price it the day it appears.
- **Mixed-type datasets blend confidence semantics.** Choice items use
  `P(chosen option)`; binary items use `max(p, 1−p)`. ECE/Brier over a mixed
  set pool both. Prefer per-type runs when the distinction matters.
- **Latency numbers are client-side** (includes your network RTT to Cloudflare).
  Do not compare them 1:1 with Cloudflare's published infra-side medians.
- **Fail-soft evaluation.** Items that error after retries are excluded from
  metrics and counted in `result.failures`. The CLI gate fails on any
  failure, but direct library users should check `failures` or risk silent drift.
- **Local inference is out of scope for most machines.** Clef is a 27B model
  with a custom joint-schema head (reference hardware: a single H200; weights
  ~55 GB fp16). No GGUF/vLLM-quantized path is published. These benchmarks
  target the hosted Workers AI API.
- **v0.x API.** Expect small breaking changes before 1.0; the v0.1 names
  `ClefEvalResult`, `ece`, `brier_score` remain importable.

## Development

```bash
make install    # editable install with dev extras
make test       # pytest with coverage (>90% enforced); integration tests excluded
make test-integration   # real-API tests: needs CLEF_ACCOUNT_ID / CLEF_API_TOKEN
make lint       # ruff
make build      # wheel + sdist
```

Project layout: `src/clef_evals/` (config, client, models, metrics, judge, cli) ·
`tests/` (unit + fixtures with real API shapes) · `evals/` (datasets, runner,
committed results) · `scripts/check_regression.py` + `.github/actions/regression-gate/`
(CI gate) · `kaggle-kernel/` (cloud reproduction) · `research/` (calibration
walkthrough notebook) · `docs/` + `brag-output/` (visuals, video renderer).

## License

Apache 2.0, see [LICENSE](LICENSE). Clef models are Apache 2.0 on
[Hugging Face](https://huggingface.co/Cloudflare/clef).
