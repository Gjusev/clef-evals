"""Kaggle kernel: clone clef-evals, install it, and benchmark on Kaggle runtime.

Two modes:
- Live (Kaggle secrets CLEF_ACCOUNT_ID / CLEF_API_TOKEN set via Add-ons ->
  Secrets): runs evals/run_eval.py against the hosted Clef API and writes
  benchmark-results.json to /kaggle/working.
- Dry run (no secrets): proves the whole pipeline on Kaggle with a mock
  transport over the committed dataset and runs the test suite; writes
  dry-run-results.json.

Verified host-side loop (from the laya-evals reproduction):
    KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel
    KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/clef-evals-benchmark
    KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/clef-evals-benchmark -p out/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")


def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def load_secret(name: str) -> str:
    """Read a Kaggle secret when available, else fall back to the environment."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name, "")


REPO = "clef-evals"
run("git clone --depth 1 https://github.com/Gjusev/clef-evals.git")
run(f"{sys.executable} -m pip install -q --no-input './{REPO}[dev]'")

account_id = load_secret("CLEF_ACCOUNT_ID")
api_token = load_secret("CLEF_API_TOKEN")

if account_id and api_token:
    print("secrets found: running the LIVE benchmark", flush=True)
    os.environ["CLEF_ACCOUNT_ID"] = account_id
    os.environ["CLEF_API_TOKEN"] = api_token
    run(
        f"{sys.executable} {REPO}/evals/run_eval.py --concurrency 8 "
        f"--output {WORKING / 'benchmark-results.json'}"
    )
else:
    print("no secrets: running the DRY RUN (mock transport, no API calls)", flush=True)
    dry_run = f'''
import json, sys
from pathlib import Path
sys.path.insert(0, "{REPO}/src")
import httpx
from clef_evals import ClefConfig, ClefClient, ClefJudge
from clef_evals.judge import load_eval_set

def mock_clef(request):
    body = json.loads(request.content)
    answers = {{}}
    for qid, q in body["questions"].items():
        if q["type"] == "choice":
            options = list(q["criteria"])
            probs = {{o: round(1/len(options), 3) for o in options}}
            probs[options[0]] = round(1 - sum(probs[o] for o in options[1:]), 3)
            answers[qid] = {{"type": "choice", "choice": options[0],
                             "probabilities": probs, "confidence": probs[options[0]]}}
        else:
            answers[qid] = {{"type": "noul", "noul": 0.5}}
    return httpx.Response(200, json={{"result": {{
        "model": "@cf/cloudflare/clef", "answers": answers,
        "usage": {{"input_tokens": 120, "output_tokens": 8}}}},
        "success": True, "errors": [], "messages": []}})

config = ClefConfig(account_id="dry-run", api_token="not-used")
client = ClefClient(config, http_client=httpx.Client(transport=httpx.MockTransport(mock_clef)))
judge = ClefJudge(config, client=client)
eval_set = load_eval_set("{REPO}/tests/fixtures/eval_set.json")
result = judge.evaluate(eval_set)
report = {{"mode": "mock-dry-run", **result.to_dict()}}
Path("{WORKING}").mkdir(parents=True, exist_ok=True)
Path("{WORKING}/dry-run-results.json").write_text(json.dumps(report, indent=2))
print(result.summary())
'''
    dry_run_file = WORKING / "dry_run_bench.py"
    dry_run_file.parent.mkdir(parents=True, exist_ok=True)
    dry_run_file.write_text(dry_run.strip() + "\n", encoding="utf-8")
    run(f"{sys.executable} {dry_run_file}")
    run(f"{sys.executable} -m pytest {REPO}/tests -q")

print("done: outputs in /kaggle/working", flush=True)
