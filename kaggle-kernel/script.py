"""Kaggle kernel: clone clef-evals, install it, run the benchmark, print the table.

Credentials are read from Kaggle secrets (Add-ons -> Secrets) with the keys
CLEF_ACCOUNT_ID and CLEF_API_TOKEN, falling back to environment variables.
The kernel pushes results to /kaggle/working for `kaggle kernels output`.

Verified push/monitor/output loop (from the host machine):
    KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel
    KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/clef-evals-benchmark
    KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/clef-evals-benchmark -p out/
"""

import os
import subprocess
import sys


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


ACCOUNT_ID = load_secret("CLEF_ACCOUNT_ID")
API_TOKEN = load_secret("CLEF_API_TOKEN")
if not ACCOUNT_ID or not API_TOKEN:
    sys.exit("Set Kaggle secrets CLEF_ACCOUNT_ID and CLEF_API_TOKEN before pushing.")

os.environ["CLEF_ACCOUNT_ID"] = ACCOUNT_ID
os.environ["CLEF_API_TOKEN"] = API_TOKEN

run("git clone --depth 1 https://github.com/Gjusev/clef-evals.git")
run(f"{sys.executable} -m pip install -q --no-input ./clef-evals")
run(
    f"{sys.executable} clef-evals/evals/run_eval.py --concurrency 8 "
    "--output /kaggle/working/benchmark-results.json"
)
run("cp -r clef-evals/evals/results /kaggle/working/ 2>/dev/null || true")
print("done: results in /kaggle/working", flush=True)
