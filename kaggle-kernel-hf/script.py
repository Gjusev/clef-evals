"""Kaggle GPU kernel: run clef-evals against clef-flash loaded from HuggingFace weights.

No Cloudflare credentials needed: the model runs locally on the kernel GPU and
is wired into ClefJudge through a transport shim that mimics the REST envelope.

Reality check (why this is experimental):
- clef-flash weights are 19.1 GB (fp16 + joint head). Kaggle T4 x2 offers
  2 x 16 GB, so the model must shard across both GPUs.
- The model card pins torch 2.11 + transformers 5.10.2 and shows single-device
  loading. Multi-GPU sharding may or may not work with the custom
  joint_schema_model code; if it does not, the kernel fails fast with the
  reason instead of burning quota silently.

Host-side loop:
    KAGGLE_API_TOKEN=... python -m kaggle kernels push -p kaggle-kernel-hf
    KAGGLE_API_TOKEN=... python -m kaggle kernels status gjusev/clef-flash-hf-benchmark
    KAGGLE_API_TOKEN=... python -m kaggle kernels output gjusev/clef-flash-hf-benchmark -p out/
"""

import json
import pathlib
import subprocess
import sys

WORKING = pathlib.Path("/kaggle/working")


def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def die(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


# 1. GPU present? (fails fast: unverified accounts get no GPU but keep settings)
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    gpus = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
except Exception as error:  # noqa: BLE001
    gpus = []
print(f"GPUs: {gpus}", flush=True)
if not gpus:
    die("no GPU visible. Verify the Kaggle account by phone, or the push silently drops GPU.")
total_vram_gb = 0.0
for g in gpus:
    mem = g.split(",")[1].strip().split()[0].replace("MiB", "")
    total_vram_gb += float(mem) / 1024
print(f"total VRAM: {total_vram_gb:.1f} GB (need >= 20 for clef-flash fp16 + head)", flush=True)
if total_vram_gb < 20:
    die(f"insufficient VRAM across {len(gpus)} GPU(s): {total_vram_gb:.1f} GB")

# 2. Pinned stack from the model card, then the repo
#    torchvision must upgrade in the SAME pip call or Kaggle's preinstalled
#    cu128 build shadows the cu130 torch and AutoProcessor import explodes.
run(f"{sys.executable} -m pip install -q --no-input --upgrade --upgrade-strategy eager "
    "'torch==2.11.0' torchvision 'transformers==5.10.2' accelerate pillow huggingface_hub")
run("git clone --depth 1 https://github.com/Gjusev/clef-evals.git")
run(f"{sys.executable} -m pip install -q --no-input ./clef-evals")

# 3. Judge the committed dataset against the LOCAL weights via the systemone shape
WORKING.mkdir(parents=True, exist_ok=True)
(WORKING / "gpus.json").write_text(json.dumps(gpus), encoding="utf-8")

bench = r'''
import json, sys
from pathlib import Path

sys.path.insert(0, "clef-evals/src")
snapshot = __import__("huggingface_hub").snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, snapshot)
import torch  # noqa: F401  (model code expects torch imported first)
from joint_schema_model import load_release_model, systemone

model, processor = None, None
def die(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)

try:
    # First try: shard across both T4s via accelerate, if the custom loader
    # forwards device_map to the backbone. Single-device load OOMs: 19.1 GB
    # of weights do not fit in one T4's 14.6 GB usable VRAM.
    model, processor = load_release_model(snapshot, device_map="auto")
    print("loaded with device_map=auto (multi-GPU sharding works)", flush=True)
except TypeError:
    die("custom loader rejects device_map: sharding unsupported; clef-flash "
        "fp16 needs an 80GB-class GPU (or the hosted API at $0.24/M input tokens)")
except torch.OutOfMemoryError:
    die("single-device load OOMed and device_map sharding is not available")

import httpx
from clef_evals import ClefConfig, ClefClient, ClefJudge
from clef_evals.judge import load_eval_set

def local_clef(request: httpx.Request) -> httpx.Response:
    """Translate a REST-shaped body into a local systemone() call."""
    body = json.loads(request.content)
    body["model"] = "clef-flash"
    response = systemone(model, processor, body)
    return httpx.Response(200, json={"result": response, "success": True,
                                     "errors": [], "messages": []})

config = ClefConfig(account_id="local-hf", api_token="not-used",
                    model="@cf/cloudflare/clef-flash")
transport = httpx.MockTransport(local_clef)
client = ClefClient(config, http_client=httpx.Client(transport=transport))
judge = ClefJudge(config, client=client)

eval_set = load_eval_set("clef-evals/evals/data/support_routing.jsonl")
result = judge.evaluate(eval_set)
report = {"mode": "local-hf-clef-flash",
          "gpu": json.loads(Path("/kaggle/working/gpus.json").read_text()),
          **result.to_dict()}
Path("/kaggle/working/hf-benchmark-results.json").write_text(json.dumps(report, indent=2))
print(result.summary())
'''
bench_file = WORKING / "hf_bench.py"
bench_file.write_text(bench.strip() + "\n", encoding="utf-8")
run(f"{sys.executable} {bench_file}")
