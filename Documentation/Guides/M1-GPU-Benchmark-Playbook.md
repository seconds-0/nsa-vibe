# M1 GPU Benchmark & Threshold Tuning Playbook

This playbook explains how to run Native Sparse Attention (NSA) FA‑2 GPU benchmarks, validate parity, and tune thresholds to enable FA‑2 paths by default with confidence.

Audience: engineer or infra operator with access to a CUDA GPU host.

---

## 1) Objectives & Outcomes

- Validate FA‑2 parity vs SDPA within documented tolerances on GPU.
- Measure performance for sliding and compressed branches.
- Tune `runtime.fa2_min_len_win` and `runtime.fa2_min_len_cmp` for your hardware.
- Provide a short report (tables + chosen thresholds) and land config updates.

Deliverables:
- A short markdown note with speedups and chosen thresholds (template in §10).
- A commit updating `configs/base.yaml` thresholds and optionally `PRD.md` notes.

---

## 2) What You Need

- NVIDIA GPU with recent architecture (recommended):
  - H100 (SM90), L40/L40S (SM89), L4 (SM89), A100 (SM80), A10/A10G (SM86)
- OS: Ubuntu 22.04 or 20.04
- NVIDIA driver compatible with your CUDA runtime (PyTorch 2.3.x + CUDA 12.x or 11.8)
- Disk: ≥ 50 GB; RAM: ≥ 16 GB
- SSH access (preferred) or provider web console

Time/Cost ballpark:
- Bench sweep takes minutes per configuration; total < 1 GPU‑hour
- Provider on‑demand prices vary (~$0.4–$2.0/hr for L4/A10; more for L40S/A100/H100)

---

## 3) Provider Quickstarts

Pick any one. These steps assume you can create an instance and SSH in.

### 3.1 Lambda Cloud
- Choose instance: L4 (24–48 GB), L40S (48 GB), A10 (24 GB), A100 (40/80 GB)
- Image: “Lambda Stack” (Ubuntu + CUDA + PyTorch preinstalled)
- Launch → SSH into the instance (documented in Lambda UI)

### 3.2 RunPod (On‑Demand)
- Choose GPU: L4/L40S/A10/A100
- Template/Container: “PyTorch 2.x CUDA 12.1” or similar
- Expose SSH; start pod; connect via SSH

### 3.3 AWS EC2
- Instance: g6/g6e (L4/L40S), g5 (A10G), p4/p5 (A100/H100)
- AMI: “AWS Deep Learning AMI GPU PyTorch” or an NGC PyTorch image
- Launch with sufficient volume; SSH via key pair

---

## 4) Repository & Environment Setup

```bash
# Ensure GPU visible
nvidia-smi

# Install uv (if missing)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo and install deps
git clone https://github.com/seconds-0/nsa-vibe.git
cd nsa-vibe
uv venv -p 3.11 .venv
uv pip sync -r requirements.txt

# CUDA/PyTorch sanity
PYTHONPATH=. ./.venv/bin/python - <<'PY'
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY

# FlashAttention‑2 probe
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from nsa.kernels.flash_wrappers import is_flash_available, fa2_supported
import torch
print('flash-attn available:', is_flash_available())
print('fa2_supported(fp16, head_dim=64):', fa2_supported(torch.device('cuda'), torch.float16, 64) if torch.cuda.is_available() else False)
PY
```

If `flash-attn available` is `False`, install a wheel matching your PyTorch CUDA build:
```bash
# Example for CUDA 12.1 wheels; adjust to your torch/cu version
./.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 flash-attn==2.*
```

---

## 5) Parity Validation (GPU)

Run opt‑in GPU parity suite to ensure numerics are in‑tolerance on your device:
```bash
NSA_TEST_FA2=1 PYTHONPATH=. ./.venv/bin/python -m pytest -q -k "fa2_gpu_varlen" | cat
```
Expect pass/skip. If skipped, check FA‑2 availability and head‑dim alignment (multiples of 8/16 by dtype/device).

---

## 6) Benchmarking

We provide `bench/bench_fa2.py` to measure masked SDPA vs FA‑2 for sliding and compressed branches.

Baseline benches:
```bash
PYTHONPATH=. NSA_USE_FA2=1 NSA_DEBUG_TIMING=1 ./.venv/bin/python bench/bench_fa2.py | tee bench.out
```
Outputs include:
- Sliding lines: `sliding masked XXX ms  fa2 YYY ms  speedup xZ.ZZ`
- Compressed lines: `compressed masked XXX ms  fa2 YYY ms  speedup xZ.ZZ`
- Optional timing logs per bucket if `NSA_DEBUG_TIMING=1`

FA‑2 path comparisons (optional):
```bash
# Force varlen-all path
echo 'VARLEN' && PYTHONPATH=. NSA_USE_FA2=1 NSA_FA2_FORCE_VARLEN=1 ./.venv/bin/python bench/bench_fa2.py | tee bench_varlen.out
# Force dense per-bucket path
echo 'DENSE' && PYTHONPATH=. NSA_USE_FA2=1 NSA_FA2_FORCE_DENSE=1  ./.venv/bin/python bench/bench_fa2.py | tee bench_dense.out
```

---

## 7) Threshold Tuning Procedure

Goal: choose minimal lengths where FA‑2 consistently beats masked SDPA.

1) Sliding threshold `runtime.fa2_min_len_win`:
   - Inspect sliding results across S ∈ {256, 512, 1024} and w ∈ {32, 64, 128, 256}.
   - Identify the smallest window length w where FA‑2 wins consistently (speedup > 1.0 with headroom).
   - Set `runtime.fa2_min_len_win: <w>` in `configs/base.yaml`.

2) Compressed threshold `runtime.fa2_min_len_cmp`:
   - With (l,d) ≈ (32,16), note the smallest effective `num_cmp` = floor((t+1‑l)/d)+1 where FA‑2 wins.
   - Map that to an integer length L; set `runtime.fa2_min_len_cmp: <L>`.

3) Re‑run benches with thresholds set (and without forcing) to confirm FA‑2 engages above the thresholds and masked SDPA is used below them.

4) Commit updated thresholds and, optionally, add a short summary to `PRD.md` (§Thresholds) with the tuned values.

---

## 8) Automation Options

### 8.1 Self‑Hosted GPU Runner (GitHub Actions)

If you have a persistent GPU host, register it as a self‑hosted runner and add a job that runs benches and uploads artifacts.

Example workflow job (snippet):
```yaml
jobs:
  gpu-bench:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Setup uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install deps
        run: |
          uv venv -p 3.11 .venv
          uv pip sync -r requirements.txt
      - name: Bench
        env:
          NSA_USE_FA2: "1"
          NSA_DEBUG_TIMING: "1"
        run: |
          PYTHONPATH=. ./.venv/bin/python bench/bench_fa2.py | tee bench.out
      - uses: actions/upload-artifact@v4
        with:
          name: bench-outputs
          path: bench.out
```

Security: restrict runner to your repo; avoid exposing tokens; rotate regularly.

### 8.2 RunPod Serverless Job API (headless)

RunPod supports programmatic job execution. Minimal flow:
- Create a template with an image that includes CUDA/PyTorch and `uv`.
- POST an API job that: clones the repo, installs deps, runs benches, uploads results to S3/GCS.
- Poll job status; fetch logs and artifacts.

Pseudo‑script (Python):
```python
import requests, json, os
RUNPOD_API = os.environ["RUNPOD_API_KEY"]
headers = {"Authorization": f"Bearer {RUNPOD_API}", "Content-Type": "application/json"}
body = {
  "name": "nsa-bench",
  "gpu": "L4",
  "image": "nvcr.io/nvidia/pytorch:24.06-py3",
  "command": "bash -lc 'git clone https://github.com/seconds-0/nsa-vibe.git && cd nsa-vibe && curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv -p 3.11 .venv && uv pip sync -r requirements.txt && NSA_USE_FA2=1 PYTHONPATH=. ./.venv/bin/python bench/bench_fa2.py | tee bench.out && aws s3 cp bench.out s3://your-bucket/path/'"
}
r = requests.post("https://api.runpod.io/graphql", headers=headers, data=json.dumps({"query": "..."}))
# Refer to RunPod API docs for exact endpoint/body; handle job id, status polling, and logs fetch.
```

(Implement fully only if you intend to automate; APIs change—consult provider docs.)

---

## 9) Troubleshooting

- `flash-attn available: False`: install a matching wheel for your PyTorch CUDA version.
- Head‑dim unsupported: ensure `d_k` multiple of 8/16 depending on dtype/device (we gate via `fa2_supported`).
- OOM: lower S or w, or use smaller Dk/Dv; ensure only one bench job runs at a time; close other GPU apps.
- Performance variance: run multiple iterations; avoid noisy neighbors; pin `CUDA_VISIBLE_DEVICES=0`.
- Parity fails: loosen to documented tolerance for FP16/BF16 (≤ 2e‑4) and re‑check inputs; ensure no dropout/bias.

---

## 10) Results Template (paste into PR/Doc)

```markdown
### GPU Bench Results (Device: <GPU Model>, Torch <ver>, CUDA <ver>)

Sliding (S,w; masked vs FA‑2):
- S=256: w=32 x1.xx, w=64 x1.xx, w=128 x1.xx, w=256 x1.xx
- S=512: ...
- S=1024: ...

Compressed (S; masked vs FA‑2):
- S=256: x1.xx
- S=512: x1.xx
- S=1024: x1.xx

Chosen thresholds:
- runtime.fa2_min_len_win: <value>
- runtime.fa2_min_len_cmp: <value>

Notes:
- Bucket histograms show ...
- Determinism observations ...
```

---

## 11) Acceptance Criteria

- GPU parity suites pass (opt‑in) within tolerances (FP32 ≤ 5e‑5; FP16/BF16 ≤ 2e‑4 where applicable).
- FA‑2 shows speedups for realistic lengths; thresholds set accordingly and committed.
- Default GPU behavior: FA‑2 enabled with tuned thresholds; CPU remains SDPA‑only.

---

## 12) Final Checklist

- [ ] GPU host provisioned (Lambda/RunPod/AWS/etc.)
- [ ] Repo cloned; env created with `uv`; CUDA + flash‑attn verified
- [ ] GPU parity suite (opt‑in) passes/skips as expected
- [ ] Benches run; outputs collected (`bench.out`, optional varlen/dense)
- [ ] Thresholds chosen; `configs/base.yaml` updated
- [ ] Optional PRD notes updated with tuned values
- [ ] Full test suite green; changes committed

---

For assistance or to automate this process, consider a self‑hosted GitHub GPU runner (§8.1) or a provider job API (§8.2).
