# M7 — Training Showcase Plan

Status: Proposed

Objective
- Train a tiny, self-contained LLM-style decoder using NSAAttention to demonstrate end-to-end trainability with convergent loss and qualitative samples. Keep it fast, reproducible, and runnable on a single 4090/A100.

Scope
- Model: 0.5–15M params (Track A) and 20–40M params (Track B); depth 4–8; dim 128–512; heads 4–8; GQA groups 1–4; context 512–1024 (A) and 8k (B).
- Attention: NSAAttention (cmp/sel/win) with PRD defaults (l=32, d=16, l’=64, n=16, w=512). FA‑2 optional for cmp/win; Triton off by default on 4090.
- Data: FineWeb‑Edu (Hugging Face) as primary corpus; Tiny Shakespeare as tiny smoke. Byte‑level fallback or GPT‑2‑style BPE for efficiency.
- Training: Single GPU (A100‑40G preferred; 4090 acceptable). BF16/FP32; fixed seeds; configurable batch size/steps.

Deliverables
- Config: configs/train_showcase.yaml (model, optimizer, data, runtime flags).
- Script: reuse scripts/train_toy.py (extend to accept showcase config) or add scripts/train_showcase.py.
- Artifacts:
  - Checkpoint: checkpoints/showcase_final.pt (weights + optimizer state + config).
  - Logs/curves: training.csv (loss, lr, ppl), curves.png (loss vs steps).
  - Samples: samples/before.txt, samples/after.txt (seeded generations at fixed prompt).
  - Report: Documentation/Test-Reports/M7-Training-Showcase-Report.md (env, config, metrics, curves, samples).
  - Publishing: Hugging Face model card + weights (README, config.json, .safetensors or .pt) and a minimal `nsa_loader.py` + `scripts/generate.py` usage snippet.

Acceptance Criteria
- A1: No NaNs; smooth decreasing loss; run time ≤ 2h on 4090 at ~1–3M params.
- A2: Validation PPL improves ≥ 30% vs init; suggested target PPL < 3.0 on Tiny Shakespeare char-level at ~1–3M params.
- A3: Deterministic: fixed seed; loss variance ≤ ±5% across two reruns.
- A4: Observability: gate mean/std finite; decode counters monotonic in eval; group-consistency/causality tests green.

Datasets — FineWeb‑Edu
- Source: `HuggingFaceFW/fineweb-edu` (curated educational/textbook‑like split of FineWeb).
- Why: cleaner, expository text → faster convergence for small models; long articles → good for NSA long‑context.
- Access: Hugging Face Datasets with streaming to avoid full downloads.
  - Python: `from datasets import load_dataset; ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)`
- Tokenization: start with byte‑level (zero‑dep) for Track A; prefer GPT‑2‑style BPE for Track B to improve efficiency.
- Packing: fixed‑length contiguous chunks (A: S=512/1024; B: S=8192). Deterministic seeds and stable shard order.

Training Tracks & Targets
- Track A (Tiny, usable now): 5–15M params, S=512–1024, batch∼8–16. 50k–200k steps on A100‑40G (≈1–3h). Deliver a “NSA‑Tiny‑512/1024” checkpoint on HF with sample generations.
- Track B (Long‑context demo): 20–40M params, S=8192, batch 1–2, gradient checkpointing optional. 10k–20k steps on A100‑40G (≈3–6h) to demonstrate multi‑page continuations and NSA selection at scale.

How to Use FineWeb‑Edu in this repo
- Data loader (plan): add `scripts/datasets/fineweb_edu_loader.py` to stream, tokenize, and pack batches for `[B, S]` training.
- Wiring: add a CLI flag to `scripts/train_showcase.py` (e.g., `--dataset fineweb_edu`), defaulting to synthetic; when selected, use the streaming loader.
- Determinism: fixed random seeds; shard → pack order stable across runs; document in the report.

Example Commands (A100‑40G)
- Env: `python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-gpu-cu121-torch24.txt`
- Track A (byte‑level):
  - `CONFIG=configs/train_showcase.yaml PYTHONPATH=. NSA_USE_FA2=1 python scripts/train_showcase.py` (set model.dim≈256–384; n_layers 4–6; seq_len 512–1024; batch 8–16; precision bf16)
- Track B (BPE; long‑context):
  - Same command with model.dim≈384–512; n_layers 6–8; seq_len 8192; batch 1–2; enable gradient checkpointing if wired.

Checkpoints & Publishing
- Save final weights + config: `artifacts/train_showcase/model.safetensors` + `config.json` (NSA hyperparams included).
- HF upload: model weights, `nsa_loader.py` (instantiates `LlamaBlockNSA` and loads state), and `scripts/generate.py` (top‑k/p sampling CLI).
- Model card: architecture summary (cmp/sel/win + gate), PRD invariants (causality, GQA), NSA counters, gate stats plots, and two generation examples.

Validation on Real Text
- Long‑context “needle”: run selection mapping coverage on FineWeb‑Edu articles (not just synthetic data) at 64k where feasible; report selection coverage stats and counters.

Milestones
- M7‑A: HF release — NSA‑Tiny‑512 (byte‑level), with runnable local demo and model card.
- M7‑B: HF release — NSA‑Tiny‑8k (BPE), with long‑context demo and selection coverage plots.

Runtime Flags
- NSA_USE_FA2=1 (optional), NSA_FA2_MIN_LEN_{WIN,CMP} to avoid tiny-length slowdowns.
- NSA_TRITON_SEL_FORCE=0 (default on 4090). NSA_FORCE_BRANCH for ablations (cmp|sel|win).
- NSA_DEBUG_LOG=0 by default; enable selectively.

Tasks
1) Config + script wiring
- Add configs/train_showcase.yaml; ensure scripts/train_toy.py accepts it.
- Byte-level dataset loader (Tiny Shakespeare) with train/val split and fixed seed shuffling.

2) Sanity passes
- Evaluate forward/backward on a tiny batch; run a 200-step smoke to verify stability.

3) Full run
- Launch training per config; save artifacts and report.

4) Report
- Curves (CSV/PNG), final metrics (train/val loss, PPL), environment (GPU/driver, Torch/Triton/FA-2), samples before/after.

Risks & Mitigation
- FA‑2 instability on 4090 → default to SDPA; thresholds high by default.
- Overfit → val split, early stopping option; small weight decay.
- Throughput variance → fix seeds, stable batch shapes; report exact setup.

Timeline
- Wiring + config: 0.5 day
- Sanity + tuning: 0.5–1 day
- Full run + report: 0.5–1 day
