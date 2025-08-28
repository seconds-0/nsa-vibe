# 2025-08-26 Test Engineer Report - DDP Performance Validation v2

This report captures results. The canonical procedure lives under Documentation/Test-Plans.

Procedure
- See: `Documentation/Test-Plans/DDP-Performance-Validation-Runbook.md`

Environment
- Branch: `feat/nsa-training-breakthrough-stable-a100`
- Python/Torch/CUDA: record via `artifacts/collect_env.txt`
- GPUs: 2×A100 80GB PCIe

Results (to be filled by Test Engineer)
- Baseline DDP throughput: <value> toks/s (global)
- Bucket sweep (mean toks/s over last 50 steps):
  - 25 MB: <mean> ± <stdev>
  - 50 MB: <mean> ± <stdev>
  - 100 MB: <mean> ± <stdev>
- SDPA audit outcome: <summary>
- Selection v2 parity/perf: <summary>
- NVTX attribution notes: <summary>
- Single‑GPU isolation: <per‑GPU toks/s>, comparison vs DDP
- Fused AdamW A/B (optional): <baseline vs fused>
- Stability 1,200‑step: <steady/any anomalies>

Artifacts
- Attach or link: `<OUT>/training.csv`, `<OUT>/heartbeat_rank*.jsonl`, `<OUT>/env.json`, memory dumps
- System: `artifacts/git_sha.txt`, `artifacts/collect_env.txt`, `artifacts/nvidia_smi.xml`

Decision
- GO/NO‑GO with recommended envs: `NSA_SEL_RANGES_V2=1`, `NSA_DDP_COMPRESS=bf16`, `NSA_DDP_BUCKET_MB=<best>`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`

