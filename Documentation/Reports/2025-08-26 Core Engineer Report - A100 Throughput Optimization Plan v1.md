Title: NSA A100 Throughput Optimization Plan (Consultant Brief)
Date: 2025-08-26
Owner: Core Engineering

Executive Summary
- Stability: Training is now stable. The step‑5 hang was eliminated by switching to batched prefill (NSA_PREFILL_BATCHED=1) and disabling end‑of‑step auxiliary stats (NSA_DISABLE_AUX_STATS=1). A100 bf16 runs complete without crashes.
- Throughput: Current throughput is ~39 tokens/second on 2×A100 80GB PCIe (no NVLink), seq_len=2048, batch_size=2, bf16. This tracks with our PCIe Guide’s expected Gen3 band (35–40 toks/s) but is far below literature numbers for optimized/NVLink setups.
- Gap vs literature: Even accounting for sequence length (×4), NSA overhead (×10), and PCIe (×2–3), we remain orders of magnitude below highly optimized references. We have identified a specific code bottleneck likely responsible for large fractions of step time: Python‑side range conversion in the selection pipeline at S=2048.
- Recommendation: Pause the 50k run to avoid runaway cost. Run a short optimization sprint (24–48h) to remove Python loops from the selection pipeline (vectorize on GPU), profile, and retune DDP/NCCL. Then re‑launch 50k on A100; consider NVLink hardware for further gains.

Current Configuration
- Hardware: 2×A100 80GB PCIe (no NVLink)
- Software: PyTorch 2.5.1+cu121, CUDA 12.1, Python 3.10
- Model: 125M (GPT‑2 scale) — dim=768, n_layers=12, n_heads=12, n_kv_groups=2, d_k=64, d_v=64
- NSA architecture (key features):
  - Three attention branches: selection (sparse block attention via compressed scoring + top‑k ranges), compressed (ϕ‑pooled KV), sliding window; mixed by a small gate MLP.
  - Additional steps vs standard attention: compress pooling (ϕ), sparse CSR/COO mapping and scatter, top‑k + tie‑break per token, range merge per token.
- Runtime hyperparameters:
  - precision: bf16; seq_len: 2048; batch_size: 2 (global; 1/GPU)
  - optimizer: AdamW (lr 2e‑4), cosine schedule; warmup 2000; grad_clip 1.0; weight_decay 0.01
  - gradient checkpointing: disabled under DDP for stability (re‑enable later for memory if needed)
- Stability toggles (mandatory):
  - NSA_PREFILL_BATCHED=1 (bypass sequential prefill hang)
  - NSA_DISABLE_AUX_STATS=1 (avoid end‑of‑step overhead)
  - Logging off for smokes: NSA_TB_DISABLE=1, NSA_DISABLE_CSV_LOGS=1
- DDP/NCCL performance toggles:
  - NSA_DDP_STATIC_GRAPH=1, NSA_DDP_FIND_UNUSED=0, NSA_DDP_BUCKET_MB=25
  - NCCL_ALGO=Ring, NCCL_PROTO=Simple, NCCL_IB_DISABLE=1 (PCIe, no IB)

Measured Performance
- ~39 toks/s; ~105 s/step at batch_size=2, seq_len=2048; ~15.6 GB/GPU memory use.
- Increasing per‑GPU batch_size to 4 degraded throughput (33 toks/s; −15%) and dropped utilization to ~35% — PCIe sync dominates (DDP communication overhead).
- Conclusion: On PCIe, keep per‑GPU batch small (1–2) and scale effective batch with gradient accumulation.

Comparison to Literature
- Reported: GPT‑2 124M on single A100 PCIe ~178k toks/s at ~49% MFU; NVLink clusters can process 10B tokens very quickly.
- Key differences: inference vs training benchmarks; shorter sequence (1024) vs our 2048 (O(n^2) cost); standard attention vs NSA; NVLink vs PCIe; heavy kernel fusion in production code vs research code.
- Despite these differences, our implementation is likely under‑optimized. We can aim for a pragmatic 2–10× improvement with targeted fixes, while acknowledging PCIe ceilings.

Identified Throughput Bottlenecks
1) Selection range conversion (primary)
   - File: nsa/core/selection_scorer.py
   - Function: convert_indices_to_ranges_batched(indices, meta, S)
   - Issue: Python loop over (B × S × G) to deduplicate and merge contiguous block indices into ranges. At S=2048 this forces CPU iteration, GPU syncs (.item()), and blocks parallelism.
   - Impact: Inflates step time even though core scoring and mapping are on GPU. The longer the sequence, the worse the impact.

2) Sparse mapping/scatter hygiene
   - map_pcmp_to_pslc_batched uses COO/CSR scatter_add to accumulate selection scores. With Long indices and contiguous tensors, it should be fine; we must verify no inadvertent host syncs.

3) SDPA kernel routing
   - On A100 bf16 we should be on Flash SDPA; we need to confirm no fallbacks to math/mem_efficient, and ensure q/k/v/masks are contiguous.

4) DDP overlap / bucket sizing
   - We set static graph and bucket_cap_mb=25. Profiling will confirm comm/comp overlap; we can adjust buckets and try gradient accumulation to reduce sync frequency.

What We’ve Implemented (to aid profiling)
- NVTX toggles: NSA_NVTX=1 gates lightweight NVTX ranges around key stages in batched prefill (projections+RoPE, pcmp_all, map_pcmp_to_pslc, topk+ranges, branch_attn+gate).
- Profiler harness: scripts/profiler_smoke.py captures a Chrome trace (CPU+CUDA) and a textual summary. Default schedule waits 10, warms 10, records 50 steps.
- PCIe NCCL tuning: Runner sets NCCL_ALGO=Ring, NCCL_PROTO=Simple, NCCL_IB_DISABLE=1 (no IB), as recommended in our PCIe guide.

Optimization Plan (24–48h)
1) Profile and attribute
   - Run profiler_smoke for a 200‑step synthetic smoke on A100 bf16 (PCIe), capture a 50‑step Chrome trace on rank 0, with NSA_NVTX=1.
   - Attribute wall time to: selection scoring, sparse mapping, range conversion, SDPA kernels, backward, optimizer, DDP comm.

2) Remove Python loops from selection/range pipeline (highest ROI)
   - Strategy A (mask consumption): Avoid range materialization entirely. Carry selected block indices (B,S,G,k) into attention and compute masks/offsets on GPU. This bypasses dedup/merge.
   - Strategy B (vectorized GPU merge): Implement a segment‑based merge on GPU to turn sorted indices into contiguous [start,end) ranges without Python. Use torch ops to identify contiguous runs (equality with shifted arrays), then cumulative grouping.
   - Ship behind an env flag (e.g., NSA_SEL_RANGES_V2=1) for A/B testing.

3) SDPA/DDP refinements (post‑vectorization)
   - Verify Flash SDPA everywhere; NVTX/profiler should show flash ops.
   - Revisit DDP bucket sizes with new compute balance; try {25, 50} MB.
   - Prefer gradient accumulation to increase effective batch without PCIe penalties.

4) Optional environment lever
   - If available, validate on an NVLink (SXM) box to quantify expected uplifts vs PCIe; this directly changes the cost curve.

Targets and Expectations
- Short‑term (post vectorization): 2×–10× improvement plausible if Python range merging dominates. PCIe ceiling still applies; treat Gen3 35–40 toks/s as baseline, Gen4 45–55 toks/s, NVLink 2–3× relative to PCIe.

Risks and Mitigations
- Stability regressions from refactoring selection/range code → A/B flag, unit tests on selection equivalence (Eq. 9 verifier), staged rollout.
- SDPA kernel drift across PyTorch versions → pin version for production; add one‑time logs if fallbacks occur.
- DDP behavior sensitivity → keep static graph + tuned buckets; use accum instead of larger per‑GPU batch on PCIe.

Proposed Deliverables to Consultant
- Chrome trace (50 steps) with NVTX ranges; text summary of top kernels/CPU ops.
- Before/after A/B for selection range path (wall times per stage, step time, toks/s).
- A short design note for the chosen vectorization strategy (mask‑based consumption vs GPU merge), with implementation sketch and test plan.
- Updated runbook snippet for PCIe and NVLink variants (expected bandwidth‑bound targets).

Appendix A: Commands
- Profiler harness (A100 bf16, synthetic):
  NSA_PREFILL_BATCHED=1 NSA_DISABLE_AUX_STATS=1 NSA_NVTX=1 \
  CONFIG=configs/m7c_125m_2xa100_production.yaml \
  python -u scripts/profiler_smoke.py --dataset synthetic --steps 200

- Manual smoke (300 steps; PCIe tuned):
  NSA_PREFILL_BATCHED=1 NSA_DISABLE_AUX_STATS=1 NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 \
  NSA_DDP_STATIC_GRAPH=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_BUCKET_MB=25 \
  NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_IB_DISABLE=1 \
  CONFIG=configs/m7c_125m_2xa100_production.yaml \
  torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --steps 300 --precision bf16

Appendix B: Files of Interest
- nsa/core/selection_scorer.py — convert_indices_to_ranges_batched (primary Python hotspot to remove); map_pcmp_to_pslc_batched (GPU scatter hygiene).
- nsa/core/nsa_attention.py — batched prefill path; SDPA routing; NVTX hooks.
- scripts/profiler_smoke.py — profiling harness.
- Documentation/Guides/PCIe-GPU-Optimization.md — expected PCIe bands and NCCL guidance.

