PRD / System Plan — Native Sparse Attention (NSA)

Status: Approved for implementation
Driver: Lead Engineer (you)
Objective: Implement NSA as a drop‑in attention module for decoder‑only Transformers with trainable, hardware‑aligned sparse attention that runs locally.
Core idea (one line): For each query, combine Compressed, Selected, and Sliding branches via learned gates; selection is blockwise and group‑consistent (GQA/MQA‑friendly); only the selection branch needs a custom kernel; compressed & sliding run on FlashAttention‑2 (FA‑2).

1) Goals & Non‑Goals
1.1 Goals

G1. Correctness: NSA matches full attention when configured to cover all tokens (small‑S equivalence).

G2. Trainability: End‑to‑end gradients flow through branches; selection is chosen via scores from the compressed branch (no auxiliary losses).

G3. Hardware alignment:

Compressed & Sliding branches run on FA‑2.

Selection uses a custom group‑centric kernel (Triton) that shares KV across heads inside a GQA group (Figure 3).

G4. Decode efficiency: Per‑step tokens loaded ≈ ((S‑l)/d) + n*l' + w, driving near‑linear speedup vs full attention at long context (Table 4).

G5. Long‑context competence: Perfect 64k needle‑in‑a‑haystack retrieval (Figure 5).

1.2 Non‑Goals

Replicating exact pretraining curves or absolute speed numbers from the paper; we match trends (Figure 6) and mechanics.

Implementing MoE specifics from the 27B reference backbone (we integrate into a LLaMA‑style block).

2) User Stories

As an ML engineer, I can swap MultiheadAttention with NSAAttention in a LLaMA‑style block and run prefill+decode locally.

As a researcher, I can toggle sparsity knobs (l, d, l', n, w) to evaluate trade‑offs and visualize selection/gates.

As a perf engineer, I can compare decode token‑reads and timing vs FA‑2 at 8k–64k contexts (trend like Figure 6 / Table 4).

3) Product Requirements
3.1 Functional Requirements (FR)

FR1: Provide three parallel branches—Compressed (cmp), Selected (slc), Sliding (win)—and learned gates to combine their outputs (Eq. 5; Figure 2).

FR2: Compression builds overlapped blocks of size l with stride d<l and maps each block via a learnable ϕ to a compressed token (Eq. 7). (M0: avg pooling ϕ; M2: Conv1d+MLP).

FR3: Selection reuses p_cmp = softmax(Q·K_cmp^T) to compute block scores for selection blocks of size l'; if blockings differ, apply Eq. 9 mapping; sum across heads in each GQA group (Eq. 10); choose top‑n blocks (deterministic ties go to lower index), force‑include block 0 and the two local blocks, de‑duplicate, then merge adjacent selections into contiguous [start,end) ranges; attend over raw tokens in those ranges (Eqs. 8–12; Figure 2 right).

FR4: Sliding attends over the last w tokens with separate K/V to avoid shortcut learning (Section 3.3.3) and uses distinct cache tensors (`K_win`,`V_win`) separate from selection raw caches (`K_sel`,`V_sel`).

FR5: GQA consistency: all heads in a group share the same selected blocks (Eq. 10; kernel concept in Figure 3).

FR6: Causality: All three branches must not read indices > t (Figure 2, right panel).

FR7: Prefill + Decode support with per‑branch KV caches and compressed‑stream increments every d steps once warmed by l.

FR8: Selection kernel: custom Triton forward/backward with group‑centric Q loads and shared KV fetch per GQA group; inner loop iterates selected KV block chunks Bk | l' (Figure 3).

3.2 Non‑Functional Requirements (NFR)

NFR1 – Accuracy: NSA ≈ full attention for small sequences when configured to cover all tokens (MAE < 1e‑5 FP32).

NFR2 – Trainability: gradcheck passes on tiny dims; toy LM loss decreases smoothly (qualitatively like Figure 4).

NFR3 – Efficiency signals:

Decode token‑reads match formula and scale towards Table 4 expectations (rate‑of‑change).

Prefill/backward timing trend approaches Figure 6 curves (shape, not exact ms).

M0 implementation note:
- Masked varlen SDPA (sliding/compressed) and selection packing are enabled by default for performance, with `NSA_FORCE_PARITY=1` to force SDPA/gather reference paths.
- CPU fallback uses SDPA-only reference paths; FA‑2/Triton are not used in tests.

NFR4 – Stability: No future leakage; gates do not saturate early.

NFR5 – Local run: Single‑GPU A100/4090; BF16/FP32 fallback; CPU path for tiny debug.

4) References to the Paper (authoritative anchors)

Figure 2 (p.3) — Three‑branch architecture & gating; visualization of compute regions.

Eqs. 7–12 (pp.6–8) — Compression operator, compressed‑to‑selection mapping, group reduction, top‑n selection.

Section 3.3.3 (p.8) — Sliding window with per‑branch K/V to prevent shortcut learning.

Figure 3 (p.9) — Group‑centric kernel design for selection.

Section 4.1 (p.9–10) — Default hyperparameters: l=32, d=16, l'=64, n=16 (incl. 1 init + 2 local), w=512; G=4, H=64, d_k=192, d_v=128.

Table 4 (p.14) — Decode memory budget & expected speedup trend.

Figure 5 (p.12) — Perfect 64k needle retrieval.

Figure 6 (p.13) — Forward/backward speed curves (trend).

§6.1 + Figure 7 (p.15) — Why we avoid auxiliary losses/heuristics.

5) System Architecture
5.1 High‑Level Flow (per token t, per GQA group)
flowchart LR
  X[Hidden x_t] -->|W_Q + RoPE| Q[Q_t (all heads in group)]
  subgraph Branches
    direction TB
    Q --> CmpScore[p_cmp = softmax(Q · K_cmp^T)]:::op
    CmpScore --> MapM[p_slc = p_cmp @ M (Eq.9)]:::op
    MapM --> GroupSum[p'_slc = sum_h p_slc^(h)]:::op
    GroupSum --> Topk[I_t = top-n (force init + 2 local)]:::op
    Topk --> GatherSel[Gather raw K_sel,V_sel by blocks]:::mem
    GatherSel --> SelAttn[Attn(Q, K_sel, V_sel) — SDPA→Triton]:::op
    Kcmp[K_cmp,V_cmp via ϕ over (l,d)]:::mem
    Q --> CmpAttn[Attn(Q, K_cmp,V_cmp) — FA-2]:::op
    Q --> WinGather[Window K_w,V_w (last w)]:::mem
    WinGather --> WinAttn[Attn(Q, K_w,V_w) — FA-2]:::op
  end
  CmpAttn --> Gate[Gate MLP → softmax]:::op
  SelAttn --> Gate
  WinAttn --> Gate
  Gate --> O[Output O_t]
classDef op fill:#eef,stroke:#88a,stroke-width:1px;
classDef mem fill:#efe,stroke:#7a7,stroke-width:1px;


Source: Architecture & gating summarized in Figure 2 (paper p.3); selection equations on pp.7–8.

5.4 Sequence Length Handling

- M0–M3 assume fixed sequence length per batch (no ragged var‑length). All masks are causal.
- M6 may add var‑length/padded batch support via FA‑2 varlen wrappers; until then, fall back to dense SDPA for varlen experiments.

5.2 Decode Step Ordering
sequenceDiagram
  participant T as Token t
  participant W as Sliding window cache (K_win,V_win)
  participant Φ as Compressed stream (ϕ on RoPE(K_cmp,V_cmp))
  participant S as Selector (M_csl CSR, group-reduce, top-n)
  participant R as Raw KV cache (K_sel,V_sel)
  participant A as Attn branches + Gate

  T->>W: Read last w K/V (≤ t)
  T->>Φ: Emit compressed token every d steps after warmup l (≤ t); ϕ runs on RoPE-applied K/V
  T->>S: p_cmp from Q·K_cmp; map to p_slc via M; sum across heads; choose top-n + forced blocks
  S->>R: Gather contiguous raw KV ranges for selected blocks (≤ t), with dedup + merge
  T->>A: Run cmp / sel / win attentions; gate; sum → O_t


Source: Compression (Eq. 7), selection (Eqs. 8–12), sliding (§3.3.3).

5.3 Kernel Execution Model (Selection Branch)
flowchart TB
  subgraph GridLoop[Grid Loop: iterate (B, G, t)]
    Qload[Load all heads' Q for the group] --> Inner
  end
  subgraph Inner[Inner Loop: iterate selected KV block chunks (Bk | l')]
    Kfetch[Fetch K,V block chunk to SRAM] --> Dot[Q@K^T + LSE stats]
    Dot --> Acc[Accumulate P·V; update m,l]
    Acc --> Inner
  end
  Inner --> Write[Write O,m,l to HBM]


Source: Figure 3 shows group‑centric Q load, shared KV fetch, and inner loop over contiguous blocks (p.9).

6) Interfaces & Data Contracts
6.1 PyTorch Module API
class NSAAttention(nn.Module):
    def __init__(self, dim:int, n_heads:int, n_kv_groups:int, d_k:int, d_v:int,
                 l:int=32, d:int=16, l_sel:int=64, n_sel:int=16, w:int=512,
                 gate_hidden:int|None=None, gate_temp:float=1.0,
                 rope_impl:str="llama",
                 use_flash:bool=False, use_triton_sel:bool=False):
        ...

    def forward(self, x: Tensor, kv: "NSA_KV", *, prefill: bool) -> tuple[Tensor, "NSA_KV"]:
        """x: [B, S, dim] (prefill) or [B, 1, dim] (decode). Returns (y, updated kv).
        CPU fallback: selection path uses SDPA gather; cmp/win use SDPA; Triton/FA‑2 are optional.
        """

6.2 KV Caches
@dataclass
class NSA_KV:
    # selection branch raw KV cache (used only by selection)
    K_sel: Tensor  # [B,G,S,Dk]
    V_sel: Tensor  # [B,G,S,Dv]
    # sliding branch KV cache (distinct projection weights)
    K_win: Tensor  # [B,G,S,Dk]
    V_win: Tensor  # [B,G,S,Dv]
    # sliding window metadata
    win_ptr: Tensor  # [B,G] int positions
    # compressed stream & emission schedule
    K_cmp: Tensor    # [B,G,S_cmp,Dk]
    V_cmp: Tensor    # [B,G,S_cmp,Dv]
    cmp_emit_next: Tensor  # [B,G] int next emission
    # precomputed indices & mapping
    meta: "BlockMeta"

    # layout & ownership
    # Tensors are laid out as [B,G,S,D*], contiguous in D, group-major in memory.
    # Caches are owned per layer; no sharing across layers.

6.3 Block Metadata
@dataclass
class BlockMeta:
    l:int; d:int; l_sel:int; n_sel:int; w:int
    cmp_starts: Tensor  # [S_cmp] block start indices
    sel_starts: Tensor  # [S_sel] block start indices
    M_csl: Tensor       # sparse CSR map cmp->sel for Eq. 9 (fractional overlap weights)
    # M_csl encodes, for each compressed block, the selection blocks it overlaps with,
    # weighted by fractional overlap length; stored FP32 CSR on device and rebuilt
    # only when (l,d,l_sel) change.

6.4 Selection Scorer
def select_blocks_from_pcmp(
    p_cmp: Tensor, meta: BlockMeta,
    n_groups:int, h_per_group:int, n_top:int,
    force_init: bool=True, force_local:int=2
) -> Tensor:
    """Return [B, G, n_top, 2] start/end (exclusive) raw-token ranges."""

6.5 Triton Kernel Entry Points (M4/M5)
def sel_attn_fwd(Q: Tensor, K: Tensor, V: Tensor, sel_ranges: Tensor,
                 scale: float, causal: bool=True) -> tuple[Tensor, Tensor, Tensor]:
    """Q:[B,G,h,Dk]; K/V:[B,G,S,D*]; sel_ranges:[B,G,n_sel,2]; returns (O, m, l)."""

def sel_attn_bwd(dO: Tensor, Q: Tensor, K: Tensor, V: Tensor,
                 sel_ranges: Tensor, m: Tensor, l: Tensor, scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dQ, dK, dV with writes only to selected ranges."""

7) Default Hyperparameters

Blocks & windows: l=32, d=16, l'=64, n=16 (includes 1 initial + 2 local blocks), w=512.

Heads & dims: GQA groups G=4, total heads H=64 (16/head per group), d_k=192, d_v=128.
Source: Section 4.1 pretraining setup and NSA config.

8) Implementation Plan (Steel Thread → Full)
Milestones (each keeps tests green)

M0 — Steel Thread

SDPA everywhere; ϕ = average pooling; prefill + 1‑step decode; GQA selection with forced init + 2 local.

Acceptance: Causality tests, Eq.9 mapping property tests (CSR weighted overlap), group consistency test, small‑S equivalence, decode‑counter math (exact integer formula with early‑step handling).

M1 — FA‑2 for Compressed & Sliding

Swap SDPA→FA‑2 for cmp/win (paper: these branches compatible).

M1 notes:
- Integrate FA‑2 varlen/dense for cmp/win with packing buckets; keep SDPA reference and `NSA_FORCE_PARITY` fallback.
- Parity tolerance vs SDPA oracle: FP32 ≤ 5e‑5 (BF16/FP16 ≤ 2e‑4 if tested); identical softmax scale; no dropout/bias.
- Head_dim/device constraints: assert/xfail and fall back to SDPA when unsupported; CPU uses SDPA.

M2 — Learnable ϕ & Trainable Gates

Replace pooling with Conv1d(l,stride=d, causal) + tiny MLP; gate MLP learns (softmax).

Needle@64k must remain perfect.

M3 — Full Decode Caches

Rolling window; compressed emission every d after warmup l; selection reads raw K/V per top‑n blocks.

Decode token‑reads match formula trend (Table 4).

M4 — Triton Selection Kernel (Forward)

Figure‑3 schedule; numerics match SDPA gather (FP32 tol ≤ 1e‑4).

M5 — Triton Backward (Trainable End-to-End)

FA‑style LSE stats; dQ/dK/dV sparse writes; gradcheck passes; toy LM loss down (Figure 4 trend).

M6 — Perf & Robustness

Timing trends like Figure 6; dtype & shape stress; OOM resilience.

9) Development Practices

Frameworks: PyTorch ≥2.3; Triton ≥3.0; optional FlashAttention‑2 (≥2.x) for cmp/win; Hydra configs; PyTest + Hypothesis; ruff + mypy. Use `uv` for environment and commands.

Precision: BF16 weights / FP32 accumulators; FP32 reference path for tests.

torch.compile: fullgraph=False; disable in debug mode. Compile cmp/win branches optionally; keep selection path eager unless Triton is enabled.

Determinism for tests: fixed seeds; disable cudnn.benchmark; no dropout in tests.

Config‑driven: All l, d, l', n, w, G, H, d_k, d_v in YAML; ablations allowed.

CI: CPU unit tests & static checks (CPU fallback uses SDPA gather for selection); GPU CI job for small equivalence, decode counters, 8–16k needle smoke.

10) Repository Layout
nsa/
  core/
    nsa_attention.py
    compress_pool.py
    block_index.py
    selection_scorer.py
    rope.py
  kernels/
    flash_wrappers.py
    triton_sel_kernel/
      __init__.py
      sel_fwd.triton
      sel_bwd.triton
      microbench.py
  model/
    llama_block_nsa.py
    rmsnorm.py
    mlp.py
  cache/
    kv_cache.py
  tests/
    test_masks.py
    test_block_math.py
    test_group_consistency.py
    test_equiv_small.py
    test_ablation.py
    test_decode_counters.py
    test_gradcheck.py
    test_needle.py
  bench/
    bench_prefill.py
    bench_decode.py
  cli/
    demo_infer.py
  configs/
    base.yaml
    ablations.yaml
  scripts/
    train_toy.py
  README.md
  CONTRIBUTING.md
  pyproject.toml
  requirements.txt

11) Testing & Verification (kept from day 1)
A) Unit / Property (fast)

Causality (all branches): No index > t participates (Figure 2 right).

Eq. 9 mapping: Random p_cmp, verify p_slc = p_cmp @ M equals overlap sum; assert d|l and d|l'.

Eq. 10 group consistency: Selected I_t equal across heads in the same GQA group.

Shape/dtype checks across BF16/FP32.

B) Equivalence (deterministic)

Full‑coverage ≈ full attention: Configure w ≥ S and n*l' ≥ S → MAE < 1e‑5 FP32 (Eq. 5 aggregation).

C) Long‑Context Functional

Decode token‑reads vs Table 4: use the exact integer formula with early‑step handling; trend lines up with Table 4.

Needle@64k: Perfect retrieval across positions (Figure 5).

Needle dataset spec: deterministic generator (fixed vocab/alphabet, fixed seed), single “needle” token embedded at controlled depths; report retrieval accuracy across a grid of depths matching Figure 5; document dataset parameters in tests.

D) Trainability (M5+)

gradcheck: FP32 only, tiny dims; disable compile; set tolerances (e.g., rtol=1e‑3, atol=1e‑4). Run CPU reference and GPU kernel variants.

toy LM loss down (qualitative trend like Figure 4).

E) Perf (M4+)

Prefill/backward timing trend approximates Figure 6; decode latency scales with reduced reads (Table 4).

12) Bench & Observability

Benchmarks

bench_prefill.py: batch prefill (8k,16k,32k,64k), NSA vs FA‑2 (cmp/win).

bench_decode.py: per‑step latency & token‑reads vs S; compute expected vs actual tokens.

Instrumentation

Heatmaps for p_cmp and p_slc; overlay I_t (compare to Figure 2’s green regions).

Gate histograms (mean±std) to detect early saturation; log per‑branch means and std.

Selection distance histogram (local vs far).

Per‑step counters per branch: tokens read (and, where possible, bytes read) for cmp/sel/win.

13) Algorithms & Patterns (implementation details)
13.1 Compression ϕ

M0: Average pooling over [i*d : i*d+l) → produces K_cmp, V_cmp. Apply RoPE to K per branch before ϕ; ϕ runs on RoPE‑transformed K/V so p_cmp aligns with Q.

M2: Depthwise Conv1d(l, stride=d, causal) over time + tiny MLP on channel; update incrementally: when (t‑l) % d == 0, emit a compressed token from the last l raw K/V. Source: Eq. 7 + §3.3.1.

13.2 Selection Scoring & Mapping

Compute p_cmp = softmax(Q·K_cmp^T) (Eq. 8). Prefill: compute in tiled batches over compressed time with FA‑2/SDPA to bound workspace; tests run with deterministic flags.

Map to selection block scores with a precomputed sparse CSR matrix M_csl with fractional overlap weights (Eq. 9), then sum over heads in group (Eq. 10). Rebuild M_csl only when (l,d,l') change.

Top‑n with forced inclusion: block 0 + 2 local blocks covering [t‑2*l', t] (Eqs. 11–12). Deterministic tie‑break (lower index), then force‑include and de‑duplicate; finally merge adjacent blocks into contiguous [start,end) ranges. Clamp all ranges ≤ t.

13.3 Sliding Branch

Last w tokens, own K/V projections to avoid shortcut learning; attend with FA‑2. Source: §3.3.3. Sliding uses its own cache tensors (K_win,V_win); selection uses (K_sel,V_sel).

13.4 Gating

Group‑pooled query → MLP(Dk→Dk/2→3) with last‑layer weights initialized to zeros → softmax(gates/τ) with temperature τ (default 1.0) over 3 gates (paper uses sigmoid; we normalize for stability). Source: Eq. 5 defines gated sum.

13.5 Selection Kernel (Triton)

Grid loop over (B,G,t); inner loop iterates contiguous selected KV blocks in chunks of Bk with Bk | l'.

Load all heads’ Q for group to SRAM, share K/V fetch across heads; FA‑style LSE numerics with saved m,l.

Backward recomputes logits chunk‑wise and writes sparse dK,dV for selected ranges only. Source: Figure 3 & §3.4.

CPU fallback: Gather the selected raw token ranges and run PyTorch SDPA as the reference implementation (used in CPU CI and as correctness oracle for Triton numerics).

14) Config Examples

configs/base.yaml

model:
  dim: 2560
  n_heads: 64
  n_kv_groups: 4
  d_k: 192
  d_v: 128
nsa:
  l: 32
  d: 16
  l_sel: 64
  n_sel: 16
  w: 512
  phi: "avg"        # M2: "conv_mlp"
runtime:
  precision: "bf16"
  use_flash: false      # M1 -> true
  use_triton_sel: false # M4 -> true
  compile_cmp_win: false # optional torch.compile for cmp/win only

15) Acceptance Criteria (Definition of Done)

DoD‑M0

All unit tests (masks, Eq.9, Eq.10, shapes) pass.

Small‑S equivalence test passes (MAE < 1e‑5 FP32).

Decode counter equals predicted exact integer formula on synthetic runs (Table 4):
- num_cmp(S) = 0 if S < l else floor((S - l) / d) + 1
- reads(S) = num_cmp(S) + n*l' + min(w, S)
Document early‑step corrections when S < l or S < w.

DoD‑M2

Learnable ϕ integrated; gates train; Needle@64k = 100% retrieval (Figure 5).

DoD‑M4

Triton selection forward matches SDPA gather on selected tokens (FP32 tol ≤ 1e‑4) and shows lower HBM reads.

DoD‑M5

gradcheck passes; toy LM loss decreases (trend like Figure 4).

DoD‑M6

Prefill/backward trend approximates Figure 6; decode token‑reads trend matches Table 4.

16) Risk Register & Mitigations

R1: Block mapping (Eq. 9) mistakes → poor selection.
Mitigation: Property tests across random (l,d,l') with d|l, d|l'; assert invariants.

R2: GQA inconsistency → high KV traffic.
Mitigation: Group‑reduce before top‑k; test identical I_t across heads (Eq. 10).

R3: Future leakage via forced local blocks.
Mitigation: All ranges clamped to ≤ t; branch‑level causal masks.

R4: Gate collapse (one branch dominates).
Mitigation: Softmax gates + temperature; monitor histograms; ensure per‑branch K/V. Source: §3.3.3 warns of shortcut learning.

R5: Kernel mismatch vs SDPA gather.
Mitigation: FP32 first; Bk | l'; verify LSE stats; compare numerics in tests. Source: Figure 3.

17) Developer Tasks (by file / agent)

Agent A — block_index.py: Build compression/selection block indices; precompute sparse M (Eq. 9); pack [start,end) ranges.

Agent B — compress_pool.py: M0 avg pooling ϕ; M2 Conv1d(l,stride=d,causal)+MLP; decode emission schedule (Eq. 7).

Agent C — selection_scorer.py: p_cmp (Eq. 8) → p_slc (Eq. 9) → group sum (Eq. 10) → top‑n with forced blocks (Eqs. 11–12).

Agent D — nsa_attention.py: Wire 3 branches; gates (softmax, τ, zero‑init last layer); RoPE; prefill + 1‑step decode; causal masks; CPU fallback for selection; compile cmp/win optionally.

Agent E — tests/: Unit/property tests (A), small equivalence (B), decode counters (C), needle harness (C), gradcheck (D).

Agent F — triton_sel_kernel/: Kernel forward (M4) then backward (M5) per Figure 3; micro‑autotune Bk, num_warps, num_stages.

18) Pseudocode (Forward pass: M0 logic)
def forward_nsattention(x, kv, prefill: bool):
    # Projections
    Q = rope(apply_W_Q(x))               # [B,S,Hq,Dk]
    K_raw_cmp, V_raw_cmp = apply_W_KV_cmp(x)  # [B,G,S,D*]
    K_raw_slc, V_raw_slc = apply_W_KV_slc(x)
    K_raw_win, V_raw_win = apply_W_KV_win(x)

    # Update caches (prefill or decode-step)
    kv.update_selection_raw(K_raw_slc, V_raw_slc)  # selection-only raw cache
    kv.update_window(K_raw_win, V_raw_win)         # maintain last w tokens (sliding cache)
    kv.update_compressed(K_raw_cmp, V_raw_cmp, phi, l, d)  # emit every d after l

    # ==== Compressed branch ====
    K_cmp, V_cmp = kv.K_cmp, kv.V_cmp
    O_cmp, p_cmp = attn_and_scores(Q, K_cmp, V_cmp)  # p_cmp: softmax logits per Eq. 8

    # ==== Selection branch ====
    p_slc = map_cmp_to_sel(p_cmp, kv.meta.M_csl)                 # Eq. 9 (CSR weighted)
    p_grp = group_reduce(p_slc)                                  # Eq. 10
    sel_ranges = topn_ranges(p_grp, n_sel, force_init=True, local=2,
                             tie_break="low", dedup=True, merge_adjacent=True,
                             clamp_to_t=True)                    # Eqs. 11–12
    O_slc = sdpa_over_ranges(Q, kv.K_sel, kv.V_sel, sel_ranges)  # M4→ triton or SDPA

    # ==== Sliding branch ====
    win_ranges = kv.get_window_ranges(w)
    O_win = sdpa_over_ranges(Q, kv.K_win, kv.V_win, win_ranges)

    # ==== Gating & merge ====
    g = softmax(gate_mlp(group_pool(Q)), dim=-1)     # Eq. 5 (paper uses sigmoid; we normalize)
    O = g[...,0]*O_cmp + g[...,1]*O_slc + g[...,2]*O_win
    return O, kv


Anchors: Eqs. 5, 7–12, §3.3.3; Figure 2 flow.

19) Commands / Dev Ergonomics
# Env setup (first time)
uv venv -p 3.10 .venv
uv pip sync -r requirements.txt

# Fast tests
uv run -q pytest

# Long-context tests (needle & counters)
uv run -q pytest -m long

# Benchmarks
uv run python bench/bench_prefill.py --config configs/base.yaml
uv run python bench/bench_decode.py  --config configs/base.yaml

# Demo
uv run python cli/demo_infer.py --config configs/base.yaml

# CPU fallback
# On CPU or without Triton/FA‑2, selection falls back to SDPA gather; cmp/win use SDPA.

20) Open Questions (tracked; default answers chosen)

Gating activation: Paper uses sigmoids per branch; we use softmax for scale stability. Decision: keep softmax; add temperature.

ϕ operator detail: Paper states “MLP with intra‑block position encoding”; we implement Conv1d+MLP for hardware friendliness. Decision: adopt Conv1d+MLP.

Divisibility constraints: Paper prefers d|l and d|l'. Decision: enforce; fall back to general overlap map only when explicitly configured.

FA‑2 wrapper specifics: Use varlen/dense packed wrappers as available in FA‑2 ≥2.x. If unavailable, fallback to SDPA with causal masks. Minimum supported FA‑2 version documented in requirements.

21) Appendix — Why this design is sound

Three‑branch NSA with learned gates balances global awareness (compressed), precise retrieval (selected), and locality (sliding). Ref: Figure 2; Eq. 5.

Trainable selection via p_cmp → p_slc → top‑n avoids aux‑loss pitfalls and supports end‑to‑end training. Ref: Eqs. 8–12; §6.1 critique of aux/heuristics (Figure 7).

Hardware alignment: cmp/win reuse FA‑2; selection kernel is group‑centric with shared KV, chunked inner loop Bk | l'. Ref: §3.4; Figure 3.

Efficiency: Decode memory budget and speedup trend follow Table 4; training/prefill trends like Figure 6.

Sequence length stance: M0–M3 use fixed sequence length per batch. Var‑length/padded sequences and FA‑2 varlen APIs can be introduced in M6.

Visual Anchors from Paper (use while implementing)

Figure 2 (p.3) shows left‑panel 3‑branch pipeline and right‑panel green compute regions—mirror our heatmaps / selected ranges.

Figure 3 (p.9) illustrates kernel loops; follow that for Triton layout and autotuning knobs.

Figure 5 (p.12) needle plot → target = 100% across depths @64k.

Table 4 (p.14) decode access budget table — use as the acceptance curve.

22) File Structure (final copy/paste)
nsa/
  core/{nsa_attention.py, compress_pool.py, block_index.py, selection_scorer.py, rope.py}
  kernels/{flash_wrappers.py, triton_sel_kernel/{__init__.py, sel_fwd.triton, sel_bwd.triton, microbench.py}}
  model/{llama_block_nsa.py, rmsnorm.py, mlp.py}
  cache/kv_cache.py
  tests/*.py
  bench/{bench_prefill.py, bench_decode.py}
  cli/demo_infer.py
  configs/{base.yaml, ablations.yaml}
  scripts/train_toy.py
  README.md, CONTRIBUTING.md, pyproject.toml, requirements.txt

requirements.txt (versions pinned; example)
- torch==2.3.*  # match CUDA toolkit installed
- triton==3.*
- flash-attn==2.*  # optional; guard imports
- numpy
- pydantic
- hydra-core
- pytest
- hypothesis
- ruff
- mypy

Final Reminder

Use the paper defaults: l=32, d=16, l'=64, n=16 (incl. 1 init + 2 local), w=512. These are explicitly stated in Section 4.1 and are the parameters assumed by the kernel and decode economics in Table 4. Keep GQA group‑consistent selection as non‑negotiable to preserve the bandwidth wins.