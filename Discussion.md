***** CONTEXT *****

THIS TRANSCRIPT IS THE DISCUSSION THAT LEAD TO THE PRD AND MAY CONTAIN ADDITIONAL INFORMATION.

**********************





Hello there! We are going to embark upon a grand journey where we are going to re-implement native sparse attention in here as designed. It is going to be very challenging, and I need you to think extremely hard and make a detailed implementation plan for how to do this. Notably this is going to be done entirely by coding agents, which means our analysis, discussion, and documentation should be specific for those coding agents to implement this. Please review this paper, and remember you are the lead engineer here, so I want you to be confident and strong in your opinions of how you want to implement this. The goal being that we should be able to run this implementation locally and actually get it to work. 
ChatGPT said:
Request for GPT-5 Pro
Reasoned for 5m 1s

Got it. I’ll lead with a concrete, opinionated plan the coding agents can implement end‑to‑end to get Native Sparse Attention (NSA) running locally. I’ll reference the paper explicitly so our choices stay faithful to the design. Where the paper is underspecified, I’ll make pragmatic, hardware‑aligned decisions and mark them as such.

0) What we’re building (in one screen)

NSA module for decoder‑only Transformers that replaces full attention with three parallel branches per layer—Compressed, Selected, Sliding—whose outputs are gated and summed. Selection is blockwise and group‑consistent (GQA/MQA‑friendly), compressed tokens are produced by a learnable intra‑block ϕ mapping (MLP/Conv), and sliding covers local recency. The selection branch needs a custom Triton kernel; compressed and sliding can ride FlashAttention‑2. See Figure 2 (page 3) and §3 for the algorithm; Figure 3 (page 9) for the kernel schedule.

Key paper specs we will honor by default:

Three branches with learned gates (sigmoid/logit → softmax) aggregating branch outputs (Eq. 5, §3.2, p.6–7).

Compression over overlapping blocks (length l, stride d < l) via learnable ϕ (Eq. 7, §3.3.1, p.6). We’ll set l=32, d=16.

Selection reuses the compressed attention scores to score larger selection blocks (size l', top‑n blocks), with head‑aggregation inside GQA group (Eqs. 8–12, §3.3.2, p.7–8). Defaults: l'=64, n=16 including “always‑on” initial + 2 local blocks.

Sliding window over last w tokens; default w=512 (§3.3.3, p.8). Separate K/V for each branch to avoid shortcutting.

Kernel: group‑centric query load, shared KV fetch per GQA group, inner loop over selected KV blocks (Figure 3, §3.4, p.8–9).

This is designed to train end‑to‑end and to speed up forward/backward and decode for long contexts (Figure 6 and Table 4 show the shape of expected wins). We’ll keep the arithmetic‑intensity logic in mind for kernel layout (pp.5–6, §3.1; §5).

1) Deliverables & local run target

Repository layout

nsa/
  core/
    nsa_attention.py          # PyTorch module with three branches & gating
    compress_pool.py          # ϕ implementation (Conv1d or MLP pooling)
    block_index.py            # block partitioning & index transforms
    selection_scorer.py       # p_cmp -> p_slc transforms (Eqs. 8–10)
    triton_sel_kernel/
      sel_fwd.triton          # selection attention forward (Figure 3)
      sel_bwd.triton          # backward
  kernels/
    flash_wrappers.py         # wrap FA-2 for cmp & win branches
  model/
    llama_block_nsa.py        # LLaMA-style block w/ GQA + NSA
  cache/
    kv_cache.py               # per-branch KV caches + incremental updates
  tests/
    test_shapes.py
    test_block_math.py
    test_equivalence_small.py
    test_needle.py            # 64k “needle-in-haystack” harness
    test_gradcheck.py
  bench/
    bench_prefill.py
    bench_decode.py
  cli/
    demo_infer.py             # run on sample prompts
  README.md
  requirements.txt


Environment

Python ≥3.10, PyTorch ≥2.3 (SDPA & Triton integration), Triton ≥3.0, optional FlashAttention‑2 for cmp/win branches.

GPU: A100/4090 class (BF16/FP16). CPU fallback (slow) is included for debugging (baseline correctness).

2) Data layout & notation (decisions you can code against)

We implement GQA (grouped query attention); per group we share KV caches and selection. This is required by the paper’s kernel and for decode memory‑bandwidth wins (§2.1 and §3.4).

Shapes (per layer, per batch):

B: batch, S: seq, Hq: query heads, G: GQA groups, Hq = G * h (h heads per group), Hkv = G.

Dk, Dv: head dims.

Projections:

Shared Q: Q = x @ W_Q → shape [B, S, Hq, Dk] with RoPE.

Branch‑specific K/V: (K^cmp,V^cmp), (K^slc,V^slc), (K^win,V^win) use distinct W_K/W_V per branch (§3.3.3). Each stored per group as [B, G, S, Dk] and [B, G, S, Dv].

Block partitioning (time axis):

Compression: overlap blocks [i*d : i*d + l) with l=32, d=16. Outputs compressed K/V at indices i=0..⌊(S-l)/d⌋.

Selection: non‑overlapping blocks of size l' (default 64).

Sliding: last w tokens (default 512).

Index maps:

For a query position t, define:

cmp_blocks(t): all compression blocks that end ≤ t (causal).

slc_blocks(t): all selection blocks ≤ t.

win_range(t): [max(0, t-w+1) … t].

3) Algorithmic pipeline per token/block
3.1 Compressed branch (Eq. 7)

Goal: produce a compact set of keys/values for coarse global scan.
ϕ implementation (opinionated): use a light depthwise Conv1d over time on K (and V) with kernel=l, stride=d, causal padding so each compressed vector summarizes its block with learned positional mixing; follow with a tiny MLP on channel Dk/Dv. This matches “MLP with intra‑block position encoding” while being kernel‑friendly.

Prefill/training:

Build K_cmp, V_cmp for entire prefix by convolving K/V (per group).

Compute Attn(Q_t, K_cmp_≤t, V_cmp_≤t) using FA‑2 (blockwise), causal mask applied at the compressed time scale (§3.4 “Compression and Window compatible with FA‑2”).

Decode (incremental):

Maintain a ring buffer of last l per group. When (t - l) % d == 0, emit a new compressed token from K[t-l+1:t] via the same ϕ and append to K_cmp. (Paper’s overlapping blocks with d<l; this keeps compressed stream up to date.)

3.2 Selection branch (Eqs. 8–12)

Key idea: Don’t score every raw block with Q. Reuse compressed attention scores p_cmp to induce selection block scores p_slc. This makes selection cheap and trainable end‑to‑end.

Steps per token t, per GQA group:

Score compressed keys
p_cmp(t) = softmax(Q_t · K_cmp_≤t / √Dk) (Eq. 8). (Use FA‑2 matmul or a small GEMM since K_cmp is short.)

Map to selection‑block scores
If compression and selection granularity match, p_slc = p_cmp. Else compute via the overlap mapping in Eq. 9 (sum of p_cmp that overlap a given selection block); we’ll precompute a sparse mapping matrix M of shape [#cmp_blocks, #slc_blocks] respecting l, d, l' and implement p_slc = p_cmp @ M.

Group consistency for GQA
Sum p_slc across heads in the group to get shared block scores p'_slc (Eq. 10). This ensures one block set per group and maximal KV reuse.

Top‑n selection
Build I_t = topk_indices(p'_slc, n), forcing inclusion of block 0 and the two local blocks that cover [t-2*l', t] (paper default). Gather contiguous slices of raw K_slc, V_slc for those blocks (Eqs. 11–12).

Attention
Run attention per group on the concatenated selected tokens using the custom Triton selection kernel (below). Output shape [B, h, Dv] per group, then reshape to heads.

3.3 Sliding branch (§3.3.3)

Gather K_win = K[t-w+1:t], V_win = V[t-w+1:t] (per group).

Compute FA‑2 attention on this small window.

Important: Separate K/V projections per branch to prevent “shortcut learning” (paper), i.e., sliding doesn’t fully dominate.

3.4 Gating and merge (Eq. 5)

For each query head vector Q_t^(h), compute gate logits u ∈ ℝ^3 via an MLP([Dk→Dk/2→3]) on the group‑pooled query (mean over h in group) for group consistency and lower overhead.

Apply softmax on u to get g_cmp, g_slc, g_win (paper uses sigmoid; we normalize to sum‑1 for numerics—clear, defendable choice).

Output: O_t = g_cmp * O_cmp + g_slc * O_slc + g_win * O_win. (Eq. 5, §3.2).

4) Selection kernel (Triton) — what to code

Match Figure 3 (p.9): load by GQA group & time (one position), iterate selected KV blocks in the inner loop.

FWD kernel sketch (per B,G,t)

Grid loop: tiles over (B, G, t) and head subtiles (load all h heads’ queries for the group at position t into SRAM).

Inner loop: iterate selected blocks I_t in block size Bk that divides l' (e.g., 32).

For each block chunk: coalesced load K/V slices [Bk, Dk] / [Bk, Dv] into SRAM.

Compute Q @ K^T for all h heads sharing K; apply causal mask if the block overlaps future (won’t, by construction), scale and row‑wise logsumexp (like FA‑2) for stability.

Accumulate softmax‑weighted V into O and keep the usual FA‑style stats (m, l) for backward.

HBM writes: write final O and stats.

BWD kernel

Mirror FA‑style backward: reconstruct softmax using saved stats and recompute P = softmax(QK^T) on the fly, accumulate dQ, dK, dV for only the selected positions (sparse) and coalesce writes per selected block.

Why this layout:

Minimizes KV HBM reads (shared across heads) and balances arithmetic intensity; that is the paper’s main hardware point (§3.4; §5.1) and enables the reported forward/backward gains (Figure 6).

5) Prefill vs decode paths (systems behavior)

Prefill/training (compute‑bound):

Build K_cmp/V_cmp once per layer (Conv/ϕ).

Compute p_cmp in big tiles; map to p_slc with prebuilt M; select I_t batched; run selection kernel; run FA‑2 for cmp & win branches. This preserves FlashAttention‑level throughput for cmp/win (§3.4) and lets selection piggyback the compressed scoring (§3.3.2).

Decode (memory‑bound):

Maintain three caches per branch:

Sliding: rolling window of last w K/V.

Compressed: incremental emission every d steps once the last l are available.

Selection: raw K/V cache by block; we load only the top‑n blocks per step.

Expected HBM reads drop from S (full attention) to ~( (S-l)/d + n*l' + w ) tokens per step (Table 4), enabling ~4×–11.6× speedups as context grows (§5.2). We’ll verify these with counters.

6) Backward pass (training‑ready)

Gates: gradients flow through the 3‑way softmax MLP.

Branch grads:

Sliding/Compressed: use FA‑2 backward or PyTorch SDPA backward; standard.

Selection: our kernel computes dQ/dK/dV only for the selected tokens; no grads to unselected K/V. This is correct because the selection op is not discrete in our graph—the scores used to choose blocks come from p_cmp, which are in the graph. The top‑k itself remains non‑diff, but the paper sidesteps auxiliary losses by letting training adjust p_cmp and ϕ so the chosen blocks trend optimal (§3.3.2 + §6.1 contrast to aux‑loss methods).

Rationale: The paper emphasizes trainable end‑to‑end without aux losses and highlights the pitfalls of non‑trainable or high‑overhead selection (cluster/hash/aux‑loss), which we avoid (§2.2; §6.1, Figure 7).

7) Exact APIs for coding agents
7.1 NSAAttention module
class NSAAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_groups, d_k, d_v,
                 l=32, d=16, l_sel=64, n_sel=16, w=512,
                 rope_impl="llama", use_flash=True, use_triton_sel=True):
        ...

    def forward(self, x, kv_caches, attn_mask=None, prefill=False):
        """
        x: [B, S, dim]
        kv_caches: dict with per-branch K/V caches & indices (see below)
        prefill: True for training/prefill; False for decode (single or few tokens)

        returns: y [B, S, dim], updated kv_caches
        """

7.2 KV caches (per layer)
@dataclass
class NSA_KVCache:
    # raw per-branch KV for selection & sliding
    K_raw: Tensor  # [B,G,S,Dk]
    V_raw: Tensor  # [B,G,S,Dv]
    # compressed streams + index maps
    K_cmp: Tensor  # [B,G,S_cmp,Dk]
    V_cmp: Tensor  # [B,G,S_cmp,Dv]
    cmp_meta:  Any # mapping tensors for (t)->cmp_block_range
    # selection block index metadata
    sel_meta:  Any # mapping tensors (p_cmp->p_slc), block starts/ends of size l'
    # sliding window pointers
    win_meta:  Any # headless ring-buffer cursors

7.3 Selection scorer
def selection_blocks_from_pcmp(p_cmp, sel_meta, group_reduce=True, k=n_sel,
                               force_blocks=("init","local2")) -> List[BlockIdx]:
    """
    p_cmp: [B,G,heads_in_group?, S_cmp]
    returns per (B,G,t): top-n selection block ids (integers)
    """

7.4 Triton selection kernel entry
def sel_attn_forward(Q_t, K_raw, V_raw, sel_blocks, l_sel, causal=True):
    """
    Q_t: [B,G,h,Dk]    # heads in group
    K_raw/V_raw: [B,G,S,Dk/Dv]
    sel_blocks: list/packed tensor of [B,G,n_sel,2] (start,end) block ranges
    returns O_t: [B,G,h,Dv], saved_stats
    """

8) Default hyperparameters (paper-faithful)

Use the paper’s 27B config for defaults where applicable, but our module remains size‑agnostic:

GQA groups: e.g., 4 groups, Hq=64, h=16 per group, Dk=192, Dv=128. (§4.1, p.9–10)

Blocks: l=32, d=16, l'=64, n=16 (including block 0 and 2 local), w=512. (§4.1, Table text p.10)

Gates: softmax over 3 branches (paper uses sigmoid; we normalize).

9) Tests & acceptance criteria

Correctness

Shape/Mask tests: causal masking honored across all branches; selection never crosses t.

Small‑S equivalence: for S≤w and n_sel large enough to cover all history, NSA ≈ Full Attention (MAE < 1e‑5 in FP32).

Gradcheck: run PyTorch gradcheck on tiny dims to verify backward stability (FP32).

Block math tests:

Verify M (cmp→sel map, Eq. 9) is a proper overlap transform for varied l,d,l'.

Verify GQA reduce (Eq. 10) yields one shared block set per group.

Functionality
5. Needle‑in‑a‑Haystack (64k): achieve perfect retrieval at all depths like Figure 5 (§4.3). Our harness will generate the dataset and assert 100% accuracy.
6. Gate behavior: ablate branches (force gate to one‑hot) and verify each branch’s partial competence: sliding handles local; selection recovers distant blocks once p_cmp spikes; compressed alone yields coarse recall.

Performance (indicative, not gated)
7. Prefill/Backward timing: show monotonic speedup vs FA‑2 as S rises, qualitatively matching Figure 6 trends.
8. Decode HBM reads: count loaded tokens per step and confirm near‑linear reduction with Table 4 expectations as S grows (§5.2).

10) Implementation order of work (for coding agents)

Scaffold & baselines

Implement full‑attention LLaMA‑style block with GQA using PyTorch SDPA.

Add branch‑specific K/V projections and shared Q (with RoPE). Gate MLP returns uniform weights (sanity).

Compression branch

Implement ϕ as Conv1d+MLP; prefill path: build K_cmp/V_cmp.

Integrate FA‑2 (or SDPA) to attend Q→K_cmp.

Unit tests: cmp stream length and masking; compare Conv1d vs reference average pooling to ensure determinism.

Selection scorer (training‑ready)

Compute p_cmp (Eq. 8).

Build mapping M for Eq. 9 and validate with tests.

Implement group reduce (Eq. 10) and top‑n with forced blocks (Eq. 11–12).

Selection kernel (decode‑first)

Implement forward Triton kernel per Figure 3 (group‑centric); start with FP32 accumulation, move to BF16.

Build Python wrapper packing contiguous [start:end) blocks; validate vs SDPA on gathered tokens.

Sliding branch

Rolling window cache + FA‑2/SDPA attention.

Gating

Gate MLP + softmax; confirm gradients flow; add temperature knob for exploration during training.

Backward

Triton bwd for selection (FA‑style); use saved stats.

Confirm end‑to‑end gradcheck.

Decode caches

Implement incremental update rules:

Sliding ring buffer;

Compressed emission every d steps after l warm‑up;

Selection uses raw KV; only read top‑n blocks each step.

Add token‑read counters to verify Table 4 curve.

End‑to‑end demos & Long tests

Run needle test (64k).

Benchmark prefill/decoder (qualitative trend vs FA‑2/SDPA).

11) Design decisions & justifications (strong opinions)

ϕ as Conv1d (not gigantic MLP): preserves locality, vectorizes well, and is trivial to update incrementally at decode. The paper only requires “learnable MLP with intra‑block position encoding” (Eq. 7); Conv1d+MLP is a hardware‑aligned instance.

Gates normalized by softmax: The paper uses sigmoids and a gate per branch (§3.2). We normalize to keep output scale stable across context and training regimes and to reduce the burden on following MLPs.

Group‑consistent selection is mandatory: Without it, GQA’s KV sharing breaks and you lose the memory‑bandwidth advantage highlighted in §2.1 and exploited in §3.4. We enforce this at the API level.

No auxiliary losses: The paper shows aux‑loss or heuristic selectors underperform or add overhead (Figure 7, §6.1). We keep the training path clean and let p_cmp learn to be predictive.

FA‑2 for cmp & win only: Exactly as the paper states—our custom kernel is only for selection (§3.4).

12) Edge cases & numerics

Early tokens: When t < w or t < l, sliding and compressed branches operate on shorter effective windows; selection’s forced initial block guarantees coverage.

Overlap mapping (Eq. 9): requires d | l and d | l' (paper assumption). We’ll assert these at init; otherwise we rebuild M from general overlaps (slower).

Precision: Use BF16 storage with FP32 accumulators in kernels; store FA‑style stats (m, l) per (B,G,h).

RoPE application: apply after Q/K projection, per branch for K (and Q shared), to preserve comparable phases across branches.

13) Minimal runnable example (included in cli/demo_infer.py)

Build a 2‑layer, 8‑head, 2‑group toy model with NSAAttention.

Tokenize a long synthetic prompt (80k tokens on dummy vocab), run prefill + a short decode; print token throughput, loaded‑token counters, and verify selection is sparse.

Run the needle test at 64k (expect 100% retrieval like Figure 5).

14) What “done” looks like

Correctness: unit tests pass; 64k needle succeeds at 100% retrieval.

Trainability: gradient checks pass, loss on a tiny LM decreases smoothly (paper reports stable convergence and even better loss vs full attention—Figure 4—our toy should show stability).

Efficiency signals: decode reads scale close to Table 4; prefill/grad timing shows widening gap vs full attention with longer context (trends in Figure 6).

15) Appendix: Paper anchors for implementers

Architecture overview & three branches: Figure 2, §3.2–3.3 (pp.6–8).

Selection math: Eqs. 8–12, block scoring, GQA aggregation (pp.7–8).

Kernel schedule: Figure 3, §3.4 (p.8–9).

Hyperparameters proven in 27B run: §4.1 (p.9–10).

Decode memory economics: Table 4, §5.2 (p.14).

Why not aux‑loss/heuristics/clustering: §2.2, §6.1 & Figure 7 (pp.4–5, 15).

If you want, I can start by drafting the exact NSAAttention PyTorch class skeleton and the Triton kernel headers so the agents can fill in the internals.

You said:
Okay, we need a concept of a steel thread, a core workflow that we implement and validate early and then enhance with the rest so that we know that at least the basic functionality works. We need a testing and verification structure that we can validate as we add or improve the capabilities of this paper so that we don't get too off base. Does all of that make sense? Please push back if it doesn't.
ChatGPT said:
Request for GPT-5 Pro
Reasoned for 4m 11s

Absolutely—this makes sense, and I’m glad you called it out now. Here’s the steel thread I want the coding agents to build first, plus the verification scaffold that will let us iterate confidently without drifting from the paper.

Anchor to the paper: NSA combines compressed, selected, and sliding branches, gated and summed per query (Figure 2 and §3.2–3.3). Selection reuses compressed scores (Eqs. 8–12). Only the selection branch needs a custom kernel later; compressed and sliding can use FA‑2/SDPA now. Figure 3 shows the group‑centric kernel we’ll implement after the steel thread. Table 4 defines decode memory economics we’ll track from day one.

Steel Thread (M0): a minimal, verifiable, end‑to‑end NSA

Goal: Replace a full‑attention layer with a working NSA module that runs prefill and 1‑step decode, produces sane outputs, and is already testable against the paper’s math—even before any custom kernels.

Decisions for M0 (narrow, deliberate):

Queries shared; K/V per branch (as in §3.3.3) with GQA grouping already wired (KV shared within a group).

Compression ϕ (temporary): non‑parametric average pooling over overlapping blocks (l=32, d=16) for determinism; we’ll swap to learnable ϕ (Conv1d+tiny MLP) in M2. (Eq. 7, §3.3.1.)

Selection scoring: compute p_cmp from compressed attention (Eq. 8), map to selection‑block scores with a precomputed sparse matrix M (Eq. 9), group‑reduce across heads (Eq. 10), top‑n block IDs (l' = 64, n=16, with forced initial + 2 local blocks) and gather raw K/V for attention using PyTorch SDPA (no Triton yet). (Eqs. 8–12, §3.3.2.)

Sliding window: gather last w=512 tokens per group; attend with SDPA. (§3.3.3.)

Gating: simple softmax over 3 branches from a small MLP on the group‑pooled query; initialize weights to return ~uniform gates (stable outputs; §3.2 describes gating). We’ll keep softmax (paper shows sigmoid; we normalize for scale stability).

Prefill + 1‑step decode only: per‑branch KV caches; compressed stream updates every d tokens once l are available (decode rule from Eq. 7). We’ll generalize decode in M3.

Why this is the right vertical slice: it exercises all three branches, the block math, GQA group consistency, and the p_cmp→p_slc pathway that is central to NSA’s trainability, while deferring only the Triton selection kernel. It’s faithful to Figure 2 and Eqs. 7–12 but uses SDPA/FA‑2 where the paper says it’s compatible (compressed & sliding) and a PyTorch gather for selected blocks.

Verification scaffold (works from M0 and scales with features)

We’ll build tests, instruments, and acceptance gates that we keep green as we add capabilities.

A) Invariants & unit tests (fast, run on every commit)

Causal mask safety

For random shapes, assert no attention beyond t in all branches; selection never includes future blocks. Derived from Figure 2 right panel.

Block math correctness

Property tests for M (compression→selection mapping, Eq. 9): sums of p_cmp covering a selection block equal p_slc for varied (l,d,l') that satisfy d|l and d|l'. We assert those constraints as per §3.3.2.

GQA group consistency

After group‑reduce (Eq. 10), all heads in the group must see identical I_t.

Shapes & dtype stability

Check [B,S,Hq,D*] consistency, mixed precision (BF16/FP32 accumulators).

B) Equivalence & sanity tests (run locally + in CI “gpu” job)

Full‑coverage equivalence (small‑S)

Set w ≥ S, n*l' ≥ S, and include block 0 → NSA must match full attention within 1e‑5 (FP32). This pins down the merge/gate math (Eq. 5).

Branch ablation sanity

Force gates one‑hot:
a) only sliding → matches full attn when w ≥ S;
b) only selection with n*l' ≥ S;
c) only compression when we make l=d (no overlap) for a small S. Ties back to §3.3.1–3.3.3.

C) Long‑context functional tests (nightly or on demand)

Needle‑in‑a‑Haystack @ 64k (prefill)

Our harness mirrors Figure 5: ensure 100% retrieval at all positions with the M0 pipeline (gates uniform or lightly trained). This validates hierarchical retrieval (compressed→selected). (Figure 5, §4.3.)

Decode token‑read counter

Instrument: per decode step, count tokens loaded: ((S-l)/d) + n*l' + w. Track the curve vs Table 4 expectations as S grows (we expect ~4×→11.6× effective reduction). (Table 4, §5.2.)

D) Training‑readiness (once backprop lands)

Gradcheck (tiny dims)

Run PyTorch gradcheck on M2+ to validate stability (paper emphasizes end‑to‑end trainability, §3, §6.1).

Loss‑down test

Over a tiny LM corpus, verify loss decreases, echoing the stable pretraining loss trend in Figure 4 (qualitative).

Instrumentation views:

p_cmp vs p_slc heatmaps per token;

Selected block IDs overlayed on the timeline;

Per‑branch gate histograms;

Decode token‑reads chart vs S (compared against Table 4 line). These directly reflect Figure 2/3 behavior and §5.2 claims. 

Milestone ladder (each step keeps tests green)

M0 — Steel Thread (this sprint)

PyTorch NSA module with three branches, GQA, SDPA attention everywhere.

ϕ = average pooling; mapping M implemented; group‑consistent top‑n selection.

Prefill + 1‑step decode caches; counters for token‑reads.

Tests A1–A4, B5–B6, C8 on small S; basic needle harness scaffold.

M1 — FlashAttention‑2 in cmp & win

Swap SDPA→FA‑2 for compressed + sliding branches (paper notes compatibility, §3.4). Keep selection on SDPA. Update throughput benchmarks.

M2 — Learnable ϕ (+ gating training)

Replace pooling with Conv1d(l,stride=d, causal) + tiny MLP per §3.3.1; enable gate MLP learning (§3.2). Re‑run B5/B6 and needle @ 64k.

M3 — Full decode path

Proper rolling caches: sliding ring buffer; compressed emission every d; selection reading raw K/V per chosen blocks. Expand decode tests C8 over many steps (compare counters to Table 4).

M4 — Triton selection kernel (forward)

Implement Figure 3 schedule: group‑centric Q load, shared KV inner loop over selected blocks, FA‑style numerics. Validate vs SDPA gather (max error <1e‑4 FP32). (§3.4 + Figure 3.)

M5 — Backward for selection (trainable end‑to‑end)

FA‑style backward with saved stats (m,l), grads only for selected tokens; p_cmp path remains trainable by design (§3.3.2, §6.1). Add D9–D10.

M6 — Perf & robustness

Bench prefill/decoding vs FA‑2; expect trend lines similar to Figure 6. Stress tests on dtypes/shapes.

Test/bench harness the agents can implement now

Repo targets

pytest -q                          # fast unit tests (A1–A4)
pytest -q -m long                  # long-context (C7–C8)
python bench/bench_prefill.py      # throughput vs FA-2 on cmp/win
python bench/bench_decode.py       # token-reads curve (Table 4 check)
python cli/demo_infer.py           # sample prompt, print selected blocks + gates


Key fixtures

make_toy_model(): 2 layers, 8 heads, 2 GQA groups.

rand_inputs(B,S,dim,seed) ensuring reproducibility.

full_attention_reference(x) for equivalence test B5.

needle_dataset(L=64_000, depth_grid=…) replicating Figure 5 distribution.

Metrics captured per run

tokens_loaded_decode(t): actual vs expected ((S-l)/d)+n*l'+w. (Table 4.)

selected_blocks_hist: distribution of block distances (sanity: mixture of local + far).

gate_means: per-branch average gate; alerts if one branch saturates early.

Roles & work split for coding agents (M0)

Agent A (block_index.py): Build overlapping compression blocks (l=32,d=16) and non‑overlapping selection blocks (l′=64); precompute sparse M for Eq. 9; utilities to pack [start,end) for top‑n blocks.

Agent B (compress_pool.py): ϕ = average pooling (causal, overlap). Return K_cmp,V_cmp + decode‑time incremental emission logic. (Eq. 7.)

Agent C (selection_scorer.py): From Q & K_cmp, compute p_cmp (Eq. 8) → p_slc via M (Eq. 9) → group‑reduced scores (Eq. 10) → top‑n with forced initial + 2 local (Eqs. 11–12).

Agent D (nsa_attention.py): Wire three branches with SDPA, add gate MLP (softmax), and return combined outputs (Eq. 5). Maintain minimal per‑branch caches for prefill + 1‑step decode.

Agent E (tests/): Implement A1–A4, B5–B6, C8 (small grids), golden‑seed fixtures, and the token‑reads counter checks (Table 4).

Pushback & guardrails (so we don’t drift)

We will not start with the Triton kernel. The paper’s core algorithm must be validated first (p_cmp→p_slc→top‑n, group consistency). Kernel work comes after M0/M1 so we can test numerics against a known‑good SDPA gather. (Figure 3 is our M4 blueprint.)

Gating normalization: Paper shows sigmoids per branch; we’ll use softmax (sum‑to‑1) for stability. If we observe under‑utilization of branches during training (M2+), we can add a temperature. (§3.2.)

ϕ progression: Start with average pooling (M0) → learnable Conv1d+MLP (M2) to converge with the paper’s learnable compression (§3.3.1).

GQA as a non‑negotiable: Group‑consistent selection is essential to achieve the decode memory savings the paper reports (§3.4, §5.2). We enforce it in the scorer API.

What “green” looks like at M0

Functional: Prefill forward pass + 1‑step decode returns finite outputs; counters report token‑reads consistent with the formula (±0). (Table 4.)

Correctness: Equivalence test passes when NSA is configured to cover all tokens (B5); causal masks proven; block mapping property tests pass. (Eqs. 5, 7–12.)

Observability: p_cmp/p_slc heatmaps render; selected blocks and gate histograms export.

If you’re good with this, I’ll have the agents start with M0 immediately and organize the work as above. The moment M0 is green, we move to M1 (FA‑2) and M2 (learnable ϕ + trainable gates), then M4/M5 for the Triton kernel and backward to align with the Figure 6 speedup arc.

You said:
Okay, please review your implementation, pressure test it, and then add any additional detail you think is important to make sure this gets done correctly. This includes frameworks, development practices, patterns, mermaid diagrams if you need to, an example file structure, anything you can think of that will dramatically improve the implementation.
ChatGPT said:
Request for GPT-5 Pro
Reasoned for 3m 55s

Excellent—here’s a pressure‑tested, opinionated blueprint that takes our steel thread (M0) and makes it execution‑ready. I’m adding: stricter APIs, development practices, CI/bench scaffolding, failure‑mode checklists, and diagrams to keep the coding agents aligned and productive. Citations to the paper anchor each design choice so we don’t drift.

1) Pressure test of the steel thread (M0)

M0 recap: Three branches (compressed, selected, sliding) with shared Q, per‑branch K/V, GQA group‑consistent selection, gating merge, prefill + 1‑step decode, all using PyTorch SDPA (FlashAttention‑2 later). Selection scores are reused from compressed attention (p_cmp → p_slc → top‑n blocks); forced inclusion of initial + local blocks; block sizes l=32, d=16, l’=64, n=16, w=512. This matches Figure 2 and §3.2–3.3; selection is derived from Eq. 8–12; sliding is §3.3.3.

The big risks & how we mitigate them (M0)

Block‑math drift (cmp→sel mapping): If Eq. 9 is misapplied, selection will be incoherent.
Mitigation: Precompute a sparse mapping matrix M (cmp‑block → selection‑block) and validate with property tests across random (l,d,l') that satisfy d|l and d|l'. Assert these divisibility constraints at init (paper assumes them). (§3.3.2, Eq. 9).

GQA inconsistency: If heads in a group pick different blocks, we lose decode memory wins.
Mitigation: Group‑reduce block scores (Eq. 10) before top‑k; unit test that selected indices are identical across heads in a group. (§3.3.2).

Selection not truly causal: If forced local blocks cross t, future leakage breaks training.
Mitigation: Construct selection windows strictly ≤ t and test no‑future‑access invariants across branches (Figure 2).

Gate instability early on: Paper shows per‑branch gates via sigmoid; we normalize with softmax to keep output scale stable.
Mitigation: Gate MLP → softmax, initialize to ~uniform; add temperature knob; track per‑branch gate histograms to catch collapse. (§3.2, Eq. 5).

Decode cache coherence: Compressed tokens must be emitted every d steps once l are available; sliding window ring buffer must not drift.
Mitigation: Reference counters at each step and test that loaded tokens per decode step equal ((S-l)/d) + n*l' + w. Compare against Table 4 expectations; this is the simplest high‑signal SLO for correctness of sparsity. (§5.2, Table 4).

Performance illusions: SDPA on selection may be slow; that’s fine for M0 because the goal is correctness. Speed comes with FA‑2 (M1) and custom Triton selection kernel (M4), which follows Figure 3’s group‑centric load schedule. (§3.4, Figure 3; Figure 6 later for expected speed curves).

2) Architecture & data‑flow (diagrams)

High‑level pipeline (per token 
𝑡
t, per GQA group)

flowchart LR
  X[Hidden x_t] -->|W_Q + RoPE| Q[Q_t (all heads in group)]
  subgraph Branches
    direction TB
    Q --> CmpScore[p_cmp = softmax(Q • K_cmp^T)]:::op
    CmpScore --> MapM[p_slc = p_cmp @ M (Eq.9)]:::op
    MapM --> GroupSum[p'_slc = sum_h p_slc^(h)]:::op
    GroupSum --> Topk[I_t = top-n blocks (force init + local)]:::op
    Topk --> GatherSel[Gather raw K_sel,V_sel]:::mem
    GatherSel --> SelAttn[Attn(Q, K_sel, V_sel)]:::op
    
    Q --> WinGather[Window K_w,V_w (last w)]:::mem
    WinGather --> WinAttn[Attn(Q, K_w,V_w)]:::op
    
    Kcmp[K_cmp,V_cmp from ϕ over (l,d)]:::mem
    Q --> CmpAttn[Attn(Q, K_cmp,V_cmp)]:::op
  end
  CmpAttn --> Gate[Gate MLP → softmax]:::op
  SelAttn --> Gate
  WinAttn --> Gate
  Gate --> Out[O_t]
classDef op fill:#eef,stroke:#88a,stroke-width:1px;
classDef mem fill:#efe,stroke:#7a7,stroke-width:1px;


Paper anchors: three branches and gating in Figure 2 (p.3); selection score reuse Eq. 8–12 (pp.7–8); sliding branch design §3.3.3 (p.8).

Decode step timeline (what must happen in order)

sequenceDiagram
  participant T as Token t
  participant W as Sliding cache (w)
  participant R as Raw KV cache
  participant Φ as Compressed stream (ϕ)
  participant S as Selector (M, top-n)
  participant A as Attn branches + Gate

  T->>W: Read last w K/V (≤ t)
  T->>Φ: Ensure compressed K_cmp/V_cmp up-to-date<br/>(emit if (t-l) % d == 0)
  T->>S: p_cmp from Q·K_cmp; p_slc = p_cmp @ M; sum across heads; top-n + forced blocks
  S->>R: Gather contiguous selected raw blocks (≤ t)
  T->>A: Run 3 attentions (cmp/sel/win), gate, sum → O_t


Selection kernel (M4/M5) execution model (from Figure 3, p.9)

flowchart TB
  subgraph GridLoop[Grid Loop: over (B, G, t)]
    Qload[Load all heads' Q for group] --> Inner[Inner Loop over selected KV blocks]
  end
  subgraph Inner[Inner Loop: coalesced KV block chunks (Bk | l')]
    Kfetch[Load K[V] block to SRAM] --> Dot[Q @ K^T + softmax stats]
    Dot --> Acc[Accumulate P·V, update m,l]
    Acc -->|repeat for blocks| Inner
  end
  Inner --> Write[Write O, m, l to HBM]


Rationale & constraints: group‑centric Q load, shared KV fetch, and Bk divides l' to keep coalesced accesses and Tensor Cores busy; this is the paper’s hardware‑aligned plan. (§3.4; see also Figure 6, p.13 for speed curves we should roughly match later.)

3) Codebase blueprint (finalized)
nsa/
  core/
    nsa_attention.py               # public module; wires 3 branches + gates
    compress_pool.py               # ϕ implementations: avg, Conv1d+MLP
    block_index.py                 # block builders, mapping M (cmp→sel)
    selection_scorer.py            # Eq.8–12 scoring + top-k (group-consistent)
    rope.py                        # RoPE utilities (per-branch K; shared Q)
  kernels/
    flash_wrappers.py              # FA-2 wrapper for cmp/win branches
    triton_sel_kernel/
      __init__.py
      sel_fwd.triton               # Figure 3 forward
      sel_bwd.triton               # FA-style backward w/ saved stats
      microbench.py                # autotune Bk, num_warps, num_stages
  model/
    llama_block_nsa.py             # LLaMA-style block with GQA + NSA
    rmsnorm.py, mlp.py             # fused RMSNorm/MLP stubs (optional)
  cache/
    kv_cache.py                    # ring buffer (win), raw blocks, cmp stream
  tests/
    test_masks.py                  # causal guarantees on all branches
    test_block_math.py             # properties for Eq.9 & divisibility
    test_group_consistency.py      # Eq.10 invariant
    test_equiv_small.py            # B5 full-coverage ≈ full-attn
    test_ablation.py               # branch one-hot sanity
    test_decode_counters.py        # token-reads vs Table 4
    test_gradcheck.py              # once kernels/bwd exist
    test_needle.py                 # 64k retrieval harness
  bench/
    bench_prefill.py               # prefill/grad wall-time vs FA-2
    bench_decode.py                # token-read curve & latency
  cli/
    demo_infer.py                  # prints selected blocks/gates
  configs/
    base.yaml                      # dimensions, blocks, windows
    ablations.yaml                 # swap ϕ, gates, n,l',w
  scripts/
    train_toy.py                   # tiny LM; loss-down smoke test
  README.md
  CONTRIBUTING.md
  requirements.txt
  pyproject.toml                   # ruff+mypy+pytest config

4) Frameworks, build flags & dev practices

Frameworks: PyTorch ≥2.3 (SDPA & torch.compile), Triton ≥3.0, FlashAttention‑2 for cmp/win branches (paper says these branches are FA‑compatible), Hydra for configs, PyTest+Hypothesis for property tests, NumPy, Pydantic (for config validation), ruff (lint), mypy (typing), pre‑commit hooks. (§3.4 mentions FA‑2 compatibility for cmp & win).

Precision: default BF16 params with FP32 accumulators; test path supports FP32 reference for equivalence.

torch.compile: wrap forward paths (disable for debug), set fullgraph=False.

Determinism: set seeds, disable torch.backends.cudnn.benchmark, fix dropout off for tests.

CI: CPU unit tests + type/lint; GPU job (self‑hosted) for test_equiv_small, test_decode_counters, and 8–16k needle smoke.

Style: no hidden global state; pass caches explicitly; avoid in‑place ops that break autograd.

5) Crystal‑clear interfaces (signatures agents must honor)
# nsa/core/nsa_attention.py
class NSAAttention(nn.Module):
    def __init__(self, dim:int, n_heads:int, n_kv_groups:int, d_k:int, d_v:int,
                 l:int=32, d:int=16, l_sel:int=64, n_sel:int=16, w:int=512,
                 gate_hidden:int=None, rope_impl:str="llama",
                 use_flash:bool=False, use_triton_sel:bool=False):
        ...

    def forward(self, x: Tensor, kv: "NSA_KV", *, prefill: bool) -> tuple[Tensor, "NSA_KV"]:
        """x: [B, S, dim] (prefill) or [B, 1, dim] (decode); returns (y, updated kv)"""

# nsa/cache/kv_cache.py
@dataclass
class NSA_KV:
    # raw per-branch KV (selection/sliding share the same raw cache tensors)
    K_raw: Tensor  # [B,G,S,Dk]
    V_raw: Tensor  # [B,G,S,Dv]
    # rolling window
    win_ptr: Tensor  # [B,G] positions (int)
    # compressed stream
    K_cmp: Tensor    # [B,G,S_cmp,Dk]
    V_cmp: Tensor    # [B,G,S_cmp,Dv]
    cmp_emit_next: Tensor  # [B,G] next emission position (int)
    # meta & precomputed index maps
    meta: "BlockMeta"

# nsa/core/block_index.py
@dataclass
class BlockMeta:
    l:int; d:int; l_sel:int; n_sel:int; w:int
    cmp_starts: Tensor     # [S_cmp] start indices
    sel_starts: Tensor     # [S_sel] start indices
    M_csl: Tensor          # sparse mapping (cmp->sel) for Eq.9

# nsa/core/selection_scorer.py
def select_blocks_from_pcmp(
    p_cmp: Tensor, meta: BlockMeta, n_groups:int, h_per_group:int, n_top:int,
    force_init: bool=True, force_local:int=2
) -> Tensor:
    """Return [B, G, n_top, 2] start/end (exclusive) in raw-token indices."""


Triton kernel headers (M4/M5)

# nsa/kernels/triton_sel_kernel/__init__.py
def sel_attn_fwd(Q: Tensor, K: Tensor, V: Tensor, sel_ranges: Tensor,
                 scale: float, causal: bool=True) -> tuple[Tensor, Tensor, Tensor]:
    """Q:[B,G,h,Dk]; K/V:[B,G,S,D*]; sel_ranges:[B,G,n_sel,2]; returns (O, m, l)"""

def sel_attn_bwd(dO: Tensor, Q: Tensor, K: Tensor, V: Tensor,
                 sel_ranges: Tensor, m: Tensor, l: Tensor, scale: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return dQ, dK, dV sparse grads (only for selected ranges)."""

6) Testing & verification structure (expanded)
A) Unit/property tests (fast)

Masks: for random shapes, ensure no attention index > t. (Figure 2 causal depiction).

Mapping Eq. 9: Build random p_cmp; verify p_slc equals reference sum of overlapping cmp blocks → selection block; assert d|l and d|l'. (§3.3.2).

Group consistency Eq. 10: Selected I_t identical over heads in a group. (§3.3.2).

Ablations: Force gates one‑hot → selection path matches full‑attn when n*l' ≥ S; sliding matches when w ≥ S; compressed matches when l=d on small S. (§3.3.1–3.3.3).

B) Equivalence (deterministic)

Full coverage ≈ full attention: With config w ≥ S and n*l' ≥ S, NSA output must match SDPA full‑attn within 1e‑5 (FP32). Validates Eq. 5 aggregation. (§3.2).

C) Long‑context functionality

Token‑reads counter vs Table 4: Per decode step, compute actual tokens loaded; compare to formula ((S-l)/d) + n*l' + w. Inspect scaling at S ∈ {8k,16k,32k,64k}; expected speedup grows ~linearly with the reduction (Table 4). (§5.2, Table 4).

Needle‑in‑a‑Haystack @ 64k: Perfect retrieval like Figure 5. We’ll reproduce that harness. (§4.3, Figure 5).

D) Trainability (after M5)

Gradcheck small dims: FP32; checks bwd numerics stability.

Loss‑down smoke: Tiny LM for a few thousand steps; loss decreases smoothly similar to Figure 4 trend (qualitatively). (§4.1, Figure 4).

E) Perf checks (after M4)

Prefill/Backward vs FA‑2: Speed curves should look like Figure 6: rising advantage with context; don’t chase exact numbers, match shape. (§5.1, Figure 6).

7) Kernel guidance (ahead of M4)

Tiling: choose Bk ∈ {16,32,64} such that Bk | l' (paper requirement) and num_warps ∈ {2,4,8}, num_stages ∈ {1,2,3}; autotune on real dims. (§3.4).

Loop structure: outer grid over (B, G, t); inner loop over selected KV block chunks; Q for all heads in a group resides in SRAM; K/V fetched coalesced and reused across heads. (Figure 3).

Numerics: FA‑style log‑sum‑exp with saved m,l stats; FP32 accumulators; fused scale 1/√d_k.

Backward: recompute logits per chunk with saved m,l; accumulate dQ,dK,dV only for selected ranges (sparse writes, coalesced per block).

Correctness oracles: Compare to SDPA on gathered tokens (exactly the same slices) at FP32 with tolerance 1e‑4; seed fixed.

8) Decode caches (concrete rules)

Sliding window: ring buffer per (B,G) with w capacity; on step t, window = [max(0,t-w+1), t]. (§3.3.3).

Compressed stream: keep a FIFO of recent l raw K to produce one compressed token via ϕ every d steps once warmed (t ≥ l) → append to K_cmp,V_cmp. (§3.3.1, Eq. 7).

Selection read: raw K/V live contiguously in [i*l' : (i+1)*l') blocks; after deciding I_t, gather slices and run attention. (§3.3.2, Eq. 12).

Invariants: all three branches only read indices ≤ t. Enforce via masks even if an index slips (defensive programming).

9) Observability & debug hooks

Heatmaps: dump p_cmp and p_slc for a set of tokens; overlay I_t ranges—this should visually resemble Figure 2 (right) where green regions track computed areas.

Gate histograms: mean±std per branch; alert if any gate saturates (>0.98) early. (§3.2).

Token‑reads counters: per decode step, log total reads per branch; chart vs Table 4 curve. (§5.2, Table 4).

Selection distance: histogram of selected block distances (should mix local+far).

10) Development patterns that will save us pain

Make sparsity data explicit: carry sel_ranges as [start,end) pairs (int32) instead of indices; it simplifies coalesced loads and kernel writes.

Avoid hidden recompute: never silently rebuild M; compute it once per (S,l,d,l') set; include a version hash in meta.

Config‑driven: Hydra config sets (l,d,l',n,w) and GQA layout; keep ablations (e.g., ϕ="avg" vs ϕ="conv_mlp") in configs/ablations.yaml.

No length‑dependent parameters in modules: caches hold sequence‑dependent state.

Strict type checks: use typing.Annotated[Tensor, "B S H D"] comments + runtime asserts for key shapes.

Kill switches for tests: env flags NSA_FORCE_BRANCH=cmp|sel|win to force gate one‑hot in tests.

11) Example configs & commands

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
  n_sel: 16        # includes forced init + 2 local
  w: 512
  ϕ: "avg"         # M2: "conv_mlp"
runtime:
  precision: "bf16"
  use_flash: false  # M1 -> true
  use_triton_sel: false  # M4 -> true


Run the steel thread

pytest -q                               # fast unit/property tests
pytest -q -m long                       # long-context tests (needle, counters)
python cli/demo_infer.py --config configs/base.yaml
python bench/bench_decode.py --config configs/base.yaml

12) Extended milestone plan (with acceptance gates)

M0 (steel thread): SDPA everywhere, ϕ=avg; tests: masks, Eq.9/10, equivalence (small‑S), decode counters.

M1 (cmp/win on FA‑2): swap SDPA→FA‑2 for cmp & win (paper: FA‑compatible paths). Expect prefill speedup. (§3.4).

M2 (learnable ϕ + trainable gates): Conv1d(l, stride=d, causal) + tiny MLP; train gates; re‑run needle@64k—should remain perfect (Figure 5). (§3.3.1).

M3 (full decode): multi‑step decode; strict Table‑4 counters match across S; add sliding/selection stress. (§5.2).

M4 (Triton selection forward): numerical match to SDPA gather; microbench autotune; latency curves begin to mirror Figure 6 (left). (§5.1).

M5 (Triton backward): end‑to‑end trainable; gradcheck; tiny‑LM loss down akin to Figure 4 trend. (§4.1).

M6 (robustness/perf polish): dtype stress, edge cases (t<l, t<w), GPU occupancy, OOM resilience.

13) Failure‑mode checklist (fast triage)

Needle fails at deep positions:
Check Eq. 9 mapping; ensure forced blocks (init + 2 local) are included; verify ϕ is not erasing signal; inspect p_cmp heatmap (should spike near the needle’s block). (§3.3.2; Figure 5).

Decode counters off:
Compressed emission schedule wrong (should be every d after warmup l); sliding window bounds off‑by‑one; selection blocks exceeding t. (§3.3.1; §5.2).

Gate collapse:
Reduce gate temperature; normalize (softmax) if you flipped to sigmoids; verify per‑branch K/V are distinct to avoid local shortcutting. (§3.3.3).

Kernel mismatch vs SDPA:
Check Bk | l', causal masking per block, and saved m,l usage. Reproduce at FP32 first. (Figure 3).

14) Rationale anchors (why these choices are “right”)

Hierarchical three‑path design with gated merge is the core algorithm (Figure 2, Eq. 5). It balances global context (compressed), precise retrieval (selected), and locality (sliding).

Selection scores reuse from compressed attention (Eq. 8) → block mapping (Eq. 9) → group‑consistent aggregation (Eq. 10) → top‑n continuous blocks (Eq. 11–12). This is the paper’s key “natively trainable” mechanism that avoids aux losses.

Hardware alignment: cmp/win run on FA‑2; selection has a group‑centric kernel (Figure 3) to keep arithmetic intensity high and memory accesses coalesced—this is why NSA hits the speed curves in Figure 6 and decode reductions in Table 4.

15) One more diagram (end‑to‑end stack)
graph TD
  A[Tokenizer/Embedding] --> B[RMSNorm + RoPE]
  B --> C[NSAAttention]
  subgraph C_detail[NSAAttention]
    C1[Compressed (ϕ, FA-2)] --> G[Gate]
    C2[Selected (SDPA→Triton)] --> G
    C3[Sliding (FA-2)] --> G
    G --> Out[O]
  end
  C --> D[MLP block]
  D --> E[Residual + Next Layer]

Final note

Everything above is designed so agents can implement incrementally while keeping correctness visible at each step. The paper references (figures/tables/equations and page numbers) are embedded where a design choice binds to the method:

System overview and branches (Figure 2, p.3; §3.2–3.3),

Selection math (Eq. 8–12, pp.7–8),

Sliding branch and per‑branch K/V (§3.3.3, p.8),

Kernel schedule (Figure 3, p.9; §3.4),

Hyperparameters as used in 27B runs (§4.1, p.9–10),

Efficiency claims: prefill/backward curves and decode budget (Figure 6, p.13; Table 4, p.14; §5),

Why trainable selection and no aux losses (§6.1, Figure 7).

If you’re aligned with this, the agents can start committing M0 code against these signatures today, with the tests and hooks ready to keep us honest.

You said:
Can you please take all of this and reproduce it in full in an organized, detailed PRD / System plan in canvas? It must be complete and thorough, i will copy paste it into my coding agent
ChatGPT said:
Request for GPT-5 Pro
Reasoned for 5m 12s

Below is a single, self‑contained PRD / System Plan (“canvas”) you can paste directly into your coding agent. It is opinionated and executable, with exact APIs, milestones, tests, diagrams, and acceptance gates. Citations anchor each requirement to the NSA paper so we stay faithful to the design.

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

FR3: Selection reuses p_cmp = softmax(Q·K_cmp^T) to compute block scores for selection blocks of size l'; if blockings differ, apply Eq. 9 mapping; sum across heads in each GQA group (Eq. 10); choose top‑n contiguous blocks and attend over raw tokens in those blocks (Eqs. 8–12; Figure 2 right).

FR4: Sliding attends over the last w tokens with separate K/V to avoid shortcut learning (Section 3.3.3).

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

5.2 Decode Step Ordering
sequenceDiagram
  participant T as Token t
  participant W as Sliding window cache
  participant Φ as Compressed stream (ϕ)
  participant S as Selector (M, group-reduce, top-n)
  participant R as Raw KV cache
  participant A as Attn branches + Gate

  T->>W: Read last w K/V (≤ t)
  T->>Φ: Emit compressed token every d steps after warmup l (≤ t)
  T->>S: p_cmp from Q·K_cmp; map to p_slc via M; sum across heads; choose top-n + forced blocks
  S->>R: Gather contiguous raw KV for selected blocks (≤ t)
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
                 gate_hidden:int|None=None, rope_impl:str="llama",
                 use_flash:bool=False, use_triton_sel:bool=False):
        ...

    def forward(self, x: Tensor, kv: "NSA_KV", *, prefill: bool) -> tuple[Tensor, "NSA_KV"]:
        """x: [B, S, dim] (prefill) or [B, 1, dim] (decode). Returns (y, updated kv)."""

6.2 KV Caches
@dataclass
class NSA_KV:
    # raw cache used by selection & sliding
    K_raw: Tensor  # [B,G,S,Dk]
    V_raw: Tensor  # [B,G,S,Dv]
    # sliding window metadata
    win_ptr: Tensor  # [B,G] int positions
    # compressed stream & emission schedule
    K_cmp: Tensor    # [B,G,S_cmp,Dk]
    V_cmp: Tensor    # [B,G,S_cmp,Dv]
    cmp_emit_next: Tensor  # [B,G] int next emission
    # precomputed indices & mapping
    meta: "BlockMeta"

6.3 Block Metadata
@dataclass
class BlockMeta:
    l:int; d:int; l_sel:int; n_sel:int; w:int
    cmp_starts: Tensor  # [S_cmp] block start indices
    sel_starts: Tensor  # [S_sel] block start indices
    M_csl: Tensor       # sparse map cmp->sel for Eq. 9

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

Acceptance: Causality tests, Eq.9 mapping property tests, group consistency test, small‑S equivalence, decode‑counter math.

M1 — FA‑2 for Compressed & Sliding

Swap SDPA→FA‑2 for cmp/win (paper: these branches compatible).

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

Frameworks: PyTorch ≥2.3; Triton ≥3.0; optional FlashAttention‑2 for cmp/win; Hydra configs; PyTest + Hypothesis; ruff + mypy.

Precision: BF16 weights / FP32 accumulators; FP32 reference path for tests.

torch.compile: fullgraph=False; disable in debug mode.

Determinism for tests: fixed seeds; disable cudnn.benchmark; no dropout in tests.

Config‑driven: All l, d, l', n, w, G, H, d_k, d_v in YAML; ablations allowed.

CI: CPU unit tests & static checks; GPU CI job for small equivalence, decode counters, 8–16k needle smoke.

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

Decode token‑reads vs Table 4: ((S‑l)/d) + n*l' + w predicted vs actual reads, trend lines up with Table 4.

Needle@64k: Perfect retrieval across positions (Figure 5).

D) Trainability (M5+)

gradcheck (tiny dims) passes.

toy LM loss down (qualitative trend like Figure 4).

E) Perf (M4+)

Prefill/backward timing trend approximates Figure 6; decode latency scales with reduced reads (Table 4).

12) Bench & Observability

Benchmarks

bench_prefill.py: batch prefill (8k,16k,32k,64k), NSA vs FA‑2 (cmp/win).

bench_decode.py: per‑step latency & token‑reads vs S; compute expected vs actual tokens.

Instrumentation

Heatmaps for p_cmp and p_slc; overlay I_t (compare to Figure 2’s green regions).

Gate histograms (mean±std) to detect early saturation.

Selection distance histogram (local vs far).

13) Algorithms & Patterns (implementation details)
13.1 Compression ϕ

M0: Average pooling over [i*d : i*d+l) → produces K_cmp, V_cmp.

M2: Depthwise Conv1d(l, stride=d, causal) over time + tiny MLP on channel; update incrementally: when (t‑l) % d == 0, emit a compressed token from the last l raw K/V. Source: Eq. 7 + §3.3.1.

13.2 Selection Scoring & Mapping

Compute p_cmp = softmax(Q·K_cmp^T) (Eq. 8).

Map to selection block scores with a precomputed sparse M (Eq. 9), then sum over heads in group (Eq. 10).

Top‑n with forced inclusion: block 0 + 2 local blocks covering [t‑2*l', t] (Eqs. 11–12). Blockwise selection is key for hardware efficiency & attention continuity.

13.3 Sliding Branch

Last w tokens, own K/V projections to avoid shortcut learning; attend with FA‑2. Source: §3.3.3.

13.4 Gating

Group‑pooled query → MLP → softmax over 3 gates (paper uses sigmoid; we normalize for stability). Source: Eq. 5 defines gated sum.

13.5 Selection Kernel (Triton)

Grid loop over (B,G,t); inner loop iterates contiguous selected KV blocks in chunks of Bk with Bk | l'.

Load all heads’ Q for group to SRAM, share K/V fetch across heads; FA‑style LSE numerics with saved m,l.

Backward recomputes logits chunk‑wise and writes sparse dK,dV for selected ranges only. Source: Figure 3 & §3.4.

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

15) Acceptance Criteria (Definition of Done)

DoD‑M0

All unit tests (masks, Eq.9, Eq.10, shapes) pass.

Small‑S equivalence test passes (MAE < 1e‑5 FP32).

Decode counter equals predicted ((S‑l)/d) + n*l' + w on synthetic runs. (Table 4).

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

Agent D — nsa_attention.py: Wire 3 branches; gates (softmax); RoPE; prefill + 1‑step decode; causal masks.

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
    kv.update_raw(K_raw_slc, V_raw_slc)     # raw shared for slc/win
    kv.update_window(K_raw_win, V_raw_win)  # maintain last w tokens
    kv.update_compressed(K_raw_cmp, V_raw_cmp, phi, l, d)  # emit every d after l

    # ==== Compressed branch ====
    K_cmp, V_cmp = kv.K_cmp, kv.V_cmp
    O_cmp, p_cmp = attn_and_scores(Q, K_cmp, V_cmp)  # p_cmp: softmax logits per Eq. 8

    # ==== Selection branch ====
    p_slc = map_cmp_to_sel(p_cmp, kv.meta.M_csl)     # Eq. 9
    p_grp = group_reduce(p_slc)                      # Eq. 10
    sel_ranges = topn_ranges(p_grp, n_sel, force_init=True, local=2)  # Eqs. 11–12
    O_slc = sdpa_over_ranges(Q, kv.K_raw, kv.V_raw, sel_ranges)  # M4→ triton

    # ==== Sliding branch ====
    win_ranges = kv.get_window_ranges(w)
    O_win = sdpa_over_ranges(Q, kv.K_raw, kv.V_raw, win_ranges)

    # ==== Gating & merge ====
    g = softmax(gate_mlp(group_pool(Q)), dim=-1)     # Eq. 5 (paper uses sigmoid; we normalize)
    O = g[...,0]*O_cmp + g[...,1]*O_slc + g[...,2]*O_win
    return O, kv


Anchors: Eqs. 5, 7–12, §3.3.3; Figure 2 flow.

19) Commands / Dev Ergonomics
# Fast tests
pytest -q

# Long-context tests (needle & counters)
pytest -q -m long

# Benchmarks
python bench/bench_prefill.py --config configs/base.yaml
python bench/bench_decode.py  --config configs/base.yaml

# Demo
python cli/demo_infer.py --config configs/base.yaml

20) Open Questions (tracked; default answers chosen)

Gating activation: Paper uses sigmoids per branch; we use softmax for scale stability. Decision: keep softmax; add temperature.

ϕ operator detail: Paper states “MLP with intra‑block position encoding”; we implement Conv1d+MLP for hardware friendliness. Decision: adopt Conv1d+MLP.

Divisibility constraints: Paper prefers d|l and d|l'. Decision: enforce; fall back to general overlap map only when explicitly configured.

21) Appendix — Why this design is sound

Three‑branch NSA with learned gates balances global awareness (compressed), precise retrieval (selected), and locality (sliding). Ref: Figure 2; Eq. 5.

Trainable selection via p_cmp → p_slc → top‑n avoids aux‑loss pitfalls and supports end‑to‑end training. Ref: Eqs. 8–12; §6.1 critique of aux/heuristics (Figure 7).

Hardware alignment: cmp/win reuse FA‑2; selection kernel is group‑centric with shared KV, chunked inner loop Bk | l'. Ref: §3.4; Figure 3.

Efficiency: Decode memory budget and speedup trend follow Table 4; training/prefill trends like Figure 6.

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

Final Reminder

Use the paper defaults: l=32, d=16, l'=64, n=16 (incl. 1 init + 2 local), w=512. These are explicitly stated in Section 4.1 and are the parameters assumed by the kernel and decode economics in Table 4. Keep GQA group‑consistent selection as non‑negotiable to preserve the bandwidth wins.

Paste this canvas into your coding agent as the authoritative plan. It includes the steel thread, acceptance gates, diagrams, and exact APIs needed to implement NSA correctly and verifiably.

ChatGPT can make mistakes. Check important info.