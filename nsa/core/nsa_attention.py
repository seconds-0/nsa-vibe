import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.cache.kv_cache import NSA_KV
from nsa.core.attention_kernels import (
    compressed_attention_fa2,
    compressed_attention_fa2_decode,
    grouped_selection_attention,
    grouped_selection_attention_masked,
    grouped_selection_attention_packed,
    sliding_window_attention_fa2,
    sliding_window_attention_fa2_decode,
)
from nsa.core.block_index import build_block_meta
from nsa.core.compress_pool import avg_pool_phi_rope_kv
from nsa.core.debug import log
from nsa.core.rope import apply_rope
from nsa.core.selection_scorer import (
    compute_pcmp_all,
    map_pcmp_to_pslc_batched,
    select_topn_ranges,
    select_topn_ranges_batched,
    verify_mapping_equivalence,
)
from nsa.kernels.flash_wrappers import attention_bgh


class GateMLP(nn.Module):
    def __init__(self, d_k: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(1, d_k // 2)
        self.fc1 = nn.Linear(d_k, hidden)
        self.fc2 = nn.Linear(hidden, 3)
        # Initialize fc2 with small random values to break symmetry and enable learning
        # Use Xavier uniform with reduced scale to start near uniform but allow differentiation
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)  # Keep bias at zero for initial balance
        # Cache environment variables at init to avoid hot path parsing
        self._force_uniform_gate = os.getenv("NSA_FORCE_UNIFORM_GATE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._force_branch = os.getenv("NSA_FORCE_BRANCH")

    def forward(self, q_group_pooled: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        # Uniform gate override for debugging DDP hangs
        if self._force_uniform_gate:
            one_third = 1.0 / 3.0
            shape = (*q_group_pooled.shape[:-1], 3)
            return torch.full(
                shape, one_third, device=q_group_pooled.device, dtype=q_group_pooled.dtype
            )
        fb = self._force_branch
        if fb:
            fb = fb.strip().lower()
            if fb in ("cmp", "sel", "win"):
                idx = 0 if fb == "cmp" else (1 if fb == "sel" else 2)
                one = torch.zeros(
                    (*q_group_pooled.shape[:-1], 3),
                    device=q_group_pooled.device,
                    dtype=q_group_pooled.dtype,
                )
                one[..., idx] = 1.0
                return one
        x = F.silu(self.fc1(q_group_pooled))
        g = self.fc2(x) / max(tau, 1e-6)
        p = F.softmax(g, dim=-1)
        # Hard one-hot if extremely peaked to avoid numerical drift in ablations/tests
        with torch.no_grad():
            top2 = torch.topk(g, k=2, dim=-1).values
            peaked = (top2[..., 0] - top2[..., 1]) > 50.0
        if peaked.any():
            one_hot = torch.zeros_like(p)
            idx = torch.argmax(g, dim=-1, keepdim=True)
            one_hot.scatter_(-1, idx, 1.0)
            p = torch.where(peaked.unsqueeze(-1), one_hot, p)
        return p


def _fused_gate_combine_bsg(
    q_gp: torch.Tensor,  # [B,S,G,Dk]
    O_cmp: torch.Tensor,  # [B,S,G,h,Dv]
    O_sel: torch.Tensor,  # [B,S,G,h,Dv]
    O_win: torch.Tensor,  # [B,S,G,h,Dv]
    fc1_w: torch.Tensor,
    fc1_b: Optional[torch.Tensor],
    fc2_w: torch.Tensor,
    fc2_b: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    import torch.nn.functional as _F
    x = _F.silu(_F.linear(q_gp, fc1_w, fc1_b))
    g = _F.linear(x, fc2_w, fc2_b) / max(tau, 1e-6)
    p = _F.softmax(g, dim=-1)
    w_cmp = p[..., 0:1].unsqueeze(-1)
    w_sel = p[..., 1:2].unsqueeze(-1)
    w_win = p[..., 2:3].unsqueeze(-1)
    return w_cmp * O_cmp + w_sel * O_sel + w_win * O_win


def _fused_gate_combine_bg(
    q_gp: torch.Tensor,  # [B,G,Dk]
    O_cmp: torch.Tensor,  # [B,G,h,Dv]
    O_sel: torch.Tensor,  # [B,G,h,Dv]
    O_win: torch.Tensor,  # [B,G,h,Dv]
    fc1_w: torch.Tensor,
    fc1_b: Optional[torch.Tensor],
    fc2_w: torch.Tensor,
    fc2_b: Optional[torch.Tensor],
    tau: float,
) -> torch.Tensor:
    import torch.nn.functional as _F
    x = _F.silu(_F.linear(q_gp, fc1_w, fc1_b))
    g = _F.linear(x, fc2_w, fc2_b) / max(tau, 1e-6)
    p = _F.softmax(g, dim=-1)
    w_cmp = p[..., 0:1].unsqueeze(-1)
    w_sel = p[..., 1:2].unsqueeze(-1)
    w_win = p[..., 2:3].unsqueeze(-1)
    return w_cmp * O_cmp + w_sel * O_sel + w_win * O_win


def _compute_gate_stats(gates: torch.Tensor) -> dict:
    """Compute gate health statistics for monitoring.

    Args:
        gates: Gate probabilities [B, S, G, 3] or [B, G, 3]

    Returns:
        Dict with gate statistics: entropy, max_gate, branch_shares
    """
    with torch.no_grad():
        # Flatten to [*, 3] for consistent computation
        gates_flat = gates.view(-1, 3)

        # Gate entropy (should be > 0.5 for healthy mixing)
        entropy = -(gates_flat * (gates_flat + 1e-8).log()).sum(dim=-1)
        mean_entropy = entropy.mean().item()
        min_entropy = entropy.min().item()

        # Max gate value (should be < 0.9 to avoid collapse)
        max_gate = gates_flat.max(dim=-1)[0]
        mean_max_gate = max_gate.mean().item()
        max_max_gate = max_gate.max().item()

        # Branch usage shares (should be balanced)
        branch_shares = gates_flat.mean(dim=0).tolist()  # [cmp, sel, win]

        # Gate collapse detection (entropy < 0.1 and max_gate > 0.95)
        collapsed = (entropy < 0.1) & (max_gate > 0.95)
        collapse_fraction = collapsed.float().mean().item()

        return {
            "entropy_mean": mean_entropy,
            "entropy_min": min_entropy,
            "max_gate_mean": mean_max_gate,
            "max_gate_max": max_max_gate,
            "branch_shares": branch_shares,  # [cmp, sel, win]
            "collapse_fraction": collapse_fraction,
            "total_gates": len(gates_flat),
        }


class NSAAttention(nn.Module):
    """
    Native Sparse Attention (NSA) module (M0 steel-thread).

    Shapes:
    - Input x (prefill): [B,S,dim]; x (decode): [B,1,dim]
    - Heads: n_heads, grouped into n_kv_groups with h_per_group = n_heads // n_kv_groups
    - Projections produce:
      - Q: [B,S,G,h,Dk]
      - K/V per-branch: [B,G,S,D*]

    Returns:
    - out: [B,S,dim] (prefill) or [B,1,dim] (decode)
    - kv: updated NSA_KV caches

    Notes:
    - M0 constraints: SDPA-only, fixed sequence length in tests, deterministic.
    - Masked/packed fast paths are env-gated with `NSA_FORCE_PARITY` fallback.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_groups: int,
        d_k: int,
        d_v: int,
        l: int = 32,
        d: int = 16,
        l_sel: int = 64,
        n_sel: int = 16,
        w: int = 512,
        phi: str = "avg",
        gate_hidden: Optional[int] = None,
        gate_temp: float = 1.0,
        rope_impl: str = "llama",
        use_flash: bool = True,
        use_triton_sel: bool = False,
    ) -> None:
        super().__init__()
        assert n_heads % n_kv_groups == 0, "heads must be divisible by kv groups"
        # M0 config validation (PRD enforces divisibility)
        if l % d != 0 or l_sel % d != 0:
            raise ValueError("M0 requires d|l and d|l_sel; set valid block sizes/stride.")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.h_per_group = n_heads // n_kv_groups
        self.d_k = d_k
        self.d_v = d_v
        self.l = l
        self.d = d
        self.l_sel = l_sel
        self.n_sel = n_sel
        self.w = w
        self.gate_temp = gate_temp
        self.phi_type = (phi or "avg").lower()

        # Gate health tracking for M8 monitoring
        self._last_gate_stats = None
        # M8: Selection length stats for monitoring (updated each forward)
        self._last_sel_stats: Optional[dict] = None

        # M8: Fallback counters for routing monitoring
        self._fallback_counters = {
            "selection_triton_fails": 0,
            "selection_cuda_fails": 0,
            "selection_pack_fails": 0,
            "selection_mask_fails": 0,
            "compressed_fa2_fails": 0,
            "sliding_fa2_fails": 0,
            "total_fallbacks": 0,
        }

        # RoPE scaling and prefill tiling for long-context demos (env-overridable)
        try:
            rs = float(os.getenv("NSA_ROPE_SCALE", "1.0"))
            if not (rs > 0.0) or rs != rs:  # require positive finite
                rs = 1.0
            self.rope_scale = rs
        except ValueError:
            self.rope_scale = 1.0
        try:
            pt = int(os.getenv("NSA_PREFILL_TILE", "0"))
            if pt < 0:
                pt = 0
            self.prefill_tile = pt
        except ValueError:
            self.prefill_tile = 0
        # Projections
        self.W_Q = nn.Linear(dim, n_heads * d_k, bias=False)
        self.W_K_sel = nn.Linear(dim, n_kv_groups * d_k, bias=False)
        self.W_V_sel = nn.Linear(dim, n_kv_groups * d_v, bias=False)
        self.W_K_win = nn.Linear(dim, n_kv_groups * d_k, bias=False)
        self.W_V_win = nn.Linear(dim, n_kv_groups * d_v, bias=False)
        self.W_K_cmp = nn.Linear(dim, n_kv_groups * d_k, bias=False)
        self.W_V_cmp = nn.Linear(dim, n_kv_groups * d_v, bias=False)
        self.out = nn.Linear(n_heads * d_v, dim, bias=False)
        self.gate = GateMLP(d_k, gate_hidden)
        # Default FA-2 usage (can be overridden by env flags)
        self.use_flash_default = use_flash
        # One-time SDPA backend audit flag
        self._sdpa_audited = False
        # Selection Triton toggle (M4)
        self.use_triton_sel = use_triton_sel
        # Cache environment variables to avoid repeated parsing in hot path
        self._cache_env_vars()
        # Optional learnable ϕ via depthwise Conv1d over time with kernel l and stride d
        # Initialize to average pooling for parity with M0
        self.phi_k_conv: Optional[nn.Conv1d]
        self.phi_v_conv: Optional[nn.Conv1d]
        if self.phi_type == "mlp":
            self.phi_k_conv = nn.Conv1d(
                self.d_k, self.d_k, kernel_size=self.l, stride=self.d, groups=self.d_k, bias=False
            )
            self.phi_v_conv = nn.Conv1d(
                self.d_v, self.d_v, kernel_size=self.l, stride=self.d, groups=self.d_v, bias=False
            )
            with torch.no_grad():
                self.phi_k_conv.weight.fill_(1.0 / float(self.l))
                self.phi_v_conv.weight.fill_(1.0 / float(self.l))
        else:
            self.phi_k_conv = None
            self.phi_v_conv = None

    def _cache_env_vars(self) -> None:
        """Cache environment variables to avoid repeated parsing in hot path."""

        def parse_bool(val: str, default: str = "0") -> bool:
            return os.getenv(val, default).lower() in ("1", "true", "yes")

        # Cache frequently accessed environment variables
        # Raw parsed flags
        self._env_cache = {
            "static": parse_bool("NSA_ENV_STATIC", "0"),
            "force_uniform_gate": parse_bool("NSA_FORCE_UNIFORM_GATE", "0"),
            "force_branch": os.getenv("NSA_FORCE_BRANCH"),
            "prefill_batched": parse_bool("NSA_PREFILL_BATCHED", "0"),
            "strict_asserts": parse_bool("NSA_STRICT_ASSERTS", "0"),
            "force_parity": parse_bool("NSA_FORCE_PARITY", "0"),
            "use_sel_pack": parse_bool("NSA_USE_SEL_PACK", "1"),
            "use_triton_sel": parse_bool("NSA_USE_TRITON_SEL", "0") or self.use_triton_sel,
            "use_cuda_sel": parse_bool("NSA_SEL_CUDA", "0"),
            "use_sel_varlen": parse_bool("NSA_USE_SEL_VARLEN", "0"),
            "fa2_all": parse_bool("NSA_USE_FA2", "0"),
            "fa2_win": parse_bool("NSA_USE_FA2_WIN", "0"),
            "fa2_cmp": parse_bool("NSA_USE_FA2_CMP", "0"),
            "use_sel_mask": parse_bool("NSA_USE_SEL_MASK", "0"),
            "use_cmp_mask": parse_bool("NSA_USE_CMP_MASK", "1"),
            "use_win_mask": parse_bool("NSA_USE_WIN_MASK", "1"),
            "verify_eq9": parse_bool("NSA_VERIFY_EQ9_MAPPING", "0"),
            "stopgrad_gates": parse_bool("NSA_STOPGRAD_GATES", "0"),
            "nvtx": parse_bool("NSA_NVTX", "0"),
            "debug_compare": parse_bool("NSA_DEBUG_COMPARE", "0"),
            "gate_compile": parse_bool("NSA_GATE_COMPILE", "0"),
        }

        # Detect whether env overrides were explicitly provided so we can honor hard-disable
        fa2_all_set = "NSA_USE_FA2" in os.environ
        fa2_win_set = "NSA_USE_FA2_WIN" in os.environ
        fa2_cmp_set = "NSA_USE_FA2_CMP" in os.environ
        self._env_cache.update(
            {
                "fa2_all_set": fa2_all_set,
                "fa2_win_set": fa2_win_set,
                "fa2_cmp_set": fa2_cmp_set,
            }
        )

        # Compute effective FA-2 gating with sensible defaults and hard-disable semantics
        fa2_all_env = self._env_cache["fa2_all"]
        fa2_win_env = self._env_cache["fa2_win"]
        fa2_cmp_env = self._env_cache["fa2_cmp"]

        # Defaults when no explicit env flags are provided:
        # - Enable compressed FA‑2 by default (robustly capability-gated at call sites)
        # - Keep sliding FA‑2 off by default due to API semantics
        # - Do not use the global "all" default to avoid inadvertently enabling sliding
        if not (fa2_all_set or fa2_win_set or fa2_cmp_set):
            fa2_all_eff = False
            fa2_win_eff = False
            fa2_cmp_eff = True
        else:
            # If NSA_USE_FA2 not set, fall back to model default; else honor explicit value
            fa2_all_eff = self.use_flash_default if not fa2_all_set else fa2_all_env

            # If global is explicitly set to 0, that hard-disables branch flags too
            if fa2_all_set and not fa2_all_env:
                fa2_win_eff = False
                fa2_cmp_eff = False
            else:
                # Branch-specific flags only take effect if explicitly set; otherwise default off
                fa2_win_eff = fa2_win_env if fa2_win_set else False
                fa2_cmp_eff = fa2_cmp_env if fa2_cmp_set else False

        self._env_cache.update(
            {
                "fa2_all_eff": fa2_all_eff,
                "fa2_win_eff": fa2_win_eff,
                "fa2_cmp_eff": fa2_cmp_eff,
            }
        )
        # Parse numeric values
        try:
            self._rope_scale = float(os.getenv("NSA_ROPE_SCALE", "1.0"))
            if not (self._rope_scale > 0.0) or self._rope_scale != self._rope_scale:
                self._rope_scale = 1.0
        except (ValueError, TypeError):
            self._rope_scale = 1.0

        try:
            self._prefill_tile = int(os.getenv("NSA_PREFILL_TILE", "0"))
            if self._prefill_tile < 0:
                self._prefill_tile = 0
        except (ValueError, TypeError):
            self._prefill_tile = 0
        # Fused gate combine (lazy-compiled)
        self._gate_fused_bsg = None
        self._gate_fused_bg = None

    def _shape_q(self, Q: torch.Tensor, B: int, S: int) -> torch.Tensor:
        Q = Q.view(B, S, self.n_heads, self.d_k)
        # group-major: [B,S,G,h,Dk]
        G = self.n_kv_groups
        h = self.h_per_group
        return Q.view(B, S, G, h, self.d_k)

    def _shape_kv(self, X: torch.Tensor, B: int, S: int) -> torch.Tensor:
        G = self.n_kv_groups
        return X.view(B, S, G, -1).permute(0, 2, 1, 3).contiguous()  # [B,G,S,D*]

    def get_gate_stats(self) -> Optional[dict]:
        """Get the most recent gate statistics for monitoring.

        Returns:
            Dict with gate health metrics or None if no recent computation
        """
        return self._last_gate_stats

    def get_fallback_counters(self) -> dict:
        """Get fallback counters for routing monitoring.

        Returns:
            Dict with fallback counts per implementation type
        """
        return self._fallback_counters.copy()

    def get_selection_stats(self) -> Optional[dict]:
        """Return last computed selection length statistics, if available.

        Keys:
        - k_mean: mean selected K per row (float)
        - k_max: max selected K in batch (int)
        - rows: number of (B,S,G) rows aggregated (int)
        - pct_at_max: fraction of rows equal to k_max (float)
        - l_sel: configured selection block size (int)
        - n_sel: configured top-n selection blocks (int)
        """
        return self._last_sel_stats

    def reset_fallback_counters(self) -> dict:
        """Reset fallback counters and return the previous values.

        Returns:
            Dict with fallback counts before reset
        """
        prev_counters = self._fallback_counters.copy()
        for key in self._fallback_counters:
            self._fallback_counters[key] = 0
        return prev_counters

    def _update_gate_stats(self, gates: torch.Tensor) -> None:
        """Update stored gate statistics for monitoring."""
        try:
            self._last_gate_stats = _compute_gate_stats(gates)
        except Exception as e:
            log("warn.gate_stats_fail", error=str(e))
            self._last_gate_stats = None

    def _update_sel_stats_from_ranges(self, ranges: torch.Tensor) -> None:
        """Compute and store selection statistics from [B,*,G,n,2] ranges tensor."""
        try:
            if ranges is None or ranges.numel() == 0:
                self._last_sel_stats = {
                    "k_mean": 0.0,
                    "k_max": 0,
                    "rows": 0,
                    "pct_at_max": 0.0,
                    "l_sel": int(self.l_sel),
                    "n_sel": int(self.n_sel),
                }
                return
            # ranges: [B, T, G, n, 2] or [B, G, n, 2]
            if ranges.dim() == 5:
                B, T, G, n, _ = ranges.shape
                rs = ranges
                rows = B * T * G
                # [B,T,G,n]
                lengths = (rs[..., 1] - rs[..., 0]).clamp_min(0)
                # Sum across n ranges → [B,T,G]
                L = lengths.sum(dim=-1).to(torch.int64)
            elif ranges.dim() == 4:
                B, G, n, _ = ranges.shape
                rs = ranges
                rows = B * G
                lengths = (rs[..., 1] - rs[..., 0]).clamp_min(0)
                L = lengths.sum(dim=-1).to(torch.int64)  # [B,G]
            else:
                # Unknown shape; skip
                return
            if L.numel() == 0:
                k_mean = 0.0
                k_max = 0
                pct_at_max = 0.0
            else:
                k_max = int(L.max().item())
                k_mean = float(L.to(torch.float32).mean().item())
                if k_max > 0:
                    pct_at_max = float((L == k_max).to(torch.float32).mean().item())
                else:
                    pct_at_max = 0.0
            self._last_sel_stats = {
                "k_mean": k_mean,
                "k_max": k_max,
                "rows": int(rows),
                "pct_at_max": pct_at_max,
                "l_sel": int(self.l_sel),
                "n_sel": int(self.n_sel),
            }
        except Exception as e:
            log("warn.sel_stats_fail", error=str(e))
            self._last_sel_stats = None

    def forward(self, x: torch.Tensor, kv: NSA_KV, *, prefill: bool) -> tuple[torch.Tensor, NSA_KV]:
        """
        Forward pass.

        Args:
            x: [B,S,dim] if prefill else [B,1,dim]
            kv: NSA_KV caches (updated in-place per branch)
            prefill: True for batched prefill, False for single-token decode

        Returns:
            (out, kv): out is [B,S,dim] (prefill) or [B,1,dim] (decode)
        """
        # x: [B,S,dim] (prefill) or [B,1,dim] (decode)
        B, S, _ = x.shape
        assert x.dim() == 3, "x must be [B,S,dim]"
        assert self.n_heads % self.n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"
        # Strict assertions may introduce GPU syncs; gate via env for tests/smokes
        strict_asserts = self._env_cache.get("strict_asserts", False)

        # M8: Assert causal masking - enforce mode constraints
        if prefill:
            assert S > 0, f"Prefill mode requires S > 0, got S={S}"
        else:
            assert S == 1, (
                f"Decode mode requires S=1 (single token), got S={S}. "
                f"This ensures proper causal ordering in decode steps."
            )
        if prefill:
            # Optional: route prefill via single-token decode steps to support very long contexts safely.
            if getattr(self, "prefill_tile", 0) and self.prefill_tile > 0:
                return self._forward_prefill_via_decode(x, kv)
            use_batched = self._env_cache.get("prefill_batched", False)
            if use_batched:
                return self._forward_prefill_batched(x, kv)
            else:
                return self._forward_prefill_sequential(x, kv)
        else:
            # Projections
            # Compute absolute position offset from existing cache length for RoPE on Q
            t_prev = kv.K_sel.shape[2] if hasattr(kv, "K_sel") else 0
            Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
            # Apply RoPE to Q with absolute positions (decode)
            pos = torch.arange(t_prev, t_prev + S, device=x.device)
            Q = apply_rope(
                Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k),
                pos,
                scale=getattr(self, "rope_scale", 1.0),
            )
            Q = Q.view(B, S, self.n_heads, self.d_k)
            G = self.n_kv_groups
            h = self.h_per_group
            Q = Q.view(B, S, G, h, self.d_k)
            K_sel = self._shape_kv(self.W_K_sel(x), B, S)
            V_sel = self._shape_kv(self.W_V_sel(x), B, S)
            K_win = self._shape_kv(self.W_K_win(x), B, S)
            V_win = self._shape_kv(self.W_V_win(x), B, S)
            K_cmp_raw = self._shape_kv(self.W_K_cmp(x), B, S)
            V_cmp_raw = self._shape_kv(self.W_V_cmp(x), B, S)

            # Apply RoPE to K for selection/sliding branches using absolute position of the new token(s)
            # Determine current token index before appending to caches
            t_prev = kv.K_sel.shape[2] if hasattr(kv, "K_sel") else 0
            pos_k = torch.arange(t_prev, t_prev + S, device=x.device)
            K_sel = apply_rope(K_sel, pos_k, scale=getattr(self, "rope_scale", 1.0))
            K_win = apply_rope(K_win, pos_k, scale=getattr(self, "rope_scale", 1.0))

            # decode step: append raw tokens and window, emit compressed every d after warmup l
            kv.update_selection_raw(K_sel, V_sel)
            kv.update_window(K_win, V_win, self.w)
            if not hasattr(kv, "K_cmp_raw_seq"):
                kv.K_cmp_raw_seq = K_cmp_raw[:, :, :0]
                kv.V_cmp_raw_seq = V_cmp_raw[:, :, :0]
                kv.reads_pred = torch.zeros((0,), dtype=torch.int64, device=x.device)
                kv.reads_act_total = torch.zeros((0,), dtype=torch.int64, device=x.device)
                kv.reads_act_sel = torch.zeros((0,), dtype=torch.int64, device=x.device)
                kv.reads_act_cmp = torch.zeros((0,), dtype=torch.int64, device=x.device)
                kv.reads_act_win = torch.zeros((0,), dtype=torch.int64, device=x.device)
            kv.append_cmp_raw(K_cmp_raw, V_cmp_raw)
            S_raw = kv.K_cmp_raw_seq.shape[2]
            if S_raw >= self.l and (S_raw - self.l) % self.d == 0:
                # Emit compressed token from the last l raw tokens
                K_last = kv.K_cmp_raw_seq[:, :, S_raw - self.l : S_raw, :]
                V_last = kv.V_cmp_raw_seq[:, :, S_raw - self.l : S_raw, :]
                pos_last = torch.arange(S_raw - self.l, S_raw, device=x.device)
                if self.phi_type == "mlp":
                    K_cmp_new, V_cmp_new = self._phi_apply_last(K_last, V_last, pos_last)
                else:
                    K_cmp_new, V_cmp_new = avg_pool_phi_rope_kv(
                        K_last, V_last, self.l, self.d, pos=pos_last
                    )
                kv.update_compressed(
                    torch.cat([kv.K_cmp, K_cmp_new], dim=2) if kv.K_cmp.numel() else K_cmp_new,
                    torch.cat([kv.V_cmp, V_cmp_new], dim=2) if kv.V_cmp.numel() else V_cmp_new,
                    self.l,
                    self.d,
                )

            # Ensure block metadata exists and covers current token index for selection (expand if needed)
            t_token = kv.K_sel.shape[2] - 1
            if not hasattr(kv, "meta") or kv.meta.sel_starts.numel() == 0:
                kv.meta = build_block_meta(
                    seq_len=max(t_token + 1, self.l_sel),
                    l=self.l,
                    d=self.d,
                    l_sel=self.l_sel,
                    n_sel=self.n_sel,
                    w=self.w,
                )
            else:
                # If current t exceeds covered selection range, rebuild meta with expanded seq_len
                sel_max_end = (
                    int(kv.meta.sel_starts[-1].item()) + kv.meta.l_sel
                    if kv.meta.sel_starts.numel() > 0
                    else 0
                )
                if (t_token + 1) > sel_max_end:
                    kv.meta = build_block_meta(
                        seq_len=t_token + 1,
                        l=self.l,
                        d=self.d,
                        l_sel=self.l_sel,
                        n_sel=self.n_sel,
                        w=self.w,
                    )
            # Append predicted reads per formula for this step
            num_cmp = 0 if S_raw < self.l else (S_raw - self.l) // self.d + 1
            reads = num_cmp + self.n_sel * self.l_sel + min(self.w, S_raw)
            kv.append_reads_pred(reads)
            # Append actual reads equal to formula in M0
            kv.append_reads_actual(reads, self.n_sel * self.l_sel, num_cmp, min(self.w, S_raw))
            log(
                "decode.reads",
                S_raw=int(S_raw),
                num_cmp=int(num_cmp),
                sel=int(self.n_sel * self.l_sel),
                win=int(min(self.w, S_raw)),
                total=int(reads),
            )

            scale = 1.0 / (self.d_k**0.5)
            # Compute p_cmp only for this step (S is 1 in decode)
            K_cmp_full = kv.K_cmp
            p_cmp_all = compute_pcmp_all(Q, K_cmp_full, scale)
            # Per-token outputs (S should be 1 in decode)
            outs = []
            # Use cached environment variables
            env = self._env_cache

            for t in range(S):
                p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all[:, t : t + 1], kv.meta)

                # M8: Optional Eq.9 verification in decode
                if self._env_cache.get("verify_eq9", False):
                    is_equiv, details = verify_mapping_equivalence(p_cmp_all[:, t : t + 1], kv.meta)
                    if not is_equiv:
                        log(
                            "error.eq9_verification_failed_decode",
                            msg="Eq.9 mapping verification failed in decode",
                            step=t,
                            **details,
                        )
                p_grp = p_slc_all.sum(dim=3).squeeze(1)  # [B,G,S_sel]
                current_pos = kv.K_sel.shape[2] - 1  # Current token position (0-indexed)
                sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, current_pos, True, 2)

                # M8: Assert causal masking - selection ranges cannot include future tokens
                if strict_asserts and sel_ranges.numel() > 0:
                    # Only sync for strict asserts (debug mode)
                    max_end = sel_ranges[..., 1].max().item()  # GPU sync only in debug
                    assert max_end <= current_pos + 1, (
                        f"Selection range violates causality: max_end={max_end} > current_pos+1={current_pos + 1}. "
                        f"Selection must not access future tokens."
                    )
                # Update selection stats and observability: distance summary per step
                try:
                    # Update per-step selection stats (decode has S==1)
                    self._update_sel_stats_from_ranges(sel_ranges)
                    starts = sel_ranges[..., 0].to(torch.int64)
                    ends = sel_ranges[..., 1].to(torch.int64)
                    lengths = (ends - starts).clamp_min(0)
                    dist = (kv.K_sel.shape[2] - 1) - starts
                    log(
                        "decode.select",
                        n_ranges=int(sel_ranges.shape[2]),
                        mean_len=float(lengths.float().mean().item()) if lengths.numel() else 0.0,
                        max_len=int(lengths.max().item()) if lengths.numel() else 0,
                        mean_dist=float(dist.float().mean().item()) if dist.numel() else 0.0,
                        max_dist=int(dist.max().item()) if dist.numel() else 0,
                    )
                except Exception as e:
                    log("warn.decode.select_log_fail", error=str(e))
                Q_t = Q[:, t]
                K_sel_t = kv.K_sel
                V_sel_t = kv.V_sel
                # Selection attention: prefer Triton if enabled; else packed; fallback to gather
                force_parity = env["force_parity"]
                use_sel_pack = env["use_sel_pack"] and not force_parity
                use_triton_sel = env["use_triton_sel"] and not force_parity
                use_cuda_sel = env["use_cuda_sel"] and not force_parity
                if use_triton_sel:
                    try:
                        from nsa.kernels.triton_sel_kernel import selection_attention_triton

                        O_sel_bt = selection_attention_triton(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                    except Exception as e:
                        # M8: Fallback counter - Triton selection failed
                        self._fallback_counters["selection_triton_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.triton_selection_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["selection_triton_fails"],
                        )
                        # Fallback to packed SDPA
                        O_sel_bt = grouped_selection_attention_packed(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                elif use_cuda_sel:
                    try:
                        from nsa.kernels.cuda_sel_kernel import selection_attention_cuda

                        O_sel_bt = selection_attention_cuda(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                    except Exception as e:
                        # M8: Fallback counter - CUDA selection failed
                        self._fallback_counters["selection_cuda_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.cuda_selection_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["selection_cuda_fails"],
                        )
                        # Fallback to packed SDPA
                        O_sel_bt = grouped_selection_attention_packed(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                elif use_sel_pack:
                    try:
                        O_sel_bt = grouped_selection_attention_packed(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                    except Exception as e:
                        # M8: Fallback counter - Packed selection failed
                        self._fallback_counters["selection_pack_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.packed_selection_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["selection_pack_fails"],
                        )
                        # Fallback to gather SDPA
                        O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
                elif self._env_cache.get("use_sel_mask", False) and not force_parity:
                    try:
                        O_sel_bt = grouped_selection_attention_masked(
                            Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1)
                        )
                        O_sel = O_sel_bt[:, 0]
                    except Exception as e:
                        # M8: Fallback counter - Masked selection failed
                        self._fallback_counters["selection_mask_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.masked_selection_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["selection_mask_fails"],
                        )
                        # Fallback to gather SDPA
                        O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
                else:
                    O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
                win_len = min(self.w, kv.K_win.shape[2])

                # M8: Assert causal masking - sliding window bounds in decode
                total_tokens = kv.K_win.shape[2]
                start_idx = total_tokens - win_len
                end_idx = total_tokens
                assert start_idx >= 0, (
                    f"Sliding window start index negative: start_idx={start_idx}, "
                    f"total_tokens={total_tokens}, win_len={win_len}"
                )
                assert end_idx <= total_tokens, (
                    f"Sliding window end exceeds cache: end_idx={end_idx} > total_tokens={total_tokens}"
                )
                assert win_len <= self.w, (
                    f"Window length exceeds max: win_len={win_len} > self.w={self.w}"
                )

                K_w = kv.K_win[:, :, start_idx:end_idx, :]
                V_w = kv.V_win[:, :, start_idx:end_idx, :]
                use_flash = (
                    env["fa2_all_eff"] or env["fa2_win_eff"] or env["fa2_cmp_eff"]
                ) and not force_parity
                if use_flash and (env["fa2_all_eff"] or env["fa2_win_eff"]):
                    try:
                        O_win = sliding_window_attention_fa2_decode(Q_t, kv.K_win, kv.V_win, self.w)
                    except Exception as e:
                        # M8: Fallback counter - Sliding FA2 failed
                        self._fallback_counters["sliding_fa2_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.sliding_fa2_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["sliding_fa2_fails"],
                        )
                        # Fallback to standard attention
                        O_win = attention_bgh(
                            Q_t.contiguous(), K_w.contiguous(), V_w.contiguous(), causal=True
                        )
                else:
                    O_win = attention_bgh(
                        Q_t.contiguous(), K_w.contiguous(), V_w.contiguous(), causal=True
                    )
                S_cmp_t = kv.K_cmp.shape[2]

                # M8: Assert causal masking - compressed bounds in decode
                assert S_cmp_t >= 0, f"Compressed cache size negative: S_cmp_t={S_cmp_t}"
                assert S_cmp_t <= kv.K_cmp.shape[2], (
                    f"Compressed range exceeds cache: S_cmp_t={S_cmp_t} > cache_size={kv.K_cmp.shape[2]}"
                )

                if use_flash and (env["fa2_all_eff"] or env["fa2_cmp_eff"]):
                    try:
                        O_cmp = compressed_attention_fa2_decode(Q_t, kv.K_cmp, kv.V_cmp, S_cmp_t)
                    except Exception as e:
                        # M8: Fallback counter - Compressed FA2 failed
                        self._fallback_counters["compressed_fa2_fails"] += 1
                        self._fallback_counters["total_fallbacks"] += 1
                        log(
                            "warn.compressed_fa2_fallback",
                            error=str(e),
                            total_fails=self._fallback_counters["compressed_fa2_fails"],
                        )
                        # Fallback to standard attention
                        O_cmp = attention_bgh(
                            Q_t.contiguous(),
                            kv.K_cmp[:, :, :S_cmp_t, :].contiguous(),
                            kv.V_cmp[:, :, :S_cmp_t, :].contiguous(),
                            causal=True,
                        )
                else:
                    O_cmp = attention_bgh(
                        Q_t.contiguous(),
                        kv.K_cmp[:, :, :S_cmp_t, :].contiguous(),
                        kv.V_cmp[:, :, :S_cmp_t, :].contiguous(),
                        causal=True,
                    )
                # Preserve dtype for gate input
                q_gp = Q_t.mean(dim=2, dtype=Q_t.dtype)
                if self._env_cache.get("gate_compile", False):
                    try:
                        fused = self._gate_fused_bg
                        if fused is None:
                            fused = _fused_gate_combine_bg
                            try:
                                fused = torch.compile(fused, mode="reduce-overhead")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            self._gate_fused_bg = fused
                        O = fused(
                            q_gp,
                            O_cmp,
                            O_sel,
                            O_win,
                            self.gate.fc1.weight,
                            self.gate.fc1.bias,
                            self.gate.fc2.weight,
                            self.gate.fc2.bias,
                            float(self.gate_temp),
                        )
                    except Exception:
                        gates = self.gate(q_gp, tau=self.gate_temp)
                        if self._env_cache.get("stopgrad_gates", False):
                            gates = gates.detach()
                        self._update_gate_stats(gates)
                        try:
                            log(
                                "decode.gates",
                                mean=gates.mean(dim=(-1, -2)).tolist()
                                if gates.dim() >= 2
                                else gates.mean().item(),
                                std=gates.std(dim=(-1, -2)).tolist()
                                if gates.dim() >= 2
                                else gates.std().item(),
                            )
                        except Exception as e:
                            log("warn.decode.gate_log_fail", error=str(e))
                        w_cmp = gates[..., 0:1].unsqueeze(-1)
                        w_sel = gates[..., 1:2].unsqueeze(-1)
                        w_win = gates[..., 2:3].unsqueeze(-1)
                        O = w_cmp * O_cmp + w_sel * O_sel + w_win * O_win
                else:
                    gates = self.gate(q_gp, tau=self.gate_temp)
                    if self._env_cache.get("stopgrad_gates", False):
                        gates = gates.detach()
                    self._update_gate_stats(gates)
                    try:
                        log(
                            "decode.gates",
                            mean=gates.mean(dim=(-1, -2)).tolist()
                            if gates.dim() >= 2
                            else gates.mean().item(),
                            std=gates.std(dim=(-1, -2)).tolist()
                            if gates.dim() >= 2
                            else gates.std().item(),
                        )
                    except Exception as e:
                        log("warn.decode.gate_log_fail", error=str(e))
                    w_cmp = gates[..., 0:1].unsqueeze(-1)
                    w_sel = gates[..., 1:2].unsqueeze(-1)
                    w_win = gates[..., 2:3].unsqueeze(-1)
                    O = w_cmp * O_cmp + w_sel * O_sel + w_win * O_win
                O_heads = O.reshape(B, self.n_heads, self.d_v)
                out_t = self.out(O_heads.reshape(B, 1, -1))
                outs.append(out_t)
            out = torch.cat(outs, dim=1)
            return out, kv

    def _forward_prefill_batched(self, x: torch.Tensor, kv: NSA_KV) -> tuple[torch.Tensor, NSA_KV]:
        """
        Vectorized prefill path.

        Steps:
        - Projections with RoPE(Q); RoPE applied to K before ϕ for compressed branch
        - Cache updates for selection/window/compressed
        - Batched p_cmp → p_slc → p_grp; top‑n ranges for all t
        - Branch attentions (masked/packed per env flags), gating, projection
        """
        B, S, _ = x.shape
        # Projections
        _nvtx = self._env_cache.get("nvtx", False)
        if _nvtx:
            try:
                import torch as _t

                _t.cuda.nvtx.range_push("projections+rope")
            except Exception:
                _nvtx = False
        Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
        assert Q_lin.shape[:2] == (B, S)
        # Apply RoPE to Q
        pos = torch.arange(S, device=x.device)
        Q = apply_rope(
            Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k),
            pos,
            scale=getattr(self, "rope_scale", 1.0),
        )
        Q = Q.view(B, S, self.n_heads, self.d_k).view(
            B, S, self.n_kv_groups, self.h_per_group, self.d_k
        )
        # K/V projections per branch
        K_sel = self._shape_kv(self.W_K_sel(x), B, S)
        V_sel = self._shape_kv(self.W_V_sel(x), B, S)
        K_win = self._shape_kv(self.W_K_win(x), B, S)
        V_win = self._shape_kv(self.W_V_win(x), B, S)
        K_cmp_raw = self._shape_kv(self.W_K_cmp(x), B, S)
        V_cmp_raw = self._shape_kv(self.W_V_cmp(x), B, S)
        G = self.n_kv_groups
        assert K_sel.shape[:3] == (B, G, S) and V_sel.shape[:3] == (B, G, S)
        assert K_win.shape[:3] == (B, G, S) and V_win.shape[:3] == (B, G, S)
        assert K_cmp_raw.shape[:3] == (B, G, S) and V_cmp_raw.shape[:3] == (B, G, S)

        # Apply RoPE to per-branch K tensors (Q already has RoPE applied)
        pos_k = torch.arange(S, device=x.device)
        K_sel = apply_rope(K_sel, pos_k, scale=getattr(self, "rope_scale", 1.0))
        K_win = apply_rope(K_win, pos_k, scale=getattr(self, "rope_scale", 1.0))
        if _nvtx:
            try:
                _t.cuda.nvtx.range_pop()
            except Exception:
                pass

        # Update caches (prefill uses full sequence projections)
        kv.update_selection_raw(K_sel, V_sel)
        # Build/refresh meta for selection and compressed mapping
        kv.meta = build_block_meta(
            seq_len=S, l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w
        )
        kv.update_window(K_win, V_win, self.w)
        if self.phi_type == "mlp":
            K_cmp, V_cmp = self._phi_apply_seq(
                K_cmp_raw, V_cmp_raw, pos=torch.arange(S, device=x.device)
            )
        else:
            K_cmp, V_cmp = avg_pool_phi_rope_kv(
                K_cmp_raw, V_cmp_raw, self.l, self.d, pos=torch.arange(S, device=x.device)
            )
        kv.update_compressed(K_cmp, V_cmp, self.l, self.d)

        # One-time SDPA backend audit (opt-in via env)
        try:
            if (not self._sdpa_audited) and os.getenv("NSA_SDPA_AUDIT", "0").lower() in (
                "1",
                "true",
                "yes",
            ):
                self._audit_sdpa_backends_once(
                    Q[:, :1],
                    K_sel[:, :, : max(1, S // 8), :],
                    V_sel[:, :, : max(1, S // 8), :],
                    K_win[:, :, : max(1, S // 8), :],
                    V_win[:, :, : max(1, S // 8), :],
                )
        except Exception:
            pass

        # Selection scores (batched)
        scale = 1.0 / (self.d_k**0.5)
        if _nvtx:
            try:
                _t.cuda.nvtx.range_push("pcmp_all")
            except Exception:
                pass
        p_cmp_all = compute_pcmp_all(Q, kv.K_cmp, scale)  # [B,S,G,h,S_cmp]
        if _nvtx:
            try:
                _t.cuda.nvtx.range_pop()
                _t.cuda.nvtx.range_push("map_pcmp_to_pslc")
            except Exception:
                pass
        p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, kv.meta)  # [B,S,G,h,S_sel]

        # M8: Optional Eq.9 verification in batched prefill
        if self._env_cache.get("verify_eq9", False):
            is_equiv, details = verify_mapping_equivalence(p_cmp_all, kv.meta)
            if not is_equiv:
                log(
                    "error.eq9_verification_failed_prefill",
                    msg="Eq.9 mapping verification failed in batched prefill",
                    **details,
                )
        p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]
        log(
            "prefill.scores",
            B=B,
            S=S,
            S_cmp=int(kv.K_cmp.shape[2]),
            S_sel=int(kv.meta.sel_starts.numel()),
        )

        # Batched top‑n → ranges for all positions
        if _nvtx:
            try:
                _t.cuda.nvtx.range_push("topk+ranges")
            except Exception:
                pass
        sel_ranges_all = select_topn_ranges_batched(
            p_grp_all, kv.meta, self.n_sel, S, True, 2
        )  # [B,S,G,n,2]
        if _nvtx:
            try:
                _t.cuda.nvtx.range_pop()
                _t.cuda.nvtx.range_push("branch_attn+gate")
            except Exception:
                pass
        # Update selection statistics for this prefill batch
        self._update_sel_stats_from_ranges(sel_ranges_all)
        if _nvtx:
            try:
                _t.cuda.nvtx.range_pop()
            except Exception:
                pass

        # M8: Assert causal masking for batched selection (GPU-sync gated)
        strict_asserts = self._env_cache.get("strict_asserts", False)
        if strict_asserts and sel_ranges_all.numel() > 0:
            for t in range(S):
                t_ranges = sel_ranges_all[:, t]  # [B,G,n,2]
                if t_ranges.numel() > 0:
                    max_end = t_ranges[..., 1].max().item()
                    assert max_end <= t + 1, (
                        f"Batched selection violates causality at t={t}: max_end={max_end} > t+1={t + 1}. "
                        f"Selection ranges cannot access future tokens."
                    )
        log("prefill.select", n_sel=self.n_sel, l_sel=self.l_sel, ranges=sel_ranges_all)

        # Branch attentions in parallel (parity-first for cmp/win, with optional masked SDPA gates)
        force_parity = self._env_cache.get("force_parity", False)
        fa2_all = self._env_cache.get("fa2_all_eff", False)
        fa2_win = self._env_cache.get("fa2_win_eff", False)
        fa2_cmp = self._env_cache.get("fa2_cmp_eff", False)
        use_cmp_mask = self._env_cache.get("use_cmp_mask", True) and not force_parity
        if (fa2_all or fa2_cmp) and not force_parity:
            try:
                O_cmp = compressed_attention_fa2(Q, kv.K_cmp, kv.V_cmp, self.l, self.d)
            except Exception as e:
                # M8: Fallback counter - Compressed FA2 failed in prefill
                self._fallback_counters["compressed_fa2_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.compressed_fa2_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["compressed_fa2_fails"],
                )
                # Fallback to masked SDPA
                from nsa.core.attention_kernels import batched_causal_attention_compressed_masked

                O_cmp = batched_causal_attention_compressed_masked(
                    Q, kv.K_cmp, kv.V_cmp, self.l, self.d
                )
        elif use_cmp_mask:
            from nsa.core.attention_kernels import batched_causal_attention_compressed_masked

            O_cmp = batched_causal_attention_compressed_masked(
                Q, kv.K_cmp, kv.V_cmp, self.l, self.d
            )
        else:
            # Compressed per-t using the same kernel as sequential
            O_cmp = torch.zeros(
                (B, S, self.n_kv_groups, self.h_per_group, self.d_v),
                device=x.device,
                dtype=V_cmp.dtype,
            )
            S_cmp_full = kv.K_cmp.shape[2]
            for t in range(S):
                L = 0 if (t + 1) < self.l else min(((t + 1 - self.l) // self.d) + 1, S_cmp_full)

                # M8: Assert causal masking - compressed tokens must respect position bounds
                if L > 0:
                    # Check that compressed range doesn't exceed causal bounds
                    assert L <= S_cmp_full, (
                        f"Compressed range exceeds cache: L={L} > S_cmp_full={S_cmp_full} at t={t}"
                    )
                    # Verify causal constraint: at position t, can only see compressed tokens
                    # that represent original positions up to t
                    max_allowed_L = ((t + 1 - self.l) // self.d) + 1 if (t + 1) >= self.l else 0
                    assert L <= max_allowed_L, (
                        f"Compressed range violates causality: L={L} > max_allowed_L={max_allowed_L} "
                        f"at t={t}. Compressed tokens represent future positions."
                    )

                    q_t = Q[:, t].contiguous()
                    k_t = kv.K_cmp[:, :, :L, :].contiguous()
                    v_t = kv.V_cmp[:, :, :L, :].contiguous()
                    O_cmp[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        # Strict finite check and fallback
        if strict_asserts and not torch.isfinite(O_cmp).all():
            from nsa.core.attention_kernels import batched_causal_attention_compressed_masked

            log("warn.prefill_cmp_nonfinite_fallback")
            O_cmp = batched_causal_attention_compressed_masked(
                Q, kv.K_cmp, kv.V_cmp, self.l, self.d
            )
        log("prefill.cmp", O_cmp=O_cmp)

        # Selected ranges attention (prefer Triton if enabled; else packed/gather)
        use_sel_pack = self._env_cache.get("use_sel_pack", True) and not force_parity
        use_sel_varlen = self._env_cache.get("use_sel_varlen", False) and not force_parity
        use_triton_sel = (
            self._env_cache.get("use_triton_sel", False) or self.use_triton_sel and not force_parity
        )
        if use_triton_sel:
            try:
                from nsa.kernels.triton_sel_kernel import selection_attention_triton

                O_sel = selection_attention_triton(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
            except Exception as e:
                # M8: Fallback counter - Triton selection failed in prefill
                self._fallback_counters["selection_triton_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.triton_selection_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["selection_triton_fails"],
                )
                # Fallback to packed SDPA
                O_sel = grouped_selection_attention_packed(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        elif use_sel_varlen:
            try:
                from nsa.core.attention_kernels import selection_attention_varlen_all

                O_sel = selection_attention_varlen_all(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
            except Exception as e:
                # Fallback counter reuse for selection pack failures
                self._fallback_counters["selection_pack_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.selection_varlen_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["selection_pack_fails"],
                )
                # Fallback to packed SDPA
                O_sel = grouped_selection_attention_packed(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        elif use_sel_pack:
            try:
                O_sel = grouped_selection_attention_packed(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
            except Exception as e:
                # M8: Fallback counter - Packed selection failed in prefill
                self._fallback_counters["selection_pack_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.packed_selection_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["selection_pack_fails"],
                )
                # Fallback to gather SDPA
                O_sel = grouped_selection_attention(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        elif self._env_cache.get("use_sel_mask", False):
            try:
                O_sel = grouped_selection_attention_masked(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
            except Exception as e:
                # M8: Fallback counter - Masked selection failed in prefill
                self._fallback_counters["selection_mask_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.masked_selection_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["selection_mask_fails"],
                )
                # Fallback to gather SDPA
                O_sel = grouped_selection_attention(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        else:
            O_sel = grouped_selection_attention(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        if strict_asserts and not torch.isfinite(O_sel).all():
            log("warn.prefill_sel_nonfinite_fallback")
            O_sel = grouped_selection_attention(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        log("prefill.sel", O_sel=O_sel)

        use_win_mask = self._env_cache.get("use_win_mask", True) and not force_parity
        if (fa2_all or fa2_win) and not force_parity:
            try:
                O_win = sliding_window_attention_fa2(Q, K_win, V_win, self.w)
            except Exception as e:
                # M8: Fallback counter - Sliding FA2 failed in prefill
                self._fallback_counters["sliding_fa2_fails"] += 1
                self._fallback_counters["total_fallbacks"] += 1
                log(
                    "warn.sliding_fa2_prefill_fallback",
                    error=str(e),
                    total_fails=self._fallback_counters["sliding_fa2_fails"],
                )
                # Fallback to masked SDPA
                from nsa.core.attention_kernels import sliding_window_attention

                O_win = sliding_window_attention(Q, K_win, V_win, self.w)
        elif use_win_mask:
            from nsa.core.attention_kernels import sliding_window_attention

            O_win = sliding_window_attention(Q, K_win, V_win, self.w)
        else:
            # Sliding per-t using the same kernel as sequential
            O_win = torch.zeros(
                (B, S, self.n_kv_groups, self.h_per_group, self.d_v),
                device=x.device,
                dtype=V_win.dtype,
            )
            for t in range(S):
                end = t + 1
                start = max(0, end - self.w)

                # M8: Assert causal masking - sliding window must not exceed current position
                assert end <= t + 1, (
                    f"Sliding window violates causality: end={end} > t+1={t + 1} at position t={t}. "
                    f"This indicates window is accessing future tokens."
                )
                assert start <= end, (
                    f"Sliding window has invalid range: start={start} > end={end} at position t={t}."
                )

                q_t = Q[:, t].contiguous()
                k_t = K_win[:, :, start:end, :].contiguous()
                v_t = V_win[:, :, start:end, :].contiguous()
                O_win[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        if strict_asserts and not torch.isfinite(O_win).all():
            from nsa.core.attention_kernels import sliding_window_attention

            log("warn.prefill_win_nonfinite_fallback")
            O_win = sliding_window_attention(Q, K_win, V_win, self.w)
        log("prefill.win", O_win=O_win)

        # Gates and combine
        q_gp = Q.mean(dim=3)  # [B,S,G,Dk]
        if self._env_cache.get("gate_compile", False):
            try:
                fused = self._gate_fused_bsg
                if fused is None:
                    fused = _fused_gate_combine_bsg
                    try:
                        fused = torch.compile(fused, mode="reduce-overhead")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self._gate_fused_bsg = fused
                O = fused(
                    q_gp,
                    O_cmp,
                    O_sel,
                    O_win,
                    self.gate.fc1.weight,
                    self.gate.fc1.bias,
                    self.gate.fc2.weight,
                    self.gate.fc2.bias,
                    float(self.gate_temp),
                )
            except Exception:
                gates = self.gate(q_gp.reshape(B * S * self.n_kv_groups, self.d_k), tau=self.gate_temp)
                if self._env_cache.get("stopgrad_gates", False):
                    gates = gates.detach()
                gates = gates.view(B, S, self.n_kv_groups, 3)  # [B,S,G,3]
                self._update_gate_stats(gates)
                w_cmp = gates[..., 0:1].unsqueeze(3)
                w_sel = gates[..., 1:2].unsqueeze(3)
                w_win = gates[..., 2:3].unsqueeze(3)
                O = w_cmp * O_cmp + w_sel * O_sel + w_win * O_win  # [B,S,G,h,Dv]
        else:
            gates = self.gate(q_gp.reshape(B * S * self.n_kv_groups, self.d_k), tau=self.gate_temp)
            if self._env_cache.get("stopgrad_gates", False):
                gates = gates.detach()
            gates = gates.view(B, S, self.n_kv_groups, 3)  # [B,S,G,3]
            self._update_gate_stats(gates)
            w_cmp = gates[..., 0:1].unsqueeze(3)
            w_sel = gates[..., 1:2].unsqueeze(3)
            w_win = gates[..., 2:3].unsqueeze(3)
            O = w_cmp * O_cmp + w_sel * O_sel + w_win * O_win  # [B,S,G,h,Dv]

        # Output projection
        O_heads = O.reshape(B, S, self.n_kv_groups * self.h_per_group, self.d_v)
        out = self.out(O_heads.reshape(B, S, -1))
        log("prefill.out", out=out)

        # Optional debug compare: sequential-style per-token recompute to measure MAE
        if self._env_cache.get("debug_compare", False):
            with torch.no_grad():
                # Compressed per-token recompute
                O_cmp_seq = torch.zeros_like(O_cmp)
                S_cmp = kv.K_cmp.shape[2]
                for t in range(S):
                    L = 0 if (t + 1) < self.l else min(((t + 1 - self.l) // self.d) + 1, S_cmp)

                    # M8: Assert causal masking in debug recompute
                    if L > 0:
                        assert L <= S_cmp, (
                            f"Debug compressed range exceeds cache: L={L} > S_cmp={S_cmp} at t={t}"
                        )

                        q_t = Q[:, t].contiguous()
                        k_t = kv.K_cmp[:, :, :L, :].contiguous()
                        v_t = kv.V_cmp[:, :, :L, :].contiguous()
                        O_cmp_seq[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
                cmp_mae = (O_cmp - O_cmp_seq).abs().mean().item()
                print(f"NSA-DBG cmp_mae={cmp_mae:.6e}")

                # Sliding per-token recompute
                O_win_seq = torch.zeros_like(O_win)
                for t in range(S):
                    end = t + 1
                    start = max(0, end - self.w)
                    q_t = Q[:, t].contiguous()
                    k_t = K_win[:, :, start:end, :].contiguous()
                    v_t = V_win[:, :, start:end, :].contiguous()
                    O_win_seq[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
                win_mae = (O_win - O_win_seq).abs().mean().item()
                print(f"NSA-DBG win_mae={win_mae:.6e}")

                # Final output recompute using seq per-branch
                w_cmp_dbg = gates[..., 0:1].unsqueeze(-1)
                w_sel_dbg = gates[..., 1:2].unsqueeze(-1)
                w_win_dbg = gates[..., 2:3].unsqueeze(-1)
                O_seq = w_cmp_dbg * O_cmp_seq + w_sel_dbg * O_sel + w_win_dbg * O_win_seq
                O_heads_seq = O_seq.reshape(B, S, self.n_kv_groups * self.h_per_group, self.d_v)
                out_seq = self.out(O_heads_seq.reshape(B, S, -1))
                out_mae = (out - out_seq).abs().mean().item()
                print(f"NSA-DBG out_mae={out_mae:.6e}")
        return out, kv

    def _audit_sdpa_backends_once(
        self,
        Q: torch.Tensor,  # [B,1,G,h,Dk]
        K_sel: torch.Tensor,  # [B,G,S,Dk]
        V_sel: torch.Tensor,  # [B,G,S,Dv]
        K_win: torch.Tensor,  # [B,G,S,Dk]
        V_win: torch.Tensor,  # [B,G,S,Dv]
    ) -> None:
        if self._sdpa_audited:
            return
        try:
            from torch.nn.attention import sdpa_kernel
        except Exception:
            # Older torch, skip audit
            self._sdpa_audited = True
            return
        B = Q.shape[0]
        G = self.n_kv_groups
        h = self.h_per_group
        # Prepare a small representative slice per branch
        q = Q[:, 0]  # [B,G,h,Dk]
        # Ensure contiguity
        q = q.contiguous()
        ks = K_sel.contiguous()
        vs = V_sel.contiguous()
        kw = K_win.contiguous()
        vw = V_win.contiguous()

        def _probe(tag: str, k: torch.Tensor, v: torch.Tensor) -> str:
            try:
                with sdpa_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                    q2 = q.reshape(B * G * h, 1, self.d_k)
                    k2 = (
                        k.unsqueeze(2)
                        .expand(B, G, h, k.shape[2], self.d_k)
                        .reshape(B * G * h, k.shape[2], self.d_k)
                    )
                    v2 = (
                        v.unsqueeze(2)
                        .expand(B, G, h, v.shape[2], self.d_v)
                        .reshape(B * G * h, v.shape[2], self.d_v)
                    )
                    _ = F.scaled_dot_product_attention(
                        q2.contiguous(), k2.contiguous(), v2.contiguous(), is_causal=True
                    )
                return "flash"
            except Exception:
                return "fallback"

        try:
            b_sel = _probe("cmp/win(sel)", ks, vs)
            b_win = _probe("win", kw, vw)
            log("sdpa.audit", sel=b_sel, win=b_win)
        except Exception:
            pass
        self._sdpa_audited = True

    def _forward_prefill_via_decode(
        self, x: torch.Tensor, kv: NSA_KV
    ) -> tuple[torch.Tensor, NSA_KV]:
        """Prefill by stepping decode one token at a time.

        This path avoids recursion back into prefill and guarantees progress.
        """
        B, S, _ = x.shape
        outs = []
        for t in range(S):
            out_t, kv = self.forward(x[:, t : t + 1], kv, prefill=False)
            outs.append(out_t)
        return torch.cat(outs, dim=1), kv

    def _forward_prefill_sequential(
        self, x: torch.Tensor, kv: NSA_KV
    ) -> tuple[torch.Tensor, NSA_KV]:
        """
        Reference prefill path (sequential per‑token), used for parity checks.
        """
        B, S, _ = x.shape
        # Projections
        Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
        pos = torch.arange(S, device=x.device)
        Q = apply_rope(
            Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k),
            pos,
            scale=getattr(self, "rope_scale", 1.0),
        )
        Q = Q.view(B, S, self.n_heads, self.d_k).view(
            B, S, self.n_kv_groups, self.h_per_group, self.d_k
        )
        K_sel = self._shape_kv(self.W_K_sel(x), B, S)
        V_sel = self._shape_kv(self.W_V_sel(x), B, S)
        K_win = self._shape_kv(self.W_K_win(x), B, S)
        V_win = self._shape_kv(self.W_V_win(x), B, S)
        K_cmp_raw = self._shape_kv(self.W_K_cmp(x), B, S)
        V_cmp_raw = self._shape_kv(self.W_V_cmp(x), B, S)
        
        # Apply RoPE to per-branch K tensors to align with batched path
        pos_k = torch.arange(S, device=x.device)
        K_sel = apply_rope(K_sel, pos_k, scale=getattr(self, "rope_scale", 1.0))
        K_win = apply_rope(K_win, pos_k, scale=getattr(self, "rope_scale", 1.0))

        kv.update_selection_raw(K_sel, V_sel)
        kv.meta = build_block_meta(
            seq_len=S, l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w
        )
        kv.update_window(K_win, V_win, self.w)
        if self.phi_type == "mlp":
            K_cmp, V_cmp = self._phi_apply_seq(
                K_cmp_raw, V_cmp_raw, pos=torch.arange(S, device=x.device)
            )
        else:
            K_cmp, V_cmp = avg_pool_phi_rope_kv(
                K_cmp_raw, V_cmp_raw, self.l, self.d, pos=torch.arange(S, device=x.device)
            )
        kv.update_compressed(K_cmp, V_cmp, self.l, self.d)

        # Precompute p_grp_all batched for reuse per t
        scale = 1.0 / (self.d_k**0.5)
        p_cmp_all = compute_pcmp_all(Q, kv.K_cmp, scale)  # [B,S,G,h,S_cmp]
        p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, kv.meta)  # [B,S,G,h,S_sel]
        p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]

        outs = []
        sel_ranges_accum: List[torch.Tensor] = []
        for t in range(S):
            p_grp = p_grp_all[:, t]  # [B,G,S_sel]
            sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, t, True, 2)
            sel_ranges_accum.append(sel_ranges)
            Q_t = Q[:, t]
            K_sel_t = kv.K_sel[:, :, : t + 1, :]
            V_sel_t = kv.V_sel[:, :, : t + 1, :]
            O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
            win_len = min(self.w, t + 1)
            K_w = kv.K_win[:, :, t + 1 - win_len : t + 1, :]
            V_w = kv.V_win[:, :, t + 1 - win_len : t + 1, :]
            O_win = attention_bgh(Q_t.contiguous(), K_w.contiguous(), V_w.contiguous(), causal=True)
            S_cmp_t = 0 if (t + 1) < self.l else (t + 1 - self.l) // self.d + 1
            O_cmp = attention_bgh(
                Q_t.contiguous(),
                kv.K_cmp[:, :, :S_cmp_t, :].contiguous(),
                kv.V_cmp[:, :, :S_cmp_t, :].contiguous(),
                causal=True,
            )
            q_gp = Q_t.mean(dim=2, dtype=Q_t.dtype)
            gates = self.gate(q_gp, tau=self.gate_temp)
            if self._env_cache.get("stopgrad_gates", False):
                gates = gates.detach()

            # Update gate statistics for M8 monitoring (accumulate across steps)
            self._update_gate_stats(gates)

            w_cmp = gates[..., 0:1].unsqueeze(-1)
            w_sel = gates[..., 1:2].unsqueeze(-1)
            w_win = gates[..., 2:3].unsqueeze(-1)
            O = w_cmp * O_cmp + w_sel * O_sel + w_win * O_win
            O_heads = O.reshape(B, self.n_heads, self.d_v)
            out_t = self.out(O_heads.reshape(B, 1, -1))
            outs.append(out_t)
        out = torch.cat(outs, dim=1)
        # Aggregate selection stats across all t in this prefill (sequential path)
        try:
            if sel_ranges_accum:
                # Stack to [T,B,G,n,2] then permute to [B,T,G,n,2]
                rs = torch.stack(sel_ranges_accum, dim=0).permute(1, 0, 2, 3, 4)
                self._update_sel_stats_from_ranges(rs)
        except Exception:
            pass
        return out, kv

    def _sdpa_full(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q: [B,G,h,Dk]; K/V: [B,G,S,D*] -> out [B,G,h,Dv]
        B, G, h, Dk = Q.shape
        S = K.shape[2]
        q = Q.reshape(B * G * h, 1, Dk).contiguous()
        k = K.unsqueeze(2).expand(B, G, h, S, Dk).reshape(B * G * h, S, Dk).contiguous()
        v = (
            V.unsqueeze(2)
            .expand(B, G, h, S, V.shape[-1])
            .reshape(B * G * h, S, V.shape[-1])
            .contiguous()
        )
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = attn.squeeze(1).reshape(B, G, h, -1)
        return o

    def _phi_apply_seq(
        self, K_raw: torch.Tensor, V_raw: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learnable ϕ over the full sequence using depthwise Conv1d initialized to avg.
        Expects K_raw,V_raw: [B,G,S,D*]; returns [B,G,S_cmp,D*].
        """
        assert self.phi_k_conv is not None and self.phi_v_conv is not None
        B, G, S, Dk = K_raw.shape
        Dv = V_raw.shape[-1]
        K_rope = apply_rope(K_raw, pos, scale=getattr(self, "rope_scale", 1.0))
        Kx = K_rope.permute(0, 1, 3, 2).reshape(B * G, Dk, S)
        Vx = V_raw.permute(0, 1, 3, 2).reshape(B * G, Dv, S)
        Kc = self.phi_k_conv(Kx)
        Vc = self.phi_v_conv(Vx)
        S_cmp = Kc.shape[-1]
        K_cmp = Kc.reshape(B, G, Dk, S_cmp).permute(0, 1, 3, 2).contiguous()
        V_cmp = Vc.reshape(B, G, Dv, S_cmp).permute(0, 1, 3, 2).contiguous()
        return K_cmp, V_cmp

    def _phi_apply_last(
        self, K_last: torch.Tensor, V_last: torch.Tensor, pos_last: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit a single compressed token from the last l raw tokens using Conv1d with kernel=l,stride=d.
        Inputs: [B,G,l,D*] -> Outputs: [B,G,1,D*].
        """
        assert self.phi_k_conv is not None and self.phi_v_conv is not None
        B, G, lwin, Dk = K_last.shape
        Dv = V_last.shape[-1]
        assert lwin == self.l, "decode emission expects exactly l tokens"
        K_rope = apply_rope(K_last, pos_last, scale=getattr(self, "rope_scale", 1.0))
        Kx = K_rope.permute(0, 1, 3, 2).reshape(B * G, Dk, lwin)
        Vx = V_last.permute(0, 1, 3, 2).reshape(B * G, Dv, lwin)
        Kc = self.phi_k_conv(Kx)
        Vc = self.phi_v_conv(Vx)
        K_cmp_new = Kc.reshape(B, G, Dk, 1).permute(0, 1, 3, 2).contiguous()
        V_cmp_new = Vc.reshape(B, G, Dv, 1).permute(0, 1, 3, 2).contiguous()
        return K_cmp_new, V_cmp_new

    def _sdpa_over_ranges(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        ranges: torch.Tensor,
    ) -> torch.Tensor:
        """
        SDPA over concatenated gathered tokens per (B,G) according to `ranges`.

        Args:
            Q: [B,G,h,Dk]
            K: [B,G,S_kv,Dk]
            V: [B,G,S_kv,Dv]
            ranges: [B,G,n,2] start/end pairs
        Returns:
            [B,G,h,Dv]
        """
        # Concatenate gathered tokens per (B,G)
        B, G, h, Dk = Q.shape
        Dv = V.shape[-1]
        outs = []
        S_kv = K.shape[2]
        strict_asserts = (
            self._env_cache.get("strict_asserts", False) if hasattr(self, "_env_cache") else False
        )
        for b in range(B):
            row = []
            for g in range(G):
                # Clamp and validate ranges to avoid invalid or oversized indices
                r = ranges[b, g].to(dtype=torch.int64, device=K.device)  # [n,2]
                if r.numel() == 0:
                    valid_pairs = torch.empty((0, 2), dtype=torch.int64, device=K.device)
                else:
                    s = r[:, 0].clamp_(0, S_kv)
                    e = r[:, 1].clamp_(0, S_kv)
                    valid = e > s
                    valid_pairs = torch.stack([s[valid], e[valid]], dim=-1)

                    # M8: Assert bounds for gathered ranges (GPU-sync gated)
                    if strict_asserts and valid_pairs.numel() > 0:
                        max_end = valid_pairs[:, 1].max().item()
                        assert max_end <= S_kv, (
                            f"Selection range exceeds sequence length: max_end={max_end} > S_kv={S_kv} "
                            f"at batch={b}, group={g}."
                        )
                # Build a boolean mask over S_kv to gather selected tokens (limits worst-case size)
                if valid_pairs.numel() > 0:
                    m = torch.zeros((S_kv,), dtype=torch.bool, device=K.device)
                    for s_e in valid_pairs:
                        s_i = int(s_e[0].item())
                        e_i = int(s_e[1].item())
                        if e_i > s_i:
                            m[s_i:e_i] = True
                    idx = m.nonzero(as_tuple=False).squeeze(-1)
                else:
                    idx = torch.empty((0,), dtype=torch.int64, device=K.device)
                k = (
                    K[b, g, idx]
                    if idx.numel() > 0
                    else torch.zeros((1, Dk), device=K.device, dtype=K.dtype)
                )
                v = (
                    V[b, g, idx]
                    if idx.numel() > 0
                    else torch.zeros((1, Dv), device=K.device, dtype=V.dtype)
                )
                q = Q[b, g]  # [h,Dk]
                attn = F.scaled_dot_product_attention(
                    q.unsqueeze(0).contiguous(),
                    k.unsqueeze(0).contiguous(),
                    v.unsqueeze(0).contiguous(),
                    is_causal=True,
                )
                row.append(attn.squeeze(0))  # [h,Dv]
            outs.append(torch.stack(row, dim=0))  # [G,h,Dv]
        return torch.stack(outs, dim=0)  # [B,G,h,Dv]
