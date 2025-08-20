from dataclasses import dataclass
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.cache.kv_cache import NSA_KV
from nsa.core.compress_pool import avg_pool_phi_rope_kv
from nsa.core.rope import apply_rope
from nsa.core.selection_scorer import (
    compute_pcmp,
    compute_pcmp_all,
    map_pcmp_to_pslc,
    map_pcmp_to_pslc_batched,
    group_reduce_pslc,
    select_topn_ranges,
)
from nsa.kernels.flash_wrappers import attention_bgh
from nsa.core.attention_kernels import (
    batched_causal_attention_compressed,
    sliding_window_attention,
    sliding_window_attention_fa2,
    grouped_selection_attention,
    grouped_selection_attention_packed,
    grouped_selection_attention_masked,
    compressed_attention_fa2,
    sliding_window_attention_fa2_decode,
    compressed_attention_fa2_decode,
)
from nsa.core.selection_scorer import select_topn_ranges_batched
from nsa.core.debug import log
from nsa.core.block_index import build_block_meta


class GateMLP(nn.Module):
    def __init__(self, d_k: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(1, d_k // 2)
        self.fc1 = nn.Linear(d_k, hidden)
        self.fc2 = nn.Linear(hidden, 3)
        # zero-init last layer per PRD
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, q_group_pooled: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
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
        # Selection Triton toggle (M4)
        self.use_triton_sel = use_triton_sel
        # Optional learnable ϕ via depthwise Conv1d over time with kernel l and stride d
        # Initialize to average pooling for parity with M0
        self.phi_k_conv: Optional[nn.Conv1d]
        self.phi_v_conv: Optional[nn.Conv1d]
        if self.phi_type == "mlp":
            self.phi_k_conv = nn.Conv1d(self.d_k, self.d_k, kernel_size=self.l, stride=self.d, groups=self.d_k, bias=False)
            self.phi_v_conv = nn.Conv1d(self.d_v, self.d_v, kernel_size=self.l, stride=self.d, groups=self.d_v, bias=False)
            with torch.no_grad():
                self.phi_k_conv.weight.fill_(1.0 / float(self.l))
                self.phi_v_conv.weight.fill_(1.0 / float(self.l))
        else:
            self.phi_k_conv = None
            self.phi_v_conv = None

    def _shape_q(self, Q: torch.Tensor, B: int, S: int) -> torch.Tensor:
        Q = Q.view(B, S, self.n_heads, self.d_k)
        # group-major: [B,S,G,h,Dk]
        G = self.n_kv_groups
        h = self.h_per_group
        return Q.view(B, S, G, h, self.d_k)

    def _shape_kv(self, X: torch.Tensor, B: int, S: int) -> torch.Tensor:
        G = self.n_kv_groups
        return X.view(B, S, G, -1).permute(0, 2, 1, 3).contiguous()  # [B,G,S,D*]

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
        if prefill:
            use_batched = os.getenv("NSA_PREFILL_BATCHED", "0").lower() in ("1", "true", "yes")
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
            Q = apply_rope(Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k), pos)
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
            K_sel = apply_rope(K_sel, pos_k)
            K_win = apply_rope(K_win, pos_k)

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
                    K_cmp_new, V_cmp_new = avg_pool_phi_rope_kv(K_last, V_last, self.l, self.d, pos=pos_last)
                kv.update_compressed(
                    torch.cat([kv.K_cmp, K_cmp_new], dim=2) if kv.K_cmp.numel() else K_cmp_new,
                    torch.cat([kv.V_cmp, V_cmp_new], dim=2) if kv.V_cmp.numel() else V_cmp_new,
                    self.l,
                    self.d,
                )

            # Ensure block metadata exists and covers current token index for selection (expand if needed)
            t_token = kv.K_sel.shape[2] - 1
            if not hasattr(kv, "meta") or kv.meta.sel_starts.numel() == 0:
                kv.meta = build_block_meta(seq_len=max(t_token + 1, self.l_sel), l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w)
            else:
                # If current t exceeds covered selection range, rebuild meta with expanded seq_len
                sel_max_end = int(kv.meta.sel_starts[-1].item()) + kv.meta.l_sel if kv.meta.sel_starts.numel() > 0 else 0
                if (t_token + 1) > sel_max_end:
                    kv.meta = build_block_meta(seq_len=t_token + 1, l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w)
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

            scale = 1.0 / (self.d_k ** 0.5)
            # Compute p_cmp only for this step (S is 1 in decode)
            K_cmp_full = kv.K_cmp
            p_cmp_all = compute_pcmp_all(Q, K_cmp_full, scale)
            # Per-token outputs (S should be 1 in decode)
            outs = []
            for t in range(S):
                p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all[:, t : t + 1], kv.meta)
                p_grp = p_slc_all.sum(dim=3).squeeze(1)  # [B,G,S_sel]
                sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, kv.K_sel.shape[2] - 1, True, 2)
                # Observability: selection distance summary per step
                try:
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
                except Exception:
                    pass
                Q_t = Q[:, t]
                K_sel_t = kv.K_sel
                V_sel_t = kv.V_sel
                # Selection attention: prefer Triton if enabled; else packed; fallback to gather
                force_parity = os.getenv("NSA_FORCE_PARITY", "0").lower() in ("1", "true", "yes")
                use_sel_pack = os.getenv("NSA_USE_SEL_PACK", "1").lower() in ("1", "true", "yes") and not force_parity
                use_triton_sel = (
                    os.getenv("NSA_USE_TRITON_SEL", "0").lower() in ("1", "true", "yes") or self.use_triton_sel
                ) and not force_parity
                use_cuda_sel = os.getenv("NSA_SEL_CUDA", "0").lower() in ("1", "true", "yes") and not force_parity
                if use_triton_sel:
                    from nsa.kernels.triton_sel_kernel import selection_attention_triton
                    O_sel_bt = selection_attention_triton(Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1))
                    O_sel = O_sel_bt[:, 0]
                elif use_cuda_sel:
                    from nsa.kernels.cuda_sel_kernel import selection_attention_cuda
                    O_sel_bt = selection_attention_cuda(Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1))
                    O_sel = O_sel_bt[:, 0]
                elif use_sel_pack:
                    O_sel_bt = grouped_selection_attention_packed(Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1))
                    O_sel = O_sel_bt[:, 0]
                elif os.getenv("NSA_USE_SEL_MASK", "0").lower() in ("1", "true", "yes") and not force_parity:
                    O_sel_bt = grouped_selection_attention_masked(Q_t.unsqueeze(1), K_sel_t, V_sel_t, sel_ranges.unsqueeze(1))
                    O_sel = O_sel_bt[:, 0]
                else:
                    O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
                win_len = min(self.w, kv.K_win.shape[2])
                K_w = kv.K_win[:, :, kv.K_win.shape[2] - win_len : kv.K_win.shape[2], :]
                V_w = kv.V_win[:, :, kv.V_win.shape[2] - win_len : kv.V_win.shape[2], :]
                fa2_all = os.getenv("NSA_USE_FA2", "0").lower() in ("1", "true", "yes")
                fa2_win = os.getenv("NSA_USE_FA2_WIN", "0").lower() in ("1", "true", "yes")
                fa2_cmp = os.getenv("NSA_USE_FA2_CMP", "0").lower() in ("1", "true", "yes")
                use_flash = (fa2_all or fa2_win or fa2_cmp) and not force_parity
                if use_flash and (fa2_all or fa2_win):
                    O_win = sliding_window_attention_fa2_decode(Q_t, kv.K_win, kv.V_win, self.w)
                else:
                    O_win = attention_bgh(Q_t, K_w, V_w, causal=True)
                S_cmp_t = kv.K_cmp.shape[2]
                if use_flash and (fa2_all or fa2_cmp):
                    O_cmp = compressed_attention_fa2_decode(Q_t, kv.K_cmp, kv.V_cmp, S_cmp_t)
                else:
                    O_cmp = attention_bgh(Q_t, kv.K_cmp[:, :, :S_cmp_t, :], kv.V_cmp[:, :, :S_cmp_t, :], causal=True)
                q_gp = Q_t.mean(dim=2)
                gates = self.gate(q_gp, tau=self.gate_temp)
                # Observability: gate stats
                try:
                    log(
                        "decode.gates",
                        mean=gates.mean(dim=(-1, -2)).tolist() if gates.dim() >= 2 else gates.mean().item(),
                        std=gates.std(dim=(-1, -2)).tolist() if gates.dim() >= 2 else gates.std().item(),
                    )
                except Exception:
                    pass
                O = gates[..., 0:1] * O_cmp + gates[..., 1:2] * O_sel + gates[..., 2:3] * O_win
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
        Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
        assert Q_lin.shape[:2] == (B, S)
        # Apply RoPE to Q
        pos = torch.arange(S, device=x.device)
        Q = apply_rope(Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k), pos)
        Q = Q.view(B, S, self.n_heads, self.d_k).view(B, S, self.n_kv_groups, self.h_per_group, self.d_k)
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

        # Align RoPE application across branches for batched path (Q already RoPE'd)
        pos_k = torch.arange(S, device=x.device)
        K_sel = apply_rope(K_sel, pos_k)
        K_win = apply_rope(K_win, pos_k)

        # Update caches (prefill uses full sequence projections)
        kv.update_selection_raw(K_sel, V_sel)
        # Build/refresh meta for selection and compressed mapping
        kv.meta = build_block_meta(seq_len=S, l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w)
        kv.update_window(K_win, V_win, self.w)
        if self.phi_type == "mlp":
            K_cmp, V_cmp = self._phi_apply_seq(K_cmp_raw, V_cmp_raw, pos=torch.arange(S, device=x.device))
        else:
            K_cmp, V_cmp = avg_pool_phi_rope_kv(K_cmp_raw, V_cmp_raw, self.l, self.d, pos=torch.arange(S, device=x.device))
        kv.update_compressed(K_cmp, V_cmp, self.l, self.d)

        # Selection scores (batched)
        scale = 1.0 / (self.d_k ** 0.5)
        p_cmp_all = compute_pcmp_all(Q, kv.K_cmp, scale)  # [B,S,G,h,S_cmp]
        p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, kv.meta)  # [B,S,G,h,S_sel]
        p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]
        log(
            "prefill.scores",
            B=B,
            S=S,
            S_cmp=int(kv.K_cmp.shape[2]),
            S_sel=int(kv.meta.sel_starts.numel()),
        )

        # Batched top‑n → ranges for all positions
        sel_ranges_all = select_topn_ranges_batched(p_grp_all, kv.meta, self.n_sel, S, True, 2)  # [B,S,G,n,2]
        log("prefill.select", n_sel=self.n_sel, l_sel=self.l_sel, ranges=sel_ranges_all)

        # Branch attentions in parallel (parity-first for cmp/win, with optional masked SDPA gates)
        force_parity = os.getenv("NSA_FORCE_PARITY", "0").lower() in ("1", "true", "yes")
        env_all = os.environ.get("NSA_USE_FA2")
        fa2_all = (
            (env_all.lower() in ("1", "true", "yes")) if env_all is not None else self.use_flash_default
        )
        fa2_win = os.getenv("NSA_USE_FA2_WIN", "0").lower() in ("1", "true", "yes")
        fa2_cmp = os.getenv("NSA_USE_FA2_CMP", "0").lower() in ("1", "true", "yes")
        use_cmp_mask = os.getenv("NSA_USE_CMP_MASK", "1").lower() in ("1", "true", "yes") and not force_parity
        if (fa2_all or fa2_cmp) and not force_parity:
            O_cmp = compressed_attention_fa2(Q, kv.K_cmp, kv.V_cmp, self.l, self.d)
        elif use_cmp_mask:
            from nsa.core.attention_kernels import batched_causal_attention_compressed_masked
            O_cmp = batched_causal_attention_compressed_masked(Q, kv.K_cmp, kv.V_cmp, self.l, self.d)
        else:
            # Compressed per-t using the same kernel as sequential
            O_cmp = torch.zeros((B, S, self.n_kv_groups, self.h_per_group, self.d_v), device=x.device, dtype=V_cmp.dtype)
            S_cmp_full = kv.K_cmp.shape[2]
            for t in range(S):
                L = 0 if (t + 1) < self.l else min(((t + 1 - self.l) // self.d) + 1, S_cmp_full)
                if L > 0:
                    q_t = Q[:, t]
                    k_t = kv.K_cmp[:, :, :L, :]
                    v_t = kv.V_cmp[:, :, :L, :]
                    O_cmp[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        log("prefill.cmp", O_cmp=O_cmp)

        # Selected ranges attention (prefer Triton if enabled; else packed/gather)
        use_sel_pack = os.getenv("NSA_USE_SEL_PACK", "1").lower() in ("1", "true", "yes") and not force_parity
        use_triton_sel = (
            os.getenv("NSA_USE_TRITON_SEL", "0").lower() in ("1", "true", "yes") or self.use_triton_sel
        ) and not force_parity
        if use_triton_sel:
            from nsa.kernels.triton_sel_kernel import selection_attention_triton
            O_sel = selection_attention_triton(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        elif use_sel_pack:
            O_sel = grouped_selection_attention_packed(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        elif os.getenv("NSA_USE_SEL_MASK", "0").lower() in ("1", "true", "yes"):
            O_sel = grouped_selection_attention_masked(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        else:
            O_sel = grouped_selection_attention(Q, kv.K_sel, kv.V_sel, sel_ranges_all)
        log("prefill.sel", O_sel=O_sel)

        use_win_mask = os.getenv("NSA_USE_WIN_MASK", "1").lower() in ("1", "true", "yes") and not force_parity
        if (fa2_all or fa2_win) and not force_parity:
            O_win = sliding_window_attention_fa2(Q, K_win, V_win, self.w)
        elif use_win_mask:
            from nsa.core.attention_kernels import sliding_window_attention_masked
            O_win = sliding_window_attention_masked(Q, K_win, V_win, self.w)
        else:
            # Sliding per-t using the same kernel as sequential
            O_win = torch.zeros((B, S, self.n_kv_groups, self.h_per_group, self.d_v), device=x.device, dtype=V_win.dtype)
            for t in range(S):
                end = t + 1
                start = max(0, end - self.w)
                q_t = Q[:, t]
                k_t = K_win[:, :, start:end, :]
                v_t = V_win[:, :, start:end, :]
                O_win[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        log("prefill.win", O_win=O_win)

        # Gates and combine
        q_gp = Q.mean(dim=3)  # [B,S,G,Dk]
        gates = self.gate(q_gp.reshape(B * S * self.n_kv_groups, self.d_k), tau=self.gate_temp)
        gates = gates.view(B, S, self.n_kv_groups, 3).unsqueeze(3)  # [B,S,G,1,3]
        O = (
            gates[..., 0:1] * O_cmp +
            gates[..., 1:2] * O_sel +
            gates[..., 2:3] * O_win
        )  # [B,S,G,h,Dv]

        # Output projection
        O_heads = O.reshape(B, S, self.n_kv_groups * self.h_per_group, self.d_v)
        out = self.out(O_heads.reshape(B, S, -1))
        log("prefill.out", out=out)

        # Optional debug compare: sequential-style per-token recompute to measure MAE
        if os.getenv("NSA_DEBUG_COMPARE", "0").lower() in ("1", "true", "yes"):
            with torch.no_grad():
                # Compressed per-token recompute
                O_cmp_seq = torch.zeros_like(O_cmp)
                S_cmp = kv.K_cmp.shape[2]
                for t in range(S):
                    L = 0 if (t + 1) < self.l else min(((t + 1 - self.l) // self.d) + 1, S_cmp)
                    if L > 0:
                        q_t = Q[:, t]
                        k_t = kv.K_cmp[:, :, :L, :]
                        v_t = kv.V_cmp[:, :, :L, :]
                        O_cmp_seq[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
                cmp_mae = (O_cmp - O_cmp_seq).abs().mean().item()
                print(f"NSA-DBG cmp_mae={cmp_mae:.6e}")

                # Sliding per-token recompute
                O_win_seq = torch.zeros_like(O_win)
                for t in range(S):
                    end = t + 1
                    start = max(0, end - self.w)
                    q_t = Q[:, t]
                    k_t = K_win[:, :, start:end, :]
                    v_t = V_win[:, :, start:end, :]
                    O_win_seq[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
                win_mae = (O_win - O_win_seq).abs().mean().item()
                print(f"NSA-DBG win_mae={win_mae:.6e}")

                # Final output recompute using seq per-branch
                O_seq = gates[..., 0:1] * O_cmp_seq + gates[..., 1:2] * O_sel + gates[..., 2:3] * O_win_seq
                O_heads_seq = O_seq.reshape(B, S, self.n_kv_groups * self.h_per_group, self.d_v)
                out_seq = self.out(O_heads_seq.reshape(B, S, -1))
                out_mae = (out - out_seq).abs().mean().item()
                print(f"NSA-DBG out_mae={out_mae:.6e}")
        return out, kv

    def _forward_prefill_sequential(self, x: torch.Tensor, kv: NSA_KV) -> tuple[torch.Tensor, NSA_KV]:
        """
        Reference prefill path (sequential per‑token), used for parity checks.
        """
        B, S, _ = x.shape
        # Projections
        Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
        pos = torch.arange(S, device=x.device)
        Q = apply_rope(
            Q_lin.view(B, S, self.n_heads, self.d_k).reshape(B, S, self.n_heads * self.d_k), pos
        )
        Q = Q.view(B, S, self.n_heads, self.d_k).view(B, S, self.n_kv_groups, self.h_per_group, self.d_k)
        K_sel = self._shape_kv(self.W_K_sel(x), B, S)
        V_sel = self._shape_kv(self.W_V_sel(x), B, S)
        K_win = self._shape_kv(self.W_K_win(x), B, S)
        V_win = self._shape_kv(self.W_V_win(x), B, S)
        K_cmp_raw = self._shape_kv(self.W_K_cmp(x), B, S)
        V_cmp_raw = self._shape_kv(self.W_V_cmp(x), B, S)

        kv.update_selection_raw(K_sel, V_sel)
        kv.meta = build_block_meta(seq_len=S, l=self.l, d=self.d, l_sel=self.l_sel, n_sel=self.n_sel, w=self.w)
        kv.update_window(K_win, V_win, self.w)
        if self.phi_type == "mlp":
            K_cmp, V_cmp = self._phi_apply_seq(K_cmp_raw, V_cmp_raw, pos=torch.arange(S, device=x.device))
        else:
            K_cmp, V_cmp = avg_pool_phi_rope_kv(K_cmp_raw, V_cmp_raw, self.l, self.d, pos=torch.arange(S, device=x.device))
        kv.update_compressed(K_cmp, V_cmp, self.l, self.d)

        # Precompute p_grp_all batched for reuse per t
        scale = 1.0 / (self.d_k ** 0.5)
        p_cmp_all = compute_pcmp_all(Q, kv.K_cmp, scale)  # [B,S,G,h,S_cmp]
        p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, kv.meta)  # [B,S,G,h,S_sel]
        p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]

        outs = []
        for t in range(S):
            p_grp = p_grp_all[:, t]  # [B,G,S_sel]
            sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, t, True, 2)
            Q_t = Q[:, t]
            K_sel_t = kv.K_sel[:, :, : t + 1, :]
            V_sel_t = kv.V_sel[:, :, : t + 1, :]
            O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)
            win_len = min(self.w, t + 1)
            K_w = kv.K_win[:, :, t + 1 - win_len : t + 1, :]
            V_w = kv.V_win[:, :, t + 1 - win_len : t + 1, :]
            O_win = attention_bgh(Q_t, K_w, V_w, causal=True)
            S_cmp_t = 0 if (t + 1) < self.l else (t + 1 - self.l) // self.d + 1
            O_cmp = attention_bgh(Q_t, kv.K_cmp[:, :, :S_cmp_t, :], kv.V_cmp[:, :, :S_cmp_t, :], causal=True)
            q_gp = Q_t.mean(dim=2)
            gates = self.gate(q_gp, tau=self.gate_temp)
            O = gates[..., 0:1] * O_cmp + gates[..., 1:2] * O_sel + gates[..., 2:3] * O_win
            O_heads = O.reshape(B, self.n_heads, self.d_v)
            out_t = self.out(O_heads.reshape(B, 1, -1))
            outs.append(out_t)
        out = torch.cat(outs, dim=1)
        return out, kv

    def _sdpa_full(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Q: [B,G,h,Dk]; K/V: [B,G,S,D*] -> out [B,G,h,Dv]
        B, G, h, Dk = Q.shape
        S = K.shape[2]
        q = Q.reshape(B * G * h, 1, Dk)
        k = K.repeat_interleave(h, dim=1).reshape(B * G * h, S, Dk)
        v = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = attn.squeeze(1).reshape(B, G, h, -1)
        return o

    def _phi_apply_seq(self, K_raw: torch.Tensor, V_raw: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply learnable ϕ over the full sequence using depthwise Conv1d initialized to avg.
        Expects K_raw,V_raw: [B,G,S,D*]; returns [B,G,S_cmp,D*].
        """
        assert self.phi_k_conv is not None and self.phi_v_conv is not None
        B, G, S, Dk = K_raw.shape
        Dv = V_raw.shape[-1]
        K_rope = apply_rope(K_raw, pos)
        Kx = K_rope.permute(0, 1, 3, 2).reshape(B * G, Dk, S)
        Vx = V_raw.permute(0, 1, 3, 2).reshape(B * G, Dv, S)
        Kc = self.phi_k_conv(Kx)
        Vc = self.phi_v_conv(Vx)
        S_cmp = Kc.shape[-1]
        K_cmp = Kc.reshape(B, G, Dk, S_cmp).permute(0, 1, 3, 2).contiguous()
        V_cmp = Vc.reshape(B, G, Dv, S_cmp).permute(0, 1, 3, 2).contiguous()
        return K_cmp, V_cmp

    def _phi_apply_last(self, K_last: torch.Tensor, V_last: torch.Tensor, pos_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Emit a single compressed token from the last l raw tokens using Conv1d with kernel=l,stride=d.
        Inputs: [B,G,l,D*] -> Outputs: [B,G,1,D*].
        """
        assert self.phi_k_conv is not None and self.phi_v_conv is not None
        B, G, lwin, Dk = K_last.shape
        Dv = V_last.shape[-1]
        assert lwin == self.l, "decode emission expects exactly l tokens"
        K_rope = apply_rope(K_last, pos_last)
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
        for b in range(B):
            row = []
            for g in range(G):
                r = ranges[b, g]  # [n,2]
                idxs = []
                for s, e in r.tolist():
                    if e > s:
                        idxs.append(torch.arange(s, e, device=K.device))
                if idxs:
                    idx = torch.cat(idxs, dim=0)
                else:
                    idx = torch.empty((0,), dtype=torch.int64, device=K.device)
                k = K[b, g, idx] if idx.numel() > 0 else torch.zeros((1, Dk), device=K.device)
                v = V[b, g, idx] if idx.numel() > 0 else torch.zeros((1, Dv), device=K.device)
                q = Q[b, g]  # [h,Dk]
                attn = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True)
                row.append(attn.squeeze(0))  # [h,Dv]
            outs.append(torch.stack(row, dim=0))  # [G,h,Dv]
        return torch.stack(outs, dim=0)  # [B,G,h,Dv]
