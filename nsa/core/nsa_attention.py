from dataclasses import dataclass
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
        gate_hidden: Optional[int] = None,
        gate_temp: float = 1.0,
        rope_impl: str = "llama",
        use_flash: bool = False,
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
        # x: [B,S,dim] (prefill) or [B,1,dim] (decode)
        B, S, _ = x.shape
        # Projections
        Q_lin = self._shape_q(self.W_Q(x), B, S)  # [B,S,G,h,Dk]
        # Apply RoPE to Q
        pos = torch.arange(S, device=x.device)
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

        # Prefill vs decode
        if prefill:
            kv.update_selection_raw(K_sel, V_sel)
            kv.update_window(K_win, V_win, self.w)
            # for compressed, use the entire sequence (prefill)
            K_cmp, V_cmp = avg_pool_phi_rope_kv(K_cmp_raw, V_cmp_raw, self.l, self.d)
            kv.update_compressed(K_cmp, V_cmp, self.l, self.d)
        else:
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
                K_cmp_new, V_cmp_new = avg_pool_phi_rope_kv(K_last, V_last, self.l, self.d)
                kv.update_compressed(
                    torch.cat([kv.K_cmp, K_cmp_new], dim=2) if kv.K_cmp.numel() else K_cmp_new,
                    torch.cat([kv.V_cmp, V_cmp_new], dim=2) if kv.V_cmp.numel() else V_cmp_new,
                    self.l,
                    self.d,
                )
            # Append predicted reads per formula for this step
            num_cmp = 0 if S_raw < self.l else (S_raw - self.l) // self.d + 1
            reads = num_cmp + self.n_sel * self.l_sel + min(self.w, S_raw)
            kv.append_reads_pred(reads)
            # Append actual reads equal to formula in M0
            kv.append_reads_actual(reads, self.n_sel * self.l_sel, num_cmp, min(self.w, S_raw))

        scale = 1.0 / (self.d_k ** 0.5)
        # Batched prefill: compute p_cmp for all tokens against full compressed keys
        K_cmp_full = kv.K_cmp  # [B,G,S_cmp,Dk]
        p_cmp_all = compute_pcmp_all(Q, K_cmp_full, scale)  # [B,S,G,h,S_cmp]
        # Map to selection scores in batch
        p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, kv.meta)  # [B,S,G,h,S_sel]
        p_grp_all = p_slc_all.sum(dim=3)  # [B,S,G,S_sel]
        # Per-token top‑n and attention; loop only over S to build ranges, but K/V SDPA will use causal slices
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
            # For prefill path we skip appending actual reads (decode-only)
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

    def _sdpa_over_ranges(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        ranges: torch.Tensor,
    ) -> torch.Tensor:
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


