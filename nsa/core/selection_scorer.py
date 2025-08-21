
import torch
import torch.nn.functional as F

from .block_index import BlockMeta


def compute_pcmp(Q: torch.Tensor, K_cmp: torch.Tensor, scale: float) -> torch.Tensor:
    # Q: [G,h,Dk]; K_cmp: [B,G,S_cmp,Dk] with implicit B=1 for this path
    if Q.dim() == 3:
        G, h, Dk = Q.shape
        S_cmp = K_cmp.shape[2]
        q = Q.reshape(G * h, 1, Dk)
        k = K_cmp.reshape(1 * G, S_cmp, Dk).repeat_interleave(h, dim=0)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * scale
        return F.softmax(logits, dim=-1).reshape(1, G, h, S_cmp)
    else:
        B, G, h, Dk = Q.shape
        S_cmp = K_cmp.shape[2]
        q = Q.reshape(B * G * h, 1, Dk)
        k = K_cmp.reshape(B * G, S_cmp, Dk).repeat_interleave(h, dim=0)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * scale
        p = F.softmax(logits, dim=-1)
        return p.reshape(B, G, h, S_cmp)


def compute_pcmp_all(Q_all: torch.Tensor, K_cmp: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Q_all: [B,S,G,h,Dk], K_cmp: [B,G,S_cmp,Dk] -> p_cmp_all: [B,S,G,h,S_cmp]
    """
    # Transpose K to align for einsum
    Kt = K_cmp.permute(0, 1, 3, 2)  # [B,G,Dk,S_cmp]
    # Use distinct subscript for compressed axis to avoid collision with token axis
    logits = torch.einsum("bsghd,bgdc->bsghc", Q_all, Kt)  # [B,S,G,h,S_cmp]
    logits = logits * scale
    return F.softmax(logits, dim=-1)


def map_pcmp_to_pslc(p_cmp: torch.Tensor, meta: BlockMeta) -> torch.Tensor:
    # p_cmp: [B,G,h,S_cmp]
    B, G, h, S_cmp = p_cmp.shape
    indptr = meta.M_csl_indptr
    indices = meta.M_csl_indices
    values = meta.M_csl_values
    S_sel = meta.sel_starts.numel()
    device = p_cmp.device
    p_slc = torch.zeros((B, G, h, S_sel), device=device, dtype=p_cmp.dtype)
    # CSR row-wise multiply-add
    for r in range(S_cmp):
        start, end = int(indptr[r].item()), int(indptr[r + 1].item())
        if start == end:
            continue
        cols = indices[start:end]
        w = values[start:end].to(p_cmp.dtype)  # [nnz_r]
        contrib = p_cmp[..., r].unsqueeze(-1) * w  # [B,G,h,nnz_r]
        p_slc.index_add_(dim=-1, index=cols.to(device), source=contrib)
    return p_slc


def map_pcmp_to_pslc_batched(p_cmp_all: torch.Tensor, meta: BlockMeta) -> torch.Tensor:
    """
    p_cmp_all: [B,S,G,h,S_cmp] -> p_slc_all: [B,S,G,h,S_sel]
    Vectorized over B,S,G,h while looping CSR rows over S_cmp.
    """
    B, S, G, h, S_cmp = p_cmp_all.shape
    device = p_cmp_all.device
    S_sel = meta.sel_starts.numel()
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, S_sel), device=device, dtype=p_cmp_all.dtype)
    # COO sparse matmul: for each nnz (r,c,w), add p_cmp[..., r]*w to p_slc[..., c]
    rows, cols = meta.M_csl_coo_indices.to(device)
    w = meta.M_csl_coo_values.to(device=device, dtype=p_cmp_all.dtype)
    # Filter mapping rows to those < current S_cmp to avoid out-of-bounds in early decode
    valid_mask = rows < S_cmp
    if valid_mask.dim() == 0:
        valid_mask = valid_mask.unsqueeze(0)
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    w = w[valid_mask]
    if rows.numel() == 0:
        return torch.zeros((B, S, G, h, S_sel), device=device, dtype=p_cmp_all.dtype)
    p_src = p_cmp_all[..., rows] * w  # [B,S,G,h,nnz]
    p_slc = torch.zeros((B, S, G, h, S_sel), device=device, dtype=p_cmp_all.dtype)
    p_slc.index_add_(-1, cols, p_src)
    return p_slc


def group_reduce_pslc(p_slc: torch.Tensor) -> torch.Tensor:
    # Sum across heads in group (Eq. 10)
    return p_slc.sum(dim=2)


def select_topn_ranges(
    p_grp: torch.Tensor,
    meta: BlockMeta,
    n_top: int,
    t_token: int,
    force_init: bool = True,
    force_local: int = 2,
) -> torch.Tensor:
    # p_grp: [B,G,S_sel]
    B, G, S_sel = p_grp.shape
    device = p_grp.device
    # Determine candidate blocks ≤ t
    sel_starts = meta.sel_starts.to(device)
    # mask future blocks
    valid = sel_starts + meta.l_sel - 1 <= t_token
    masked = p_grp.masked_fill(~valid.view(1, 1, -1), float("-inf"))
    # force-includes set
    forced_list = []
    if force_init:
        forced_list.append(torch.zeros((B, G), dtype=torch.int64, device=device))
    if force_local > 0:
        last_block = torch.clamp((torch.tensor(t_token, device=device) // meta.l_sel), min=0)
        for i in range(force_local):
            forced_list.append(torch.clamp(last_block - i, min=0).expand(B, G))
    forced_idx = (
        torch.stack(forced_list, dim=-1)
        if forced_list
        else torch.empty((B, G, 0), device=device, dtype=torch.int64)
    )
    # Exclude forced from top-k candidates by setting their scores to -inf
    if forced_idx.numel() > 0:
        forced_mask = torch.zeros_like(masked, dtype=torch.bool)
        forced_mask.scatter_(-1, forced_idx, True)
        masked = masked.masked_fill(forced_mask, float("-inf"))
    # pick remaining to fill up to n_top
    k_rest = torch.clamp(torch.tensor(n_top - forced_idx.shape[-1], device=device), min=0).item()
    if k_rest > 0:
        # Deterministic tie-breaker to lower index on ties via tiny negative index bias
        eps = torch.finfo(masked.dtype).eps
        base_idx = torch.arange(S_sel, device=device).view(1, 1, S_sel).to(masked.dtype)
        composite = masked - (base_idx * eps)
        _, top_idx = torch.topk(composite, k=min(k_rest, S_sel), dim=-1, largest=True, sorted=True)
        sel_idx = torch.cat([forced_idx, top_idx], dim=-1)
    else:
        sel_idx = forced_idx
    # sort selected indices ascending
    sel_idx = torch.sort(sel_idx, dim=-1).values
    # merge adjacent into contiguous ranges
    ranges = []
    for b in range(B):
        bg = []
        for g in range(G):
            blocks = sel_starts[sel_idx[b, g]]  # [k]
            blocks = torch.unique(blocks, sorted=True)
            if blocks.numel() == 0:
                bg.append(torch.zeros((n_top, 2), dtype=torch.int32, device=device))
                continue
            cur_s = int(blocks[0].item())
            cur_e = cur_s + meta.l_sel
            merged: list[tuple[int, int]] = []
            for x in blocks[1:].tolist():
                if x == cur_e:  # adjacent
                    cur_e += meta.l_sel
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = x, x + meta.l_sel
            merged.append((cur_s, cur_e))
            # pad/truncate to n_top
            out = torch.zeros((n_top, 2), dtype=torch.int32, device=device)
            for i, (s, e) in enumerate(merged[:n_top]):
                e = min(e, t_token + 1)
                out[i, 0] = s
                out[i, 1] = e
            bg.append(out)
        ranges.append(torch.stack(bg, dim=0))
    return torch.stack(ranges, dim=0)  # [B,G,n_top,2]


# ===== Batched selection (prefill fast path) =====


def select_topn_ranges_batched(
    p_grp_all: torch.Tensor,  # [B,S,G,S_sel]
    meta: BlockMeta,
    n_top: int,
    S: int,
    force_init: bool = True,
    force_local: int = 2,
) -> torch.Tensor:  # [B,S,G,n_ranges,2]
    """
    Deterministic batched selection:
    - Mask future blocks per position t via block end ≤ t+1
    - Force include block 0 and last k local blocks (dedup)
    - Exclude forced from scored top‑k
    - Deterministic tie‑break to lower index on equal scores
    - Convert to merged contiguous [start,end) ranges clamped to ≤ t+1
    """
    B, S_q, G, S_sel = p_grp_all.shape
    device = p_grp_all.device

    sel_starts = meta.sel_starts.to(device)
    sel_ends = sel_starts + meta.l_sel
    tpos = torch.arange(S, device=device).view(S, 1)
    valid = sel_ends.view(1, -1) <= (tpos + 1)  # [S,S_sel]
    disallowed = ~valid
    masked = p_grp_all.masked_fill(disallowed.view(1, S, 1, S_sel), float("-inf"))

    # Forced blocks (dedup across 0 and locals)
    forced_list = []
    if force_init:
        forced_list.append(torch.zeros((B, S, G, 1), dtype=torch.long, device=device))
    if force_local > 0:
        tpos1 = torch.arange(S, device=device)
        last_block = (tpos1 // meta.l_sel).clamp_min(0)
        for k in range(force_local):
            idx = (last_block - k).clamp_min(0).view(1, S, 1, 1).expand(B, S, G, 1)
            forced_list.append(idx)
    forced = (
        torch.cat(forced_list, dim=-1).unique(sorted=True, dim=-1)
        if forced_list
        else torch.empty((B, S, G, 0), dtype=torch.long, device=device)
    )

    if forced.numel() > 0:
        forced_mask = torch.zeros_like(masked, dtype=torch.bool)
        forced_mask.scatter_(-1, forced, True)
        masked = masked.masked_fill(forced_mask, float("-inf"))

    # Deterministic top‑k using composite key with tiny index bias
    k_rest = max(0, n_top - forced.shape[-1])
    if k_rest > 0:
        # Deterministic tie-breaker to prefer lower indices on ties
        base_idx = torch.arange(S_sel, device=device).view(1, 1, 1, S_sel).expand(B, S, G, S_sel)
        eps = torch.finfo(masked.dtype).eps
        composite = masked - (base_idx.to(masked.dtype) * eps)
        _, top_idx = torch.topk(composite, k=min(k_rest, S_sel), dim=-1, largest=True)
        selected = torch.cat([forced, top_idx], dim=-1)
    else:
        selected = forced[..., :n_top]

    # Keep only valid (≤ t) indices; drop disallowed fill-ins
    valid_full = valid.view(1, S, 1, S_sel).expand(B, S, G, S_sel)
    is_valid_pick = torch.gather(valid_full, -1, selected)
    # Replace invalid with -1 sentinel
    selected = torch.where(is_valid_pick, selected, torch.full_like(selected, -1))
    # Special-case: if requested n_top ≥ number of valid blocks at t, select exactly all valid blocks [0..t]
    num_valid = valid.sum(dim=1)  # [S]
    # Build ascending [0..S_sel-1] to pick prefix per t
    all_idx = torch.arange(S_sel, device=device).view(1, 1, 1, S_sel).expand(B, S, G, S_sel)
    pick_mask = all_idx < num_valid.view(1, S, 1, 1)
    if n_top >= S_sel:
        selected = torch.where(pick_mask, all_idx, torch.full_like(all_idx, -1))
    selected = torch.sort(selected, dim=-1).values
    ranges = convert_indices_to_ranges_batched(selected, meta, S)
    return ranges


def convert_indices_to_ranges_batched(
    indices: torch.Tensor,  # [B,S,G,k]
    meta: BlockMeta,
    S: int,
) -> torch.Tensor:  # [B,S,G,n_max,2]
    B, S_q, G, k = indices.shape
    device = indices.device
    sel_starts = meta.sel_starts.to(device)

    all_ranges = []
    for b in range(B):
        for t in range(S_q):
            clamp_end = int(t) + 1
            for g in range(G):
                block_ids = [int(x) for x in indices[b, t, g].tolist() if int(x) >= 0]
                spans = []
                last_s, last_e = None, None
                prev = None
                for bid in block_ids:
                    if prev is not None and bid == prev:
                        continue
                    prev = bid
                    s0 = int(sel_starts[bid].item())
                    e0 = min(s0 + meta.l_sel, clamp_end)
                    if e0 <= s0:
                        continue
                    if last_s is None:
                        last_s, last_e = s0, e0
                    elif s0 == last_e:
                        last_e = e0
                    else:
                        spans.append((last_s, last_e))
                        last_s, last_e = s0, e0
                if last_s is not None:
                    spans.append((last_s, last_e))
                all_ranges.append(spans)

    max_ranges = max((len(r) for r in all_ranges), default=0)
    out = torch.zeros((B, S_q, G, max_ranges, 2), dtype=torch.int32, device=device)
    idx = 0
    for b in range(B):
        for t in range(S_q):
            for g in range(G):
                spans = all_ranges[idx]
                for i, (s0, e0) in enumerate(spans):
                    out[b, t, g, i, 0] = s0
                    out[b, t, g, i, 1] = e0
                idx += 1
    return out
