
import torch
import torch.nn.functional as F

import os
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
    # Out-of-place accumulation to avoid in-place versioning issues under GC/DDP
    p_slc = torch.zeros((B, G, h, S_sel), device=device, dtype=p_cmp.dtype)
    acc = torch.zeros_like(p_slc)
    # CSR row-wise multiply-add
    for r in range(S_cmp):
        start, end = int(indptr[r].item()), int(indptr[r + 1].item())
        if start == end:
            continue
        cols = indices[start:end].to(device)
        w = values[start:end].to(device=device, dtype=p_cmp.dtype)  # [nnz_r]
        contrib = p_cmp[..., r].unsqueeze(-1) * w  # [B,G,h,nnz_r]
        # Ensure Long dtype for scatter_add indices
        idx = cols.view(1, 1, 1, -1).expand(B, G, h, -1).long()
        acc = acc.scatter_add(-1, idx, contrib)
    return acc


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
    # Ensure Long dtype for scatter_add indices
    idx = cols.view(1, 1, 1, 1, -1).expand(B, S, G, h, -1).long()
    p_slc = p_slc.scatter_add(-1, idx, p_src)
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
    _skip_validation: bool = False,
) -> torch.Tensor:
    """Select top-n block ranges with deterministic tie-breaking.
    
    M8: Enhanced with robust deterministic tie-breaking for training reproducibility.
    Uses scaled epsilon bias to prefer lower indices on score ties, ensuring
    identical selection across runs with the same inputs.
    
    Args:
        p_grp: Group probabilities [B,G,S_sel] 
        meta: Block metadata with selection ranges
        n_top: Number of top blocks to select
        t_token: Current token position (0-indexed)
        force_init: Whether to force include block 0
        force_local: Number of local blocks to force include
        
    Returns:
        Selected ranges [B,G,n_top,2] as [start,end) pairs
    """
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
        # M8: Deterministic tie-breaker - prefer lower indices for reproducible selection
        # Use a tiny, fixed bias in float32 space to avoid overwhelming scores in low-precision
        # dtypes (e.g., bf16/FP16). We perform ranking in float32 regardless of input dtype.
        tie_break_scale = torch.tensor(1e-8, device=device, dtype=torch.float32)
        base_idx = torch.arange(S_sel, device=device, dtype=torch.float32).view(1, 1, S_sel)
        composite = masked.to(torch.float32) - (base_idx * tie_break_scale)
        # Ensure deterministic topk with sorted=True for consistent ordering
        k_actual = min(k_rest, S_sel)
        _, top_idx = torch.topk(composite, k=k_actual, dim=-1, largest=True, sorted=True)
        
        # M8: Assert tie-breaking worked - check for potential numerical issues
        if torch.is_grad_enabled():
            # Only check during training when gradients are enabled
            with torch.no_grad():
                orig_scores = torch.gather(masked, -1, top_idx).to(torch.float32)
                if orig_scores.numel() > 1:
                    # Check if adjacent scores are suspiciously close (potential tie-break failure)
                    score_diffs = torch.diff(orig_scores, dim=-1)
                    very_close = torch.abs(score_diffs) < (float(tie_break_scale.item()) * 0.1)
                    if very_close.any():
                        from nsa.core.debug import log
                        log("warn.selection_tiebreak", 
                            msg="Close scores detected in selection - potential tie-break instability",
                            min_diff=float(torch.abs(score_diffs).min().item()),
                            tie_break_scale=float(tie_break_scale))
        sel_idx = torch.cat([forced_idx, top_idx], dim=-1)
    else:
        sel_idx = forced_idx
    # sort selected indices ascending for consistent range merging
    sel_idx = torch.sort(sel_idx, dim=-1).values
    
    # M8: Optional determinism validation (skip if called from validation itself)
    if not _skip_validation and os.getenv("NSA_VALIDATE_SELECTION_DETERMINISM", "0").lower() in ("1", "true", "yes"):
        validate_selection_determinism(p_grp, meta, n_top, t_token)
    # merge adjacent into contiguous ranges
    ranges = []
    for b in range(B):
        bg = []
        for g in range(G):
            blocks = sel_starts[sel_idx[b, g]]  # [k], sorted non-decreasing
            # Deduplicate without extra sort (faster on GPU for small k)
            blocks = torch.unique_consecutive(blocks)
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
    M8: Deterministic batched selection with enhanced tie-breaking:
    - Mask future blocks per position t via block end ≤ t+1
    - Force include block 0 and last k local blocks (dedup)
    - Exclude forced from scored top‑k
    - Robust deterministic tie‑break to lower index on equal scores
    - Convert to merged contiguous [start,end) ranges clamped to ≤ t+1
    - Validation hooks for training reproducibility
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
        torch.cat(forced_list, dim=-1)
        if forced_list
        else torch.empty((B, S, G, 0), dtype=torch.long, device=device)
    )
    if forced.numel() > 0:
        # Ensure ascending per trailing dim then drop duplicates consecutively
        forced = torch.sort(forced, dim=-1).values
        forced = torch.unique_consecutive(forced, dim=-1)

    if forced.numel() > 0:
        forced_mask = torch.zeros_like(masked, dtype=torch.bool)
        forced_mask.scatter_(-1, forced, True)
        masked = masked.masked_fill(forced_mask, float("-inf"))

    # Deterministic top‑k using composite key with tiny index bias
    k_rest = max(0, n_top - forced.shape[-1])
    if k_rest > 0:
        # M8: Deterministic tie-breaker - prefer lower indices; rank in float32 to avoid
        # overwhelming biases under low-precision dtypes.
        tie_break_scale = torch.tensor(1e-8, device=device, dtype=torch.float32)
        base_idx = torch.arange(S_sel, device=device, dtype=torch.float32).view(1, 1, 1, S_sel).expand(B, S, G, S_sel)
        composite = masked.to(torch.float32) - (base_idx * tie_break_scale)
        # Ensure deterministic topk with explicit sorted=True for batched path
        k_actual = min(k_rest, S_sel) 
        _, top_idx = torch.topk(composite, k=k_actual, dim=-1, largest=True, sorted=True)
        
        # M8: Optional validation for tie-breaking effectiveness in training
        if torch.is_grad_enabled() and k_actual > 1:
            with torch.no_grad():
                orig_scores = torch.gather(masked, -1, top_idx).to(torch.float32)
                # Check last dimension for potential tie-break issues
                score_diffs = torch.diff(orig_scores, dim=-1) 
                very_close = torch.abs(score_diffs) < (float(tie_break_scale.item()) * 0.1)
                if very_close.any():
                    from nsa.core.debug import log
                    log("warn.batched_selection_tiebreak",
                        msg="Close scores in batched selection - potential instability",
                        batch_close_count=int(very_close.sum().item()),
                        tie_break_scale=float(tie_break_scale))
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


def map_pcmp_to_pslc_slow_path(p_cmp_all: torch.Tensor, meta: BlockMeta) -> torch.Tensor:
    """
    M8: Eq.9 slow path verifier - explicit mathematical computation.
    
    This function implements the exact mathematical definition by using the
    CSR mapping directly instead of recomputing overlaps. This ensures it
    matches the fast path exactly.
    
    Args:
        p_cmp_all: [B,S,G,h,S_cmp] compressed probabilities
        meta: Block metadata with overlap mapping
        
    Returns:
        p_slc_all: [B,S,G,h,S_sel] selection probabilities
    """
    B, S, G, h, S_cmp = p_cmp_all.shape
    device = p_cmp_all.device
    S_sel = meta.sel_starts.numel()
    
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, S_sel), device=device, dtype=p_cmp_all.dtype)
    
    # Use CSR mapping directly (same as fast path but with explicit loops)
    p_slc_all = torch.zeros((B, S, G, h, S_sel), device=device, dtype=p_cmp_all.dtype)
    
    indptr = meta.M_csl_indptr.to(device)
    indices = meta.M_csl_indices.to(device) 
    values = meta.M_csl_values.to(device, dtype=p_cmp_all.dtype)
    
    # For each compressed block (CSR row)
    for cmp_i in range(min(S_cmp, len(indptr) - 1)):
        start = int(indptr[cmp_i].item())
        end = int(indptr[cmp_i + 1].item())
        
        if start == end:
            continue
            
        # Get the selection blocks this compressed block contributes to
        sel_cols = indices[start:end]
        weights = values[start:end]
        
        # Add weighted contribution to each selection block
        for j, (sel_idx, weight) in enumerate(zip(sel_cols, weights)):
            sel_idx = int(sel_idx.item())
            if sel_idx < S_sel:
                p_slc_all[..., sel_idx] += p_cmp_all[..., cmp_i] * float(weight.item())
    
    return p_slc_all


def verify_mapping_equivalence(
    p_cmp_all: torch.Tensor, 
    meta: BlockMeta, 
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> tuple[bool, dict]:
    """
    M8: Verify fast COO path matches slow mathematical path (Eq.9 verification).
    
    Args:
        p_cmp_all: Compressed probabilities to test
        meta: Block metadata
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        (is_equivalent, details): True if paths match, plus diagnostic info
    """
    # Only run verification if explicitly requested via env flag
    if not os.getenv("NSA_VERIFY_EQ9_MAPPING", "0").lower() in ("1", "true", "yes"):
        return True, {"status": "skipped", "reason": "NSA_VERIFY_EQ9_MAPPING not set"}
    
    with torch.no_grad():
        # Compute both paths
        fast_result = map_pcmp_to_pslc_batched(p_cmp_all, meta)
        slow_result = map_pcmp_to_pslc_slow_path(p_cmp_all, meta)
        
        # Compare results
        is_close = torch.allclose(fast_result, slow_result, rtol=rtol, atol=atol)
        
        # Compute diagnostic metrics
        abs_diff = (fast_result - slow_result).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        rel_diff = abs_diff / (slow_result.abs() + atol)
        max_rel_diff = rel_diff.max().item()
        
        details = {
            "status": "verified" if is_close else "mismatch",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "shape": list(p_cmp_all.shape),
            "rtol": rtol,
            "atol": atol,
        }
        
        if not is_close:
            from nsa.core.debug import log
            log("error.eq9_mapping_mismatch",
                msg="Fast COO path does not match slow mathematical path",
                **details)
        
        return is_close, details


def validate_selection_determinism(
    p_grp: torch.Tensor,
    meta: BlockMeta, 
    n_top: int,
    t_token: int,
    num_trials: int = 5
) -> bool:
    """Validate that selection is deterministic by running multiple times.
    
    Args:
        p_grp: Group probabilities [B,G,S_sel]
        meta: Block metadata
        n_top: Number of top blocks to select
        t_token: Current token position
        num_trials: Number of trials to test determinism
        
    Returns:
        True if all trials produce identical results
    """
    # Only run validation if explicitly requested via env flag
    if not os.getenv("NSA_VALIDATE_SELECTION_DETERMINISM", "0").lower() in ("1", "true", "yes"):
        return True
        
    if p_grp.requires_grad:
        # Don't validate during training to avoid affecting gradients
        return True
        
    with torch.no_grad():
        results = []
        for trial in range(num_trials):
            ranges = select_topn_ranges(p_grp.clone(), meta, n_top, t_token, True, 2, _skip_validation=True)
            results.append(ranges.clone())
        
        # Check if all results are identical
        for i in range(1, num_trials):
            if not torch.equal(results[0], results[i]):
                from nsa.core.debug import log
                log("error.selection_nondeterministic",
                    msg=f"Selection non-deterministic: trial 0 != trial {i}",
                    trial_0_shape=list(results[0].shape),
                    trial_i_shape=list(results[i].shape))
                return False
                
    return True
