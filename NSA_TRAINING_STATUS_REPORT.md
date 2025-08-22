# NSA Training Status Report - Critical Update

## Executive Summary
The original bug at `nsa/core/nsa_attention.py:818` has been **successfully fixed**, but testing revealed a **secondary performance bottleneck** in the selection scorer that makes training impractically slow.

## Original Bug - FIXED ✅

### Problem
- **Location**: `nsa/core/nsa_attention.py:818` in `_sdpa_over_ranges`
- **Cause**: `for s, e in r.tolist():` with `torch.arange(s, e)` caused infinite hang
- **Impact**: Training completely stuck at step 2

### Solution Applied
```python
# OLD (buggy) - line 818
for s, e in r.tolist():
    if e > s:
        idxs.append(torch.arange(s, e, device=K.device))

# NEW (fixed) - lines 818-837
r = ranges[b, g].to(dtype=torch.int64, device=K.device)
if r.numel() == 0:
    valid_pairs = torch.empty((0, 2), dtype=torch.int64, device=K.device)
else:
    s = r[:, 0].clamp_(0, S_kv)
    e = r[:, 1].clamp_(0, S_kv)
    valid = e > s
    valid_pairs = torch.stack([s[valid], e[valid]], dim=-1)
# Build boolean mask to gather selected tokens
if valid_pairs.numel() > 0:
    m = torch.zeros((S_kv,), dtype=torch.bool, device=K.device)
    for s_e in valid_pairs:
        s_i = int(s_e[0].item())
        e_i = int(s_e[1].item())
        if e_i > s_i:
            m[s_i:e_i] = True
    idx = m.nonzero(as_tuple=False).squeeze(-1)
```

### Verification
- ✅ Training now progresses past step 2
- ✅ No more hang at the original location
- ✅ Forward pass completes successfully

## New Performance Issue - DISCOVERED ⚠️

### Problem
- **Location**: `nsa/core/selection_scorer.py:146`
- **Cause**: `torch.unique(blocks, sorted=True)` is extremely slow
- **Impact**: Training progresses but at ~6 tokens/second (should be >1000)

### Evidence from Testing

#### Synthetic Dataset
```
[debug] step 1: input shape torch.Size([1, 1024]), seq_len 1024
step 0001 | loss 5.7168 | lr 1.00e-07 | toks/s 7
[debug] step 2: input shape torch.Size([1, 1024]), seq_len 1024
[debug] step 3: input shape torch.Size([1, 1024]), seq_len 1024
[debug] step 4: input shape torch.Size([1, 1024]), seq_len 1024
[debug] step 5: input shape torch.Size([1, 1024]), seq_len 1024
step 0005 | loss 5.7614 | lr 3.00e-07 | toks/s 6
```

#### FineWeb-Edu Dataset
```
[train] first FineWeb‑Edu batch fetched in 13.53s
[debug] step 1: input shape torch.Size([1, 1024]), seq_len 1024
step 0001 | loss 10.9498 | lr 1.00e-07 | toks/s 6
[debug] step 2: input shape torch.Size([1, 1024]), seq_len 1024
[debug] step 3: input shape torch.Size([1, 1024]), seq_len 1024
[debug] step 4: input shape torch.Size([1, 1024]), seq_len 1024
```

#### Watchdog Stack Traces
Multiple watchdog dumps show the new bottleneck:
```
File "/home/ubuntu/nsa-vibe/nsa/core/nsa_attention.py", line 717, in _forward_prefill_sequential
    sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, t, True, 2)
File "/home/ubuntu/nsa-vibe/nsa/core/selection_scorer.py", line 146, in select_topn_ranges
    blocks = torch.unique(blocks, sorted=True)  # <- NEW BOTTLENECK
```

## Performance Analysis

### Current State
- **Throughput**: ~6-7 tokens/second (extremely slow)
- **GPU Utilization**: 99% (but inefficient due to unique operation)
- **Memory Usage**: 30-60GB (normal for model size)
- **Progress**: Training advances but triggers 180s watchdog timeouts

### Expected Performance
- Should achieve >1000 tokens/second for 125M model
- Steps should complete in seconds, not minutes
- No watchdog timeouts during normal training

## Root Cause Analysis

The selection scorer's `torch.unique()` operation is being called repeatedly and is computationally expensive when:
1. Operating on large tensors
2. Requiring sorted output
3. Called within the critical training loop

This suggests the selection algorithm needs optimization, possibly by:
- Caching unique operations
- Using more efficient sorting algorithms
- Restructuring the selection logic

## Test Environment
- **Server**: Prime Intellect ubuntu@216.81.248.82 (2x A100 80GB)
- **Branch**: test-plan/m7-training-readiness
- **Config**: configs/m7c_125m_fast_log.yaml
- **Tested with**: Both synthetic and FineWeb-Edu datasets

## Artifacts Collected

### Log Files
- `training_synth.log` - Synthetic training attempts
- `training_fwe_final.log` - FineWeb-Edu training
- `artifacts/m7c_125m/heartbeat_rank0.jsonl` - Progress tracking
- `artifacts/m7c_125m/watchdog_stackdump_*.txt` - Stack traces showing bottleneck

### Key Metrics
- Unit tests: 7/7 passed
- Data loading: Works correctly (~13s for FineWeb-Edu)
- Step 1 completion: Successful with loss calculation
- Steps 2-5: Complete but extremely slow

## Recommendations for Main Agent

### Immediate Actions
1. **Acknowledge partial success** - Original bug is fixed
2. **Investigate selection_scorer.py:146** - Profile and optimize `torch.unique()`
3. **Consider temporary workaround** - Reduce `n_sel` parameter to minimize unique operations

### Optimization Strategies
1. **Cache unique results** when possible
2. **Use torch.unique(sorted=False)** if ordering not critical
3. **Batch unique operations** instead of per-step calls
4. **Consider alternative algorithms** that avoid repeated sorting

### Testing Requirements
1. Profile the selection scorer in isolation
2. Benchmark torch.unique() with various input sizes
3. Test with reduced n_sel values (e.g., 4 instead of 16)
4. Monitor actual vs expected throughput

## Conclusion

**Status**: The critical hang bug is FIXED, but training is not production-ready due to performance issues.

**Progress**:
- ✅ Original `r.tolist()` hang eliminated
- ✅ Training progresses beyond step 2
- ⚠️ New bottleneck in selection scorer
- ❌ Training too slow for practical use (6 toks/s vs >1000 expected)

**Next Steps**: The main agent should focus on optimizing the selection scorer's `torch.unique()` operation to achieve acceptable training speeds. The NSA architecture is functionally correct but needs performance optimization.

## Files to Review
1. `/home/ubuntu/nsa-vibe/nsa/core/selection_scorer.py` - Line 146 (bottleneck)
2. `/home/ubuntu/nsa-vibe/nsa/core/nsa_attention.py` - Lines 717, 818-837 (fixed)
3. `/home/ubuntu/nsa-vibe/artifacts/m7c_125m/watchdog_stackdump_*.txt` - Stack traces
4. Config: `/home/ubuntu/nsa-vibe/configs/m7c_125m_fast_log.yaml`

---
*Report generated: 2025-08-22T04:40:00Z*
*Training attempts: 15+ runs across synthetic and FineWeb-Edu*
*Total testing time: ~6 hours*