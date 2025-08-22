# NSA Training Bug Analysis & Handoff

## Executive Summary

Training for M7C configuration consistently hangs during step 2 due to a bug in the NSA attention implementation. The issue is specifically in the selection branch computation at `nsa/core/nsa_attention.py:818` where `r.tolist()` appears to hang indefinitely or contains invalid/infinite ranges.

## Environment
- **Server**: Prime Intellect ubuntu@216.81.248.82 (2x A100 80GB)
- **Branch**: test-plan/m7-training-readiness  
- **Python**: 3.10 with PyTorch in venv at `/home/ubuntu/nsa-vibe/.venv`
- **Config**: `configs/m7c_125m_fast_log.yaml`

## Timeline of Investigation

### Phase 1: Initial Training Attempts (22:00-01:00)
- Multiple training attempts failed with tokenizer warnings
- Loss was increasing instead of decreasing
- Learning rate stuck at 1e-07 (floor issue in cosine schedule)
- Attempted fixes:
  - Added sequence length validation
  - Modified tokenizer to use truncation
  - Fixed learning rate schedule (removed 0.1 floor)

### Phase 2: Enhanced Debugging (01:00-03:00)
- Other agent enhanced scripts with:
  - Heartbeat monitoring (`artifacts/*/heartbeat_rank0.jsonl`)
  - Stack dump signals (SIGUSR1, SIGTERM)
  - Watchdog thread (180s timeout)
  - FineWeb-Edu smoke test with timeout
  - Progress logging in data loader

### Phase 3: Bug Isolation (03:00-04:00)
- Training consistently hangs after step 1
- Gets through data loading successfully (~15s)
- Processes first batch (step 1, loss 10.95)
- **Hangs during step 2 forward pass**
- Stack dump revealed exact location of hang

## Root Cause Analysis

### Stack Trace (Critical Path)
```
File "/home/ubuntu/nsa-vibe/nsa/core/nsa_attention.py", line 818, in _sdpa_over_ranges
    for s, e in r.tolist():  # <-- HANGS HERE
```

### Full Call Stack
1. `train_showcase.py:545` - `logits = model(x)`
2. `llama_block_nsa.py:103` - `out, _kv = self.attn(xn, kv=kv, prefill=True)`
3. `nsa_attention.py:211` - `return self._forward_prefill_sequential(x, kv)`
4. `nsa_attention.py:721` - `O_sel = self._sdpa_over_ranges(Q_t, K_sel_t, V_sel_t, sel_ranges)`
5. **`nsa_attention.py:818`** - `for s, e in r.tolist()` **← HANG POINT**

### Code Context
```python
# nsa/core/nsa_attention.py:815-824
for g in range(G):
    r = ranges[b, g]  # [n,2]
    idxs = []
    for s, e in r.tolist():  # Line 818 - HANGS HERE
        if e > s:
            idxs.append(torch.arange(s, e, device=K.device))
    if idxs:
        idx = torch.cat(idxs, dim=0)
    else:
        idx = torch.empty((0,), dtype=torch.int64, device=K.device)
```

### Configuration That Triggers Bug
```yaml
model:
  dim: 768
  n_layers: 12
  n_heads: 12
  n_kv_groups: 2  # GQA with 2 groups
  d_k: 64
  d_v: 64
nsa:
  l: 32       # Compression block size
  d: 16       # Compression stride
  l_sel: 64   # Selection block size
  n_sel: 16   # Number of selected blocks
  w: 512      # Window size
  phi: "avg"  # Compression operator
train:
  seq_len: 1024
  batch_size: 1
```

## Evidence Summary

1. **Data Loading Works**: FineWeb-Edu loads successfully in ~15s
2. **Model Initialization Works**: No errors during model construction
3. **First Step Works**: Step 1 completes with loss 10.95
4. **Step 2 Hangs**: Consistently hangs at same location
5. **Dataset Independent**: Same hang with synthetic data
6. **GPU Memory Allocated**: 30GB reserved, process shows active
7. **CPU at 100%**: Suggests infinite loop rather than deadlock

## Hypothesis

The `ranges` tensor at line 816 likely contains:
- Invalid values (negative, NaN, or infinity)
- Extremely large ranges causing memory/computation issues
- Malformed shape causing `.tolist()` to fail

This is likely related to the selection scoring mechanism producing invalid block indices, possibly due to:
- GQA group consistency issues (2 KV groups, 12 heads)
- Block index calculation errors with seq_len=1024
- Edge cases in the selection algorithm

## Collected Artifacts

### Key Files on Remote
- `/home/ubuntu/nsa-vibe/artifacts/m7c_125m/stackdump_1755834515.txt` - Main stack trace
- `/home/ubuntu/nsa-vibe/artifacts/m7c_125m/heartbeat_rank0.jsonl` - Training progress
- `/home/ubuntu/nsa-vibe/artifacts/m7c_125m/watchdog_stackdump_*.txt` - Watchdog dumps
- `/home/ubuntu/nsa-vibe/training_fresh.log` - Latest training attempt log

### Reproduction Steps
```bash
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82
cd /home/ubuntu/nsa-vibe
source .venv/bin/activate
export CONFIG=configs/m7c_125m_fast_log.yaml
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
python -u scripts/train_showcase.py --dataset synthetic --ddp 0
# Will hang after step 1
```

## Handoff Recommendations for Research Agent

### Immediate Debugging Steps

1. **Add debug logging before line 818**:
   ```python
   # In nsa/core/nsa_attention.py:817
   print(f"[DEBUG] ranges shape: {r.shape}, dtype: {r.dtype}")
   print(f"[DEBUG] ranges values: {r}")
   print(f"[DEBUG] ranges min/max: {r.min()}, {r.max()}")
   ```

2. **Check selection scorer output**:
   - Verify `sel_ranges` tensor in `_forward_prefill_sequential` (line 721)
   - Check if selection scores are producing valid indices
   - Validate block index calculations

3. **Test with smaller config**:
   ```yaml
   nsa:
     n_sel: 4  # Reduce from 16
     w: 128    # Reduce from 512
   train:
     seq_len: 256  # Reduce from 1024
   ```

4. **Bypass selection temporarily**:
   - Comment out selection branch
   - Test if compressed + sliding branches work alone

### Likely Fix Areas

1. **Block Index Calculation** (`block_index.py`)
   - Check compression/selection block mapping
   - Verify index bounds with seq_len=1024

2. **Selection Scorer** (`selection_scorer.py`)
   - Validate score computation
   - Check GQA group consistency enforcement
   - Verify forced block inclusion

3. **Range Generation**
   - Ensure ranges are within [0, seq_len)
   - Check for integer overflow with large sequences
   - Validate tensor shapes and dtypes

### Testing Strategy

1. **Unit test the failing function**:
   ```python
   # Test _sdpa_over_ranges in isolation
   def test_sdpa_over_ranges():
       # Create minimal tensors
       # Call _sdpa_over_ranges directly
       # Check where it hangs
   ```

2. **Add timeout wrapper**:
   ```python
   import signal
   def timeout_handler(signum, frame):
       raise TimeoutError("Operation timed out")
   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(5)  # 5 second timeout
   try:
       for s, e in r.tolist():
           # ...
   finally:
       signal.alarm(0)
   ```

3. **Test edge cases**:
   - seq_len = power of 2 (512, 1024, 2048)
   - seq_len not divisible by block sizes
   - Single vs multiple KV groups

## Success Criteria

Training should:
1. Complete step 2 without hanging
2. Show decreasing loss over initial steps
3. Maintain ~1000+ tokens/sec throughput
4. Successfully checkpoint after `save_every` steps

## Contact & Access

- SSH: `ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82`
- Training command: See "Reproduction Steps" above
- TensorBoard: `ssh -L 6006:localhost:6006` then http://localhost:6006
- Kill stuck process: `pkill -f 'python.*train_showcase'`
- Get stack dump: `kill -USR1 <PID>` (dumps to artifacts/*/stackdump_*.txt)

## Conclusion

The NSA implementation has a critical bug in the selection attention branch that causes training to hang indefinitely. The issue is specifically at the range iteration in `_sdpa_over_ranges` and is likely due to invalid range values being produced by the selection scoring mechanism. This needs immediate attention as it blocks all training with the M7C configuration.