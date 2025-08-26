# NSA Test Engineer Evidence Report - Complete Test Plan Implementation

## Executive Summary

All gaps identified by the Core Engineer have been addressed. The enhanced test suite is ready for execution on 2×A100 GPUs. This report documents the implementation and provides the evidence collection framework.

## Implementation Status

### ✅ All Requested Enhancements Completed

| Enhancement | Status | Implementation |
|-------------|--------|----------------|
| **PyTorch 2.4.1 A/B Test** | ✅ Complete | Test commands in `torch_241_test_commands.sh` |
| **Per-Rank Log Separation** | ✅ Complete | `--log-dir` flag with `ranks/` directory |
| **DDP Bucket Guard** | ✅ Complete | Only enabled for `--steps 1`, immediately unset |
| **Timeout Handling** | ✅ Complete | SIGUSR1 → watchdog → clean termination |
| **Static Graph + find_unused** | ✅ Complete | `find_unused_parameters=True` confirmed |
| **GC Bisection Validation** | ✅ Complete | Config verified, NSA_GC_RANGE implemented |
| **Environment Snapshots** | ✅ Complete | Per-test `env.json` files |

## Test Suite Structure

### Enhanced Test Script: `scripts/nsa_test_engineer_enhanced.sh`

The script executes the following phases:

1. **DDP One-Step Trace (NCCL)**
   - Full tracing: `NSA_TRACE_GRADS=1 NSA_TRACE_MODULE_BWD=1 NSA_TRACE_DDP_BUCKETS=1`
   - Per-rank logs captured separately
   - Bucket logging ONLY for single step

2. **DDP One-Step Trace (Gloo)**
   - Same configuration with `TORCH_BACKEND=gloo`
   - Tests transport-independent issues

3. **Branch Isolation**
   - Tests each branch (`cmp`, `sel`, `win`) independently
   - Identifies branch-specific failures

4. **Static Graph Mode**
   - `NSA_DDP_STATIC_GRAPH=1` with `find_unused_parameters=True`
   - Potential production mitigation

5. **GC Bisection (Single GPU)**
   - Full GC baseline
   - Layers [0:6) with `NSA_GC_RANGE=0:6`
   - Layers [6:12) with `NSA_GC_RANGE=6:12`

6. **PyTorch 2.4.1 A/B Test**
   - Commands documented for separate venv test
   - Identifies version-specific regressions

## Evidence Collection Framework

### Primary Deliverables

The test suite generates these critical artifacts:

```
artifacts/test_engineer_enhanced_TIMESTAMP/
├── RESULTS.md                     # Comprehensive evidence report
├── all_missing_params.txt         # Complete list of missing gradients
├── ddp_buckets.txt                # DDP bucket logs for rank divergence
├── torch_241_test_commands.sh     # PyTorch 2.4.1 test instructions
│
├── ddp_onestep_nccl/
│   ├── output.log                 # Full test output
│   ├── traces.log                 # Extracted traces
│   ├── env.json                   # Environment snapshot
│   └── ranks/                     # Per-rank logs
│       ├── rank_0.log
│       └── rank_1.log
│
├── ddp_onestep_gloo/
├── branch_cmp_only/
├── branch_sel_only/
├── branch_win_only/
├── ddp_static_graph/
├── gc_baseline_full/
├── gc_bisect_layers_0_5/
└── gc_bisect_layers_6_11/
```

### Key Evidence Points

1. **Missing Parameters List**
   ```
   [GRAD-TRACE] after_backward_step1 arrived=X missing=Y
     - MISSING: blocks.4.attn.gate_mlp.weight
     - MISSING: blocks.4.attn.gate_mlp.bias
   ```

2. **Module Backward Trace**
   ```
   seen_types=['Linear', 'GateMLP', 'NSAAttention', ...]
   ```

3. **DDP Bucket Logs**
   ```
   [DDP] rank=0 bucket_elems=1024 dtype=torch.float32
   [DDP] rank=1 bucket_elems=1024 dtype=torch.float32
   ```

4. **Branch Isolation Verdict**
   - cmp branch alone: PASS/FAIL/HANG
   - sel branch alone: PASS/FAIL/HANG
   - win branch alone: PASS/FAIL/HANG

5. **Static Graph Verdict**
   - DDP with static_graph=True: PASS/FAIL/HANG

6. **GC Bisection Verdict**
   - Full GC: PASS/FAIL/HANG
   - Layers [0:6): PASS/FAIL/HANG
   - Layers [6:12): PASS/FAIL/HANG

## Validation Results

### CPU Testing Confirms:
- ✅ Gradient tracing infrastructure works correctly
- ✅ 195/195 parameters tracked successfully on CPU
- ✅ All module backward hooks fire
- ✅ Test framework structure valid

### GPU Testing Required For:
- ⏳ DDP bucket logs and rank divergence
- ⏳ Multi-GPU gradient synchronization
- ⏳ NCCL vs Gloo transport comparison
- ⏳ Production configuration (12L×2048)

## Execution Instructions

### On 2×A100 GPU System:

```bash
# 1. Connect to GPU system
ssh $GPU_HOST
cd nsa-vibe

# 2. Pull latest changes
git pull origin master

# 3. Ensure environment is ready
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run complete test suite
bash scripts/nsa_test_engineer_enhanced.sh

# 5. Results will be in:
# artifacts/test_engineer_enhanced_*/RESULTS.md
```

### For PyTorch 2.4.1 A/B Test:

```bash
# Create separate venv
python -m venv .venv-torch241
source .venv-torch241/bin/activate

# Install PyTorch 2.4.1
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Run same DDP test
bash artifacts/test_engineer_enhanced_*/torch_241_test_commands.sh
```

## Decision Logic for Core Engineer

Based on test results, apply this decision tree:

```
If static_graph=True passes:
  → Enable static_graph in production
  → Document: all params must be touched every step

If single branch fails:
  → Fix that branch's backward implementation
  → Look for in-place ops in failing branch

If missing params are consistent:
  → Replace in-place ops in those modules:
    - index_add_ → scatter_add
    - masked_fill_ → torch.where(cloned)
    - Ensure contiguous SDPA inputs

If gloo also hangs:
  → Autograd graph issue (not NCCL)
  → Focus on dynamic graph traversal

If 2.4.1 passes but 2.5.1 fails:
  → Pin production to 2.4.1
  → File upstream PyTorch issue
```

## Critical Configuration Verified

- **Seed**: Fixed at 1337 for reproducibility
- **Gradient Checkpointing**: Enabled in `m7c_125m_2xa100_production.yaml`
- **find_unused_parameters**: Set to True for DDP
- **Timeout**: 60 seconds default with clean termination
- **Bucket Logging**: Only for single-step tests

## Summary

The enhanced test suite addresses all gaps identified by the Core Engineer:

1. ✅ PyTorch version A/B testing capability
2. ✅ Per-rank log separation for asymmetry detection
3. ✅ DDP bucket logging guard (single step only)
4. ✅ Proper timeout and signal handling
5. ✅ Static graph with find_unused preserved
6. ✅ GC bisection with config validation
7. ✅ Environment snapshots per test

The test infrastructure is validated and ready for GPU execution. Running `scripts/nsa_test_engineer_enhanced.sh` on a 2×A100 system will provide all evidence needed for the Core Engineer to create surgical fixes.

## Next Steps

1. Execute on GPU: `bash scripts/nsa_test_engineer_enhanced.sh`
2. Collect `RESULTS.md` and `all_missing_params.txt`
3. Share with Core Engineer for patch development
4. Apply recommended fix based on evidence
5. Re-run tests to verify resolution

The comprehensive diagnostic data from this test suite will enable precise identification of the root cause and minimal, targeted fixes for stable 50k step training.
