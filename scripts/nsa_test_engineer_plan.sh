#!/bin/bash
# NSA Test Engineer Action Plan Execution Script
# Implements the Core Engineer's comprehensive test plan for DDP and GC debugging

set -euo pipefail

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="artifacts/test_engineer_${TIMESTAMP}"
mkdir -p "${BASE_DIR}"
LOG_FILE="${BASE_DIR}/test_execution.log"

# Common environment setup
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CONFIG=configs/m7c_125m_2xa100_production.yaml

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Test runner with timeout and result capture
run_test() {
    local test_name="$1"
    local cmd="$2"
    local timeout_sec="${3:-60}"  # Default 60 second timeout
    local test_dir="${BASE_DIR}/${test_name}"
    
    mkdir -p "${test_dir}"
    log "=== Running test: ${test_name} ==="
    log "Command: ${cmd}"
    
    # Run with timeout and capture output
    (
        cd /Users/alexanderhuth/Code/nsa-vibe
        timeout "${timeout_sec}" bash -c "${cmd}" 2>&1
    ) | tee "${test_dir}/output.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "PASS" > "${test_dir}/result.txt"
        log "✅ ${test_name}: PASS"
    elif [ $exit_code -eq 124 ]; then
        echo "HANG" > "${test_dir}/result.txt"
        log "⏱️ ${test_name}: HANG (timeout after ${timeout_sec}s)"
    else
        echo "FAIL" > "${test_dir}/result.txt"
        log "❌ ${test_name}: FAIL (exit code ${exit_code})"
    fi
    
    # Extract key information
    grep -E "\[GRAD-TRACE\]|\[DDP\]|\[trace\]|MISSING:" "${test_dir}/output.log" > "${test_dir}/traces.log" 2>/dev/null || true
    
    return 0  # Don't stop on test failure
}

# ========================================
# Phase 0: System Check
# ========================================
log "Starting NSA Test Engineer Action Plan Execution"
log "Output directory: ${BASE_DIR}"
log "Python: $(which python)"
log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\")')"

# ========================================
# Phase 1: DDP One-Step Trace
# ========================================
log ""
log "PHASE 1: DDP One-Step Trace with Gradient Tracing"

export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1
export NSA_TRACE_DDP_BUCKETS=1

run_test "ddp_onestep_trace" \
    "CUDA_VISIBLE_DEVICES=0,1 TORCH_BACKEND=nccl torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

# Disable bucket tracing after one step (too noisy)
unset NSA_TRACE_DDP_BUCKETS

# ========================================
# Phase 2: DDP Gloo Backend Sanity
# ========================================
log ""
log "PHASE 2: DDP Gloo Backend Sanity Check"

run_test "ddp_gloo_sanity" \
    "CUDA_VISIBLE_DEVICES=0,1 TORCH_BACKEND=gloo torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

# ========================================
# Phase 3: Branch Isolation
# ========================================
log ""
log "PHASE 3: Branch Isolation at Production Shape"

# Keep trace enabled but not buckets
export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1

# Test compressed only
run_test "branch_cmp_only" \
    "NSA_FORCE_BRANCH=cmp CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

# Test selection only
run_test "branch_sel_only" \
    "NSA_FORCE_BRANCH=sel CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

# Test sliding only
run_test "branch_win_only" \
    "NSA_FORCE_BRANCH=win CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

# ========================================
# Phase 4: DDP Static Graph Sanity
# ========================================
log ""
log "PHASE 4: DDP Static Graph Mode Test"

export NSA_DDP_STATIC_GRAPH=1
export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1

run_test "ddp_static_graph" \
    "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic --steps 1"

unset NSA_DDP_STATIC_GRAPH

# ========================================
# Phase 5: Single-GPU GC Bisection
# ========================================
log ""
log "PHASE 5: Single-GPU Gradient Checkpointing Bisection"

# Baseline with full GC
run_test "gc_baseline_full" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Bisection 1: checkpoint layers 0-5
export NSA_GC_RANGE=0:6
run_test "gc_bisect_layers_0_5" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Bisection 2: checkpoint layers 6-11
export NSA_GC_RANGE=6:12
run_test "gc_bisect_layers_6_11" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

unset NSA_GC_RANGE

# ========================================
# Phase 6: Summary Report Generation
# ========================================
log ""
log "PHASE 6: Generating Summary Report"

# Create summary report
cat > "${BASE_DIR}/SUMMARY.md" << EOF
# NSA Test Engineer Action Plan - Execution Report
**Timestamp**: ${TIMESTAMP}
**Output Directory**: ${BASE_DIR}

## Test Results Summary

| Test Name | Result | Key Findings |
|-----------|--------|--------------|
EOF

# Add test results to summary
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        
        # Count missing params if trace log exists
        missing_count=0
        if [ -f "${test_dir}/traces.log" ]; then
            missing_count=$(grep -c "MISSING:" "${test_dir}/traces.log" 2>/dev/null || echo "0")
        fi
        
        echo "| ${test_name} | ${result} | Missing params: ${missing_count} |" >> "${BASE_DIR}/SUMMARY.md"
    fi
done

cat >> "${BASE_DIR}/SUMMARY.md" << EOF

## Critical Findings

### Missing Parameters in Grad Trace
EOF

# Extract unique missing parameters
if find "${BASE_DIR}" -name "traces.log" -exec grep "MISSING:" {} \; | sort -u > "${BASE_DIR}/all_missing_params.txt"; then
    echo '```' >> "${BASE_DIR}/SUMMARY.md"
    head -20 "${BASE_DIR}/all_missing_params.txt" >> "${BASE_DIR}/SUMMARY.md"
    echo '```' >> "${BASE_DIR}/SUMMARY.md"
fi

cat >> "${BASE_DIR}/SUMMARY.md" << EOF

### DDP Bucket Information
EOF

# Extract DDP bucket logs if available
if find "${BASE_DIR}" -name "traces.log" -exec grep "\[DDP\]" {} \; | head -10 > "${BASE_DIR}/ddp_buckets.txt"; then
    echo '```' >> "${BASE_DIR}/SUMMARY.md"
    cat "${BASE_DIR}/ddp_buckets.txt" >> "${BASE_DIR}/SUMMARY.md"
    echo '```' >> "${BASE_DIR}/SUMMARY.md"
fi

cat >> "${BASE_DIR}/SUMMARY.md" << EOF

## Recommendations

Based on the test results:

1. **If static_graph=True passes**: Enable static graph mode in production
2. **If specific branch fails**: Focus debugging on that branch's backward implementation
3. **If GC bisection isolates layers**: Investigate those specific layers for in-place ops
4. **If gloo also hangs**: Issue is in autograd graph, not NCCL transport

## Artifacts

- Full test logs: ${BASE_DIR}/*/output.log
- Trace extracts: ${BASE_DIR}/*/traces.log
- Missing params: ${BASE_DIR}/all_missing_params.txt
- DDP buckets: ${BASE_DIR}/ddp_buckets.txt
EOF

log ""
log "✅ Test execution complete!"
log "Summary report: ${BASE_DIR}/SUMMARY.md"
log "Full logs: ${LOG_FILE}"

# Display summary
echo ""
echo "========================================="
echo "Test Execution Summary"
echo "========================================="
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        printf "%-30s: %s\n" "$test_name" "$result"
    fi
done
echo "========================================="