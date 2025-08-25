#!/bin/bash
# NSA Test Engineer CPU-Only Tests for Local Verification
# Tests that can run without GPU to verify script structure

set -euo pipefail

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="artifacts/test_engineer_cpu_${TIMESTAMP}"
mkdir -p "${BASE_DIR}"
LOG_FILE="${BASE_DIR}/test_execution.log"

# Activate virtual environment
source .venv/bin/activate

# Common environment setup
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CONFIG=configs/m7c_125m_2k_test_cpu.yaml  # Use CPU config
export PYTHONPATH=/Users/alexanderhuth/Code/nsa-vibe

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Test runner with timeout and result capture
run_test() {
    local test_name="$1"
    local cmd="$2"
    local timeout_sec="${3:-30}"  # Shorter timeout for CPU tests
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
    grep -E "\[GRAD-TRACE\]|\[trace\]|MISSING:" "${test_dir}/output.log" > "${test_dir}/traces.log" 2>/dev/null || true
    
    return 0  # Don't stop on test failure
}

# ========================================
# Phase 0: System Check
# ========================================
log "Starting NSA Test Engineer CPU-Only Verification"
log "Output directory: ${BASE_DIR}"
log "Python: $(which python)"
log "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log "Device: CPU only"

# ========================================
# Test 1: Single-CPU Trace Test
# ========================================
log ""
log "TEST 1: Single-CPU with Gradient Tracing"

export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1

run_test "cpu_trace_test" \
    "python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# ========================================
# Test 2: Branch Isolation on CPU
# ========================================
log ""
log "TEST 2: Branch Isolation on CPU"

# Test compressed only
run_test "cpu_branch_cmp" \
    "NSA_FORCE_BRANCH=cmp python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Test selection only
run_test "cpu_branch_sel" \
    "NSA_FORCE_BRANCH=sel python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Test sliding only
run_test "cpu_branch_win" \
    "NSA_FORCE_BRANCH=win python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# ========================================
# Test 3: GC Range Test on CPU
# ========================================
log ""
log "TEST 3: Gradient Checkpointing Range Test"

# Test with layers 0-2 checkpointed
export NSA_GC_RANGE=0:2
run_test "cpu_gc_range_0_2" \
    "python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

unset NSA_GC_RANGE

# ========================================
# Summary Report
# ========================================
log ""
log "Generating Summary Report"

# Create summary report
cat > "${BASE_DIR}/SUMMARY.md" << EOF
# NSA Test Engineer CPU Verification Report
**Timestamp**: ${TIMESTAMP}
**Output Directory**: ${BASE_DIR}

## Test Results Summary

| Test Name | Result | Notes |
|-----------|--------|-------|
EOF

# Add test results to summary
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        
        # Check for traces
        trace_info=""
        if [ -f "${test_dir}/traces.log" ] && [ -s "${test_dir}/traces.log" ]; then
            trace_info="Traces captured"
        fi
        
        echo "| ${test_name} | ${result} | ${trace_info} |" >> "${BASE_DIR}/SUMMARY.md"
    fi
done

cat >> "${BASE_DIR}/SUMMARY.md" << EOF

## Trace Information

### Missing Parameters
EOF

# Extract unique missing parameters
if find "${BASE_DIR}" -name "traces.log" -exec grep "MISSING:" {} \; | sort -u > "${BASE_DIR}/all_missing_params.txt" 2>/dev/null; then
    if [ -s "${BASE_DIR}/all_missing_params.txt" ]; then
        echo '```' >> "${BASE_DIR}/SUMMARY.md"
        head -20 "${BASE_DIR}/all_missing_params.txt" >> "${BASE_DIR}/SUMMARY.md"
        echo '```' >> "${BASE_DIR}/SUMMARY.md"
    else
        echo "No missing parameters detected in traces." >> "${BASE_DIR}/SUMMARY.md"
    fi
fi

log ""
log "✅ CPU verification complete!"
log "Summary report: ${BASE_DIR}/SUMMARY.md"

# Display summary
echo ""
echo "========================================="
echo "CPU Test Execution Summary"
echo "========================================="
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        printf "%-25s: %s\n" "$test_name" "$result"
    fi
done
echo "========================================="