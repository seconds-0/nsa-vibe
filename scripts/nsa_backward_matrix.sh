#!/bin/bash
# NSA Backward Pass Test Matrix Runner
# Runs systematic tests to isolate the backward pass hang

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_SCRIPT="${SCRIPT_DIR}/nsa_backward_repro.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="artifacts/nsa_backward_matrix_${TIMESTAMP}"

# Create base output directory
mkdir -p "${BASE_OUT_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${BASE_OUT_DIR}/matrix_log.txt"
}

# Run test with timeout
run_test() {
    local test_name="$1"
    local cmd="$2"
    local timeout_sec="${3:-30}"  # Default 30 second timeout
    
    log "Running: ${test_name}"
    log "Command: ${cmd}"
    
    # Run with timeout
    timeout "${timeout_sec}" bash -c "${cmd}" 2>&1 | tee "${BASE_OUT_DIR}/${test_name}.log"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log "✓ ${test_name}: PASS"
        echo "PASS" > "${BASE_OUT_DIR}/${test_name}.result"
    elif [ $exit_code -eq 124 ]; then
        log "✗ ${test_name}: HANG (timeout after ${timeout_sec}s)"
        echo "HANG" > "${BASE_OUT_DIR}/${test_name}.result"
        
        # Capture nvidia-smi on hang
        nvidia-smi > "${BASE_OUT_DIR}/${test_name}_nvidia_smi.txt" 2>&1 || true
    else
        log "✗ ${test_name}: FAIL (exit code ${exit_code})"
        echo "FAIL" > "${BASE_OUT_DIR}/${test_name}.result"
    fi
    
    echo ""
    return 0  # Don't stop on individual test failures
}

# Ensure we're in the right directory
cd "${SCRIPT_DIR}/.."

log "Starting NSA Backward Pass Test Matrix"
log "Output directory: ${BASE_OUT_DIR}"
log "Python: $(which python)"
log "CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"

# ========================================
# Priority A: Branch Isolation (5 layers, seq_len=128)
# ========================================
log "=== Phase 1: Branch Isolation ==="

# Test each branch independently
run_test "branch_win" \
    "NSA_FORCE_BRANCH=win python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch win --out-dir ${BASE_OUT_DIR}/branch_win"

run_test "branch_sel" \
    "NSA_FORCE_BRANCH=sel python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch sel --out-dir ${BASE_OUT_DIR}/branch_sel"

run_test "branch_cmp" \
    "NSA_FORCE_BRANCH=cmp python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch cmp --out-dir ${BASE_OUT_DIR}/branch_cmp"

# ========================================
# Priority A: Selection Backend Sweep
# ========================================
log "=== Phase 2: Selection Backend Sweep ==="

run_test "sel_masked" \
    "NSA_FORCE_BRANCH=sel python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch sel --sel masked --out-dir ${BASE_OUT_DIR}/sel_masked"

run_test "sel_packed" \
    "NSA_FORCE_BRANCH=sel python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch sel --sel packed --out-dir ${BASE_OUT_DIR}/sel_packed"

run_test "sel_gather" \
    "NSA_FORCE_BRANCH=sel python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch sel --sel gather --out-dir ${BASE_OUT_DIR}/sel_gather"

# ========================================
# Priority A: Compressed Backend Sweep
# ========================================
log "=== Phase 3: Compressed Backend Sweep ==="

run_test "cmp_masked" \
    "NSA_FORCE_BRANCH=cmp python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch cmp --cmp masked --out-dir ${BASE_OUT_DIR}/cmp_masked"

run_test "cmp_parity" \
    "NSA_FORCE_BRANCH=cmp python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch cmp --cmp parity --out-dir ${BASE_OUT_DIR}/cmp_parity"

# ========================================
# Priority A: Sliding Backend Sweep
# ========================================
log "=== Phase 4: Sliding Backend Sweep ==="

run_test "win_masked" \
    "NSA_FORCE_BRANCH=win python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch win --win masked --out-dir ${BASE_OUT_DIR}/win_masked"

run_test "win_parity" \
    "NSA_FORCE_BRANCH=win python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --branch win --win parity --out-dir ${BASE_OUT_DIR}/win_parity"

# ========================================
# Priority B: Sequence Length Scaling (with default config)
# ========================================
log "=== Phase 5: Sequence Length Scaling ==="

for seq_len in 32 64 128 256; do
    run_test "seq_${seq_len}" \
        "python ${REPRO_SCRIPT} --layers 5 --seq-len ${seq_len} --out-dir ${BASE_OUT_DIR}/seq_${seq_len}"
done

# ========================================
# Priority B: Allocator Sensitivity (test with worst case from above)
# ========================================
log "=== Phase 6: Allocator Sensitivity ==="

# Test with expandable segments
run_test "alloc_expandable" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --out-dir ${BASE_OUT_DIR}/alloc_expandable"

# Test with smaller split size
run_test "alloc_small_split" \
    "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 python ${REPRO_SCRIPT} --layers 5 --seq-len 128 --out-dir ${BASE_OUT_DIR}/alloc_small_split"

# ========================================
# Generate Summary Report
# ========================================
log "=== Generating Summary Report ==="

cat > "${BASE_OUT_DIR}/summary.txt" << EOF
NSA Backward Pass Test Matrix Summary
=====================================
Timestamp: ${TIMESTAMP}
Output Directory: ${BASE_OUT_DIR}

Test Results:
-------------
EOF

for result_file in "${BASE_OUT_DIR}"/*.result; do
    if [ -f "$result_file" ]; then
        test_name=$(basename "$result_file" .result)
        result=$(cat "$result_file")
        printf "%-30s: %s\n" "$test_name" "$result" >> "${BASE_OUT_DIR}/summary.txt"
    fi
done

echo "" >> "${BASE_OUT_DIR}/summary.txt"
echo "Detailed logs available in: ${BASE_OUT_DIR}/*.log" >> "${BASE_OUT_DIR}/summary.txt"

# Display summary
cat "${BASE_OUT_DIR}/summary.txt"

log "Test matrix complete. Results in: ${BASE_OUT_DIR}/summary.txt"