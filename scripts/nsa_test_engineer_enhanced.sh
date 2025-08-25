#!/bin/bash
# NSA Test Engineer Enhanced Action Plan - Complete GPU Test Suite
# Addresses all gaps identified by Core Engineer

set -euo pipefail

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="artifacts/test_engineer_enhanced_${TIMESTAMP}"
mkdir -p "${BASE_DIR}"
LOG_FILE="${BASE_DIR}/test_execution.log"
RESULTS_FILE="${BASE_DIR}/RESULTS.md"

# Common environment setup
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CONFIG=configs/m7c_125m_2xa100_production.yaml
export PYTHONPATH=/Users/alexanderhuth/Code/nsa-vibe

# Fixed seed for reproducibility
export PYTHONHASHSEED=1337

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Enhanced test runner with per-rank logs and timeout handling
run_test() {
    local test_name="$1"
    local cmd="$2"
    local timeout_sec="${3:-60}"
    local test_dir="${BASE_DIR}/${test_name}"
    
    mkdir -p "${test_dir}"
    log "=== Running test: ${test_name} ==="
    log "Command: ${cmd}"
    
    # Save environment snapshot
    env | grep -E "NSA_|TORCH_|NCCL_|CUDA_|CONFIG" > "${test_dir}/env.json" || true
    
    # Setup signal handler for stack dump
    trap 'kill -USR1 $! 2>/dev/null || true' TERM
    
    # Run with timeout and capture output
    (
        cd /Users/alexanderhuth/Code/nsa-vibe
        # For torchrun commands, add per-rank logging
        if [[ "$cmd" == *"torchrun"* ]]; then
            # Create rank-specific log directory
            mkdir -p "${test_dir}/ranks"
            # Modify command to add log directory
            modified_cmd=$(echo "$cmd" | sed "s/torchrun/torchrun --log-dir ${test_dir}\/ranks/")
            timeout "${timeout_sec}" bash -c "${modified_cmd}" 2>&1
        else
            timeout "${timeout_sec}" bash -c "${cmd}" 2>&1
        fi
    ) | tee "${test_dir}/output.log" &
    
    local pid=$!
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "PASS" > "${test_dir}/result.txt"
        log "✅ ${test_name}: PASS"
    elif [ $exit_code -eq 124 ]; then
        echo "HANG" > "${test_dir}/result.txt"
        log "⏱️ ${test_name}: HANG (timeout after ${timeout_sec}s)"
        
        # Try to get stack dump before killing
        pkill -USR1 -f "train_showcase.py" 2>/dev/null || true
        sleep 2
        
        # Capture any watchdog dumps
        cp artifacts/m7c_125m_2xa100_prod/watchdog_stackdump_*.txt "${test_dir}/" 2>/dev/null || true
    else
        echo "FAIL" > "${test_dir}/result.txt"
        log "❌ ${test_name}: FAIL (exit code ${exit_code})"
    fi
    
    # Extract key information
    grep -E "\[GRAD-TRACE\]|\[DDP\]|\[trace\]|MISSING:" "${test_dir}/output.log" > "${test_dir}/traces.log" 2>/dev/null || true
    
    # Extract per-rank traces if available
    if [ -d "${test_dir}/ranks" ]; then
        for rank_log in "${test_dir}"/ranks/*.log; do
            if [ -f "$rank_log" ]; then
                rank_num=$(basename "$rank_log" | grep -oE '[0-9]+')
                grep -E "\[GRAD-TRACE\]|\[DDP\]|MISSING:" "$rank_log" > "${test_dir}/rank${rank_num}_traces.log" 2>/dev/null || true
            fi
        done
    fi
    
    return 0  # Don't stop on test failure
}

# ========================================
# Phase 0: System Check & Validation
# ========================================
log "Starting NSA Test Engineer Enhanced Action Plan"
log "Output directory: ${BASE_DIR}"
log "Python: $(which python)"
log "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    log "CUDA Version: $(python -c 'import torch; print(torch.version.cuda)')"
    log "GPU Device: $(python -c 'import torch; print(torch.cuda.get_device_name())')"
fi

# Validate gradient checkpointing is enabled
if grep -q "gradient_checkpointing: true" "${CONFIG}"; then
    log "✅ Gradient checkpointing confirmed enabled in config"
else
    log "⚠️ WARNING: Gradient checkpointing not enabled in config!"
fi

# ========================================
# Phase 1: DDP One-Step Trace (NCCL)
# ========================================
log ""
log "PHASE 1: DDP One-Step Trace with NCCL Backend"
log "Note: NSA_TRACE_DDP_BUCKETS enabled ONLY for this single step"

export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1
export NSA_TRACE_DDP_BUCKETS=1  # Only for single step

run_test "ddp_onestep_nccl" \
    "CUDA_VISIBLE_DEVICES=0,1 TORCH_BACKEND=nccl torchrun --nproc_per_node=2 --master-port=29500 scripts/train_showcase.py --dataset synthetic --steps 1"

# IMPORTANT: Disable bucket tracing immediately (too noisy for multi-step)
unset NSA_TRACE_DDP_BUCKETS
log "Disabled NSA_TRACE_DDP_BUCKETS after single-step test"

# ========================================
# Phase 2: DDP One-Step Trace (Gloo)
# ========================================
log ""
log "PHASE 2: DDP One-Step Trace with Gloo Backend"

run_test "ddp_onestep_gloo" \
    "CUDA_VISIBLE_DEVICES=0,1 TORCH_BACKEND=gloo torchrun --nproc_per_node=2 --master-port=29501 scripts/train_showcase.py --dataset synthetic --steps 1"

# ========================================
# Phase 3: Branch Isolation Tests
# ========================================
log ""
log "PHASE 3: Branch Isolation at Production Shape (12L×2048)"

# Keep grad/module tracing but not buckets
export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1

# Test compressed only
run_test "branch_cmp_only" \
    "NSA_FORCE_BRANCH=cmp CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29502 scripts/train_showcase.py --dataset synthetic --steps 1"

# Test selection only
run_test "branch_sel_only" \
    "NSA_FORCE_BRANCH=sel CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29503 scripts/train_showcase.py --dataset synthetic --steps 1"

# Test sliding only
run_test "branch_win_only" \
    "NSA_FORCE_BRANCH=win CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29504 scripts/train_showcase.py --dataset synthetic --steps 1"

# ========================================
# Phase 4: DDP Static Graph Test
# ========================================
log ""
log "PHASE 4: DDP Static Graph Mode (with find_unused_parameters=True)"

export NSA_DDP_STATIC_GRAPH=1
export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1

run_test "ddp_static_graph" \
    "CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port=29505 scripts/train_showcase.py --dataset synthetic --steps 1"

unset NSA_DDP_STATIC_GRAPH

# ========================================
# Phase 5: Single-GPU GC Bisection
# ========================================
log ""
log "PHASE 5: Single-GPU Gradient Checkpointing Bisection"
log "Config has gradient_checkpointing=true, testing layer ranges"

# Baseline with full GC (all layers)
run_test "gc_baseline_full" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Bisection 1: checkpoint layers 0-5
export NSA_GC_RANGE=0:6
log "Testing GC for layers [0:6)"
run_test "gc_bisect_layers_0_5" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

# Bisection 2: checkpoint layers 6-11
export NSA_GC_RANGE=6:12
log "Testing GC for layers [6:12)"
run_test "gc_bisect_layers_6_11" \
    "CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 1"

unset NSA_GC_RANGE

# ========================================
# Phase 6: PyTorch 2.4.1 A/B Test (Optional)
# ========================================
log ""
log "PHASE 6: PyTorch Version A/B Test"

# Check if we can create a 2.4.1 venv
if command -v python3 &> /dev/null; then
    log "Checking for PyTorch 2.4.1 installation..."
    
    # Document the commands for 2.4.1 testing
    cat > "${BASE_DIR}/torch_241_test_commands.sh" << 'EOF'
#!/bin/bash
# Commands to test with PyTorch 2.4.1

# Create fresh venv
python -m venv .venv-torch241
source .venv-torch241/bin/activate

# Install PyTorch 2.4.1 with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Run the same DDP one-step trace
export NSA_TRACE_GRADS=1
export NSA_TRACE_MODULE_BWD=1
export NSA_TRACE_DDP_BUCKETS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CONFIG=configs/m7c_125m_2xa100_production.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/train_showcase.py --dataset synthetic --steps 1

# Compare with 2.5.1 results
EOF
    chmod +x "${BASE_DIR}/torch_241_test_commands.sh"
    log "PyTorch 2.4.1 test commands saved to: ${BASE_DIR}/torch_241_test_commands.sh"
else
    log "Python3 not available for creating separate venv"
fi

# ========================================
# Phase 7: Evidence Collection & Summary
# ========================================
log ""
log "PHASE 7: Collecting Evidence and Generating Report"

# Create comprehensive results report
cat > "${RESULTS_FILE}" << EOF
# NSA Test Engineer Enhanced Action Plan - Evidence Report
**Timestamp**: ${TIMESTAMP}
**Output Directory**: ${BASE_DIR}
**PyTorch Version**: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "N/A")
**CUDA**: $(python -c 'import torch; print(f"{torch.cuda.is_available()} - {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")' 2>/dev/null || echo "N/A")

## Test Results Summary

| Test Phase | Result | Key Finding |
|------------|--------|-------------|
EOF

# Collect results for each test
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        
        # Extract key findings
        finding=""
        if [ -f "${test_dir}/traces.log" ]; then
            missing_count=$(grep -c "MISSING:" "${test_dir}/traces.log" 2>/dev/null || echo "0")
            if [ "$missing_count" -gt 0 ]; then
                finding="${missing_count} missing grads"
            fi
        fi
        
        echo "| ${test_name} | **${result}** | ${finding} |" >> "${RESULTS_FILE}"
    fi
done

# Extract missing parameters
echo "" >> "${RESULTS_FILE}"
echo "## Critical Evidence" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"
echo "### 1. Missing Parameters from Grad Trace (Rank 0)" >> "${RESULTS_FILE}"

if find "${BASE_DIR}" -name "traces.log" -exec grep "MISSING:" {} \; | sort -u > "${BASE_DIR}/all_missing_params.txt"; then
    echo '```' >> "${RESULTS_FILE}"
    head -50 "${BASE_DIR}/all_missing_params.txt" >> "${RESULTS_FILE}"
    total_missing=$(wc -l < "${BASE_DIR}/all_missing_params.txt")
    if [ "$total_missing" -gt 50 ]; then
        echo "... and $((total_missing - 50)) more missing parameters" >> "${RESULTS_FILE}"
    fi
    echo '```' >> "${RESULTS_FILE}"
    echo "**Full list**: ${BASE_DIR}/all_missing_params.txt" >> "${RESULTS_FILE}"
else
    echo "No missing parameters detected." >> "${RESULTS_FILE}"
fi

# Extract module backward trace
echo "" >> "${RESULTS_FILE}"
echo "### 2. Module Backward Trace Summary" >> "${RESULTS_FILE}"

if grep -h "seen_types=" "${BASE_DIR}"/*/traces.log 2>/dev/null | head -1 > /tmp/mod_trace.txt; then
    echo '```' >> "${RESULTS_FILE}"
    cat /tmp/mod_trace.txt >> "${RESULTS_FILE}"
    echo '```' >> "${RESULTS_FILE}"
fi

# Extract DDP bucket logs
echo "" >> "${RESULTS_FILE}"
echo "### 3. DDP Bucket Logs (One Step)" >> "${RESULTS_FILE}"

if find "${BASE_DIR}/ddp_onestep_nccl" -name "*.log" -exec grep "\[DDP\]" {} \; > "${BASE_DIR}/ddp_buckets.txt" 2>/dev/null; then
    echo '```' >> "${RESULTS_FILE}"
    head -20 "${BASE_DIR}/ddp_buckets.txt" >> "${RESULTS_FILE}"
    echo '```' >> "${RESULTS_FILE}"
fi

# Branch isolation verdict
echo "" >> "${RESULTS_FILE}"
echo "### 4. Branch Isolation Verdict" >> "${RESULTS_FILE}"

for branch in cmp sel win; do
    if [ -f "${BASE_DIR}/branch_${branch}_only/result.txt" ]; then
        result=$(cat "${BASE_DIR}/branch_${branch}_only/result.txt")
        echo "- **${branch}** branch alone: ${result}" >> "${RESULTS_FILE}"
    fi
done

# Static graph verdict
echo "" >> "${RESULTS_FILE}"
echo "### 5. Static Graph Verdict" >> "${RESULTS_FILE}"

if [ -f "${BASE_DIR}/ddp_static_graph/result.txt" ]; then
    result=$(cat "${BASE_DIR}/ddp_static_graph/result.txt")
    echo "- DDP with static_graph=True: **${result}**" >> "${RESULTS_FILE}"
fi

# GC bisection verdict
echo "" >> "${RESULTS_FILE}"
echo "### 6. GC Bisection Verdict (Single GPU)" >> "${RESULTS_FILE}"

if [ -f "${BASE_DIR}/gc_baseline_full/result.txt" ]; then
    result=$(cat "${BASE_DIR}/gc_baseline_full/result.txt")
    echo "- Full GC (all layers): ${result}" >> "${RESULTS_FILE}"
fi
if [ -f "${BASE_DIR}/gc_bisect_layers_0_5/result.txt" ]; then
    result=$(cat "${BASE_DIR}/gc_bisect_layers_0_5/result.txt")
    echo "- GC layers [0:6): ${result}" >> "${RESULTS_FILE}"
fi
if [ -f "${BASE_DIR}/gc_bisect_layers_6_11/result.txt" ]; then
    result=$(cat "${BASE_DIR}/gc_bisect_layers_6_11/result.txt")
    echo "- GC layers [6:12): ${result}" >> "${RESULTS_FILE}"
fi

# PyTorch version comparison
echo "" >> "${RESULTS_FILE}"
echo "### 7. PyTorch Version A/B Test" >> "${RESULTS_FILE}"
echo "- Current version: $(python -c 'import torch; print(torch.__version__)')" >> "${RESULTS_FILE}"
echo "- Test commands for 2.4.1: See ${BASE_DIR}/torch_241_test_commands.sh" >> "${RESULTS_FILE}"

# Per-rank analysis
echo "" >> "${RESULTS_FILE}"
echo "## Per-Rank Analysis" >> "${RESULTS_FILE}"

if ls "${BASE_DIR}"/*/rank*_traces.log 1> /dev/null 2>&1; then
    echo "Per-rank traces captured:" >> "${RESULTS_FILE}"
    for rank_trace in "${BASE_DIR}"/*/rank*_traces.log; do
        rank_name=$(basename "$rank_trace")
        test_name=$(basename $(dirname "$rank_trace"))
        missing=$(grep -c "MISSING:" "$rank_trace" 2>/dev/null || echo "0")
        echo "- ${test_name}/${rank_name}: ${missing} missing params" >> "${RESULTS_FILE}"
    done
else
    echo "No per-rank traces available (may need GPU execution)" >> "${RESULTS_FILE}"
fi

# Recommendations
echo "" >> "${RESULTS_FILE}"
echo "## Recommendations Based on Results" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

cat >> "${RESULTS_FILE}" << 'EOF'
1. **If static_graph=True passes**: Enable in production with documented invariant
2. **If specific branch hangs alone**: Focus debugging on that branch's backward
3. **If GC bisection shows pattern**: Investigate layers in failing range for in-place ops
4. **If gloo also hangs**: Issue is in autograd graph, not NCCL transport
5. **If 2.4.1 passes but 2.5.1 fails**: Pin to 2.4.1 and file upstream issue

## Artifacts Structure

```
EOF

# Show artifact structure
tree -L 2 "${BASE_DIR}" 2>/dev/null >> "${RESULTS_FILE}" || ls -la "${BASE_DIR}" >> "${RESULTS_FILE}"

echo '```' >> "${RESULTS_FILE}"

log ""
log "✅ Test execution complete!"
log ""
log "=" * 60
log "KEY OUTPUTS:"
log "=" * 60
log "Results Report: ${RESULTS_FILE}"
log "Execution Log: ${LOG_FILE}"
log "Missing Params: ${BASE_DIR}/all_missing_params.txt"
log "DDP Buckets: ${BASE_DIR}/ddp_buckets.txt"
log "=" * 60

# Display summary
echo ""
echo "TEST EXECUTION SUMMARY"
echo "======================"
for test_dir in "${BASE_DIR}"/*/; do
    if [ -d "$test_dir" ] && [ -f "${test_dir}/result.txt" ]; then
        test_name=$(basename "$test_dir")
        result=$(cat "${test_dir}/result.txt")
        printf "%-30s: %s\n" "$test_name" "$result"
    fi
done
echo "======================"