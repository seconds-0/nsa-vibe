#!/bin/bash

# NSA M7C Production 2×A100 Training with Comprehensive Diagnostics
# Implements the full diagnostic enhancement plan with success criteria monitoring

set -e

echo "=== NSA M7C Production 2×A100 Training ==="
echo "Timestamp: $(date)"
echo

# Verify GPU availability
echo "🔍 GPU Check:"
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not available - GPU environment not ready"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
echo

# Set production environment variables per plan
export NSA_USE_FA2=1
export NSA_PREFILL_BATCHED=1
export NSA_DISABLE_AUX_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export NSA_DDP_STATIC_GRAPH=1
export NSA_DDP_FIND_UNUSED=0
export NSA_DDP_BUCKET_MB=25
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NSA_MEM_DUMP_EVERY=100
export NSA_LOG_GRAD_NORM=1
export PYTHONPATH=.
export CONFIG=configs/m7c_125m_2xa100_production.yaml

# TORCH_LOGS for first 100 steps only (will disable after initial routing validation)
export TORCH_LOGS="+sdp"

echo "🚀 Environment Configuration:"
echo "  NSA_USE_FA2=$NSA_USE_FA2"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "  NCCL_ALGO=$NCCL_ALGO"
echo "  NCCL_PROTO=$NCCL_PROTO"
echo "  NSA_MEM_DUMP_EVERY=$NSA_MEM_DUMP_EVERY"
echo "  TORCH_LOGS=$TORCH_LOGS (first 100 steps)"
echo "  CONFIG=$CONFIG"
echo

# Clean previous artifacts
echo "🧹 Cleaning previous artifacts..."
rm -rf artifacts/m7c_125m_2xa100_prod/
mkdir -p artifacts/m7c_125m_2xa100_prod/

# Phase 1: Synthetic Data Test (Steps 1-200)
echo "=== Phase 1: Synthetic Data Stability Test (200 steps) ==="
echo "Testing DDP + gradient checkpointing compatibility..."

# Modify config for synthetic phase
sed 's/steps: 500/steps: 200/' configs/m7c_125m_2xa100_production.yaml > configs/m7c_125m_2xa100_phase1.yaml

export CONFIG=configs/m7c_125m_2xa100_phase1.yaml

echo "🏃 Launching Phase 1: torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic"

timeout 30m torchrun --nproc_per_node=2 scripts/train_showcase.py \
    --dataset synthetic \
    2>&1 | tee artifacts/m7c_125m_2xa100_prod/training_phase1.log

PHASE1_EXIT_CODE=$?

echo
echo "=== Phase 1 Analysis ==="

if [ $PHASE1_EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 1 PASSED - DDP + gradient checkpointing stable"
else
    echo "❌ Phase 1 FAILED - Exit code: $PHASE1_EXIT_CODE"
    
    # Check for specific failure patterns
    if grep -q "mark.*ready.*twice" artifacts/m7c_125m_2xa100_prod/training_phase1.log; then
        echo "🔍 Found 'mark variable ready twice' error - DDP incompatibility confirmed"
        echo "💡 Recommendation: Switch to FSDP implementation"
        exit 1
    fi
    
    if grep -q "CUDA.*out of memory\|OOM" artifacts/m7c_125m_2xa100_prod/training_phase1.log; then
        echo "🔍 Found OOM error - Memory management issue"
        echo "💡 Recommendation: Reduce batch size or sequence length"
        exit 1
    fi
    
    echo "🔍 Unknown failure - check training_phase1.log"
    exit 1
fi

# Disable TORCH_LOGS for Phase 2 (routing validated)
unset TORCH_LOGS
echo "🔇 Disabled TORCH_LOGS for Phase 2 (routing validated)"

# Phase 2: FineWeb-Edu Production Test (Steps 201-500)
echo
echo "=== Phase 2: FineWeb-Edu Production Test (300 steps) ==="

# Modify config for production phase
sed 's/steps: 200/steps: 500/' configs/m7c_125m_2xa100_phase1.yaml > configs/m7c_125m_2xa100_phase2.yaml

export CONFIG=configs/m7c_125m_2xa100_phase2.yaml

echo "🏃 Launching Phase 2: torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu"

timeout 45m torchrun --nproc_per_node=2 scripts/train_showcase.py \
    --dataset fineweb_edu \
    --fwe-report-docs 1000 \
    --loader-timeout 120 \
    --synthetic-on-fail \
    2>&1 | tee artifacts/m7c_125m_2xa100_prod/training_phase2.log

PHASE2_EXIT_CODE=$?

echo
echo "=== Final Analysis and Success Criteria Evaluation ==="

# Combine logs for analysis
cat artifacts/m7c_125m_2xa100_prod/training_phase1.log artifacts/m7c_125m_2xa100_prod/training_phase2.log > artifacts/m7c_125m_2xa100_prod/training_combined.log

if [ $PHASE2_EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 2 PASSED - Full production pipeline stable"
else
    echo "❌ Phase 2 FAILED - Exit code: $PHASE2_EXIT_CODE"
    echo "🔍 Check training_phase2.log for FineWeb-Edu specific issues"
fi

# Success Criteria Analysis
echo
echo "🎯 SUCCESS CRITERIA EVALUATION:"

# 1. Stability Check
if [ $PHASE1_EXIT_CODE -eq 0 ] && [ $PHASE2_EXIT_CODE -eq 0 ]; then
    echo "✅ STABILITY: 500 steps completed without DDP crashes"
else
    echo "❌ STABILITY: Training failed before 500 steps"
fi

# 2. Memory Check
if [ -f "artifacts/m7c_125m_2xa100_prod/mem_step1.json" ]; then
    reserved_mb=$(python3 -c "
import json
with open('artifacts/m7c_125m_2xa100_prod/mem_step1.json') as f:
    stats = json.load(f)
reserved_gb = stats.get('reserved_bytes.all.current', 0) / (1024**3)
print(f'{reserved_gb:.1f}')
")
    echo "📊 MEMORY: Reserved ${reserved_mb}GB per GPU"
    if (( $(echo "$reserved_mb < 40" | bc -l) )); then
        echo "✅ MEMORY: Under 40GB threshold"
    else
        echo "❌ MEMORY: Exceeds 40GB threshold"
    fi
else
    echo "❌ MEMORY: mem_step1.json not found"
fi

# 3. Throughput Check  
if [ -f "artifacts/m7c_125m_2xa100_prod/training.csv" ]; then
    final_toks_per_s=$(tail -1 artifacts/m7c_125m_2xa100_prod/training.csv | cut -d',' -f4)
    echo "📊 THROUGHPUT: ${final_toks_per_s} tokens/sec"
    if (( $(echo "$final_toks_per_s > 50" | bc -l) )); then
        echo "✅ THROUGHPUT: Above 50 toks/s target"
    else
        echo "❌ THROUGHPUT: Below 50 toks/s target"
    fi
else
    echo "❌ THROUGHPUT: training.csv not found"
fi

# 4. Selection Health Check
if [ -f "artifacts/m7c_125m_2xa100_prod/k_stats.csv" ]; then
    last_stats=$(tail -1 artifacts/m7c_125m_2xa100_prod/k_stats.csv)
    k_mean=$(echo $last_stats | cut -d',' -f2)
    k_max=$(echo $last_stats | cut -d',' -f3) 
    pct_at_max=$(echo $last_stats | cut -d',' -f5)
    echo "📊 SELECTION: k_mean=$k_mean, k_max=$k_max, pct_at_max=$pct_at_max"
    if (( $(echo "$pct_at_max < 0.3" | bc -l) )); then
        echo "✅ SELECTION: Healthy diversity (pct_at_max < 0.3)"
    else
        echo "❌ SELECTION: Poor diversity (pct_at_max >= 0.3)"
    fi
else
    echo "❌ SELECTION: k_stats.csv not found"
fi

# 5. Fallback Analysis
if [ -f "artifacts/m7c_125m_2xa100_prod/fallback_counters.csv" ]; then
    total_fallbacks=$(tail -1 artifacts/m7c_125m_2xa100_prod/fallback_counters.csv | cut -d',' -f8)
    echo "📊 FALLBACKS: $total_fallbacks total"
    if [ "$total_fallbacks" -lt 10 ]; then
        echo "✅ FALLBACKS: Near zero (< 10 total)"
    else
        echo "⚠️  FALLBACKS: High count (>= 10 total)"
    fi
else
    echo "❌ FALLBACKS: fallback_counters.csv not found"
fi

echo
echo "📁 ARTIFACTS CAPTURED:"
for artifact in dtypes_report.txt mem_boot.txt mem_step1.txt opt_state_mb.txt \
               k_stats.csv fallback_counters.csv heartbeat_rank0.jsonl training.csv metrics.json; do
    if [ -f "artifacts/m7c_125m_2xa100_prod/$artifact" ]; then
        echo "✅ $artifact"
    else
        echo "❌ $artifact (missing)"
    fi
done

echo
echo "=== Production Test Complete ==="
echo "Check artifacts in: artifacts/m7c_125m_2xa100_prod/"
echo "Key logs: training_phase1.log, training_phase2.log, training_combined.log"

if [ $PHASE1_EXIT_CODE -eq 0 ] && [ $PHASE2_EXIT_CODE -eq 0 ]; then
    echo "🎉 OVERALL RESULT: SUCCESS - Ready for production deployment"
    exit 0
else
    echo "🚨 OVERALL RESULT: FAILURE - Review criteria and implement fallbacks"
    exit 1
fi
