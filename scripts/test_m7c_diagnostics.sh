#!/bin/bash

# NSA M7C Comprehensive Diagnostic Test Script
# Tests DDP + gradient checkpointing compatibility with enhanced diagnostics

set -e

echo "=== NSA M7C Diagnostic Test Suite ==="
echo "Testing DDP + gradient checkpointing with comprehensive diagnostics"
echo "Timestamp: $(date)"
echo

# Common diagnostic environment variables
export NSA_USE_FA2=1
export NSA_SDPA_FLASH_ONLY=1  # Force flash attention where possible
export NSA_MEM_DUMP_EVERY=20  # Memory dumps every 20 steps
export NSA_LOG_GRAD_NORM=1    # Log gradient norms
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONPATH=.

# Clean previous test artifacts
echo "Cleaning previous test artifacts..."
rm -rf artifacts/m7c_125m_2k_test/
mkdir -p artifacts/m7c_125m_2k_test/

echo "=== Phase 1: Single-GPU Sanity Test ==="
echo "Testing gradient checkpointing on single GPU with synthetic data..."

CUDA_VISIBLE_DEVICES=0 python -u scripts/train_showcase.py \
    --config configs/m7c_125m_2k_test.yaml \
    --dataset synthetic \
    --ddp 0 \
    2>&1 | tee artifacts/m7c_125m_2k_test/training_single.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Single-GPU test PASSED"
    # Check memory usage
    if [ -f "artifacts/m7c_125m_2k_test/mem_step1.txt" ]; then
        echo "📊 Memory usage after step 1:"
        grep -E "Reserved memory|Allocated memory" artifacts/m7c_125m_2k_test/mem_step1.txt | head -2
    fi
else
    echo "❌ Single-GPU test FAILED - aborting multi-GPU test"
    exit 1
fi

echo
echo "=== Phase 2: Multi-GPU Distributed Test ==="
echo "Testing DDP + gradient checkpointing with no_sync() fix..."

# Clean artifacts for multi-GPU test
rm -rf artifacts/m7c_125m_2k_test/*
mkdir -p artifacts/m7c_125m_2k_test/

export CONFIG=configs/m7c_125m_2k_test.yaml
export TORCH_LOGS="+sdp"  # Log SDPA backend decisions

echo "Environment variables:"
echo "  CONFIG=$CONFIG"
echo "  NSA_USE_FA2=$NSA_USE_FA2"
echo "  NSA_SDPA_FLASH_ONLY=$NSA_SDPA_FLASH_ONLY"
echo "  NSA_MEM_DUMP_EVERY=$NSA_MEM_DUMP_EVERY"
echo "  TORCH_LOGS=$TORCH_LOGS"
echo

torchrun --nproc_per_node=2 scripts/train_showcase.py \
    --dataset synthetic \
    2>&1 | tee artifacts/m7c_125m_2k_test/training_multi.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Multi-GPU test PASSED - DDP + gradient checkpointing working!"
else
    echo "❌ Multi-GPU test FAILED"
    echo "Checking for specific DDP errors..."
    if grep -q "mark.*ready.*twice" artifacts/m7c_125m_2k_test/training_multi.log; then
        echo "🔍 Found 'mark variable ready twice' error - DDP incompatibility persists"
    fi
    if grep -q "SIGABRT\|Signal 6" artifacts/m7c_125m_2k_test/training_multi.log; then
        echo "🔍 Found SIGABRT - potential data loader or memory issue"
    fi
fi

echo
echo "=== Diagnostic Artifact Summary ==="
echo "Artifacts written to: artifacts/m7c_125m_2k_test/"

for artifact in dtypes_report.txt mem_boot.txt mem_step1.txt opt_state_mb.txt \
               k_stats.csv fallback_counters.csv heartbeat_rank0.jsonl training.csv; do
    if [ -f "artifacts/m7c_125m_2k_test/$artifact" ]; then
        echo "✅ $artifact"
    else
        echo "❌ $artifact (missing)"
    fi
done

echo
echo "=== Quick Analysis ==="

# Memory analysis
if [ -f "artifacts/m7c_125m_2k_test/mem_step1.txt" ]; then
    echo "📊 Memory Usage (step 1):"
    grep -E "Reserved memory|Allocated memory" artifacts/m7c_125m_2k_test/mem_step1.txt | head -2
fi

# Optimizer footprint
if [ -f "artifacts/m7c_125m_2k_test/opt_state_mb.txt" ]; then
    echo "📊 Optimizer State: $(cat artifacts/m7c_125m_2k_test/opt_state_mb.txt)MB"
fi

# Performance check
if [ -f "artifacts/m7c_125m_2k_test/training.csv" ]; then
    echo "📊 Final Performance:"
    tail -1 artifacts/m7c_125m_2k_test/training.csv | awk -F',' '{printf "  Loss: %.4f, Tokens/sec: %.0f\n", $2, $4}'
fi

# Gate health (from heartbeat)
if [ -f "artifacts/m7c_125m_2k_test/heartbeat_rank0.jsonl" ]; then
    echo "📊 Gate Health (last step):"
    tail -1 artifacts/m7c_125m_2k_test/heartbeat_rank0.jsonl | jq -r '
        if .gate_entropy_mean then 
            "  Gate Entropy: " + (.gate_entropy_mean | tostring) + " (healthy > 0.5)"
        else 
            "  Gate stats not available"
        end'
fi

# Fallback analysis
if [ -f "artifacts/m7c_125m_2k_test/fallback_counters.csv" ]; then
    total_fallbacks=$(tail -1 artifacts/m7c_125m_2k_test/fallback_counters.csv | cut -d',' -f8)
    echo "📊 Total Fallbacks: $total_fallbacks"
fi

echo
echo "=== Diagnostic Test Complete ==="
echo "Review artifacts in: artifacts/m7c_125m_2k_test/"
echo "Key files:"
echo "  - training_single.log / training_multi.log: Full training logs"  
echo "  - heartbeat_rank0.jsonl: Real-time telemetry"
echo "  - mem_*.txt: Memory snapshots"
echo "  - fallback_counters.csv: Selection/FA2 fallback tracking"
echo "  - k_stats.csv: Selection statistics"