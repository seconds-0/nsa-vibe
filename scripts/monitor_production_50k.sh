#!/usr/bin/env bash
# Production 50K Training Monitor Script
# Tracks key metrics and alerts on issues

set -euo pipefail

OUT_DIR="artifacts/m7c_125m_2xa100_prod"

echo "================================="
echo "NSA 50K Production Run Monitor"
echo "================================="
echo "Output directory: $OUT_DIR"
echo ""

# Function to extract last value from CSV
get_last_metric() {
    local file=$1
    local col=$2
    if [ -f "$file" ]; then
        tail -1 "$file" | cut -d',' -f"$col"
    else
        echo "N/A"
    fi
}

# Function to calculate mean of last N values
get_mean_last_n() {
    local file=$1
    local col=$2
    local n=$3
    if [ -f "$file" ]; then
        tail -"$n" "$file" | cut -d',' -f"$col" | awk '{sum+=$1} END {printf "%.1f", sum/NR}'
    else
        echo "N/A"
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "NSA 50K Production Monitor - $(date)"
    echo "============================================"
    
    # GPU status
    echo ""
    echo "GPU STATUS:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv || echo "GPU info unavailable"
    
    # Training progress
    if [ -f "$OUT_DIR/training.csv" ]; then
        echo ""
        echo "TRAINING PROGRESS:"
        LAST_STEP=$(get_last_metric "$OUT_DIR/training.csv" 1)
        LAST_LOSS=$(get_last_metric "$OUT_DIR/training.csv" 2)
        LAST_LR=$(get_last_metric "$OUT_DIR/training.csv" 3)
        LAST_TOKS=$(get_last_metric "$OUT_DIR/training.csv" 4)
        MEAN_TOKS=$(get_mean_last_n "$OUT_DIR/training.csv" 4 20)
        
        echo "  Step:           $LAST_STEP / 50000"
        echo "  Loss:           $LAST_LOSS"
        echo "  Learning Rate:  $LAST_LR"
        echo "  Throughput:     $LAST_TOKS toks/s (last)"
        echo "  Mean (last 20): $MEAN_TOKS toks/s"
        
        # Progress bar
        if [ "$LAST_STEP" != "N/A" ] && [ "$LAST_STEP" -gt 0 ]; then
            PROGRESS=$((LAST_STEP * 100 / 50000))
            printf "  Progress:       ["
            for i in $(seq 1 50); do
                if [ $i -le $((PROGRESS / 2)) ]; then
                    printf "="
                else
                    printf " "
                fi
            done
            printf "] %d%%\n" $PROGRESS
        fi
    else
        echo ""
        echo "TRAINING PROGRESS: Waiting for data..."
    fi
    
    # Heartbeat telemetry (last entry)
    if [ -f "$OUT_DIR/heartbeat_rank0.jsonl" ]; then
        echo ""
        echo "HEARTBEAT (Last):"
        LAST_HB=$(tail -1 "$OUT_DIR/heartbeat_rank0.jsonl")
        
        # Extract key metrics using Python for JSON parsing
        python3 - <<EOF 2>/dev/null || echo "  Unable to parse heartbeat"
import json
data = json.loads('$LAST_HB')
print(f"  Memory Alloc:   {data.get('gpu_mem_alloc', 'N/A')} MiB")
print(f"  Memory Reserved: {data.get('gpu_mem_reserved', 'N/A')} MiB")
print(f"  Entropy Mean:   {data.get('entropy_mean', 'N/A'):.3f}" if isinstance(data.get('entropy_mean'), (int, float)) else "  Entropy Mean:   N/A")
print(f"  Max Gate Mean:  {data.get('max_gate_mean', 'N/A'):.3f}" if isinstance(data.get('max_gate_mean'), (int, float)) else "  Max Gate Mean:  N/A")
print(f"  Collapse Frac:  {data.get('collapse_fraction', 'N/A'):.3f}" if isinstance(data.get('collapse_fraction'), (int, float)) else "  Collapse Frac:  N/A")
print(f"  Data Fetch:     {data.get('dt_fetch_s', 'N/A'):.2f}s" if isinstance(data.get('dt_fetch_s'), (int, float)) else "  Data Fetch:     N/A")
EOF
    fi
    
    # Check for issues
    echo ""
    echo "HEALTH CHECKS:"
    
    # Throughput check
    if [ "$MEAN_TOKS" != "N/A" ]; then
        if (( $(echo "$MEAN_TOKS < 39" | bc -l) )); then
            echo "  ⚠️  WARNING: Throughput below target (39 toks/s)"
        else
            echo "  ✅ Throughput OK (≥39 toks/s)"
        fi
    fi
    
    # Check for .HALT file
    if [ -f "$OUT_DIR/.HALT" ]; then
        echo "  🛑 HALT FILE DETECTED - Training will stop"
    fi
    
    # Check for recent checkpoint
    LATEST_CKPT=$(ls -t "$OUT_DIR"/checkpoint_step*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "  ✅ Latest checkpoint: $(basename $LATEST_CKPT)"
    else
        echo "  ⚠️  No checkpoints saved yet"
    fi
    
    echo ""
    echo "============================================"
    echo "Press Ctrl+C to exit monitor"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done