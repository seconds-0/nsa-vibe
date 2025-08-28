#!/bin/bash
# Monitor runbook test progress

echo "=== Runbook Test Progress Monitor ==="
echo "Branch: feat/nsa-selection-varlen-packing"
echo "Commit: bb1c3bbe95a7"
echo ""

while true; do
    echo "=== Status at $(date) ==="
    
    # Check which test is currently running
    current_test=$(tmux capture-pane -t runbook -p 2>/dev/null | grep "=== Test" | tail -1)
    echo "Current: $current_test"
    
    # Check progress of each log file
    for f in run_s512_adaptive.log run_s512_v1.log run_s512_v2.log \
             run_s1024_adaptive.log run_s1024_v1.log run_s1024_v2.log \
             run_s2048_adaptive.log run_s2048_v1.log run_s2048_v2.log \
             run_ddp2_s2048_adaptive.log; do
        if [ -f "$f" ]; then
            steps=$(grep -c "^step" "$f" 2>/dev/null || echo "0")
            if [ "$steps" -gt 0 ]; then
                last_line=$(grep "^step" "$f" | tail -1)
                echo "$f: $steps/200 steps - $last_line"
            else
                echo "$f: starting..."
            fi
        fi
    done
    
    # Check GPU utilization
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    
    echo "========================================="
    sleep 300  # Check every 5 minutes
done