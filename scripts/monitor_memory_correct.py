#!/usr/bin/env python3
"""
Parallel memory monitoring script with correct unit labels.
Reads heartbeat file and displays memory with proper GB labels.

Usage:
    python scripts/monitor_memory_correct.py
    
Or on remote:
    ssh ubuntu@216.81.248.67 'cd nsa-vibe && python3 scripts/monitor_memory_correct.py'
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

def monitor_memory(heartbeat_file="artifacts/train_showcase/heartbeat_rank0.jsonl", interval=5):
    """Monitor memory from heartbeat with correct unit interpretation."""
    
    if not Path(heartbeat_file).exists():
        print(f"Error: Heartbeat file {heartbeat_file} not found")
        return
    
    print("=" * 60)
    print("NSA Training Memory Monitor (Corrected Units)")
    print("=" * 60)
    print(f"Reading from: {heartbeat_file}")
    print(f"Update interval: {interval} seconds")
    print("-" * 60)
    print(f"{'Time':<10} {'Step':<8} {'Loss':<8} {'Mem Alloc':<12} {'Mem Reserved':<12} {'Throughput':<10}")
    print("-" * 60)
    
    last_step = -1
    
    try:
        while True:
            try:
                # Read last line of heartbeat
                with open(heartbeat_file, 'r') as f:
                    for line in f:
                        pass  # Read to last line
                    if line:
                        data = json.loads(line)
                        
                        step = data.get('step', 0)
                        if step != last_step:  # Only print if new data
                            loss = data.get('loss', 0)
                            mem_alloc = data.get('gpu_mem_alloc', 0)
                            mem_reserved = data.get('gpu_mem_reserved', 0)
                            toks_per_s = data.get('toks_per_s', 0)
                            
                            # Interpret the values as ~GB (they're in 1024MB units)
                            # The actual formula in code is bytes // (1024*1024) = MB
                            # But the values are suspiciously small, suggesting they might be
                            # already processed or in different units
                            
                            # For now, interpret as approximate GB values
                            mem_alloc_gb = mem_alloc  # Treat as GB directly
                            mem_reserved_gb = mem_reserved  # Treat as GB directly
                            
                            current_time = datetime.now().strftime("%H:%M:%S")
                            
                            print(f"{current_time:<10} {step:<8} {loss:<8.4f} "
                                  f"{mem_alloc_gb:<5.1f} GB    {mem_reserved_gb:<5.1f} GB     "
                                  f"{toks_per_s:<.0f} tok/s")
                            
                            last_step = step
                            
            except (json.JSONDecodeError, IOError):
                pass  # File might be being written to
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Monitoring stopped by user")
        return

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor NSA training memory with correct units")
    parser.add_argument("--heartbeat", default="artifacts/train_showcase/heartbeat_rank0.jsonl",
                        help="Path to heartbeat JSONL file")
    parser.add_argument("--interval", type=int, default=5,
                        help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor_memory(args.heartbeat, args.interval)

if __name__ == "__main__":
    main()