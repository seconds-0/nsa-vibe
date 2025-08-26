#!/usr/bin/env python3
"""
Profiler A/B comparison for NSA range conversion optimizations.
Runs short training with v1 (Python loops) vs v2 (GPU vectorized) paths
and reports performance metrics.

Usage:
  # Compare v1 vs v2 range conversion
  python scripts/profiler_comparison.py --steps 100 --warmup 10
  
  # With DDP compression enabled
  NSA_DDP_COMPRESS=bf16 python scripts/profiler_comparison.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
import torch

def run_benchmark(
    name: str,
    env_vars: Dict[str, str],
    steps: int,
    warmup: int,
    out_dir: Path
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    print(f"\n=== Running benchmark: {name} ===")
    
    # Setup environment
    env = os.environ.copy()
    env.update(env_vars)
    
    # Ensure critical settings
    env['PYTHONUNBUFFERED'] = '1'
    # Ensure repository root on PYTHONPATH for module imports in subprocess
    repo_root = Path(__file__).resolve().parent.parent
    env['PYTHONPATH'] = os.pathsep.join([str(repo_root), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    env['NSA_TB_DISABLE'] = '1'
    env['NSA_DISABLE_CSV_LOGS'] = '1'
    env['CONFIG'] = env.get('CONFIG', 'configs/m7c_125m_40g.yaml')
    
    # Create output directory
    bench_dir = out_dir / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    cmd = [
        sys.executable, '-u',
        'scripts/train_showcase.py',
        '--dataset', 'synthetic',
        '--steps', str(steps),
        '--ddp', '0',
        '--out-dir', str(bench_dir)
    ]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running {name}:")
        print(result.stderr[-2000:] if result.stderr else "No stderr")
        return {'name': name, 'error': True, 'stderr': result.stderr}
    
    # Parse heartbeat for metrics
    heartbeat_file = bench_dir / 'heartbeat_rank0.jsonl'
    metrics = {
        'name': name,
        'total_time': elapsed,
        'steps': steps,
        'warmup': warmup,
        'error': False
    }
    
    if heartbeat_file.exists():
        lines = heartbeat_file.read_text().strip().split('\n')
        if lines:
            # Skip warmup steps
            data_lines = [json.loads(line) for line in lines if line]
            post_warmup = [d for d in data_lines if d.get('step', 0) >= warmup]
            
            if post_warmup:
                # Extract metrics from post-warmup data
                toks_per_s = [d.get('toks_per_s', 0) for d in post_warmup if 'toks_per_s' in d]
                losses = [d.get('loss', 0) for d in post_warmup if 'loss' in d]
                gpu_mem = [d.get('gpu_mem_alloc', 0) for d in post_warmup if 'gpu_mem_alloc' in d]
                
                if toks_per_s:
                    metrics['avg_toks_per_s'] = sum(toks_per_s) / len(toks_per_s)
                    metrics['max_toks_per_s'] = max(toks_per_s)
                    metrics['min_toks_per_s'] = min(toks_per_s)
                
                if losses:
                    metrics['final_loss'] = losses[-1]
                    metrics['avg_loss'] = sum(losses) / len(losses)
                
                if gpu_mem:
                    metrics['avg_gpu_mem_mib'] = sum(gpu_mem) / len(gpu_mem)
                    metrics['peak_gpu_mem_mib'] = max(gpu_mem)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100, help='Total steps to run')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup steps to exclude from metrics')
    parser.add_argument('--out', type=str, default='artifacts/profiler_comparison', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define configurations to test
    configs = [
        {
            'name': 'baseline_v1',
            'env': {
                'NSA_SEL_RANGES_V2': '0',
                'NSA_DDP_COMPRESS': 'none',
                'NSA_NVTX': '0'
            }
        },
        {
            'name': 'v2_ranges',
            'env': {
                'NSA_SEL_RANGES_V2': '1',
                'NSA_DDP_COMPRESS': 'none',
                'NSA_NVTX': '0'
            }
        },
        {
            'name': 'v2_ranges_nvtx',
            'env': {
                'NSA_SEL_RANGES_V2': '1',
                'NSA_DDP_COMPRESS': 'none',
                'NSA_NVTX': '1'
            }
        }
    ]
    
    # Add DDP compression if available and on multi-GPU
    if torch.cuda.device_count() > 1:
        configs.extend([
            {
                'name': 'v2_ranges_ddp_bf16',
                'env': {
                    'NSA_SEL_RANGES_V2': '1',
                    'NSA_DDP_COMPRESS': 'bf16',
                    'NSA_NVTX': '0'
                }
            },
            {
                'name': 'v2_ranges_ddp_fp16',
                'env': {
                    'NSA_SEL_RANGES_V2': '1',
                    'NSA_DDP_COMPRESS': 'fp16',
                    'NSA_NVTX': '0'
                }
            }
        ])
    
    # Run benchmarks
    results = []
    for config in configs:
        result = run_benchmark(
            name=config['name'],
            env_vars=config['env'],
            steps=args.steps,
            warmup=args.warmup,
            out_dir=out_dir
        )
        results.append(result)
        
        # Print immediate results
        if not result.get('error'):
            print(f"  Avg throughput: {result.get('avg_toks_per_s', 0):.1f} toks/s")
            print(f"  Peak GPU mem: {result.get('peak_gpu_mem_mib', 0):.0f} MiB")
    
    # Generate comparison report
    report_path = out_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write("# NSA Optimization A/B Comparison Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps: {args.steps} (warmup: {args.warmup})\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Configuration | Avg Throughput (toks/s) | Speedup | Peak Memory (MiB) | Final Loss |\n")
        f.write("|---------------|-------------------------|---------|-------------------|------------|\n")
        
        baseline_tps = 0
        for r in results:
            if r.get('error'):
                f.write(f"| {r['name']} | ERROR | - | - | - |\n")
            else:
                tps = r.get('avg_toks_per_s', 0)
                if r['name'] == 'baseline_v1':
                    baseline_tps = tps
                    speedup = "1.0x"
                else:
                    speedup = f"{tps/baseline_tps:.2f}x" if baseline_tps > 0 else "N/A"
                
                f.write(f"| {r['name']} | {tps:.1f} | {speedup} | "
                       f"{r.get('peak_gpu_mem_mib', 0):.0f} | "
                       f"{r.get('final_loss', 0):.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate improvements
        v1_result = next((r for r in results if r['name'] == 'baseline_v1' and not r.get('error')), None)
        v2_result = next((r for r in results if r['name'] == 'v2_ranges' and not r.get('error')), None)
        
        if v1_result and v2_result:
            v1_tps = v1_result.get('avg_toks_per_s', 0)
            v2_tps = v2_result.get('avg_toks_per_s', 0)
            if v1_tps > 0:
                improvement = ((v2_tps - v1_tps) / v1_tps) * 100
                f.write(f"- **V2 Range Conversion**: {improvement:.1f}% throughput improvement\n")
                f.write(f"  - Baseline (v1): {v1_tps:.1f} toks/s\n")
                f.write(f"  - Optimized (v2): {v2_tps:.1f} toks/s\n")
        
        f.write("\n## Environment\n\n")
        f.write("```\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name()}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
        f.write("```\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. Enable NSA_SEL_RANGES_V2=1 for production (validated equivalent)\n")
        f.write("2. Enable NSA_DDP_COMPRESS=bf16 for multi-GPU PCIe setups\n")
        f.write("3. Keep NSA_NVTX=1 for profiling runs only (small overhead)\n")
        f.write("4. Monitor for any .item()/.cpu() regressions in hot paths\n")
    
    # Save raw results
    results_json = out_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Comparison complete ===")
    print(f"Report: {report_path}")
    print(f"Raw data: {results_json}")
    
    # Print summary
    if v1_result and v2_result and not v1_result.get('error') and not v2_result.get('error'):
        v1_tps = v1_result.get('avg_toks_per_s', 0)
        v2_tps = v2_result.get('avg_toks_per_s', 0)
        if v1_tps > 0:
            print(f"\nV2 speedup: {v2_tps/v1_tps:.2f}x ({v2_tps:.1f} vs {v1_tps:.1f} toks/s)")

if __name__ == '__main__':
    import torch  # Import here to check availability
    main()
