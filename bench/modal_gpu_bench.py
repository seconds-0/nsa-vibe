"""
Modal-based GPU benchmarking for NSA FlashAttention-2 thresholds.

This script provisions GPU containers on-demand, runs benchmarks, and returns
optimized thresholds for FA-2 sliding and compressed branches.

Usage:
    modal run bench/modal_gpu_bench.py [--gpu-type L4]
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import modal

# Modal app configuration
app = modal.App("nsa-gpu-bench")

# Container image with CUDA and PyTorch
gpu_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.06-py3", add_python="3.10")
    .apt_install("git", "curl")
    .pip_install("pyyaml")  # Required for parsing configs
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=/root/.cargo/bin:$PATH' >> ~/.bashrc",
    )
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""
    device: str
    sequence_length: int
    window_size: Optional[int]
    masked_time_ms: float
    fa2_time_ms: float
    speedup: float
    branch: str  # "sliding" or "compressed"


@dataclass
class ThresholdRecommendation:
    """Recommended thresholds based on benchmark results."""
    fa2_min_len_win: int
    fa2_min_len_cmp: int
    device_name: str
    torch_version: str
    cuda_version: str
    results: List[BenchmarkResult]
    

def parse_benchmark_output(output: str) -> List[BenchmarkResult]:
    """Parse benchmark output into structured results."""
    results = []
    
    # Parse sliding results: "S=256 w=32 sliding masked X.XX ms  fa2 Y.YY ms  speedup xZ.ZZ"
    sliding_pattern = r"S=(\d+) w=(\d+) sliding masked ([\d.]+) ms\s+fa2 ([\d.]+) ms\s+speedup x([\d.]+)"
    for match in re.finditer(sliding_pattern, output):
        s, w, masked_ms, fa2_ms, speedup = match.groups()
        results.append(BenchmarkResult(
            device="cuda",
            sequence_length=int(s),
            window_size=int(w),
            masked_time_ms=float(masked_ms),
            fa2_time_ms=float(fa2_ms),
            speedup=float(speedup),
            branch="sliding"
        ))
    
    # Parse compressed results: "S=256 l=32 d=16 compressed masked X.XX ms  fa2 Y.YY ms  speedup xZ.ZZ"
    compressed_pattern = r"S=(\d+) l=\d+ d=\d+ compressed masked ([\d.]+) ms\s+fa2 ([\d.]+) ms\s+speedup x([\d.]+)"
    for match in re.finditer(compressed_pattern, output):
        s, masked_ms, fa2_ms, speedup = match.groups()
        results.append(BenchmarkResult(
            device="cuda",
            sequence_length=int(s),
            window_size=None,
            masked_time_ms=float(masked_ms),
            fa2_time_ms=float(fa2_ms),
            speedup=float(speedup),
            branch="compressed"
        ))
    
    return results


def determine_thresholds(results: List[BenchmarkResult], safety_margin: float = 1.1) -> Tuple[int, int]:
    """
    Determine optimal FA-2 thresholds from benchmark results.
    
    Args:
        results: List of benchmark results
        safety_margin: Minimum speedup required (default 1.1 = 10% faster)
    
    Returns:
        (fa2_min_len_win, fa2_min_len_cmp) thresholds
    """
    # Find minimum window size where FA-2 consistently beats masked
    sliding_results = [r for r in results if r.branch == "sliding"]
    win_threshold = 512  # Conservative default
    
    for w in sorted(set(r.window_size for r in sliding_results if r.window_size)):
        w_results = [r for r in sliding_results if r.window_size == w]
        if all(r.speedup >= safety_margin for r in w_results):
            win_threshold = w
            break
    
    # For compressed, find minimum effective length
    compressed_results = [r for r in results if r.branch == "compressed"]
    cmp_threshold = 32  # Conservative default
    
    # Group by sequence length and check speedups
    for s in sorted(set(r.sequence_length for r in compressed_results)):
        s_results = [r for r in compressed_results if r.sequence_length == s]
        if all(r.speedup >= safety_margin for r in s_results):
            # Map sequence length to effective compressed blocks
            # With l=32, d=16: num_cmp = floor((s+1-32)/16)+1
            if s >= 256:
                cmp_threshold = 16  # Aggressive threshold for larger sequences
                break
    
    return win_threshold, cmp_threshold


@app.function(
    image=gpu_image,
    gpu=modal.gpu.T4(),  # Default to T4, can be overridden
    timeout=900,  # 15 minutes
    retries=2,  # Retry once on failure
)
def run_gpu_benchmark(gpu_type: str = "T4") -> Dict:
    """
    Run GPU benchmarks in Modal container.
    
    Args:
        gpu_type: GPU type to request (T4, L4, A10, A100, etc.)
    
    Returns:
        Dictionary with benchmark results and recommendations
    """
    import torch
    
    # Clone repository (or update if exists)
    if not os.path.exists("nsa-vibe"):
        subprocess.run(["git", "clone", "https://github.com/seconds-0/nsa-vibe.git"], check=True)
    os.chdir("nsa-vibe")
    
    # Set up environment
    subprocess.run(["/root/.cargo/bin/uv", "venv", "-p", "3.10", ".venv"], check=True)
    subprocess.run(["/root/.cargo/bin/uv", "pip", "sync", "-r", "requirements.txt"], check=True)
    
    # Install flash-attn if needed
    try:
        # Try to install flash-attn for the current CUDA version
        cuda_version = torch.version.cuda.replace(".", "")[:3]  # e.g., "121" for 12.1
        subprocess.run([
            "./.venv/bin/pip", "install", "--no-deps",
            f"--index-url", f"https://download.pytorch.org/whl/cu{cuda_version}",
            "flash-attn>=2.0"
        ], check=False)  # Don't fail if install doesn't work
    except:
        pass
    
    # Verify GPU and FA-2 availability
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    # Check flash-attn availability
    fa2_check = subprocess.run([
        "./.venv/bin/python", "-c",
        "from nsa.kernels.flash_wrappers import is_flash_available; print(is_flash_available())"
    ], capture_output=True, text=True)
    device_info["flash_available"] = "True" in fa2_check.stdout
    
    if not device_info["cuda_available"]:
        return {"error": "No GPU available", "device_info": device_info}
    
    if not device_info.get("flash_available", False):
        print("Warning: FlashAttention-2 not available, attempting to continue...")
    
    # Run parity tests
    print("Running GPU parity tests...")
    parity_result = subprocess.run([
        "NSA_TEST_FA2=1", "PYTHONPATH=.", "./.venv/bin/python", "-m", "pytest",
        "-q", "-k", "fa2_gpu_varlen"
    ], capture_output=True, text=True, shell=True)
    
    # Run benchmarks
    print("Running benchmarks...")
    bench_env = os.environ.copy()
    bench_env["NSA_USE_FA2"] = "1"
    bench_env["PYTHONPATH"] = "."
    
    bench_result = subprocess.run([
        "./.venv/bin/python", "bench/bench_fa2.py"
    ], env=bench_env, capture_output=True, text=True)
    
    # Parse results
    results = parse_benchmark_output(bench_result.stdout)
    win_threshold, cmp_threshold = determine_thresholds(results)
    
    recommendation = ThresholdRecommendation(
        fa2_min_len_win=win_threshold,
        fa2_min_len_cmp=cmp_threshold,
        device_name=device_info["device_name"],
        torch_version=device_info["torch_version"],
        cuda_version=device_info["cuda_version"],
        results=results
    )
    
    return {
        "device_info": device_info,
        "parity_passed": parity_result.returncode == 0,
        "parity_output": parity_result.stdout,
        "benchmark_output": bench_result.stdout,
        "recommendation": {
            "fa2_min_len_win": recommendation.fa2_min_len_win,
            "fa2_min_len_cmp": recommendation.fa2_min_len_cmp,
        },
        "results": [
            {
                "branch": r.branch,
                "sequence_length": r.sequence_length,
                "window_size": r.window_size,
                "speedup": r.speedup,
                "masked_ms": r.masked_time_ms,
                "fa2_ms": r.fa2_time_ms,
            }
            for r in results
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


@app.local_entrypoint()
def main(gpu_type: str = "T4", output_file: Optional[str] = None):
    """
    Entry point for Modal app execution.
    
    Args:
        gpu_type: Type of GPU to use (T4, L4, A10, A100, H100)
        output_file: Optional JSON file to save results
    """
    print(f"Starting GPU benchmark on {gpu_type}...")
    
    # Map GPU types to Modal GPU objects
    gpu_map = {
        "T4": modal.gpu.T4,
        "L4": modal.gpu.L4,
        "A10": modal.gpu.A10G,
        "A100": modal.gpu.A100,
        "H100": modal.gpu.H100,
    }
    
    if gpu_type not in gpu_map:
        print(f"Unknown GPU type: {gpu_type}. Available: {list(gpu_map.keys())}")
        sys.exit(1)
    
    # Update the function's GPU configuration
    global run_gpu_benchmark
    run_gpu_benchmark = run_gpu_benchmark.with_options(gpu=gpu_map[gpu_type]())
    
    # Run the benchmark
    with app.run():
        result = run_gpu_benchmark.remote(gpu_type=gpu_type)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Device: {result['device_info']['device_name']}")
    print(f"CUDA: {result['device_info']['cuda_version']}")
    print(f"PyTorch: {result['device_info']['torch_version']}")
    print(f"Flash-Attn Available: {result['device_info']['flash_available']}")
    print(f"Parity Tests: {'PASSED' if result['parity_passed'] else 'FAILED'}")
    print("\nRecommended Thresholds:")
    print(f"  fa2_min_len_win: {result['recommendation']['fa2_min_len_win']}")
    print(f"  fa2_min_len_cmp: {result['recommendation']['fa2_min_len_cmp']}")
    
    # Print detailed results
    print("\nDetailed Results:")
    sliding_results = [r for r in result['results'] if r['branch'] == 'sliding']
    if sliding_results:
        print("\n  Sliding Window:")
        for r in sliding_results:
            print(f"    S={r['sequence_length']:4} w={r['window_size']:3}: "
                  f"speedup {r['speedup']:.2f}x "
                  f"(masked {r['masked_ms']:.2f}ms vs fa2 {r['fa2_ms']:.2f}ms)")
    
    compressed_results = [r for r in result['results'] if r['branch'] == 'compressed']
    if compressed_results:
        print("\n  Compressed:")
        for r in compressed_results:
            print(f"    S={r['sequence_length']:4}: "
                  f"speedup {r['speedup']:.2f}x "
                  f"(masked {r['masked_ms']:.2f}ms vs fa2 {r['fa2_ms']:.2f}ms)")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Print config update snippet
    print("\n" + "="*60)
    print("CONFIG UPDATE")
    print("="*60)
    print("Add to configs/base.yaml:")
    print(f"""
runtime:
  fa2_min_len_win: {result['recommendation']['fa2_min_len_win']}
  fa2_min_len_cmp: {result['recommendation']['fa2_min_len_cmp']}
""")
    
    return result


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU benchmarks on Modal")
    parser.add_argument("--gpu-type", default="T4", choices=["T4", "L4", "A10", "A100", "H100"],
                        help="GPU type to use for benchmarking")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    args = parser.parse_args()
    
    main(gpu_type=args.gpu_type, output_file=args.output)