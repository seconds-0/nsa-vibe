#!/usr/bin/env python3
"""
CLI wrapper for automated GPU benchmarking.

This script provides a user-friendly interface to run GPU benchmarks
across different providers and automatically update configurations.

Usage:
    python bench/run_automated_bench.py --provider modal --gpu T4
    python bench/run_automated_bench.py --provider local  # If you have a local GPU
    python bench/run_automated_bench.py --provider runpod --gpu A100 --api-key $RUNPOD_KEY
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Handle imports whether running from project root or bench directory
try:
    from bench.threshold_optimizer import ThresholdOptimizer
except ImportError:
    from threshold_optimizer import ThresholdOptimizer


class BenchmarkRunner:
    """Base class for benchmark runners."""
    
    def run(self, gpu_type: str = "T4") -> Dict:
        """Run benchmark and return results."""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if this runner is available."""
        raise NotImplementedError


class ModalRunner(BenchmarkRunner):
    """Run benchmarks using Modal."""
    
    def is_available(self) -> bool:
        """Check if Modal is installed and configured."""
        try:
            result = subprocess.run(
                ["modal", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                return False
            
            # Check for Modal tokens
            return (os.environ.get("MODAL_TOKEN_ID") and 
                   os.environ.get("MODAL_TOKEN_SECRET"))
        except FileNotFoundError:
            return False
    
    def run(self, gpu_type: str = "T4") -> Dict:
        """Run benchmark on Modal."""
        print(f"Starting Modal benchmark on {gpu_type}...")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = tmp.name
        
        try:
            # Run Modal benchmark
            cmd = [
                "modal", "run", "bench/modal_gpu_bench.py",
                "--gpu-type", gpu_type,
                "--output", output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running Modal benchmark: {result.stderr}")
                if "quota" in result.stderr.lower():
                    print("\nQuota exceeded. Please check your Modal usage limits.")
                    print("Visit https://modal.com/usage to view your current usage.")
                elif "payment" in result.stderr.lower() or "billing" in result.stderr.lower():
                    print("\nPayment method required for GPU usage.")
                    print("Please add a payment method at https://modal.com/settings/billing")
                elif "authentication" in result.stderr.lower():
                    print("\nAuthentication failed. Please run: modal token new")
                return {}
            
            # Load and return results
            if Path(output_file).exists():
                with open(output_file) as f:
                    return json.load(f)
            else:
                print(f"Output file not created. Check Modal logs.")
                return {}
        except FileNotFoundError:
            print("Modal CLI not found. Please install: pip install modal")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
        finally:
            # Clean up temp file
            if Path(output_file).exists():
                Path(output_file).unlink()


class LocalRunner(BenchmarkRunner):
    """Run benchmarks on local GPU."""
    
    def is_available(self) -> bool:
        """Check if local GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def run(self, gpu_type: str = "LOCAL") -> Dict:
        """Run benchmark locally."""
        print("Running benchmark on local GPU...")
        
        # Check environment
        env = os.environ.copy()
        env["NSA_USE_FA2"] = "1"
        env["PYTHONPATH"] = "."
        
        # Run parity tests
        print("Running parity tests...")
        parity_result = subprocess.run(
            ["python", "-m", "pytest", "-q", "-k", "fa2_gpu_varlen"],
            env={**env, "NSA_TEST_FA2": "1"},
            capture_output=True,
            text=True
        )
        
        # Run benchmarks
        print("Running benchmarks...")
        bench_result = subprocess.run(
            ["python", "bench/bench_fa2.py"],
            env=env,
            capture_output=True,
            text=True
        )
        
        # Parse output
        optimizer = ThresholdOptimizer()
        sliding, compressed = optimizer.parse_bench_output(bench_result.stdout)
        
        # Get device info
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        # Format results similar to Modal output
        results = []
        for r in sliding:
            results.append({
                "branch": "sliding",
                "sequence_length": r["S"],
                "window_size": r["w"],
                "speedup": r["speedup"],
                "masked_ms": r["masked_ms"],
                "fa2_ms": r["fa2_ms"],
            })
        
        for r in compressed:
            results.append({
                "branch": "compressed",
                "sequence_length": r["S"],
                "window_size": None,
                "speedup": r["speedup"],
                "masked_ms": r["masked_ms"],
                "fa2_ms": r["fa2_ms"],
            })
        
        # Determine thresholds
        optimizer = ThresholdOptimizer()
        data = type('BenchmarkData', (), {
            'sliding_results': sliding,
            'compressed_results': compressed
        })()
        win_threshold, cmp_threshold = optimizer.determine_thresholds(data)
        
        return {
            "device_info": {
                "device_name": device_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cuda_available": torch.cuda.is_available(),
            },
            "parity_passed": parity_result.returncode == 0,
            "benchmark_output": bench_result.stdout,
            "results": results,
            "recommendation": {
                "fa2_min_len_win": win_threshold,
                "fa2_min_len_cmp": cmp_threshold,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


class RunPodRunner(BenchmarkRunner):
    """Run benchmarks on RunPod (future implementation)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    
    def is_available(self) -> bool:
        """Check if RunPod API is configured."""
        return bool(self.api_key)
    
    def run(self, gpu_type: str = "L4") -> Dict:
        """Run benchmark on RunPod."""
        print(f"RunPod runner not yet implemented. Would run on {gpu_type}")
        # TODO: Implement RunPod API integration
        # This would:
        # 1. Create a serverless endpoint or pod
        # 2. Run the benchmark script
        # 3. Fetch results
        # 4. Clean up resources
        return {}


class LambdaCloudRunner(BenchmarkRunner):
    """Run benchmarks on Lambda Cloud (future implementation)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
    
    def is_available(self) -> bool:
        """Check if Lambda Cloud API is configured."""
        return bool(self.api_key)
    
    def run(self, gpu_type: str = "A10") -> Dict:
        """Run benchmark on Lambda Cloud."""
        print(f"Lambda Cloud runner not yet implemented. Would run on {gpu_type}")
        # TODO: Implement Lambda Cloud API integration
        return {}


def print_results(results: Dict) -> None:
    """Pretty print benchmark results."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    device_info = results.get("device_info", {})
    print(f"Device: {device_info.get('device_name', 'Unknown')}")
    print(f"CUDA: {device_info.get('cuda_version', 'N/A')}")
    print(f"PyTorch: {device_info.get('torch_version', 'N/A')}")
    
    if "parity_passed" in results:
        print(f"Parity Tests: {'PASSED' if results['parity_passed'] else 'FAILED'}")
    
    recommendation = results.get("recommendation", {})
    if recommendation:
        print("\nRecommended Thresholds:")
        print(f"  fa2_min_len_win: {recommendation.get('fa2_min_len_win', 'N/A')}")
        print(f"  fa2_min_len_cmp: {recommendation.get('fa2_min_len_cmp', 'N/A')}")
    
    # Print detailed results
    bench_results = results.get("results", [])
    if bench_results:
        sliding = [r for r in bench_results if r["branch"] == "sliding"]
        compressed = [r for r in bench_results if r["branch"] == "compressed"]
        
        if sliding:
            print("\nSliding Window Performance:")
            print("  S    w    Speedup  Masked(ms)  FA2(ms)")
            print("  " + "-"*45)
            for r in sorted(sliding, key=lambda x: (x["sequence_length"], x["window_size"])):
                print(f"  {r['sequence_length']:4} {r['window_size']:4} "
                      f"{r['speedup']:7.2f}x {r['masked_ms']:10.2f} {r['fa2_ms']:8.2f}")
        
        if compressed:
            print("\nCompressed Performance:")
            print("  S    Speedup  Masked(ms)  FA2(ms)")
            print("  " + "-"*37)
            for r in sorted(compressed, key=lambda x: x["sequence_length"]):
                print(f"  {r['sequence_length']:4} {r['speedup']:7.2f}x "
                      f"{r['masked_ms']:10.2f} {r['fa2_ms']:8.2f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated GPU benchmarking for NSA FA-2 thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on Modal with T4 GPU
  %(prog)s --provider modal --gpu T4
  
  # Run on local GPU
  %(prog)s --provider local
  
  # Run and update config
  %(prog)s --provider modal --gpu L4 --update-config
  
  # Run with custom safety margin
  %(prog)s --provider modal --gpu A100 --safety-margin 1.5
  
  # Save results to file
  %(prog)s --provider modal --gpu T4 --output results.json
"""
    )
    
    parser.add_argument(
        "--provider",
        choices=["modal", "local", "runpod", "lambda"],
        default="modal",
        help="Compute provider for benchmarking"
    )
    
    parser.add_argument(
        "--gpu",
        choices=["T4", "L4", "A10", "A100", "H100", "LOCAL"],
        default="T4",
        help="GPU type to use (ignored for local provider)"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Automatically update configs/base.yaml with new thresholds"
    )
    
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.2,
        help="Minimum speedup to enable FA-2 (default: 1.2 = 20%%)"
    )
    
    parser.add_argument(
        "--report",
        help="Generate markdown report file"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for cloud provider (can also use env vars)"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare results from multiple JSON files"
    )
    
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        print("Comparison mode:")
        print("="*60)
        
        for file_path in args.compare:
            if not Path(file_path).exists():
                print(f"File not found: {file_path}")
                continue
            
            with open(file_path) as f:
                data = json.load(f)
            
            device = data.get("device_info", {}).get("device_name", "Unknown")
            rec = data.get("recommendation", {})
            print(f"\n{Path(file_path).name}:")
            print(f"  Device: {device}")
            print(f"  fa2_min_len_win: {rec.get('fa2_min_len_win', 'N/A')}")
            print(f"  fa2_min_len_cmp: {rec.get('fa2_min_len_cmp', 'N/A')}")
        
        return 0
    
    # Select runner based on provider
    runners = {
        "modal": ModalRunner(),
        "local": LocalRunner(),
        "runpod": RunPodRunner(args.api_key),
        "lambda": LambdaCloudRunner(args.api_key),
    }
    
    runner = runners.get(args.provider)
    if not runner:
        print(f"Unknown provider: {args.provider}")
        return 1
    
    # Check if runner is available
    if not runner.is_available():
        print(f"Provider '{args.provider}' is not available.")
        print("Please check:")
        if args.provider == "modal":
            print("  - Modal is installed: pip install modal")
            print("  - Modal tokens are set: MODAL_TOKEN_ID, MODAL_TOKEN_SECRET")
        elif args.provider == "local":
            print("  - PyTorch is installed with CUDA support")
            print("  - A GPU is available on this machine")
        elif args.provider in ["runpod", "lambda"]:
            print(f"  - API key is set: {args.provider.upper()}_API_KEY")
        return 1
    
    # Run benchmark
    print(f"Running benchmark on {args.provider} with {args.gpu}...")
    start_time = time.time()
    
    try:
        results = runner.run(gpu_type=args.gpu)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1
    
    elapsed = time.time() - start_time
    print(f"Benchmark completed in {elapsed:.1f} seconds")
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    # Generate report if requested
    if args.report and results:
        optimizer = ThresholdOptimizer(safety_margin=args.safety_margin)
        
        # Convert results to BenchmarkData format
        sliding = []
        compressed = []
        for r in results.get("results", []):
            if r["branch"] == "sliding":
                sliding.append({
                    "S": r["sequence_length"],
                    "w": r["window_size"],
                    "speedup": r["speedup"],
                    "masked_ms": r["masked_ms"],
                    "fa2_ms": r["fa2_ms"],
                })
            else:
                compressed.append({
                    "S": r["sequence_length"],
                    "speedup": r["speedup"],
                    "masked_ms": r["masked_ms"],
                    "fa2_ms": r["fa2_ms"],
                })
        
        from types import SimpleNamespace
        data = SimpleNamespace(
            device=results["device_info"]["device_name"],
            torch_version=results["device_info"]["torch_version"],
            cuda_version=results["device_info"]["cuda_version"],
            sliding_results=sliding,
            compressed_results=compressed,
        )
        
        recommendation = results.get("recommendation", {})
        report = optimizer.generate_report(
            data,
            recommendation.get("fa2_min_len_win", 512),
            recommendation.get("fa2_min_len_cmp", 32)
        )
        
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.report}")
    
    # Update config if requested
    if args.update_config and results.get("recommendation"):
        optimizer = ThresholdOptimizer(safety_margin=args.safety_margin)
        recommendation = results["recommendation"]
        
        config_path = Path("configs/base.yaml")
        if config_path.exists():
            optimizer.update_config(
                config_path,
                recommendation["fa2_min_len_win"],
                recommendation["fa2_min_len_cmp"]
            )
            print(f"\nConfig updated: {config_path}")
            print(f"  fa2_min_len_win: {recommendation['fa2_min_len_win']}")
            print(f"  fa2_min_len_cmp: {recommendation['fa2_min_len_cmp']}")
        else:
            print(f"Config file not found: {config_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())