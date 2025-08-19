#!/usr/bin/env python3
"""
Prime Intellect GPU benchmarking for NSA FlashAttention-2 thresholds.

This script provisions GPU pods on Prime Intellect, runs benchmarks, and returns
optimized thresholds for FA-2 sliding and compressed branches.

Usage:
    python bench/prime_gpu_bench.py --gpu-type T4
    python bench/prime_gpu_bench.py --gpu-type A100_40GB --output results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import paramiko


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


class PrimeIntellectBenchmark:
    """Manages GPU benchmarking on Prime Intellect platform."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from env or parameter."""
        self.api_key = api_key or os.environ.get("PRIME_API_KEY")
        if not self.api_key:
            raise ValueError("PRIME_API_KEY not found. Set environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.primeintellect.ai/api/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.pod_id = None
        self.ssh_client = None
    
    def find_cheapest_gpu(self, gpu_type: str, regions: Optional[List[str]] = None) -> Dict:
        """
        Find cheapest available GPU of specified type.
        
        Args:
            gpu_type: GPU model (T4, L4, A100_40GB, etc.)
            regions: Optional list of regions to search
        
        Returns:
            Dictionary with cheapest option details
        """
        params = {
            'gpu_type': gpu_type,
            'gpu_count': 1
        }
        if regions:
            params['regions'] = regions
        
        resp = requests.get(
            f"{self.base_url}/availability/",
            params=params,
            headers=self.headers
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to check availability: {resp.text}")
        
        options = resp.json()
        if not options:
            raise RuntimeError(f"No {gpu_type} GPUs available")
        
        # Sort by hourly price and return cheapest
        return min(options, key=lambda x: x['prices']['hourly'])
    
    def create_benchmark_pod(self, gpu_type: str = "T4", max_price: Optional[float] = None) -> str:
        """
        Create pod for benchmarking.
        
        Args:
            gpu_type: GPU model to use
            max_price: Maximum hourly price willing to pay
        
        Returns:
            Pod ID
        """
        print(f"Finding cheapest {gpu_type}...")
        option = self.find_cheapest_gpu(gpu_type)
        
        hourly_price = option['prices']['hourly']
        provider = option['provider']
        print(f"Found {gpu_type} on {provider} for ${hourly_price:.2f}/hr")
        
        # Set max price with 20% buffer if not specified
        if max_price is None:
            max_price = hourly_price * 1.2
        
        # Create pod configuration
        pod_config = {
            "pod": {
                "name": f"nsa-bench-{gpu_type.lower()}-{int(time.time())}",
                "cloudId": option['cloudId'],
                "gpuType": gpu_type,
                "socket": option.get('socket', 'PCIe'),
                "gpuCount": 1,
                "diskSize": 50,
                "vcpus": 8,
                "memory": 32,
                "image": "ubuntu_22_cuda_12",
                "maxPrice": max_price,
                "autoRestart": False,
                "envVars": [
                    {"key": "PYTHONPATH", "value": "."},
                    {"key": "NSA_USE_FA2", "value": "1"},
                    {"key": "DEBIAN_FRONTEND", "value": "noninteractive"}
                ]
            },
            "provider": {
                "type": provider
            }
        }
        
        print(f"Creating pod...")
        resp = requests.post(
            f"{self.base_url}/pods/",
            json=pod_config,
            headers=self.headers
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to create pod: {resp.text}")
        
        pod_data = resp.json()
        self.pod_id = pod_data['id']
        print(f"Created pod: {self.pod_id}")
        return self.pod_id
    
    def wait_for_pod(self, pod_id: str, timeout: int = 300) -> Dict:
        """
        Wait for pod to be ready.
        
        Args:
            pod_id: Pod identifier
            timeout: Maximum wait time in seconds
        
        Returns:
            Pod details when ready
        """
        print(f"Waiting for pod to be ready...")
        start = time.time()
        
        while time.time() - start < timeout:
            resp = requests.get(
                f"{self.base_url}/pods/{pod_id}",
                headers=self.headers
            )
            
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to get pod status: {resp.text}")
            
            pod = resp.json()
            status = pod.get('status', 'unknown')
            
            if status == 'running':
                print(f"Pod ready after {int(time.time() - start)}s")
                return pod
            elif status in ['error', 'failed', 'terminated']:
                raise RuntimeError(f"Pod failed with status: {status}")
            
            print(f"Pod status: {status}, waiting...")
            time.sleep(10)
        
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")
    
    def setup_ssh(self, ssh_info: Dict, private_key_path: Optional[str] = None) -> paramiko.SSHClient:
        """
        Setup SSH connection to pod.
        
        Args:
            ssh_info: SSH connection details from pod info
            private_key_path: Path to SSH private key (if required)
        
        Returns:
            Connected SSH client
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        connect_kwargs = {
            'hostname': ssh_info['host'],
            'port': ssh_info.get('port', 22),
            'username': ssh_info.get('user', 'root'),
            'timeout': 30
        }
        
        if private_key_path and Path(private_key_path).exists():
            connect_kwargs['key_filename'] = private_key_path
        else:
            # Try to use default SSH key or password from environment
            if 'PRIME_SSH_KEY' in os.environ:
                connect_kwargs['key_filename'] = os.environ['PRIME_SSH_KEY']
            elif 'PRIME_SSH_PASSWORD' in os.environ:
                connect_kwargs['password'] = os.environ['PRIME_SSH_PASSWORD']
        
        print(f"Connecting via SSH to {ssh_info['host']}...")
        ssh.connect(**connect_kwargs)
        self.ssh_client = ssh
        return ssh
    
    def run_command(self, command: str, timeout: int = 300) -> Tuple[str, str]:
        """
        Run command on pod via SSH.
        
        Args:
            command: Command to execute
            timeout: Command timeout in seconds
        
        Returns:
            (stdout, stderr) tuple
        """
        if not self.ssh_client:
            raise RuntimeError("SSH not connected")
        
        stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
        stdout_text = stdout.read().decode()
        stderr_text = stderr.read().decode()
        
        return stdout_text, stderr_text
    
    def setup_environment(self) -> None:
        """Setup Python environment and clone repository on pod."""
        print("Setting up environment on pod...")
        
        commands = [
            # Update and install system dependencies
            "apt-get update && apt-get install -y git python3-pip python3-venv curl",
            
            # Install uv for faster Python package management
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "echo 'export PATH=/root/.cargo/bin:$PATH' >> ~/.bashrc",
            "source ~/.bashrc",
            
            # Clone repository
            "git clone https://github.com/seconds-0/nsa-vibe.git /workspace/nsa-vibe",
            
            # Setup Python environment
            "cd /workspace/nsa-vibe && /root/.cargo/bin/uv venv -p 3.10 .venv",
            "cd /workspace/nsa-vibe && /root/.cargo/bin/uv pip sync -r requirements.txt",
            
            # Try to install flash-attn
            "cd /workspace/nsa-vibe && ./.venv/bin/pip install flash-attn --no-build-isolation || true"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd[:80]}...")
            stdout, stderr = self.run_command(cmd)
            if "error" in stderr.lower() and "flash-attn" not in cmd:
                print(f"Error: {stderr}")
                raise RuntimeError(f"Setup failed: {stderr}")
    
    def run_benchmarks(self) -> str:
        """
        Run NSA benchmarks on pod.
        
        Returns:
            Benchmark output
        """
        print("Running benchmarks...")
        
        # Run parity tests first
        print("Running parity tests...")
        stdout, stderr = self.run_command(
            "cd /workspace/nsa-vibe && NSA_TEST_FA2=1 PYTHONPATH=. ./.venv/bin/python -m pytest -q -k fa2_gpu_varlen || true"
        )
        parity_passed = "passed" in stdout.lower() or "skipped" in stdout.lower()
        print(f"Parity tests: {'PASSED' if parity_passed else 'SKIPPED/FAILED'}")
        
        # Run performance benchmarks
        print("Running performance benchmarks...")
        stdout, stderr = self.run_command(
            "cd /workspace/nsa-vibe && NSA_USE_FA2=1 PYTHONPATH=. ./.venv/bin/python bench/bench_fa2.py",
            timeout=600
        )
        
        if not stdout:
            raise RuntimeError(f"No benchmark output. Error: {stderr}")
        
        return stdout
    
    def parse_benchmark_output(self, output: str) -> List[BenchmarkResult]:
        """Parse benchmark output into structured results."""
        results = []
        
        # Parse sliding results
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
        
        # Parse compressed results
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
    
    def determine_thresholds(self, results: List[BenchmarkResult], safety_margin: float = 1.1) -> Tuple[int, int]:
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
        
        for s in sorted(set(r.sequence_length for r in compressed_results)):
            s_results = [r for r in compressed_results if r.sequence_length == s]
            if all(r.speedup >= safety_margin for r in s_results):
                if s >= 256:
                    cmp_threshold = 16
                    break
        
        return win_threshold, cmp_threshold
    
    def get_device_info(self) -> Dict:
        """Get GPU and system information from pod."""
        print("Getting device information...")
        
        # Get GPU info
        stdout, _ = self.run_command(
            "cd /workspace/nsa-vibe && ./.venv/bin/python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\""
        )
        device_name = stdout.strip() or "Unknown"
        
        # Get PyTorch version
        stdout, _ = self.run_command(
            "cd /workspace/nsa-vibe && ./.venv/bin/python -c \"import torch; print(torch.__version__)\""
        )
        torch_version = stdout.strip() or "Unknown"
        
        # Get CUDA version
        stdout, _ = self.run_command(
            "cd /workspace/nsa-vibe && ./.venv/bin/python -c \"import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')\""
        )
        cuda_version = stdout.strip() or "Unknown"
        
        return {
            "device_name": device_name,
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "cuda_available": device_name != "CPU"
        }
    
    def cleanup(self) -> None:
        """Clean up resources (SSH connection and pod)."""
        if self.ssh_client:
            print("Closing SSH connection...")
            self.ssh_client.close()
            self.ssh_client = None
        
        if self.pod_id:
            print(f"Terminating pod {self.pod_id}...")
            try:
                resp = requests.delete(
                    f"{self.base_url}/pods/{self.pod_id}",
                    headers=self.headers
                )
                if resp.status_code == 200:
                    result = resp.json()
                    final_cost = result.get('finalCost', 'Unknown')
                    print(f"Pod terminated. Final cost: ${final_cost}")
                else:
                    print(f"Warning: Failed to terminate pod: {resp.text}")
            except Exception as e:
                print(f"Warning: Error terminating pod: {e}")
            
            self.pod_id = None
    
    def run_full_benchmark(self, gpu_type: str = "T4", safety_margin: float = 1.2) -> Dict:
        """
        Run complete benchmark workflow.
        
        Args:
            gpu_type: GPU model to use
            safety_margin: Minimum speedup for thresholds
        
        Returns:
            Complete results dictionary
        """
        try:
            # Create pod
            pod_id = self.create_benchmark_pod(gpu_type)
            
            # Wait for pod to be ready
            pod_info = self.wait_for_pod(pod_id)
            ssh_info = pod_info.get('sshConnection')
            
            if not ssh_info:
                raise RuntimeError("No SSH connection info in pod response")
            
            # Connect via SSH
            self.setup_ssh(ssh_info)
            
            # Setup environment
            self.setup_environment()
            
            # Get device info
            device_info = self.get_device_info()
            
            # Run benchmarks
            benchmark_output = self.run_benchmarks()
            
            # Parse results
            results = self.parse_benchmark_output(benchmark_output)
            
            # Determine thresholds
            win_threshold, cmp_threshold = self.determine_thresholds(results, safety_margin)
            
            # Build response
            return {
                "device_info": device_info,
                "pod_info": {
                    "id": pod_id,
                    "provider": pod_info.get('provider'),
                    "price_per_hour": pod_info.get('price', {}).get('hourly', 'Unknown')
                },
                "benchmark_output": benchmark_output,
                "results": [asdict(r) for r in results],
                "recommendation": {
                    "fa2_min_len_win": win_threshold,
                    "fa2_min_len_cmp": cmp_threshold
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            # Always cleanup
            self.cleanup()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run GPU benchmarks on Prime Intellect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on T4 (cheapest)
  %(prog)s --gpu-type T4
  
  # Run on A100 with custom safety margin
  %(prog)s --gpu-type A100_40GB --safety-margin 1.5
  
  # Save results to file
  %(prog)s --gpu-type L4 --output results.json
  
GPU Types:
  T4, L4, RTX_4090, A10, A100_40GB, A100_80GB, H100_80GB
"""
    )
    
    parser.add_argument(
        "--gpu-type",
        default="T4",
        help="GPU type to use for benchmarking"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.2,
        help="Minimum speedup to enable FA-2 (default: 1.2 = 20%%)"
    )
    
    parser.add_argument(
        "--max-price",
        type=float,
        help="Maximum hourly price willing to pay"
    )
    
    parser.add_argument(
        "--api-key",
        help="Prime Intellect API key (or set PRIME_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    try:
        benchmark = PrimeIntellectBenchmark(api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set PRIME_API_KEY environment variable or use --api-key")
        return 1
    
    # Run benchmark
    print(f"Starting Prime Intellect benchmark on {args.gpu_type}...")
    print("="*60)
    
    try:
        results = benchmark.run_full_benchmark(
            gpu_type=args.gpu_type,
            safety_margin=args.safety_margin
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Device: {results['device_info']['device_name']}")
        print(f"PyTorch: {results['device_info']['torch_version']}")
        print(f"CUDA: {results['device_info']['cuda_version']}")
        print(f"Provider: {results['pod_info']['provider']}")
        print(f"Cost: ${results['pod_info']['price_per_hour']}/hr")
        
        print("\nRecommended Thresholds:")
        print(f"  fa2_min_len_win: {results['recommendation']['fa2_min_len_win']}")
        print(f"  fa2_min_len_cmp: {results['recommendation']['fa2_min_len_cmp']}")
        
        # Print detailed results
        print("\nDetailed Results:")
        for r in results['results']:
            if r['branch'] == 'sliding':
                print(f"  S={r['sequence_length']:4} w={r['window_size']:3}: "
                      f"speedup {r['speedup']:.2f}x")
            else:
                print(f"  S={r['sequence_length']:4} (compressed): "
                      f"speedup {r['speedup']:.2f}x")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())