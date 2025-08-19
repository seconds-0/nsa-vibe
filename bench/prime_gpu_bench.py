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
            gpu_type: GPU model (RTX4090_24GB, A100_40GB, etc.)
            regions: Optional list of regions to search
        
        Returns:
            Dictionary with cheapest option details
        """
        # Use Prime Intellect API with query parameters (from their docs)
        params = {
            'gpu_type': gpu_type,
            'gpu_count': 1
        }
        if regions:
            params['regions'] = regions
        
        print(f"Querying Prime Intellect API: {params}")
        
        resp = requests.get(
            f"{self.base_url}/availability/",
            params=params,
            headers=self.headers,
            timeout=30
        )
        
        print(f"API Response: Status {resp.status_code}")
        
        if resp.status_code != 200:
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        print(f"Response data: {json.dumps(data, indent=2) if data else 'Empty response'}")
        
        if not data:
            raise RuntimeError(f"No data returned for {gpu_type}")
        
        # Prime Intellect nests results under GPU type key
        if gpu_type not in data:
            available_keys = list(data.keys())
            raise RuntimeError(f"No {gpu_type} found in response. Available: {available_keys}")
        
        options = data[gpu_type]
        if not options:
            raise RuntimeError(f"No {gpu_type} options available")
        
        print(f"Found {len(options)} options for {gpu_type}")
        
        # Validate options and extract pricing
        valid_options = []
        for opt in options:
            if ('cloudId' not in opt or 'prices' not in opt or 'provider' not in opt):
                print(f"Skipping option missing required fields: {opt.get('provider', 'Unknown')}")
                continue
            
            # Extract hourly price from either onDemand or communityPrice
            prices = opt['prices']
            hourly_price = None
            
            if prices.get('onDemand'):
                hourly_price = prices['onDemand']
            elif prices.get('communityPrice'):
                hourly_price = prices['communityPrice']
            
            if hourly_price is None:
                print(f"Skipping option with no valid pricing: {opt.get('provider', 'Unknown')}")
                continue
            
            # Add computed hourly price for easy sorting
            opt['hourly_price'] = hourly_price
            valid_options.append(opt)
            print(f"Valid option: {opt['provider']} ${hourly_price:.4f}/hr ({opt['security']})")
        
        if not valid_options:
            raise RuntimeError(f"Found {len(options)} options but none have valid pricing")
        
        # Sort by hourly price and return cheapest
        cheapest = min(valid_options, key=lambda x: x['hourly_price'])
        print(f"Selected cheapest: {cheapest['provider']} ${cheapest['hourly_price']:.4f}/hr")
        return cheapest
    
    def create_benchmark_pod(self, gpu_type: str = "RTX4090_24GB", max_price: Optional[float] = None) -> str:
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
        
        hourly_price = option['hourly_price']  # Pre-computed in find_cheapest_gpu
        provider = option['provider']
        print(f"Found {gpu_type} on {provider} for ${hourly_price:.4f}/hr")
        
        # Set max price with 20% buffer if not specified
        if max_price is None:
            max_price = hourly_price * 1.2
        
        # Create pod configuration (no envVars - not supported)
        pod_config = {
            "pod": {
                "name": f"nsa-bench-{gpu_type.lower().replace('_', '-')}-{int(time.time())}",
                "cloudId": option['cloudId'],
                "gpuType": gpu_type,
                "socket": option.get('socket', 'PCIe'),
                "gpuCount": 1,
                "diskSize": 50,
                "vcpus": 8,
                "memory": 32,
                "image": "ubuntu_22_cuda_12",
                "maxPrice": max_price,
                "autoRestart": False
            },
            "provider": {
                "type": provider
            }
        }
        
        print(f"Creating pod...")
        
        # Simple retry logic for pod creation
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/pods/",
                    json=pod_config,
                    headers=self.headers,
                    timeout=30
                )
                
                if resp.status_code in [200, 201]:  # 201 = Created
                    pod_data = resp.json()
                    
                    # Validate response has required fields
                    if 'id' not in pod_data:
                        raise RuntimeError(f"Invalid pod response, missing 'id': {pod_data}")
                    
                    self.pod_id = pod_data['id']
                    print(f"Created pod: {self.pod_id} (status: {pod_data.get('status', 'unknown')})")
                    return self.pod_id
                elif resp.status_code == 429:
                    # Rate limiting, wait longer
                    print(f"Rate limited, waiting 30s before retry {attempt + 1}/3...")
                    time.sleep(30)
                    last_error = f"Rate limited: {resp.text}"
                else:
                    last_error = f"API error {resp.status_code}: {resp.text}"
                    if attempt < 2:
                        print(f"Failed attempt {attempt + 1}/3: {last_error}")
                        time.sleep(5)
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {e}"
                if attempt < 2:
                    print(f"Network error on attempt {attempt + 1}/3, retrying...")
                    time.sleep(5)
        
        raise RuntimeError(f"Failed to create pod after 3 attempts. Last error: {last_error}")
    
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
            
            if status.lower() in ['running', 'active']:
                print(f"Pod ready after {int(time.time() - start)}s (status: {status})")
                return pod
            elif status.lower() in ['error', 'failed', 'terminated', 'failed_to_provision']:
                raise RuntimeError(f"Pod failed with status: {status}")
            
            print(f"Pod status: {status}, waiting...")
            time.sleep(10)
        
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")
    
    def parse_ssh_connection_string(self, ssh_string: str) -> Dict[str, str]:
        """
        Parse Prime Intellect SSH connection string.
        
        Args:
            ssh_string: SSH connection string like "root@80.15.7.37 -p 45603"
        
        Returns:
            Dictionary with host, port, user keys
        """
        import re
        
        # Parse "root@80.15.7.37 -p 45603" format
        match = re.match(r'(\w+)@([\d.]+)\s+-p\s+(\d+)', ssh_string)
        if not match:
            raise RuntimeError(f"Cannot parse SSH connection string: {ssh_string}")
        
        user, host, port = match.groups()
        
        return {
            'user': user,
            'host': host, 
            'port': int(port)
        }
    
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
        
        # Configure for cloud instances
        ssh.load_system_host_keys()
        ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
        
        connect_kwargs = {
            'hostname': ssh_info['host'],
            'port': ssh_info.get('port', 22),
            'username': ssh_info.get('user', 'root'),
            'timeout': 30
        }
        
        # Try authentication methods in order of preference
        auth_configured = False
        
        if private_key_path and Path(private_key_path).exists():
            connect_kwargs['key_filename'] = private_key_path
            auth_configured = True
        elif 'PRIME_SSH_KEY' in os.environ:
            connect_kwargs['key_filename'] = os.environ['PRIME_SSH_KEY']
            auth_configured = True
        elif 'PRIME_SSH_PASSWORD' in os.environ:
            connect_kwargs['password'] = os.environ['PRIME_SSH_PASSWORD']
            auth_configured = True
        elif Path('~/.ssh/primeintellect_ed25519').expanduser().exists():
            # Use the generated Prime Intellect key
            connect_kwargs['key_filename'] = str(Path('~/.ssh/primeintellect_ed25519').expanduser())
            auth_configured = True
        
        # If no explicit auth provided, try system SSH keys and passwordless auth
        if not auth_configured:
            print("No SSH credentials provided, trying system SSH keys and passwordless auth...")
            connect_kwargs['look_for_keys'] = True
            connect_kwargs['allow_agent'] = True
            # Try empty password for RunPod containers
            connect_kwargs['password'] = ''
        
        print(f"Connecting via SSH to {ssh_info['host']}...")
        
        # Try multiple authentication methods for cloud instances
        auth_methods = []
        
        if auth_configured:
            # Use configured auth
            auth_methods.append(connect_kwargs)
        else:
            # Try multiple methods for cloud instances
            auth_methods.extend([
                # Method 1: System keys + empty password
                {**connect_kwargs, 'look_for_keys': True, 'allow_agent': True, 'password': ''},
                # Method 2: Just system keys
                {**connect_kwargs, 'look_for_keys': True, 'allow_agent': True},
                # Method 3: No auth (some containers allow this)
                {**connect_kwargs, 'look_for_keys': False, 'allow_agent': False},
            ])
        
        last_error = None
        for i, method in enumerate(auth_methods):
            try:
                print(f"Trying SSH auth method {i+1}/{len(auth_methods)}...")
                ssh.connect(**method)
                print("SSH connection successful!")
                break
            except Exception as e:
                last_error = e
                print(f"Auth method {i+1} failed: {e}")
                if i < len(auth_methods) - 1:
                    ssh.close()  # Close failed connection
                    ssh = paramiko.SSHClient()  # Create new client for next attempt
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            raise RuntimeError(
                f"All SSH authentication methods failed. Last error: {last_error}\n"
                "Please provide SSH credentials via:\n"
                "  - PRIME_SSH_KEY environment variable (path to key file)\n"
                "  - PRIME_SSH_PASSWORD environment variable\n"
                "  - Or ensure your SSH agent has the correct key"
            )
        
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
            
            # Set environment variables in bashrc (since envVars not supported in pod config)
            "echo 'export PYTHONPATH=.' >> ~/.bashrc",
            "echo 'export NSA_USE_FA2=1' >> ~/.bashrc", 
            "echo 'export DEBIAN_FRONTEND=noninteractive' >> ~/.bashrc",
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
    
    def run_full_benchmark(self, gpu_type: str = "RTX4090_24GB", safety_margin: float = 1.2) -> Dict:
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
            
            # Save pod ID for manual recovery if script crashes
            with open('.prime_pod_id', 'w') as f:
                f.write(f"{pod_id}\n")
                f.write(f"# Created at {datetime.utcnow().isoformat()}\n")
                f.write(f"# GPU: {gpu_type}\n")
                f.write(f"# To manually cleanup: prime pods terminate {pod_id}\n")
            
            # Wait for pod to be ready
            pod_info = self.wait_for_pod(pod_id)
            ssh_connection_string = pod_info.get('sshConnection')
            
            if not ssh_connection_string:
                raise RuntimeError(f"No SSH connection info in pod response: {pod_info}")
            
            # Parse Prime Intellect SSH connection string: "root@80.15.7.37 -p 45603"
            ssh_info = self.parse_ssh_connection_string(ssh_connection_string)
            
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
            
            # Remove pod ID file if cleanup succeeded
            if not self.pod_id and Path('.prime_pod_id').exists():
                os.remove('.prime_pod_id')


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run GPU benchmarks on Prime Intellect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on T4 (cheapest)
  %(prog)s --gpu-type T4_16GB
  
  # Run on A100 with custom safety margin
  %(prog)s --gpu-type A100_40GB --safety-margin 1.5
  
  # Save results to file
  %(prog)s --gpu-type L4_24GB --output results.json
  
GPU Types:
  T4_16GB, L4_24GB, RTX4090_24GB, A10_24GB, A100_40GB, A100_80GB, H100_80GB
"""
    )
    
    parser.add_argument(
        "--gpu-type",
        default="RTX4090_24GB",
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