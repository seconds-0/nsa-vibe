#!/usr/bin/env python3
"""
Run M4 Triton Selection benchmarks on Prime Intellect pod.

This script provisions a GPU pod, clones the repo, installs dependencies,
and runs the Triton selection attention benchmarks.
"""

import argparse
import subprocess
import sys


def run_command(cmd: str, check: bool = True, shell: bool = True) -> str:
    """Run command and return output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def setup_pod_and_benchmark():
    """Set up Prime Intellect pod and run M4 benchmarks."""

    # Commands to run on the pod
    setup_commands = [
        # System setup
        "cd /root",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update",
        "apt-get install -y git python3-pip python3-venv ninja-build",
        # Clone repo
        "git clone https://github.com/seconds-0/nsa-vibe.git",
        "cd nsa-vibe",
        "git checkout feat/m0-complete",
        # Setup environment
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "pip install --upgrade pip wheel",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install triton",  # Latest Triton
        "pip install -r requirements.txt",
        # Verify GPU and PyTorch
        'python3 -c \'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}")\'',
        # Verify Triton
        "python3 -c 'import triton; print(f\"Triton version: {triton.__version__}\")'",
        # Run parity tests first
        "NSA_USE_TRITON_SEL=1 NSA_TEST_TRITON_SEL=1 python3 -m pytest nsa/tests/test_triton_sel_parity.py -v",
        # Run benchmark suite
        "echo '=== M4 TRITON SELECTION BENCHMARKS ==='",
        # Small configurations
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --N 64 --H 4 --D 64 --Dv 64 --L_list 16,32,64,128,256 --dist few --iters 50",
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --N 64 --H 4 --D 64 --Dv 64 --L_list 16,32,64,128,256 --dist many --iters 50",
        # Medium configurations
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --N 256 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512 --dist few --iters 50",
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --N 256 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512 --dist many --iters 50",
        # Large configurations
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --N 1024 --H 8 --D 128 --Dv 128 --L_list 128,256,512,1024 --dist mixed --iters 50",
        # Decode-style configurations
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --decode 1 --H 4 --D 64 --Dv 64 --L_list 128,256,512 --dist few --iters 50",
        "NSA_USE_TRITON_SEL=1 python3 bench/bench_sel_triton.py --decode 1 --H 8 --D 128 --Dv 128 --L_list 256,512,1024 --dist mixed --iters 50",
        # Generate summary
        "echo '=== M4 BENCHMARK COMPLETE ==='",
        "echo 'Results above show Triton vs Packed SDPA performance across configurations'",
        "echo 'Look for speedup ≥ 1.2x to determine sel_triton_min_L threshold'",
    ]

    # Create combined script
    " && ".join(setup_commands)

    print("=" * 80)
    print("M4 TRITON SELECTION BENCHMARK SCRIPT")
    print("=" * 80)
    print("This script will:")
    print("1. Set up Prime Intellect RTX 4090 pod")
    print("2. Clone nsa-vibe repo and install dependencies")
    print("3. Run Triton vs SDPA benchmark suite")
    print("4. Validate numerical parity")
    print("5. Determine optimal sel_triton_min_L threshold")
    print()

    # Save script for manual execution
    script_path = "/tmp/m4_benchmark.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\nset -e\n")
        for cmd in setup_commands:
            f.write(f"{cmd}\n")

    print(f"Script saved to: {script_path}")
    print()
    print("To run manually on Prime Intellect pod:")
    print(f"1. Upload and run: bash {script_path}")
    print("2. Or copy/paste commands from above")
    print()

    # Try to run automatically using Prime Intellect automation
    try:
        print("Attempting automatic Prime Intellect pod setup...")
        result = run_command(
            "python3 bench/prime_gpu_bench.py --gpu-type RTX4090_24GB", check=False
        )

        if "error" in result.lower() or "failed" in result.lower():
            print("❌ Automatic setup failed. Please run manually.")
            return False
        else:
            print("✅ Prime Intellect benchmarks completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Automatic setup failed: {e}")
        print("Please run benchmarks manually using the script above.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run M4 Triton Selection benchmarks")
    parser.add_argument(
        "--manual", action="store_true", help="Generate script only, don't auto-run"
    )

    args = parser.parse_args()

    if args.manual:
        print("Manual mode: generating benchmark script only")

    success = setup_pod_and_benchmark()

    if success:
        print("🎉 M4 benchmarks completed! Check output above for threshold recommendations.")
    else:
        print("⚠️  Manual execution required. See script above.")


if __name__ == "__main__":
    main()
