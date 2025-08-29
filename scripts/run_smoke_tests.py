#!/usr/bin/env python3
"""M8 smoke tests for NSA training validation.

This script runs quick validation tests to ensure training is working correctly
before long runs. It checks loss improvement, throughput, and gate health.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_training_csv(csv_path: Path) -> List[Dict]:
    """Read training CSV and return list of records."""
    records = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {
                        "step": int(row["step"]),
                        "loss": float(row["loss"]),
                        "lr": float(row["lr"]),
                        "toks_per_s": float(row["toks_per_s"]),
                    }
                )
    except Exception as e:
        print(f"[smoke] Failed to read CSV {csv_path}: {e}")
        return []
    return records


def read_heartbeat_jsonl(hb_path: Path) -> List[Dict]:
    """Read heartbeat JSONL and return list of records."""
    records = []
    try:
        with open(hb_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("msg") == "progress":
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"[smoke] Failed to read heartbeat {hb_path}: {e}")
        return []
    return records


def check_loss_improvement(records: List[Dict], min_steps: int = 100) -> Tuple[bool, str]:
    """Check if loss is improving over time."""
    if len(records) < min_steps:
        return False, f"Need at least {min_steps} steps, got {len(records)}"

    # Compare first 50 steps vs last 200 steps
    first_50 = records[:50]
    last_200 = records[-200:]

    if len(first_50) < 10 or len(last_200) < 10:
        return False, "Insufficient data for comparison"

    first_avg = sum(r["loss"] for r in first_50) / len(first_50)
    last_avg = sum(r["loss"] for r in last_200) / len(last_200)

    improvement = (first_avg - last_avg) / first_avg

    if improvement < 0.01:  # Less than 1% improvement
        return False, f"Loss barely improved: {first_avg:.4f} → {last_avg:.4f} ({improvement:.1%})"

    return True, f"Loss improved: {first_avg:.4f} → {last_avg:.4f} ({improvement:.1%})"


def check_no_nans(records: List[Dict]) -> Tuple[bool, str]:
    """Check for NaN or infinite values in loss."""
    for record in records:
        loss = record["loss"]
        if not (loss == loss and abs(loss) < float("inf")):  # NaN/inf check
            return False, f"Found NaN/inf loss at step {record['step']}: {loss}"
    return True, "No NaN/inf values found"


def check_throughput(records: List[Dict], min_tps: float = 10.0) -> Tuple[bool, str]:
    """Check minimum throughput requirement."""
    if not records:
        return False, "No throughput data"

    # Check median throughput of last 100 steps
    recent = records[-100:] if len(records) >= 100 else records
    throughputs = [r["toks_per_s"] for r in recent]
    median_tps = sorted(throughputs)[len(throughputs) // 2]

    if median_tps < min_tps:
        return False, f"Throughput too low: {median_tps:.1f} toks/s < {min_tps:.1f}"

    return True, f"Throughput OK: {median_tps:.1f} toks/s"


def check_data_fetch_health(hb_records: List[Dict]) -> Tuple[bool, str]:
    """Check data fetch timing for streaming health."""
    fetch_times = [r.get("dt_fetch_s") for r in hb_records if r.get("dt_fetch_s") is not None]

    if not fetch_times:
        return True, "No fetch timing data (synthetic mode)"

    # Check for fetch stalls (> 2s is concerning)
    slow_fetches = [t for t in fetch_times if t > 2.0]
    if len(slow_fetches) > len(fetch_times) * 0.1:  # More than 10% slow
        avg_slow = sum(slow_fetches) / len(slow_fetches)
        return (
            False,
            f"Data fetch stalls: {len(slow_fetches)}/{len(fetch_times)} slow (avg {avg_slow:.2f}s)",
        )

    avg_fetch = sum(fetch_times) / len(fetch_times)
    return True, f"Data fetch healthy: avg {avg_fetch:.3f}s"


def save_baseline_metrics(metrics: Dict, baseline_path: Path) -> bool:
    """Save metrics as baseline for future comparisons."""
    try:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[baseline] Saved baseline to {baseline_path}")
        return True
    except Exception as e:
        print(f"[baseline] Failed to save baseline: {e}")
        return False


def load_baseline_metrics(baseline_path: Path) -> Optional[Dict]:
    """Load baseline metrics for comparison."""
    try:
        if not baseline_path.exists():
            return None
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"[baseline] Loaded baseline from {baseline_path}")
        return baseline
    except Exception as e:
        print(f"[baseline] Failed to load baseline: {e}")
        return None


def extract_metrics(csv_records: List[Dict], hb_records: List[Dict]) -> Dict:
    """Extract key metrics for baseline comparison."""
    if not csv_records:
        return {}

    # Loss metrics
    final_loss = csv_records[-1]["loss"] if csv_records else float("inf")
    first_100_avg = sum(r["loss"] for r in csv_records[:100]) / min(100, len(csv_records))
    last_100_avg = sum(r["loss"] for r in csv_records[-100:]) / min(100, len(csv_records))

    # Throughput metrics
    recent_tps = [r["toks_per_s"] for r in csv_records[-100:]]
    median_tps = sorted(recent_tps)[len(recent_tps) // 2] if recent_tps else 0.0

    # Gate health metrics
    gate_records = [r for r in hb_records if "gate_entropy_mean" in r]
    gate_entropy = 0.0
    gate_collapse = 0.0
    if gate_records:
        recent_gates = gate_records[-50:]
        gate_entropy = sum(r["gate_entropy_mean"] for r in recent_gates) / len(recent_gates)
        gate_collapse = sum(r.get("gate_collapse_frac", 0.0) for r in recent_gates) / len(
            recent_gates
        )

    return {
        "final_loss": final_loss,
        "first_100_loss_avg": first_100_avg,
        "last_100_loss_avg": last_100_avg,
        "loss_improvement_pct": (first_100_avg - last_100_avg) / first_100_avg * 100
        if first_100_avg > 0
        else 0.0,
        "median_tps": median_tps,
        "gate_entropy_mean": gate_entropy,
        "gate_collapse_frac": gate_collapse,
        "total_steps": len(csv_records),
    }


def compare_with_baseline(
    current: Dict, baseline: Dict, tolerance_pct: float = 5.0
) -> Tuple[bool, List[str]]:
    """Compare current metrics with baseline, allowing tolerance."""
    if not baseline:
        return True, ["No baseline available for comparison"]

    issues = []

    # Check critical metrics with tolerance
    checks = [
        ("final_loss", "higher", "Final loss regression"),
        ("loss_improvement_pct", "lower", "Loss improvement degraded"),
        ("median_tps", "lower", "Throughput degraded"),
        ("gate_entropy_mean", "lower", "Gate entropy degraded"),
    ]

    for metric, direction, description in checks:
        if metric not in current or metric not in baseline:
            continue

        curr_val = current[metric]
        base_val = baseline[metric]

        if base_val == 0:
            continue  # Skip division by zero

        change_pct = (curr_val - base_val) / abs(base_val) * 100

        if direction == "lower" and change_pct < -tolerance_pct:
            issues.append(
                f"{description}: {curr_val:.4f} vs baseline {base_val:.4f} ({change_pct:+.1f}%)"
            )
        elif direction == "higher" and change_pct > tolerance_pct:
            issues.append(
                f"{description}: {curr_val:.4f} vs baseline {base_val:.4f} ({change_pct:+.1f}%)"
            )

    # Check gate collapse - absolute threshold
    if "gate_collapse_frac" in current and current["gate_collapse_frac"] > 0.3:
        issues.append(f"High gate collapse: {current['gate_collapse_frac']:.1%}")

    return len(issues) == 0, issues


def check_selection_determinism() -> Tuple[bool, str]:
    """Check selection determinism with identical inputs."""
    try:
        import os
        import sys

        import torch

        sys.path.append(".")

        from nsa.core.block_index import build_block_meta
        from nsa.core.selection_scorer import select_topn_ranges

        # Create test scenario with potential ties
        torch.manual_seed(12345)
        meta = build_block_meta(seq_len=128, l=32, d=16, l_sel=16, n_sel=4, w=32)
        S_sel = meta.sel_starts.numel()

        B, G = 1, 2
        p_grp = torch.randn(B, G, S_sel, dtype=torch.float32)
        # Force ties to test deterministic behavior
        if S_sel >= 4:
            p_grp[0, 0, 1:4] = 0.5  # Identical scores

        # Run multiple trials with same input
        results = []
        for trial in range(3):
            ranges = select_topn_ranges(p_grp.clone(), meta, n_top=4, t_token=60)
            results.append(ranges)

        # Check all results are identical
        for i in range(1, len(results)):
            if not torch.equal(results[0], results[i]):
                return False, f"Selection non-deterministic: trial 0 != trial {i}"

        return True, f"Selection deterministic across {len(results)} trials"

    except Exception as e:
        return False, f"Selection determinism check failed: {e}"


def check_gate_health(hb_records: List[Dict]) -> Tuple[bool, str]:
    """Check gate statistics for collapse or imbalance."""
    gate_records = [r for r in hb_records if "gate_entropy_mean" in r]

    if not gate_records:
        return True, "No gate health data available"

    recent_gates = gate_records[-50:] if len(gate_records) >= 50 else gate_records

    # Check entropy (should be > 0.5 for healthy mixing)
    entropies = [r["gate_entropy_mean"] for r in recent_gates]
    avg_entropy = sum(entropies) / len(entropies)

    if avg_entropy < 0.3:
        return False, f"Gate entropy too low: {avg_entropy:.3f} (indicates collapse)"

    # Check max gate values (should be < 0.9)
    max_gates = [r.get("gate_max_gate", 0.0) for r in recent_gates]
    avg_max_gate = sum(max_gates) / len(max_gates) if max_gates else 0.0

    if avg_max_gate > 0.85:
        return False, f"Gate max values too high: {avg_max_gate:.3f} (indicates collapse)"

    # Check collapse fraction
    collapse_fracs = [r.get("gate_collapse_frac", 0.0) for r in recent_gates]
    avg_collapse = sum(collapse_fracs) / len(collapse_fracs) if collapse_fracs else 0.0

    if avg_collapse > 0.2:  # More than 20% collapsed gates
        return False, f"High gate collapse rate: {avg_collapse:.1%}"

    # Check branch balance
    if recent_gates and "gate_branch_shares" in recent_gates[-1]:
        shares = recent_gates[-1]["gate_branch_shares"]  # [cmp, sel, win]
        if len(shares) == 3:
            min_share = min(shares)
            max_share = max(shares)
            if max_share > 0.8 or min_share < 0.05:
                return False, f"Unbalanced branches: {shares} (one branch dominates)"

    return True, f"Gate health OK: entropy={avg_entropy:.3f}, collapse={avg_collapse:.1%}"


def run_synthetic_smoke(steps: int = 1000, timeout: int = 300) -> bool:
    """Run synthetic data smoke test."""
    print(f"[smoke] Running synthetic smoke ({steps} steps, {timeout}s timeout)")

    out_dir = Path("artifacts/smoke_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "scripts/train_showcase.py",
        "--dataset",
        "synthetic",
        "--ddp",
        "0",
        "--steps",
        str(steps),
    ]

    env = os.environ.copy()
    env["CONFIG"] = "configs/train_showcase.yaml"
    env["NSA_LOG_GRAD_NORM"] = "1"  # Enable grad norm logging

    try:
        result = subprocess.run(
            cmd, timeout=timeout, env=env, capture_output=True, text=True, cwd="."
        )

        if result.returncode != 0:
            print(f"[smoke] Synthetic smoke failed: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"[smoke] Synthetic smoke timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[smoke] Synthetic smoke error: {e}")
        return False


def run_fineweb_smoke(steps: int = 1000, timeout: int = 600) -> bool:
    """Run FineWeb-Edu streaming smoke test."""
    print(f"[smoke] Running FineWeb-Edu smoke ({steps} steps, {timeout}s timeout)")

    cmd = [
        sys.executable,
        "-u",
        "scripts/train_showcase.py",
        "--dataset",
        "fineweb_edu",
        "--ddp",
        "0",
        "--fwe-report-docs",
        "500",
        "--loader-timeout",
        "120",
        "--synthetic-on-fail",  # Fallback to synthetic if streaming fails
        "--steps",
        str(steps),
    ]

    env = os.environ.copy()
    env["CONFIG"] = "configs/train_showcase.yaml"
    env["NSA_LOG_GRAD_NORM"] = "1"
    env["NSA_TOKENIZER"] = env.get("NSA_TOKENIZER", "byte")  # Default to byte tokenizer

    try:
        result = subprocess.run(
            cmd, timeout=timeout, env=env, capture_output=True, text=True, cwd="."
        )

        if result.returncode != 0:
            print(f"[smoke] FineWeb smoke failed: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"[smoke] FineWeb smoke timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[smoke] FineWeb smoke error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NSA training smoke tests")
    parser.add_argument(
        "--csv",
        type=str,
        default="artifacts/train_showcase/training.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--heartbeat",
        type=str,
        default="artifacts/train_showcase/heartbeat_rank0.jsonl",
        help="Path to heartbeat JSONL",
    )
    parser.add_argument(
        "--min-steps", type=int, default=100, help="Minimum steps required for validation"
    )
    parser.add_argument(
        "--min-tps", type=float, default=10.0, help="Minimum tokens/second threshold"
    )
    parser.add_argument(
        "--run-synthetic", action="store_true", help="Run synthetic data smoke test"
    )
    parser.add_argument("--run-fineweb", action="store_true", help="Run FineWeb-Edu smoke test")
    parser.add_argument("--smoke-steps", type=int, default=1000, help="Steps to run in smoke tests")
    parser.add_argument(
        "--smoke-timeout", type=int, default=600, help="Timeout for smoke tests (seconds)"
    )
    parser.add_argument(
        "--save-baseline", type=str, help="Save current metrics as baseline to specified file"
    )
    parser.add_argument(
        "--compare-baseline", type=str, help="Compare current metrics against baseline file"
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=5.0,
        help="Tolerance percentage for baseline comparison",
    )

    args = parser.parse_args()

    exit_code = 0

    # Run smoke tests if requested
    if args.run_synthetic:
        if not run_synthetic_smoke(args.smoke_steps, args.smoke_timeout):
            exit_code = 1

    if args.run_fineweb:
        if not run_fineweb_smoke(args.smoke_steps, args.smoke_timeout):
            exit_code = 1

    # Validate existing training data
    csv_path = Path(args.csv)
    hb_path = Path(args.heartbeat)

    if not csv_path.exists() and not (args.run_synthetic or args.run_fineweb):
        print(f"[smoke] CSV file not found: {csv_path}")
        print("[smoke] Use --run-synthetic or --run-fineweb to run smoke tests")
        return 1

    if csv_path.exists():
        print(f"[smoke] Validating training data: {csv_path}")

        # Read data
        csv_records = read_training_csv(csv_path)
        hb_records = read_heartbeat_jsonl(hb_path) if hb_path.exists() else []

        print(
            f"[smoke] Found {len(csv_records)} training steps, {len(hb_records)} heartbeat records"
        )

        # Extract metrics for baseline comparison
        current_metrics = extract_metrics(csv_records, hb_records)

        # Handle baseline operations
        baseline_passed = True
        baseline_issues = []

        if args.save_baseline and current_metrics:
            print("\n[baseline] Current metrics:")
            for key, value in current_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            save_baseline_metrics(current_metrics, Path(args.save_baseline))

        if args.compare_baseline and current_metrics:
            baseline = load_baseline_metrics(Path(args.compare_baseline))
            baseline_passed, baseline_issues = compare_with_baseline(
                current_metrics, baseline, args.baseline_tolerance
            )

        # Run checks
        checks = [
            ("Loss improvement", check_loss_improvement(csv_records, args.min_steps)),
            ("No NaN/inf values", check_no_nans(csv_records)),
            ("Throughput", check_throughput(csv_records, args.min_tps)),
            ("Data fetch health", check_data_fetch_health(hb_records)),
            ("Gate health", check_gate_health(hb_records)),
            ("Selection determinism", check_selection_determinism()),
        ]

        # Add baseline comparison as a check if enabled
        if args.compare_baseline:
            if baseline_passed:
                checks.append(
                    ("Baseline comparison", (True, f"Within {args.baseline_tolerance}% tolerance"))
                )
            else:
                checks.append(("Baseline comparison", (False, "; ".join(baseline_issues))))

        print("\n[smoke] Validation Results:")
        print("=" * 50)

        for name, (passed, message) in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {name}: {message}")
            if not passed:
                exit_code = 1

        if exit_code == 0:
            print("\n🎉 All smoke tests PASSED!")
        else:
            print("\n💥 Some smoke tests FAILED!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
