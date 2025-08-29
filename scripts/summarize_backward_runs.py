#!/usr/bin/env python3
"""
NSA Backward Pass Test Summarizer
Analyzes test results and generates comprehensive report
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path


def parse_result_file(result_path):
    """Parse a single test result file"""
    try:
        with open(result_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def analyze_test_dir(test_dir):
    """Analyze a single test directory"""
    test_name = Path(test_dir).name
    result_file = Path(test_dir) / "result.json"
    env_file = Path(test_dir) / "env.json"

    result = {"name": test_name, "status": "UNKNOWN"}

    # Check for result file
    if result_file.exists():
        data = parse_result_file(result_file)
        if data:
            result.update(data)

    # Check for .result file (from matrix runner)
    result_status_file = Path(test_dir).parent / f"{test_name}.result"
    if result_status_file.exists():
        with open(result_status_file) as f:
            status = f.read().strip()
            result["status"] = status

    # Parse environment
    if env_file.exists():
        with open(env_file) as f:
            env_data = json.load(f)
            result["config"] = env_data.get("args", {})

    return result


def generate_summary_report(results_dir, output_file=None):
    """Generate comprehensive summary report"""

    # Find all test directories
    test_dirs = glob.glob(f"{results_dir}/**/env.json", recursive=True)
    test_dirs = [Path(p).parent for p in test_dirs]

    # Also check for matrix run directories
    matrix_dirs = glob.glob(f"{results_dir}/*.result")

    all_results = []

    # Analyze each test directory
    for test_dir in test_dirs:
        result = analyze_test_dir(test_dir)
        all_results.append(result)

    # Generate report
    report = []
    report.append("# NSA Backward Pass Test Summary")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"**Results Directory**: {results_dir}")
    report.append(f"**Total Tests**: {len(all_results)}")

    # Summary statistics
    passed = sum(1 for r in all_results if r.get("result") == "PASS" or r.get("status") == "PASS")
    failed = sum(1 for r in all_results if r.get("result") == "FAIL" or r.get("status") == "FAIL")
    hung = sum(1 for r in all_results if r.get("result") == "HANG" or r.get("status") == "HANG")

    report.append("\n## Overall Results")
    report.append(f"- ✅ Passed: {passed}")
    report.append(f"- ❌ Failed: {failed}")
    report.append(f"- ⏱️ Hung: {hung}")
    report.append(f"- ❓ Unknown: {len(all_results) - passed - failed - hung}")

    # Memory scaling analysis
    report.append("\n## Memory Scaling Analysis")

    scaling_tests = [r for r in all_results if "seq" in r.get("name", "").lower()]
    if scaling_tests:
        report.append("\n| Test | Seq Length | Status | Forward Mem (MB) | Backward Mem (MB) |")
        report.append("|------|------------|--------|------------------|-------------------|")

        for test in sorted(scaling_tests, key=lambda x: x.get("config", {}).get("seq_len", 0)):
            config = test.get("config", {})
            seq_len = config.get("seq_len", "?")
            status = test.get("status", test.get("result", "UNKNOWN"))

            mem_forward = test.get("mem_after_forward", {})
            mem_backward = test.get("mem_after_backward", {})

            forward_mb = mem_forward.get("allocated_mb", "N/A")
            backward_mb = mem_backward.get("allocated_mb", "N/A")

            status_emoji = {"PASS": "✅", "FAIL": "❌", "HANG": "⏱️"}.get(status, "❓")

            report.append(
                f"| {test['name']} | {seq_len} | {status_emoji} {status} | {forward_mb} | {backward_mb} |"
            )

    # Branch analysis
    report.append("\n## Branch Analysis")

    branch_tests = [
        r for r in all_results if any(b in r.get("name", "").lower() for b in ["win", "sel", "cmp"])
    ]
    if branch_tests:
        report.append("\n| Branch | Backend | Status | Time (s) | Memory (MB) |")
        report.append("|--------|---------|--------|----------|-------------|")

        for test in branch_tests:
            config = test.get("config", {})
            branch = config.get("branch", "all")

            # Determine backend
            backend = "default"
            if config.get("sel"):
                backend = f"sel_{config['sel']}"
            elif config.get("cmp"):
                backend = f"cmp_{config['cmp']}"
            elif config.get("win"):
                backend = f"win_{config['win']}"

            status = test.get("status", test.get("result", "UNKNOWN"))
            status_emoji = {"PASS": "✅", "FAIL": "❌", "HANG": "⏱️"}.get(status, "❓")

            time = test.get("backward_time", test.get("forward_time", "N/A"))
            if isinstance(time, float):
                time = f"{time:.2f}"

            mem = test.get("mem_after_backward", test.get("mem_after_forward", {}))
            mem_mb = mem.get("allocated_mb", "N/A")
            if isinstance(mem_mb, float):
                mem_mb = f"{mem_mb:.1f}"

            report.append(f"| {branch} | {backend} | {status_emoji} {status} | {time} | {mem_mb} |")

    # Critical findings
    report.append("\n## Critical Findings")

    # Check for quadratic scaling
    if scaling_tests:
        seq_128 = next(
            (t for t in scaling_tests if t.get("config", {}).get("seq_len") == 128), None
        )
        seq_512 = next(
            (t for t in scaling_tests if t.get("config", {}).get("seq_len") == 512), None
        )

        if seq_128 and seq_512:
            mem_128 = seq_128.get("mem_after_forward", {}).get("allocated_mb", 0)
            mem_512 = seq_512.get("mem_after_forward", {}).get("allocated_mb", 0)

            if mem_128 and mem_512:
                scaling_factor = mem_512 / mem_128
                if scaling_factor > 8:  # 4x sequence should be ~16x memory for O(S²)
                    report.append("\n⚠️ **Quadratic Memory Scaling Detected**")
                    report.append(f"- seq_len=128: {mem_128:.1f} MB")
                    report.append(f"- seq_len=512: {mem_512:.1f} MB")
                    report.append(
                        f"- Scaling factor: {scaling_factor:.1f}x (expected 4x for linear)"
                    )

    # Check for branch-specific issues
    branch_status = {}
    for test in branch_tests:
        branch = test.get("config", {}).get("branch", "all")
        status = test.get("status", test.get("result", "UNKNOWN"))
        if branch not in branch_status:
            branch_status[branch] = []
        branch_status[branch].append(status)

    for branch, statuses in branch_status.items():
        if all(s in ["FAIL", "HANG"] for s in statuses):
            report.append(f"\n❌ **Branch '{branch}' consistently fails/hangs**")

    # Recommendations
    report.append("\n## Recommendations")

    if hung > 0:
        report.append("\n### For Hang Issues:")
        report.append("1. Reduce sequence length to ≤512")
        report.append("2. Use smaller model dimension (128 instead of 768)")
        report.append("3. Enable gradient checkpointing")
        report.append("4. Consider memory-efficient attention implementations")

    if failed > 0:
        report.append("\n### For Failed Tests:")
        report.append("1. Check CUDA memory allocator settings")
        report.append("2. Verify causality constraints")
        report.append("3. Review backward hook implementations")

    # Write report
    report_text = "\n".join(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Summarize NSA backward pass test results")
    parser.add_argument("results_dir", help="Directory containing test results")
    parser.add_argument("--output", "-o", help="Output file for report (default: print to stdout)")
    parser.add_argument("--json", action="store_true", help="Also save raw JSON data")

    args = parser.parse_args()

    results = generate_summary_report(args.results_dir, args.output)

    if args.json:
        json_file = args.output.replace(".md", ".json") if args.output else "results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON data saved to: {json_file}")


if __name__ == "__main__":
    main()
