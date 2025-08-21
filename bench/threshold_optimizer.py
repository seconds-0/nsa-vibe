"""
Threshold optimizer for NSA FlashAttention-2 configuration.

This module parses benchmark results from various sources (Modal, local runs, CI)
and determines optimal FA-2 thresholds, then updates the configuration.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BenchmarkData:
    """Container for benchmark data from various sources."""

    device: str
    torch_version: str
    cuda_version: str
    sliding_results: list[dict]  # List of {S, w, speedup, masked_ms, fa2_ms}
    compressed_results: list[dict]  # List of {S, speedup, masked_ms, fa2_ms}
    selection_rows: list[dict] | None = None


class ThresholdOptimizer:
    """Optimizes FA-2 thresholds based on benchmark results."""

    def __init__(self, safety_margin: float = 1.2):
        """
        Initialize optimizer.

        Args:
            safety_margin: Minimum speedup required to enable FA-2 (default 1.2 = 20% faster)
        """
        self.safety_margin = safety_margin

    def parse_bench_output(self, output: str) -> tuple[list[dict], list[dict]]:
        """
        Parse raw benchmark output from bench_fa2.py.

        Returns:
            (sliding_results, compressed_results)
        """
        sliding_results = []
        compressed_results = []

        # Parse sliding: "S=256 w=32 sliding masked X.XX ms  fa2 Y.YY ms  speedup xZ.ZZ"
        sliding_pattern = (
            r"S=(\d+) w=(\d+) sliding masked ([\d.]+) ms\s+fa2 ([\d.]+) ms\s+speedup x([\d.]+)"
        )
        for match in re.finditer(sliding_pattern, output):
            s, w, masked_ms, fa2_ms, speedup = match.groups()
            sliding_results.append(
                {
                    "S": int(s),
                    "w": int(w),
                    "speedup": float(speedup),
                    "masked_ms": float(masked_ms),
                    "fa2_ms": float(fa2_ms),
                }
            )

        # Parse compressed: "S=256 l=32 d=16 compressed masked X.XX ms  fa2 Y.YY ms  speedup xZ.ZZ"
        compressed_pattern = r"S=(\d+) l=\d+ d=\d+ compressed masked ([\d.]+) ms\s+fa2 ([\d.]+) ms\s+speedup x([\d.]+)"
        for match in re.finditer(compressed_pattern, output):
            s, masked_ms, fa2_ms, speedup = match.groups()
            compressed_results.append(
                {
                    "S": int(s),
                    "speedup": float(speedup),
                    "masked_ms": float(masked_ms),
                    "fa2_ms": float(fa2_ms),
                }
            )

        return sliding_results, compressed_results

    def load_modal_results(self, json_path: str | Path) -> BenchmarkData:
        """Load results from Modal benchmark JSON output."""
        with open(json_path) as f:
            data = json.load(f)

        sliding = []
        compressed = []

        for r in data.get("results", []):
            if r["branch"] == "sliding":
                sliding.append(
                    {
                        "S": r["sequence_length"],
                        "w": r["window_size"],
                        "speedup": r["speedup"],
                        "masked_ms": r["masked_ms"],
                        "fa2_ms": r["fa2_ms"],
                    }
                )
            elif r["branch"] == "compressed":
                compressed.append(
                    {
                        "S": r["sequence_length"],
                        "speedup": r["speedup"],
                        "masked_ms": r["masked_ms"],
                        "fa2_ms": r["fa2_ms"],
                    }
                )

        # Selection rows (optional)
        sel_rows = []
        sel_out = data.get("selection_output", "")
        if sel_out:
            # Mirror parser from modal bench
            for line in sel_out.splitlines():
                line = line.strip()
                if not (line.startswith("dense,") or line.startswith("varlen,")):
                    continue
                parts = line.split(",")
                row: dict[str, object] = {"mode": parts[0]}
                for kv in parts[1:]:
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ("N", "H", "D", "Dv", "L", "n", "streams"):
                        row[k if k != "n" else "nspans"] = int(v)
                    elif k in ("tri_ms", "ref_ms", "speedup", "mae"):
                        row[k] = float(v.replace("x", ""))
                sel_rows.append(row)

        return BenchmarkData(
            device=data["device_info"]["device_name"],
            torch_version=data["device_info"]["torch_version"],
            cuda_version=data["device_info"]["cuda_version"],
            sliding_results=sliding,
            compressed_results=compressed,
            selection_rows=sel_rows or None,
        )

    def load_text_output(self, text_path: str | Path) -> BenchmarkData:
        """Load results from raw text benchmark output."""
        with open(text_path) as f:
            output = f.read()

        sliding, compressed = self.parse_bench_output(output)

        # Try to extract device info from output
        device = "Unknown"
        torch_version = "Unknown"
        cuda_version = "Unknown"

        # Look for device info patterns
        device_match = re.search(r"GPU:\s*(.+)", output)
        if device_match:
            device = device_match.group(1)

        return BenchmarkData(
            device=device,
            torch_version=torch_version,
            cuda_version=cuda_version,
            sliding_results=sliding,
            compressed_results=compressed,
        )

    def determine_thresholds(self, data: BenchmarkData) -> tuple[int, int]:
        """
        Determine optimal thresholds from benchmark data.

        Returns:
            (fa2_min_len_win, fa2_min_len_cmp)
        """
        # Sliding window threshold
        win_threshold = self._find_sliding_threshold(data.sliding_results)

        # Compressed threshold
        cmp_threshold = self._find_compressed_threshold(data.compressed_results)

        return win_threshold, cmp_threshold

    def determine_sel_threshold(
        self, data: BenchmarkData, safety_margin: float | None = None
    ) -> int:
        rows = data.selection_rows or []
        if not rows:
            return 4096
        margin = safety_margin or self.safety_margin
        L_values = sorted(
            {int(r.get("L", 0)) for r in rows if int(r.get("streams", 1)) == 1 and "speedup" in r}
        )
        for L in L_values:
            rL = [r for r in rows if int(r.get("L", -1)) == L and int(r.get("streams", 1)) == 1]
            if rL and all(r.get("speedup", 0.0) >= margin for r in rL):
                return L
        return max(L_values[-1], 4096) if L_values else 4096

    def _find_sliding_threshold(self, results: list[dict]) -> int:
        """Find minimum window size where FA-2 consistently beats masked."""
        if not results:
            return 512  # Conservative default

        # Group by window size
        window_sizes = sorted(set(r["w"] for r in results))

        for w in window_sizes:
            w_results = [r for r in results if r["w"] == w]

            # Check if ALL sequence lengths show speedup above margin
            if all(r["speedup"] >= self.safety_margin for r in w_results):
                return w

        # If no window size meets criteria, use conservative default
        return max(window_sizes[-1], 512) if window_sizes else 512

    def _find_compressed_threshold(self, results: list[dict]) -> int:
        """Find minimum compressed length where FA-2 beats masked."""
        if not results:
            return 32  # Conservative default

        # Check different sequence lengths
        seq_lengths = sorted(set(r["S"] for r in results))

        # Find smallest S where we get consistent speedup
        for s in seq_lengths:
            s_results = [r for r in results if r["S"] == s]

            if all(r["speedup"] >= self.safety_margin for r in s_results):
                # Map to a threshold value
                # Aggressive: use 16 for S>=256, conservative: use 32
                if s <= 256:
                    return 32
                elif s <= 512:
                    return 16
                else:
                    return 8

        return 32  # Conservative default

    def update_config(
        self,
        config_path: str | Path,
        win_threshold: int,
        cmp_threshold: int,
        sel_threshold: int | None = None,
        backup: bool = True,
    ) -> None:
        """
        Update configuration file with new thresholds.

        Args:
            config_path: Path to config YAML file
            win_threshold: New fa2_min_len_win value
            cmp_threshold: New fa2_min_len_cmp value
            backup: Whether to create a backup of the original
        """
        config_path = Path(config_path)

        # Load existing config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Backup if requested
        if backup:
            backup_path = config_path.with_suffix(".yaml.bak")
            with open(backup_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Update thresholds
        if "runtime" not in config:
            config["runtime"] = {}

        config["runtime"]["fa2_min_len_win"] = win_threshold
        config["runtime"]["fa2_min_len_cmp"] = cmp_threshold
        if sel_threshold is not None:
            config["runtime"]["sel_triton_min_L"] = sel_threshold

        # Write updated config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def generate_report(
        self,
        data: BenchmarkData,
        win_threshold: int,
        cmp_threshold: int,
        sel_threshold: int | None = None,
    ) -> str:
        """Generate markdown report of benchmark results and recommendations."""
        sel_line = (
            f"- `runtime.sel_triton_min_L`: **{sel_threshold}**\n"
            if sel_threshold is not None
            else ""
        )

        report = f"""# GPU Benchmark Results

## Device Information
- **GPU**: {data.device}
- **PyTorch**: {data.torch_version}
- **CUDA**: {data.cuda_version}
- **Timestamp**: {Path(__file__).stat().st_mtime if Path(__file__).exists() else "N/A"}

## Recommended Thresholds
- `runtime.fa2_min_len_win`: **{win_threshold}**
- `runtime.fa2_min_len_cmp`: **{cmp_threshold}**
{sel_line}
## Benchmark Results

### Sliding Window Performance
| S | w | Speedup | Masked (ms) | FA-2 (ms) |
|---|---|---------|-------------|-----------|
"""
        for r in sorted(data.sliding_results, key=lambda x: (x["S"], x["w"])):
            emoji = "✅" if r["speedup"] >= self.safety_margin else "❌"
            report += f"| {r['S']} | {r['w']} | {r['speedup']:.2f}x {emoji} | {r['masked_ms']:.2f} | {r['fa2_ms']:.2f} |\n"

        report += """
### Compressed Branch Performance
| S | Speedup | Masked (ms) | FA-2 (ms) |
|---|---------|-------------|-----------|
"""
        for r in sorted(data.compressed_results, key=lambda x: x["S"]):
            emoji = "✅" if r["speedup"] >= self.safety_margin else "❌"
            report += f"| {r['S']} | {r['speedup']:.2f}x {emoji} | {r['masked_ms']:.2f} | {r['fa2_ms']:.2f} |\n"

        report += f"""
## Analysis

With a safety margin of {self.safety_margin:.1f}x:
- **Sliding**: FA-2 is faster for window sizes ≥ {win_threshold}
- **Compressed**: FA-2 is faster for effective lengths ≥ {cmp_threshold}

These thresholds ensure FA-2 is only used when it provides at least {int((self.safety_margin - 1) * 100)}% speedup.
"""
        return report


def main():
    """CLI interface for threshold optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize NSA FA-2 thresholds from benchmark results"
    )
    parser.add_argument("input", help="Input file (JSON from Modal or text output)")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file to update")
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.2,
        help="Minimum speedup to enable FA-2 (default: 1.2)",
    )
    parser.add_argument("--report", help="Output markdown report file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't update config, just show results"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = ThresholdOptimizer(safety_margin=args.safety_margin)

    # Load benchmark data
    input_path = Path(args.input)
    if input_path.suffix == ".json":
        data = optimizer.load_modal_results(input_path)
    else:
        data = optimizer.load_text_output(input_path)

    # Determine thresholds
    win_threshold, cmp_threshold = optimizer.determine_thresholds(data)
    sel_threshold = optimizer.determine_sel_threshold(data)

    # Print results
    print(f"Device: {data.device}")
    print(f"Safety Margin: {args.safety_margin:.1f}x")
    print("\nRecommended Thresholds:")
    print(f"  fa2_min_len_win: {win_threshold}")
    print(f"  fa2_min_len_cmp: {cmp_threshold}")
    print(f"  sel_triton_min_L: {sel_threshold}")

    # Generate report if requested
    if args.report:
        report = optimizer.generate_report(data, win_threshold, cmp_threshold, sel_threshold)
        with open(args.report, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.report}")

    # Update config unless dry-run
    if not args.dry_run:
        optimizer.update_config(args.config, win_threshold, cmp_threshold, sel_threshold)
        print(f"\nConfig updated: {args.config}")
    else:
        print("\n(Dry run - config not updated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
