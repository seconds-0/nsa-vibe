#!/usr/bin/env python3
"""
Generate a compact Markdown report from a one-shot artifacts directory.

Usage:
  python scripts/report_from_artifacts.py artifacts/runner/<commit> > report.md
"""
import argparse
from pathlib import Path
import json


def read_text(p: Path) -> str:
    try:
        return p.read_text().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", help="Path to artifacts/runner/<commit> directory")
    args = ap.parse_args()
    base = Path(args.artifacts)
    env = read_text(base / "env.txt")
    routing = read_text(base / "routing.json")
    sanity = read_text(base / "sanity.out")
    dec_csv = read_text(base / "decode_gpu_final.csv")
    dec_sum = read_text(base / "decode_summary.txt")
    tfwd = read_text(base / "triton_fwd.txt")
    tbwd = read_text(base / "triton_bwd.txt")
    fa2_probe = read_text(base / "fa2_probe.txt")
    fa2_var = read_text(base / "fa2_varlen.txt")

    print(f"# NSA One-Shot Report — {base.name}")
    print("\n## Environment\n")
    print("```\n" + env + "\n```")
    print("\n## Routing\n")
    try:
        info = json.loads(routing)
        print("```json\n" + json.dumps(info, indent=2) + "\n```")
    except Exception:
        print("```\n" + routing + "\n```")
    print("\n## Sanity\n")
    print("```\n" + sanity + "\n```")
    print("\n## Triton\n")
    print("### Forward\n")
    print("```\n" + tfwd + "\n```")
    print("### Backward\n")
    print("```\n" + tbwd + "\n```")
    print("\n## Decode Bench Summary\n")
    print("```\n" + dec_sum + "\n```")
    print("\n## FA-2\n")
    print("### Probe\n")
    print("```\n" + fa2_probe + "\n```")
    if fa2_var:
        print("### Varlen Parity\n")
        print("```\n" + fa2_var + "\n```")


if __name__ == "__main__":
    main()

