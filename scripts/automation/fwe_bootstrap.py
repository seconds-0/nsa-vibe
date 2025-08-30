#!/usr/bin/env python
"""
Bootstrap FineWeb‑Edu locally into a JSONL file to reduce cold start latency.

Usage:
  python scripts/automation/fwe_bootstrap.py --out /data/fwe_bootstrap.jsonl --bytes 5368709120

Notes:
  - Streams from HuggingFaceFW/fineweb-edu (train split) and writes ~N bytes locally.
  - Use with production runbook via: --dataset fineweb_edu_local --local-path <out>
"""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--bytes", type=int, default=5 * 1024 ** 3, help="Approx bytes to write (default 5GiB)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(f"datasets import failed: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)

    written = 0
    with out.open('w', encoding='utf-8') as w:
        for ex in ds:
            t = ex.get('text') or ''
            if not t:
                continue
            s = json.dumps({'text': t}, ensure_ascii=False)
            w.write(s + '
')
            written += len(s.encode('utf-8')) + 1
            if written >= args.bytes:
                break
    print(json.dumps({"out": str(out), "bytes": written}))

if __name__ == '__main__':
    main()
