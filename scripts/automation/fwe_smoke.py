#!/usr/bin/env python3
import argparse
import time
from typing import List


def main():
    ap = argparse.ArgumentParser(description="FineWeb‑Edu streaming smoke test")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--report-docs", type=int, default=1000)
    ap.add_argument("--tokenizer", type=str, default="byte", choices=["byte", "gpt2"])
    args = ap.parse_args()

    import os
    os.environ["NSA_FWE_REPORT_DOCS"] = str(int(args.report_docs))
    encode = None
    if args.tokenizer == "gpt2":
        try:
            from transformers import GPT2Tokenizer  # type: ignore
            tok = GPT2Tokenizer.from_pretrained("gpt2")
        except Exception as e:
            raise SystemExit(f"transformers/GPT2Tokenizer required for gpt2 mode: {e}")
        def encode_bytes(s: str) -> List[int]:
            t = tok.encode(s)
            return t[: args.seq_len - 1]
        encode = encode_bytes
    else:
        def encode_bytes(s: str) -> List[int]:
            t = list(s.encode("utf-8", errors="ignore"))
            return t[: args.seq_len - 1]
        encode = encode_bytes

    try:
        from scripts.datasets.fineweb_edu_loader import iter_fineweb_edu_batches  # type: ignore
    except Exception as e:
        raise SystemExit(f"failed to import loader: {e}")

    it = iter_fineweb_edu_batches(encode=encode, seq_len=args.seq_len, batch_size=args.batch, split_mod=100, split_rem=1)
    print(f"[smoke] pulling first batch: S={args.seq_len} B={args.batch} tokenizer={args.tokenizer}")
    box = {}
    import threading
    def _pull():
        nonlocal box
        t0 = time.time()
        try:
            b = next(it)
            box = {"ok": True, "dt": time.time() - t0, "shape": (len(b), len(b[0]) if b else 0)}
        except Exception as e:
            box = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    th = threading.Thread(target=_pull, daemon=True)
    th.start()
    th.join(timeout=args.timeout)
    if not box.get("ok"):
        print(f"[smoke][FAIL] loader error or timeout ({args.timeout}s): {box.get('err','timeout')}")
        raise SystemExit(2)
    print(f"[smoke][OK] first batch in {box['dt']:.2f}s shape={box['shape']}")


if __name__ == "__main__":
    main()

