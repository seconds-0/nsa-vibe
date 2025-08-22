#!/usr/bin/env python3
import sys

try:
    from scripts.datasets.fineweb_edu_loader import iter_fineweb_edu_batches  # type: ignore
except Exception as e:
    print(f"ERR: loader import failed: {e}")
    sys.exit(2)

def main():
    def encode_bytes(s: str):
        return list(s.encode("utf-8", errors="ignore"))
    it = iter_fineweb_edu_batches(encode=encode_bytes, seq_len=128, batch_size=2, split_mod=100, split_rem=1)
    for i in range(3):
        try:
            batch = next(it)
            print(f"ok batch[{i}] lens: {[len(x) for x in batch]}")
        except StopIteration:
            print("ERR: stream exhausted early")
            break

if __name__ == "__main__":
    main()

