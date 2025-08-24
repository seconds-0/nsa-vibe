#!/usr/bin/env python3
import sys

try:
    from nsa.data_pipeline import fineweb_stream_batches, Shard  # type: ignore
except Exception as e:
    print(f"ERR: pipeline import failed: {e}")
    sys.exit(2)

def main():
    def encode_bytes(s: str):
        return list(s.encode("utf-8", errors="ignore"))
    it = fineweb_stream_batches(encode=encode_bytes, seq_len=128, batch_size=2, shard=Shard(mod=100, rem=1),)
    for i in range(3):
        try:
            batch = next(it)
            print(f"ok batch[{i}] lens: {[len(x) for x in batch]}")
        except StopIteration:
            print("ERR: stream exhausted early")
            break

if __name__ == "__main__":
    main()
