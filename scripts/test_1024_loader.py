#!/usr/bin/env python3
"""Test FineWeb-Edu loader with 1024 sequence length"""

import sys

try:
    from nsa.data_pipeline import fineweb_stream_batches, Shard
except Exception as e:
    print(f"ERR: pipeline import failed: {e}")
    sys.exit(2)


def main():
    print("Testing FineWeb-Edu loader with seq_len=1024...")

    # Test with GPT-2 tokenizer (the actual use case)
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        def encode_gpt2(s: str):
            return tokenizer.encode(s)

        print("✅ GPT-2 tokenizer loaded")
    except Exception as e:
        print(f"❌ GPT-2 tokenizer failed: {e}")
        return

    # Test the data loader
    try:
        it = fineweb_stream_batches(
            encode=encode_gpt2,
            seq_len=1024,
            batch_size=1,
            shard=Shard(mod=100, rem=1),
        )

        for i in range(3):
            try:
                batch = next(it)
                seq_lens = [len(x) for x in batch]
                print(f"✅ batch[{i}] lens: {seq_lens}")

                # Validate all sequences are exactly 1024
                for j, seq_len in enumerate(seq_lens):
                    if seq_len != 1024:
                        print(f"❌ ERROR: batch[{i}] seq[{j}] has length {seq_len}, expected 1024")
                        return

            except StopIteration:
                print("❌ ERR: stream exhausted early")
                break
            except Exception as e:
                print(f"❌ ERR: {e}")
                break

        print("✅ All tests passed! Data loader working correctly.")

    except Exception as e:
        print(f"❌ Data loader test failed: {e}")


if __name__ == "__main__":
    main()
