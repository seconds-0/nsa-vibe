#!/usr/bin/env python3
"""Data pipeline utilities for streaming and local datasets.

Provides a FineWeb-Edu IterableDataset and simple local JSONL/TXT loaders.
This module is optional; scripts/train_showcase.py currently uses a simpler
loader in scripts/datasets. Migrate incrementally as needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional


Tokenizer = Callable[[str], List[int]]


@dataclass
class Shard:
    mod: int = 1
    rem: int = 0


def fineweb_stream_batches(
    encode: Tokenizer,
    seq_len: int,
    batch_size: int,
    shard: Shard = Shard(),
    report_docs: int = 1000,
) -> Iterator[List[List[int]]]:
    try:
        from datasets import load_dataset, Features, Value  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets package required. Install with: pip install datasets") from e

    features = Features(
        {
            "text": Value("string"),
            "id": Value("string"),
            "dump": Value("string"),
            "url": Value("string"),
            "file_path": Value("string"),
            "language": Value("string"),
            "language_score": Value("float64"),
            "token_count": Value("int64"),
            "score": Value("float64"),
            "int_score": Value("int64"),
        }
    )
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, features=features)
    buf: List[int] = []
    batch: List[List[int]] = []
    seen = 0
    import time as _t
    t0 = _t.time()
    last = t0
    for ex in ds:
        if seen % shard.mod != shard.rem:
            seen += 1
            continue
        seen += 1
        if report_docs and seen % report_docs == 0:
            dt = _t.time() - last
            print(f"[fwe] seen_docs={seen} dt={dt:.1f}s buf={len(buf)}", flush=True)
            last = _t.time()
        text = ex.get("text") or ""
        if not text:
            continue
        toks = encode(text)
        if not toks:
            continue
        buf.extend(toks)
        while len(buf) >= seq_len:
            seq = buf[:seq_len]
            buf = buf[seq_len:]
            batch.append(seq)
            if len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]


def fineweb_stream_batches_batched(
    encode_batch: Callable[[List[str]], List[List[int]]],
    seq_len: int,
    batch_size: int,
    shard: Shard = Shard(),
    report_docs: int = 1000,
    doc_batch: int = 64,
) -> Iterator[List[List[int]]]:
    """Streaming FineWeb‑Edu with batched tokenization and fixed-length packing.

    - encode_batch: function mapping a list of texts -> list of token id lists
    - Packs contiguous tokens from a rolling buffer into fixed seq_len examples
    - Yields Python lists of shape [batch_size][seq_len]
    """
    try:
        from datasets import load_dataset, Features, Value  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets package required. Install with: pip install datasets") from e

    features = Features(
        {
            "text": Value("string"),
            "id": Value("string"),
            "dump": Value("string"),
            "url": Value("string"),
            "file_path": Value("string"),
            "language": Value("string"),
            "language_score": Value("float64"),
            "token_count": Value("int64"),
            "score": Value("float64"),
            "int_score": Value("int64"),
        }
    )
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, features=features)

    buf: List[int] = []
    batch: List[List[int]] = []
    seen = 0
    acc_texts: List[str] = []
    import time as _t
    last = _t.time()
    for ex in ds:
        if seen % shard.mod != shard.rem:
            seen += 1
            continue
        seen += 1
        if report_docs and seen % report_docs == 0:
            dt = _t.time() - last
            print(f"[fwe] (batched) seen_docs={seen} dt={dt:.1f}s buf={len(buf)} acc_texts={len(acc_texts)}", flush=True)
            last = _t.time()
        text = ex.get("text") or ""
        if not text:
            continue
        acc_texts.append(text)
        if len(acc_texts) < max(1, int(doc_batch)):
            continue
        # Batched tokenize
        try:
            toks_list = encode_batch(acc_texts)
        except Exception:
            # Fallback to per-doc encode if batch path fails
            toks_list = []
            for t in acc_texts:
                try:
                    toks_list.append(encode_batch([t])[0])
                except Exception:
                    toks_list.append([])
        acc_texts.clear()
        # Fill rolling buffer and output fixed-length sequences
        for toks in toks_list:
            if not toks:
                continue
            buf.extend(toks)
            while len(buf) >= seq_len:
                seq = buf[:seq_len]
                buf = buf[seq_len:]
                batch.append(seq)
                if len(batch) >= batch_size:
                    yield batch[:batch_size]
                    batch = batch[batch_size:]


def local_jsonl_or_txt_batches(
    path: str,
    encode: Tokenizer,
    seq_len: int,
    batch_size: int,
) -> Iterator[List[List[int]]]:
    is_jsonl = path.endswith(".jsonl")
    buf: List[int] = []
    batch: List[List[int]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            text = line
            if is_jsonl:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        text = obj["text"]
                except Exception:
                    pass
            toks = encode(text)
            if not toks:
                continue
            buf.extend(toks)
            while len(buf) >= seq_len:
                seq = buf[:seq_len]
                buf = buf[seq_len:]
                batch.append(seq)
                if len(batch) >= batch_size:
                    yield batch[:batch_size]
                    batch = batch[batch_size:]
