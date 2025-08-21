#!/usr/bin/env python3
"""FineWeb-Edu streaming data loader for training."""
from typing import Callable, Iterator, List


def iter_fineweb_edu_batches(
    encode: Callable[[str], List[int]],
    seq_len: int,
    batch_size: int,
    split_mod: int = 1,
    split_rem: int = 0,
) -> Iterator[List[List[int]]]:
    """
    Stream FineWeb-Edu dataset in batches.
    
    Args:
        encode: Function to tokenize text (str -> List[int])
        seq_len: Sequence length for each sample
        batch_size: Number of sequences per batch
        split_mod: Modulo for sharding (for multi-GPU)
        split_rem: Remainder for sharding (rank ID)
    
    Yields:
        Batches of tokenized sequences: List[List[int]] of shape [batch_size, seq_len]
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("datasets package required for FineWeb-Edu. Install with: pip install datasets") from e
    
    # Load FineWeb-Edu in streaming mode
    # Define features to handle schema mismatch (missing 'date' field)
    from datasets import Features, Value
    features = Features({
        'text': Value('string'),
        'id': Value('string'), 
        'dump': Value('string'),
        'url': Value('string'),
        'file_path': Value('string'),
        'language': Value('string'),
        'language_score': Value('float64'),
        'token_count': Value('int64'),
        'score': Value('float64'),
        'int_score': Value('int64')
    })
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
        features=features
    )
    
    # Buffer for accumulating tokens across documents
    buffer = []
    batch = []
    doc_idx = 0
    
    for doc in dataset:
        # Shard documents across workers (for DDP)
        if doc_idx % split_mod != split_rem:
            doc_idx += 1
            continue
        doc_idx += 1
        
        # Extract and tokenize text
        text = doc.get("text", "")
        if not text:
            continue
            
        tokens = encode(text)
        if not tokens:
            continue
            
        buffer.extend(tokens)
        
        # Create sequences from buffer
        while len(buffer) >= seq_len:
            sequence = buffer[:seq_len]
            buffer = buffer[seq_len:]
            batch.append(sequence)
            
            # Yield complete batches
            if len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]