"""
Equivalence tests for GPU vectorized selection range conversion (v2).
Ensures the new NSA_SEL_RANGES_V2 path produces identical results to the original
Python loop implementation across various edge cases.
"""

import os
import torch
import pytest
from typing import Optional

from nsa.core.block_index import build_block_meta, BlockMeta
from nsa.core.selection_scorer import (
    convert_indices_to_ranges_batched,
    convert_indices_to_ranges_batched_v2,
)


def generate_test_indices(
    B: int,
    S: int,
    G: int,
    K: int,
    device: str = "cpu",
    pattern: str = "random",
    seed: int = 42,
) -> torch.Tensor:
    """Generate test indices with various patterns."""
    torch.manual_seed(seed)

    if pattern == "random":
        # Random valid indices with -1 padding
        indices = torch.randint(-1, 16, (B, S, G, K), device=device)
        # Ensure some are valid
        indices[:, :, :, 0] = torch.randint(0, 8, (B, S, G), device=device)
    elif pattern == "sequential":
        # Sequential indices [0, 1, 2, ...] with some padding
        indices = torch.arange(K, device=device).view(1, 1, 1, K).expand(B, S, G, K)
        # Add some -1 padding
        mask = torch.rand(B, S, G, K, device=device) > 0.7
        indices = torch.where(mask, torch.tensor(-1, device=device), indices)
    elif pattern == "duplicates":
        # Indices with many duplicates
        base = torch.randint(0, 4, (B, S, G, K // 2), device=device)
        indices = torch.cat([base, base], dim=-1)
        # Add -1 padding
        mask = torch.rand(B, S, G, K, device=device) > 0.8
        indices = torch.where(mask, torch.tensor(-1, device=device), indices)
    elif pattern == "gaps":
        # Indices with gaps (non-adjacent)
        indices = torch.arange(0, K * 2, 2, device=device)[:K]
        indices = indices.view(1, 1, 1, K).expand(B, S, G, K)
        mask = torch.rand(B, S, G, K, device=device) > 0.7
        indices = torch.where(mask, torch.tensor(-1, device=device), indices)
    elif pattern == "all_invalid":
        # All -1s
        indices = torch.full((B, S, G, K), -1, device=device)
    elif pattern == "single_valid":
        # Only one valid index per row
        indices = torch.full((B, S, G, K), -1, device=device)
        indices[:, :, :, 0] = torch.randint(0, 10, (B, S, G), device=device)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Sort each row (required by both implementations)
    for b in range(B):
        for s in range(S):
            for g in range(G):
                valid_mask = indices[b, s, g] >= 0
                valid_indices = indices[b, s, g][valid_mask]
                if valid_indices.numel() > 0:
                    sorted_valid, _ = torch.sort(valid_indices)
                    indices[b, s, g, : sorted_valid.numel()] = sorted_valid
                    indices[b, s, g, sorted_valid.numel() :] = -1

    return indices


def ranges_are_equivalent(
    ranges1: torch.Tensor,
    ranges2: torch.Tensor,
    tolerance: float = 0.0,
) -> bool:
    """Check if two range tensors are equivalent, ignoring zero-length padding."""
    B, S, G = ranges1.shape[:3]

    for b in range(B):
        for s in range(S):
            for g in range(G):
                # Extract non-zero ranges from both
                r1_list = []
                for i in range(ranges1.shape[3]):
                    start, end = ranges1[b, s, g, i].tolist()
                    if end > start:  # Non-zero length
                        r1_list.append((start, end))

                r2_list = []
                for i in range(ranges2.shape[3]):
                    start, end = ranges2[b, s, g, i].tolist()
                    if end > start:  # Non-zero length
                        r2_list.append((start, end))

                # Compare
                if len(r1_list) != len(r2_list):
                    return False
                for (s1, e1), (s2, e2) in zip(r1_list, r2_list):
                    if abs(s1 - s2) > tolerance or abs(e1 - e2) > tolerance:
                        return False

    return True


def verify_causality(ranges: torch.Tensor, S: int) -> bool:
    """Verify that no range extends beyond t+1 for position t."""
    B, S_q, G = ranges.shape[:3]

    for b in range(B):
        for t in range(min(S_q, S)):
            for g in range(G):
                for i in range(ranges.shape[3]):
                    start, end = ranges[b, t, g, i].tolist()
                    if end > start:  # Valid range
                        if end > t + 1:
                            print(
                                f"Causality violation at b={b}, t={t}, g={g}: range [{start}, {end}) exceeds t+1={t + 1}"
                            )
                            return False
    return True


class TestSelectionV2Equivalence:
    """Test suite for v2 selection range conversion equivalence."""

    @pytest.mark.parametrize(
        "pattern", ["random", "sequential", "duplicates", "gaps", "single_valid"]
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_equivalence_various_patterns(self, pattern: str, device: str):
        """Test that v2 produces identical results to v1 across various patterns."""
        B, S, G, K = 2, 8, 2, 12
        meta = build_block_meta(
            seq_len=64,
            l=4,
            d=2,
            l_sel=4,
            n_sel=8,
            w=16,
        )

        indices = generate_test_indices(B, S, G, K, device=device, pattern=pattern)

        # Run both implementations
        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        # Check equivalence
        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            f"Pattern {pattern}: v1 and v2 produce different results"
        )

        # Verify causality for both
        assert verify_causality(ranges_v1, S), f"Pattern {pattern}: v1 violates causality"
        assert verify_causality(ranges_v2, S), f"Pattern {pattern}: v2 violates causality"

    def test_empty_input(self):
        """Test with completely empty input."""
        B, S, G, K = 1, 4, 1, 0
        meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

        indices = torch.zeros((B, S, G, K), dtype=torch.int32)

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_v1.shape == ranges_v2.shape
        assert ranges_v1.shape[-1] == 2  # Should have 2D ranges

    def test_all_invalid(self):
        """Test with all invalid (-1) indices."""
        B, S, G, K = 2, 4, 2, 8
        meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

        indices = torch.full((B, S, G, K), -1, dtype=torch.int32)

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        # Both should produce all zero-length ranges
        assert ranges_are_equivalent(ranges_v1, ranges_v2)

        # Check that all ranges are zero-length
        for b in range(B):
            for s in range(S):
                for g in range(G):
                    for i in range(ranges_v2.shape[3]):
                        start, end = ranges_v2[b, s, g, i].tolist()
                        assert start == 0 and end == 0, (
                            "Expected zero-length range for all-invalid input"
                        )

    def test_adjacency_merging(self):
        """Test that adjacent blocks are properly merged."""
        B, S, G, K = 1, 2, 1, 6
        meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

        # Create adjacent indices [0, 1, 2] and [5, 6]
        indices = torch.tensor([[[[0, 1, 2, 5, 6, -1]]]], dtype=torch.int32)
        indices = indices.repeat(B, S, G, 1)

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            "Adjacent block merging produces different results"
        )

    def test_duplicate_handling(self):
        """Test that duplicates are properly handled."""
        B, S, G, K = 1, 2, 1, 8
        meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

        # Create duplicated indices [0, 0, 1, 1, 2, 2]
        indices = torch.tensor([[[[0, 0, 1, 1, 2, 2, -1, -1]]]], dtype=torch.int32)
        indices = indices.repeat(B, S, G, 1)

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            "Duplicate handling produces different results"
        )

    def test_causality_clamping(self):
        """Test that ranges are properly clamped to maintain causality."""
        B, S, G, K = 1, 8, 1, 4
        meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

        # Create indices that would violate causality without clamping
        indices = torch.zeros((B, S, G, K), dtype=torch.int32)
        for t in range(S):
            # Try to select blocks beyond t
            indices[0, t, 0, :] = torch.tensor([t // 2, t // 2 + 1, t // 2 + 2, -1])

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            "Causality clamping produces different results"
        )

        # Verify both maintain causality
        assert verify_causality(ranges_v1, S), "v1 violates causality"
        assert verify_causality(ranges_v2, S), "v2 violates causality"

    @pytest.mark.parametrize("B", [1, 2, 4])
    @pytest.mark.parametrize("S", [1, 8, 16, 32])
    @pytest.mark.parametrize("G", [1, 2, 4])
    def test_various_shapes(self, B: int, S: int, G: int):
        """Test with various tensor shapes."""
        K = min(16, S * 2)  # Reasonable K relative to S
        meta = build_block_meta(
            seq_len=max(64, S * 4),  # Ensure enough blocks
            l=4,
            d=2,
            l_sel=4,
            n_sel=min(8, K),
            w=16,
        )

        indices = generate_test_indices(B, S, G, K, pattern="random", seed=B * 100 + S * 10 + G)

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            f"Shape B={B}, S={S}, G={G}: v1 and v2 produce different results"
        )

    def test_large_scale(self):
        """Test with production-like dimensions."""
        B, S, G, K = 2, 256, 4, 32  # Moderate production scale
        meta = build_block_meta(
            seq_len=2048,
            l=32,
            d=16,
            l_sel=64,
            n_sel=16,
            w=512,
        )

        indices = generate_test_indices(B, S, G, K, pattern="random")

        ranges_v1 = convert_indices_to_ranges_batched(indices, meta, S)
        ranges_v2 = convert_indices_to_ranges_batched_v2(indices, meta, S)

        assert ranges_are_equivalent(ranges_v1, ranges_v2), (
            "Large scale test: v1 and v2 produce different results"
        )

        assert verify_causality(ranges_v1, S), "Large scale: v1 violates causality"
        assert verify_causality(ranges_v2, S), "Large scale: v2 violates causality"


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running quick smoke test...")
    test_suite = TestSelectionV2Equivalence()

    # Test basic equivalence
    test_suite.test_equivalence_various_patterns("random", "cpu")
    test_suite.test_adjacency_merging()
    test_suite.test_duplicate_handling()
    test_suite.test_causality_clamping()

    print("✓ All smoke tests passed!")
    print("\nRun full test suite with: pytest nsa/tests/test_selection_v2_equiv.py -v")
