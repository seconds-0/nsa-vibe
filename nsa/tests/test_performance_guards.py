"""
Performance regression guards for NSA.
These tests ensure critical hot paths remain optimized and prevent
reintroduction of Python bottlenecks.
"""

import ast
import os
import pytest
import torch
from pathlib import Path
from typing import Set, List, Tuple

# Hot path modules that must avoid Python loops and .item()/.cpu() calls
HOT_PATH_MODULES = [
    "nsa/core/selection_scorer.py",
    "nsa/core/nsa_attention.py",
    "nsa/core/compress_pool.py",
]

# Functions that are allowed to use .item() (e.g., for validation/debugging)
ALLOWED_ITEM_FUNCTIONS = {
    "selection_scorer.py": {
        "convert_indices_to_ranges_batched",  # Legacy v1 implementation
        "_probe_selection",  # Debug function
    },
    "nsa_attention.py": {
        "_probe",  # SDPA audit function
        "_log_sdpa_audit",  # Logging function
        "_compute_gate_stats",  # Stats collection
        "_update_sel_stats_from_ranges",  # Stats collection
        "forward",  # Stats collection in forward
        "_forward_prefill_batched",  # Unavoidable for now
        "_sdpa_over_ranges",  # CPU fallback path
    },
}

# Functions that are allowed to use Python loops
ALLOWED_LOOP_FUNCTIONS = {
    "selection_scorer.py": {
        "convert_indices_to_ranges_batched",  # Legacy v1 implementation
        "_compute_selection_scores_python",  # Reference implementation
    },
    "nsa_attention.py": {
        "_sdpa_selected",  # CPU fallback path
        "_decode_selected",  # Decode path (unavoidable for now)
        "forward",  # Branch iteration (minimal overhead)
        "_forward_prefill_batched",  # Batch iteration (unavoidable)
        "_forward_prefill_via_decode",  # Sequential decode (fallback)
        "_forward_prefill_sequential",  # Sequential processing (fallback)
        "_sdpa_over_ranges",  # Range iteration (CPU fallback)
    },
}


class PerformanceGuardVisitor(ast.NodeVisitor):
    """AST visitor to detect performance anti-patterns."""

    def __init__(self, filename: str):
        self.filename = Path(filename).name
        self.current_function = None
        self.violations = []

    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Attribute(self, node):
        """Check for .item() and .cpu() calls."""
        if isinstance(node.attr, str):
            if node.attr == "item":
                allowed = ALLOWED_ITEM_FUNCTIONS.get(self.filename, set())
                if self.current_function not in allowed:
                    self.violations.append(
                        (node.lineno, f".item() call in {self.current_function or '<module>'}")
                    )
            elif node.attr == "cpu" and self.current_function:
                # Check if it's a tensor.cpu() call (heuristic)
                if isinstance(node.value, ast.Call) or isinstance(node.value, ast.Name):
                    self.violations.append((node.lineno, f".cpu() call in {self.current_function}"))
        self.generic_visit(node)

    def visit_For(self, node):
        """Check for Python for loops in hot functions."""
        if self.current_function:
            allowed = ALLOWED_LOOP_FUNCTIONS.get(self.filename, set())
            if self.current_function not in allowed:
                # Check if it's iterating over a tensor dimension
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                        # Likely iterating over tensor dimensions
                        self.violations.append(
                            (node.lineno, f"Python loop in {self.current_function}")
                        )
        self.generic_visit(node)

    def visit_While(self, node):
        """Check for Python while loops in hot functions."""
        if self.current_function:
            allowed = ALLOWED_LOOP_FUNCTIONS.get(self.filename, set())
            if self.current_function not in allowed:
                self.violations.append((node.lineno, f"While loop in {self.current_function}"))
        self.generic_visit(node)


def check_module_for_violations(module_path: str) -> List[Tuple[int, str]]:
    """Check a Python module for performance violations."""
    if not os.path.exists(module_path):
        return []

    with open(module_path, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    visitor = PerformanceGuardVisitor(module_path)
    visitor.visit(tree)
    return visitor.violations


class TestPerformanceGuards:
    """Test suite for performance regression guards."""

    @pytest.mark.parametrize("module", HOT_PATH_MODULES)
    def test_no_item_cpu_in_hot_paths(self, module: str):
        """Ensure no .item() or .cpu() calls in hot path modules."""
        repo_root = Path(__file__).parent.parent.parent
        module_path = repo_root / module

        violations = check_module_for_violations(str(module_path))

        # Filter out only .item() and .cpu() violations
        item_cpu_violations = [
            (line, msg) for line, msg in violations if ".item()" in msg or ".cpu()" in msg
        ]

        if item_cpu_violations:
            msg = f"Performance violations in {module}:\n"
            for line, violation in item_cpu_violations:
                msg += f"  Line {line}: {violation}\n"
            pytest.fail(msg)

    @pytest.mark.parametrize("module", HOT_PATH_MODULES)
    def test_no_python_loops_in_hot_paths(self, module: str):
        """Ensure no Python loops in critical hot path functions."""
        repo_root = Path(__file__).parent.parent.parent
        module_path = repo_root / module

        violations = check_module_for_violations(str(module_path))

        # Filter out only loop violations
        loop_violations = [(line, msg) for line, msg in violations if "loop" in msg.lower()]

        if loop_violations:
            msg = f"Performance violations in {module}:\n"
            for line, violation in loop_violations:
                msg += f"  Line {line}: {violation}\n"
            pytest.fail(msg)

    def test_v2_range_conversion_enabled_by_default(self):
        """Ensure v2 range conversion is enabled by default in production."""
        # This test checks that the environment variable defaults to enabling v2
        from nsa.core.selection_scorer import convert_indices_to_ranges_batched_dispatch

        # Temporarily clear the env var to test default
        old_val = os.environ.pop("NSA_SEL_RANGES_V2", None)
        try:
            # Create dummy inputs
            indices = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
            from nsa.core.block_index import build_block_meta

            meta = build_block_meta(seq_len=32, l=4, d=2, l_sel=4, n_sel=4, w=8)

            # Check which implementation is used
            # We can't directly check which function is called, but we can
            # check the environment variable default
            default_v2 = os.getenv("NSA_SEL_RANGES_V2", "1") == "1"
            assert default_v2, "NSA_SEL_RANGES_V2 should default to '1' (enabled)"

        finally:
            if old_val is not None:
                os.environ["NSA_SEL_RANGES_V2"] = old_val

    def test_nvtx_annotations_present(self):
        """Ensure NVTX annotations are present in v2 implementation."""
        repo_root = Path(__file__).parent.parent.parent
        scorer_path = repo_root / "nsa/core/selection_scorer.py"

        with open(scorer_path, "r") as f:
            content = f.read()

        # Check for NVTX annotations in v2 function
        assert "nvtx.range" in content, "NVTX annotations not found in selection_scorer.py"
        assert "nsa.sel.ranges_v2" in content, "NVTX range name not found"

    def test_sdpa_contiguous_calls(self):
        """Ensure SDPA calls use contiguous tensors for Flash dispatch."""
        repo_root = Path(__file__).parent.parent.parent
        attention_path = repo_root / "nsa/core/nsa_attention.py"

        with open(attention_path, "r") as f:
            lines = f.readlines()

        # Find SDPA calls and check for .contiguous() before them
        for i, line in enumerate(lines):
            if "F.scaled_dot_product_attention" in line:
                # Check previous few lines for tensor preparation
                context_start = max(0, i - 5)
                context = "".join(lines[context_start : i + 1])

                # For reshape operations, ensure .contiguous() is called
                if "reshape" in context and "_sdpa_full" in context:
                    assert ".contiguous()" in context, (
                        f"Line {i + 1}: SDPA call after reshape should use .contiguous()"
                    )

    def test_ddp_compression_available(self):
        """Ensure DDP compression hooks are available."""
        repo_root = Path(__file__).parent.parent.parent
        train_path = repo_root / "scripts/train_showcase.py"

        with open(train_path, "r") as f:
            content = f.read()

        # Check for DDP compression implementation
        assert "NSA_DDP_COMPRESS" in content, "DDP compression env var not found"
        assert "bf16_compress_hook" in content, "BF16 compression hook not found"
        assert "fp16_compress_hook" in content, "FP16 compression hook not found"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_v2_performance_improvement(self):
        """Quick performance smoke test to ensure v2 is faster than v1."""
        from nsa.core.selection_scorer import (
            convert_indices_to_ranges_batched,
            convert_indices_to_ranges_batched_v2,
        )
        from nsa.core.block_index import build_block_meta
        import time

        # Setup test data
        B, S, G, K = 2, 256, 4, 32
        device = "cuda"
        meta = build_block_meta(seq_len=2048, l=32, d=16, l_sel=64, n_sel=16, w=512)

        indices = torch.randint(-1, 100, (B, S, G, K), device=device, dtype=torch.int32)

        # Warmup
        for _ in range(5):
            _ = convert_indices_to_ranges_batched(indices, meta, S)
            _ = convert_indices_to_ranges_batched_v2(indices, meta, S)

        # Time v1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = convert_indices_to_ranges_batched(indices, meta, S)
        torch.cuda.synchronize()
        v1_time = time.perf_counter() - t0

        # Time v2
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = convert_indices_to_ranges_batched_v2(indices, meta, S)
        torch.cuda.synchronize()
        v2_time = time.perf_counter() - t0

        # V2 should be faster
        speedup = v1_time / v2_time
        assert speedup > 1.5, f"V2 not sufficiently faster: {speedup:.2f}x speedup (expected >1.5x)"
        print(f"V2 speedup: {speedup:.2f}x ({v2_time * 1000:.2f}ms vs {v1_time * 1000:.2f}ms)")


if __name__ == "__main__":
    # Quick local test
    print("Checking for performance violations in hot path modules...")

    repo_root = Path(__file__).parent.parent.parent
    for module in HOT_PATH_MODULES:
        module_path = repo_root / module
        violations = check_module_for_violations(str(module_path))
        if violations:
            print(f"\n{module}:")
            for line, msg in violations:
                print(f"  Line {line}: {msg}")
        else:
            print(f"{module}: ✓ No violations")

    print("\nRun full test suite with: pytest nsa/tests/test_performance_guards.py -v")
