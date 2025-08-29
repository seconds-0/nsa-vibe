# 2025-08-29 Test Engineer Report - NSA Prefill Batched NaN Issue v1

## Executive Summary
**Status**: BUG IDENTIFIED - Boolean mask causing NaN with SDPA in batched prefill mode
**Severity**: High - Breaks core functionality when NSA_PREFILL_BATCHED=1
**Impact**: CI build-batched-smoke tests failing

## Issue Description

When `NSA_PREFILL_BATCHED=1` is enabled, the `sliding_window_attention` function in `nsa/core/attention_kernels.py` produces NaN values in its output, causing test failures.

## Root Cause Analysis

### Problem Location
File: `nsa/core/attention_kernels.py`, lines 133-145
Function: `sliding_window_attention`

### Technical Details

The function uses a boolean mask with `F.scaled_dot_product_attention`:

```python
# Line 133: Create boolean mask
disallowed = ~allowed  # [S,S] boolean tensor

# Line 144-145: Use boolean mask with SDPA
Mf = disallowed.view(1, 1, S, S).expand(B, G * h, S, S)
Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
```

This causes NaN values in certain PyTorch configurations when using boolean masks with SDPA.

## Reproduction

### Minimal Test Case
```python
import torch
import torch.nn.functional as F

B, S, G, h, Dk, Dv = 1, 8, 1, 4, 16, 16
Q = torch.randn(B, S, G, h, Dk)
K = torch.randn(B, G, S, Dk)
V = torch.randn(B, G, S, Dv)

# Boolean mask (FAILS)
disallowed = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
Qf = Q.reshape(B, S, G * h, Dk).transpose(1, 2)
Kf = K.unsqueeze(2).expand(B, G, h, S, Dk).reshape(B, G * h, S, Dk)
Vf = V.unsqueeze(2).expand(B, G, h, S, V.shape[-1]).reshape(B, G * h, S, V.shape[-1])
Mf = disallowed.view(1, 1, S, S).expand(B, G * h, S, S)
Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
print(f"Boolean mask NaN: {torch.isnan(Of).any()}")  # True
```

### Test Results
- **NSA_PREFILL_BATCHED=0**: ✅ PASS (uses different code path)
- **NSA_PREFILL_BATCHED=1**: ❌ FAIL (NaN values)

## Solution

Replace boolean mask with float mask using `-inf` for masked positions:

```python
# Instead of:
disallowed = ~allowed  # boolean
Mf = disallowed.view(1, 1, S, S).expand(B, G * h, S, S)

# Use:
float_mask = torch.zeros(S, S, device=device, dtype=Q.dtype)
float_mask[~allowed] = -torch.inf
Mf = float_mask.view(1, 1, S, S).expand(B, G * h, S, S)
```

### Verification
```python
# Float mask (WORKS)
float_mask = torch.zeros(S, S)
float_mask[torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)] = -torch.inf
Mf = float_mask.view(1, 1, S, S).expand(B, G * h, S, S)
Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
print(f"Float mask NaN: {torch.isnan(Of).any()}")  # False
```

## Affected Tests

### Failing Tests (with NSA_PREFILL_BATCHED=1)
- `test_equiv_small.py::test_smallS_equivalence`
- `test_phi_mlp_equiv.py::test_phi_mlp_matches_avg_prefill`

### CI Impact
- **build-batched-smoke**: Consistently failing due to this issue

## Recommendations for Core Engineer

1. **Immediate Fix**: Replace boolean mask with float mask in `sliding_window_attention` (lines 133-144)

2. **Code Change Required**:
```python
# nsa/core/attention_kernels.py, line 133
# Replace:
disallowed = ~allowed  # [S,S]
# ...
Mf = disallowed.view(1, 1, S, S).expand(B, G * h, S, S)

# With:
float_mask = torch.zeros(S, S, device=device, dtype=Q.dtype)
float_mask[~allowed] = -torch.inf
# ...
Mf = float_mask.view(1, 1, S, S).expand(B, G * h, S, S)
```

3. **Additional Considerations**:
   - This may be a PyTorch version-specific issue
   - Consider adding a test specifically for NSA_PREFILL_BATCHED=1
   - Document the mask type requirement for SDPA

## Environment Details
- PyTorch: 2.4.0
- Python: 3.9.6 (local), 3.11 (CI)
- Platform: macOS (local), Linux (CI)

## Conclusion

The NSA_PREFILL_BATCHED failure is caused by using boolean masks with `F.scaled_dot_product_attention`, which produces NaN values in certain configurations. The solution is straightforward: use float masks with `-inf` for masked positions instead of boolean masks.

This is not a regression from the recent FA2 changes but a pre-existing issue that affects the batched prefill optimization path.