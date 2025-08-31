# 2025-08-31 Test Engineer Report - Masked Selection Path Not Activated v3

## Executive Summary
**Status: IMPLEMENTATION BUG** - The masked selection path is not being activated despite correct environment variables. The NSA attention code checks `use_sel_pack` before `use_sel_mask`, and the default value for `use_sel_pack` is hardcoded as `"1"` in the env_cache initialization, preventing the masked selection path from being reached.

## Test Environment
- **Hardware**: NVIDIA A100 80GB PCIe (104.255.9.187:12600)
- **Software**: PyTorch 2.4.0+cu121, Python 3.11.13
- **Branch**: prod/a100-50k-test (commit adf9f802)

## Configuration Applied
Environment variables correctly set and verified:
```bash
NSA_USE_SEL_PACK=0  # Correctly set to disable packed
NSA_USE_SEL_MASK=1  # Correctly set to enable masked
```

Process environment confirmed:
```
$ cat /proc/3598/environ | grep NSA_USE_SEL
NSA_USE_SEL_MASK=1
NSA_USE_SEL_PACK=0
```

## Root Cause Analysis

### Code Investigation
Found in `nsa/core/nsa_attention.py`:

1. **Environment cache initialization** (line ~1021):
```python
"use_sel_pack": parse_bool("NSA_USE_SEL_PACK", "1"),  # Defaults to "1" (true)
"use_sel_mask": parse_bool("NSA_USE_SEL_MASK", "0"),  # Defaults to "0" (false)
```

2. **Selection dispatch logic** (decode path):
```python
elif use_sel_pack:  # Checked FIRST
    O_sel_bt = grouped_selection_attention_packed(...)
elif self._env_cache.get("use_sel_mask", False):  # Checked SECOND
    O_sel_bt = grouped_selection_attention_masked(...)
```

### The Bug
The conditional structure checks paths in this order:
1. Triton/CUDA selection (if enabled)
2. **Packed selection** (if `use_sel_pack`)
3. **Masked selection** (if `use_sel_mask`)
4. Default gather path

Even when `NSA_USE_SEL_PACK=0` is set, the code may not be properly parsing it as False, or the cached value is not being updated. The `elif use_sel_pack` branch is still being taken, preventing the masked selection path from ever being reached.

## Test Results

### Canary Test with Masked Selection Environment
```bash
export NSA_USE_SEL_PACK=0
export NSA_USE_SEL_MASK=1
python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --no-gc \
  --seq-len 512 --batch-size 1 --accum-batches 1
```

**Results:**
- Environment vars confirmed set correctly
- Validator passes (no warnings about selection)
- **Performance: Still 11 toks/s** (no improvement)
- CPU still at 97%, GPU at 29%
- Packed selection path still being executed

## Evidence of Bug
1. Performance identical with masked selection "enabled" (11 toks/s)
2. CPU bottleneck persists (97% usage)
3. No change in behavior despite correct env vars
4. Code shows `use_sel_pack` defaults to `"1"` and is checked first

## Required Fix

The code needs to be modified to either:

### Option 1: Fix conditional ordering
```python
if self._env_cache.get("use_sel_mask", False):  # Check masked FIRST
    O_sel = grouped_selection_attention_masked(...)
elif use_sel_pack:  # Check packed SECOND
    O_sel = grouped_selection_attention_packed(...)
```

### Option 2: Fix env_cache parsing
Ensure `NSA_USE_SEL_PACK=0` is properly parsed and respected:
```python
use_sel_pack = self._env_cache.get("use_sel_pack", True)
if not use_sel_pack and self._env_cache.get("use_sel_mask", False):
    # Use masked path
```

### Option 3: Add explicit masked-only flag
```python
if os.getenv("NSA_FORCE_SEL_MASK", "0") == "1":
    O_sel = grouped_selection_attention_masked(...)
```

## Impact
- **Current state**: Production training impossible at 11 toks/s
- **Expected with fix**: 300-800 toks/s (30-80x speedup)
- **Blocker**: Code logic prevents masked selection activation
- **Severity**: CRITICAL - No workaround available

## Recommendations

### Immediate Action Required
1. **Fix the conditional logic** in `nsa/core/nsa_attention.py`
2. **Test with forced masked path** to confirm it solves performance
3. **Add debug logging** to show which selection path is active

### Testing After Fix
1. Verify masked path is actually taken (add logging)
2. Confirm CPU usage drops below 50%
3. Confirm GPU utilization increases above 50%
4. Verify throughput reaches 300+ toks/s

## Conclusion
The masked selection implementation exists and environment configuration is correct, but a **code logic bug** prevents the masked path from being activated. The packed selection path is hardcoded as the default and checked first, making it impossible to switch to the masked implementation via environment variables alone.

This is not a configuration issue but requires a code change to fix the conditional logic or properly respect the `NSA_USE_SEL_PACK=0` setting.