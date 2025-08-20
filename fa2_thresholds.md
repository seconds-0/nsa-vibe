# GPU Benchmark Results

## Device Information
- **GPU**: Unknown
- **PyTorch**: Unknown
- **CUDA**: Unknown
- **Timestamp**: 1755717288.1231093

## Recommended Thresholds
- `runtime.fa2_min_len_win`: **512**
- `runtime.fa2_min_len_cmp`: **32**
- `runtime.sel_triton_min_L`: **4096**

## Benchmark Results

### Sliding Window Performance
| S | w | Speedup | Masked (ms) | FA-2 (ms) |
|---|---|---------|-------------|-----------|
| 128 | 512 | 0.02x ❌ | 1.02 | 66.32 |
| 256 | 512 | 0.04x ❌ | 3.60 | 100.62 |
| 512 | 512 | 0.07x ❌ | 14.09 | 209.46 |

### Compressed Branch Performance
| S | Speedup | Masked (ms) | FA-2 (ms) |
|---|---------|-------------|-----------|

## Analysis

With a safety margin of 1.2x:
- **Sliding**: FA-2 is faster for window sizes ≥ 512
- **Compressed**: FA-2 is faster for effective lengths ≥ 32

These thresholds ensure FA-2 is only used when it provides at least 19% speedup.
