# M7C Training Investigation Report

## Executive Summary
The M7C training on Prime Intellect GPUs has been experiencing critical failures. Multiple training attempts have been made with various fixes, but the training remains stuck during initialization with no actual progress despite appearing to run.

## Environment Details
- **Host**: ubuntu@216.81.248.82 (Prime Intellect)
- **GPUs**: 2x NVIDIA A100 80GB PCIe
- **SSH Key**: ~/.ssh/primeintellect_ed25519
- **Working Directory**: /home/ubuntu/nsa-vibe
- **Branch**: test-plan/m7-training-readiness

## Timeline of Events & Changes

### Initial Setup
1. Created FineWeb-Edu data loader in `scripts/datasets/fineweb_edu_loader.py`
2. Created automation scripts in `scripts/automation/`
3. Created config `configs/m7c_125m_fast_log.yaml` with:
   - 50,000 training steps
   - Sequence length: 1024 (reduced from 4096 for memory)
   - Batch size: 1 (reduced from 2)
   - Learning rate: 2.0e-4
   - FP32 precision (to avoid dtype issues)

### Problem #1: Tokenizer Warnings
**Issue**: GPT-2 tokenizer producing sequences > 1024 tokens (up to 1249), causing:
```
Token indices sequence length is longer than the specified maximum sequence length for this model (1249 > 1024). Running this sequence through the model will result in indexing errors
```

**Attempted Fixes**:
1. Added validation in data loader to check sequence lengths
2. Added input validation in training script
3. Modified `encode_bytes` function to truncate tokens (lines 204-215)
4. Changed to use tokenizer's built-in truncation: `tok.encode(s, truncation=True, max_length=1023)`

### Problem #2: Learning Rate Too Low
**Issue**: Learning rate showing as 1e-7 instead of configured 2e-4

**Root Cause**: The lr_lambda function had a floor of 0.1:
```python
# Original (problematic)
return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

# Fixed version
return 0.5 * (1.0 + math.cos(math.pi * progress))
```

### Problem #3: Training Process Stuck
**Current Status**: Training process appears to run but is actually stuck during initialization
- Process shows high CPU usage (100-107%)
- GPU memory allocated (~30GB)
- But NO output to log files (0 bytes)
- No new artifacts since 22:46
- No print statements appearing (should see "[train] dataset=fineweb_edu tokenizer=gpt2")

## Current File Changes on Remote

### Modified Files (uncommitted):
1. **scripts/train_showcase.py**:
   - Learning rate schedule fixed (line 190)
   - Tokenizer truncation added (lines 204-215)
   - Input validation added (lines 313-318)

2. **scripts/datasets/fineweb_edu_loader.py**:
   - Sequence length validation (lines 82-94)

3. **configs/m7c_125m_fast_log.yaml**:
   - Created with reduced memory settings

## Process Status

### Currently Running:
```
ubuntu     17350  /home/ubuntu/nsa-vibe/.venv/bin/tensorboard --logdir artifacts/train_showcase/tb --port 6006
ubuntu     19324  python scripts/train_showcase.py --dataset fineweb_edu --ddp 0 [STUCK - 6+ minutes]
```

### Log Files (all in /home/ubuntu/nsa-vibe/):
```
-rw-rw-r-- 1 ubuntu    0 Aug 22 02:38 training_clean_final.log    [EMPTY - Current attempt]
-rw-rw-r-- 1 ubuntu  184 Aug 22 02:18 training_final_fixed.log    [Only tokenizer warnings]
-rw-rw-r-- 1 ubuntu  184 Aug 22 01:19 training_clean.log         [Only tokenizer warnings]
-rw-rw-r-- 1 ubuntu    0 Aug 22 01:11 training_final.log         [EMPTY]
-rw-rw-r-- 1 ubuntu  184 Aug 22 01:06 training_fixed.log         [Only tokenizer warnings]
```

### Artifacts Status:
- Last update: Aug 21 22:46
- Last recorded step: 200
- Last loss: 0.120937
- No new data being written

## Diagnostic Findings

1. **Training hangs during initialization** - Never reaches the print statements
2. **High CPU usage suggests infinite loop** - Possibly in data loading or tokenization
3. **TensorBoard running but showing old data** - artifacts/train_showcase/tb has old events only
4. **SSH tunnel to TensorBoard working** - localhost:6006 accessible but shows stale data

## Critical Issues to Investigate

1. **Why is training hanging before first print?**
   - Possible issue in config loading
   - Possible issue in model initialization
   - Possible deadlock in FineWeb-Edu dataset streaming

2. **Why are logs empty despite nohup redirection?**
   - Python may be buffering output
   - Process may be crashing before flush

3. **Is the tokenizer truncation actually working?**
   - Test shows truncation works in isolation
   - But warnings still appear in some logs

## Recommended Next Steps

1. **Kill stuck process (PID 19324)**
2. **Run with explicit Python unbuffered mode**: `python -u scripts/train_showcase.py`
3. **Add debug prints before dataset loading** to identify exact hang point
4. **Test data loader in complete isolation** with the exact config
5. **Check for syntax errors** in modified train_showcase.py
6. **Consider reverting all changes** and starting fresh

## SSH Access Commands
```bash
# Connect to Prime Intellect
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82

# TensorBoard tunnel
ssh -i ~/.ssh/primeintellect_ed25519 -L 6006:localhost:6006 -N ubuntu@216.81.248.82

# Kill stuck training
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82 "kill 19324"
```

## Key Files to Review
1. `/home/ubuntu/nsa-vibe/scripts/train_showcase.py` (lines 184-220 for changes)
2. `/home/ubuntu/nsa-vibe/scripts/datasets/fineweb_edu_loader.py`
3. `/home/ubuntu/nsa-vibe/configs/m7c_125m_fast_log.yaml`

## Conclusion
The training is completely stuck, not actually running despite appearing active. The root cause appears to be in the initialization phase, likely related to the dataset loading or the tokenizer modifications. All attempted fixes have been applied but the fundamental issue of the training hanging during startup remains unresolved.