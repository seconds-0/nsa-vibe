# NSA Production Training Runbook

This document provides operational procedures, commands, and troubleshooting steps for running NSA training in production environments.

## Quick Reference Commands

### Environment Setup
```bash
# Set remote host for training pod
export REMOTE_HOST=training-pod  # or user@your.remote.host

# Essential environment variables
export CONFIG=configs/m7c_125m_fast_log.yaml
export PYTHONUNBUFFERED=1
export NSA_TOKENIZER=byte
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Training Launch (2× A100 preferred; runner auto‑detects GPUs)
```bash
# Preferred: use Prime runner (auto‑selects 24g/40g/80g configs with bf16 + checkpointing)
bash scripts/train_m7c_prime.sh

# Or launch directly if you need to override
python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 \
  --fwe-report-docs 500 --loader-timeout 120 \
  --synthetic-on-fail \
  2>&1 | tee training.log
```

### Monitoring Commands
```bash
# Monitor training progress (adjust path to match train.out_dir)
tail -f artifacts/m7c_125m/heartbeat_rank0.jsonl

# Check training metrics
python scripts/run_smoke_tests.py \
  --csv artifacts/m7c_125m/training.csv \
  --heartbeat artifacts/m7c_125m/heartbeat_rank0.jsonl

# Start watchdog
python scripts/_watchdog.py --dir artifacts/m7c_125m --halt 1
```

## Pre-Flight Checklist

### 1. Environment Validation
```bash
# Verify Python environment
python --version  # Should be 3.11+
which python
pip list | grep -E "(torch|flash-attn|triton)"

# Check CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"

# Verify NSA imports
python -c "from nsa.model.llama_block_nsa import LlamaBlockNSA; print('NSA import OK')"
python -c "from nsa.core.nsa_attention import NSAAttention; print('Attention import OK')"
```

### 2. Configuration Validation
```bash
# Check config file exists and is valid
ls -la $CONFIG
python -c "import yaml; yaml.safe_load(open('$CONFIG'))"

# Validate FA-2 policy precedence (env > runtime > default)
# Example: disable globally via env
export NSA_USE_FA2=0; python -c "import os; print('NSA_USE_FA2=', os.getenv('NSA_USE_FA2'))"
# Or rely on A100 runtime defaults: runtime.fa2_enabled: false; fa2_min_len_*: -1

# Validate dataset access
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte

# Test basic forward pass
python -c "
import torch
from nsa.model.llama_block_nsa import LlamaBlockNSA
block = LlamaBlockNSA(dim=512, n_heads=8, n_kv_groups=2, d_k=64, d_v=64, l=32, d=16, l_sel=64, n_sel=4, w=128)
x = torch.randn(1, 32, 512)
out = block(x)
print(f'Forward pass OK: {out.shape}')
"
```

### 3. Hardware Optimization
```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Set optimal CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton cache setup
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR
```

## Standard Operating Procedures

### Training Launch Procedure

1. **Pre-launch Setup**
   ```bash
   # Create artifacts directory
   mkdir -p artifacts/train_showcase
   
   # Clear any previous halt signals
   rm -f artifacts/train_showcase/.HALT artifacts/train_showcase/.anomaly_type
   
   # Start tensorboard (optional)
   tensorboard --logdir artifacts/m7c_125m/tb --port 6006 --bind_all &
   ```

2. **Launch Training**
   ```bash
   # Standard launch command
   export CONFIG=configs/m7c_125m_fast_log.yaml
   export PYTHONUNBUFFERED=1
   
   python -u scripts/train_showcase.py \
     --dataset fineweb_edu \
     --ddp 0 \
     --fwe-report-docs 500 \
     --loader-timeout 120 \
     --synthetic-on-fail \
     2>&1 | tee training.log
   ```

   FA‑2 Policy (A100 defaults)
   - Primary: set `runtime.fa2_enabled: false` (maps to `NSA_USE_FA2=0` unless env overrides).
   - Secondary: disable thresholds with sentinel `-1` via `runtime.fa2_min_len_win/cmp` (maps to `NSA_FA2_MIN_LEN_*`).
   - Precedence: env overrides config; branch env (`NSA_USE_FA2_{WIN,CMP}`) override global.
   - See Documentation/Reports/2025-08-30 Test Engineer Report - A100 FA2 vs SDPA Benchmarks v1.md for benchmark methodology and rationale.

3. **Start Monitoring**
   ```bash
   # Launch watchdog in background
   python scripts/_watchdog.py \
     --dir artifacts/train_showcase \
     --halt 1 \
     --interval 30 &
   
   # Monitor heartbeat
   tail -f artifacts/m7c_125m/heartbeat_rank0.jsonl
   ```

### Graceful Shutdown Procedure

1. **Signal Training to Stop**
   ```bash
   # Create halt signal
   touch artifacts/train_showcase/.HALT
   echo "halt: manual_stop" > artifacts/train_showcase/.HALT
   ```

2. **Wait for Graceful Exit**
   ```bash
   # Monitor for training to acknowledge halt
tail artifacts/m7c_125m/heartbeat_rank0.jsonl | grep -i halt
   
   # Check process is exiting
   ps aux | grep train_showcase
   ```

3. **Force Stop if Needed**
   ```bash
   # Get training PID
   TRAIN_PID=$(ps aux | grep train_showcase | grep -v grep | awk '{print $2}')
   
   # Send SIGTERM first
   kill -TERM $TRAIN_PID
   
   # Wait 30 seconds, then SIGKILL if still running
   sleep 30
   kill -KILL $TRAIN_PID 2>/dev/null || true
   ```

### Health Check Procedure

Run every 15 minutes during training:

```bash
#!/bin/bash
# health_check.sh

ARTIFACTS_DIR="artifacts/train_showcase"
LOG_FILE="$ARTIFACTS_DIR/health_check.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$TIMESTAMP] Starting health check" >> $LOG_FILE

# Check if training is running
if ! pgrep -f train_showcase >/dev/null; then
    echo "[$TIMESTAMP] ERROR: Training process not found" >> $LOG_FILE
    exit 1
fi

# Check for halt signal
if [[ -f "$ARTIFACTS_DIR/.HALT" ]]; then
    echo "[$TIMESTAMP] WARNING: Halt signal detected" >> $LOG_FILE
fi

# Check heartbeat freshness (should be < 3 minutes old)
if [[ -f "$ARTIFACTS_DIR/heartbeat_rank0.jsonl" ]]; then
    LAST_HB=$(stat -c %Y "$ARTIFACTS_DIR/heartbeat_rank0.jsonl")
    NOW=$(date +%s)
    AGE=$((NOW - LAST_HB))
    
    if [[ $AGE -gt 180 ]]; then
        echo "[$TIMESTAMP] ERROR: Heartbeat stale (${AGE}s old)" >> $LOG_FILE
        exit 1
    fi
fi

# Run smoke tests
if python scripts/run_smoke_tests.py \
    --csv "$ARTIFACTS_DIR/training.csv" \
    --heartbeat "$ARTIFACTS_DIR/heartbeat_rank0.jsonl" \
    --min-steps 50 >/dev/null 2>&1; then
    echo "[$TIMESTAMP] OK: Smoke tests passed" >> $LOG_FILE
else
    echo "[$TIMESTAMP] WARNING: Smoke tests failed" >> $LOG_FILE
fi

echo "[$TIMESTAMP] Health check complete" >> $LOG_FILE
```

## Troubleshooting Guide

### Training Won't Start

**Symptoms**: Training exits immediately or fails to begin
```bash
# Check basic environment
python --version
pip list | grep torch

# Test imports
python -c "import torch, nsa"

# Check config
python -c "import yaml; print(yaml.safe_load(open('$CONFIG')))"

# Test dataset connection
python scripts/automation/fwe_smoke.py --timeout 30
```

**Common Fixes**:
- Missing dependencies: `pip install -r requirements.txt`
- CUDA issues: Check `nvidia-smi` and CUDA installation
- Config errors: Validate YAML syntax
- Dataset timeout: Use `--synthetic-on-fail` flag

### Training Hangs or Stalls

**Symptoms**: No progress in heartbeat, loss not improving
```bash
# Check if process is alive
ps aux | grep train_showcase

# Check recent heartbeat
tail -5 artifacts/m7c_125m/heartbeat_rank0.jsonl

# Get stack trace
TRAIN_PID=$(pgrep -f train_showcase)
kill -USR1 $TRAIN_PID
cat artifacts/m7c_125m/stackdump_*.txt
```

**Common Causes**:
- Dataset streaming issues → Enable `--synthetic-on-fail`
- GPU memory exhaustion → Check `nvidia-smi`
- Gate collapse → Check gate health in heartbeat
- Selection hang → Disable Triton with `NSA_USE_TRITON_SEL=0`

### High Memory Usage

**Symptoms**: CUDA OOM errors, slow training
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Check batch size and sequence length
grep -E "(batch|seq)" $CONFIG

# Test with smaller config
python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 100
```

**Fixes**:
- Reduce batch size in config
- Use gradient checkpointing
- Disable packed selection: `NSA_USE_SEL_PACK=0`
- Force SDPA everywhere: `NSA_FORCE_PARITY=1`

### Performance Issues

**Symptoms**: Low tokens/second, slow training
```bash
# Check if strict asserts are enabled (kills performance)
echo $NSA_STRICT_ASSERTS  # Should be 0

# Verify FlashAttention is used
grep -i "using.*flash" training.log

# Check kernel selection
export NSA_DEBUG_COMPARE=1
```

**Optimizations**:
```bash
# Enable FlashAttention
export NSA_USE_FA2=1

# Use batched prefill
export NSA_PREFILL_BATCHED=1

# Disable debug checks
export NSA_STRICT_ASSERTS=0
export NSA_DEBUG_COMPARE=0

# Cache environment parsing
export NSA_ENV_STATIC=1
```

### Gate Collapse Detection

**Symptoms**: Watchdog reports gate collapse, training halts
```bash
# Check gate health details
cat artifacts/train_showcase/gate_collapse_details.json

# Review recent heartbeat for gate stats
tail -20 artifacts/train_showcase/heartbeat_rank0.jsonl | grep gate
```

**Recovery Actions**:
```bash
# Restart from checkpoint with lower learning rate
# Adjust gate temperature in config
# Check for gradient clipping issues
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Training Progress**
   ```bash
   # Loss trend
   tail -100 artifacts/m7c_125m/training.csv | cut -d, -f2 | sort -n
   
   # Throughput
   tail -10 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.toks_per_s'
   
   # Memory usage
   tail -10 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.gpu_mem_alloc'
   ```

2. **Gate Health**
   ```bash
   # Gate entropy (should be > 0.3)
   tail -10 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.gate_entropy_mean'
   
   # Gate collapse fraction (should be < 0.2)
   tail -10 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.gate_collapse_frac'
   
   # Branch balance
   tail -5 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.gate_branch_shares'
   ```

3. **Data Pipeline Health**
   ```bash
   # Fetch timing (should be < 1s)
   tail -10 artifacts/m7c_125m/heartbeat_rank0.jsonl | jq '.dt_fetch_s'
   
   # Documents processed
   grep "docs processed" training.log | tail -5
   ```

### Alert Conditions

Set up monitoring for these conditions:

1. **Critical Alerts** (immediate action required):
   - Training process down for > 5 minutes
   - Heartbeat stale for > 3 minutes  
   - GPU memory error
   - Loss becomes NaN/inf
   - Gate collapse detected (entropy < 0.2)

2. **Warning Alerts** (investigation needed):
   - Throughput drops > 50% for > 10 minutes
   - Data fetch time > 2s consistently
   - High gate collapse fraction (> 0.1)
   - Gradient norm spikes

### Log Analysis Commands

```bash
# Extract key metrics from logs
grep -E "(loss|toks_per_s|gate_entropy)" artifacts/m7c_125m/heartbeat_rank0.jsonl | \
  jq -r '[.step, .loss, .toks_per_s, .gate_entropy_mean] | @csv'

# Find anomalies in training
grep -i -E "(error|warning|exception|fail)" training.log

# Check for memory issues
grep -i -E "(cuda.*memory|oom)" training.log

# Find selection/attention issues
grep -i -E "(selection|attention|topk)" training.log
```

## Remote Training Operations

### SSH Connection Management
```bash
# Setup SSH config (add to ~/.ssh/config)
cat >> ~/.ssh/config << 'EOF'
Host training-pod
    HostName your.remote.host  
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
    RemoteForward 6006 localhost:6006  # Tensorboard tunnel
EOF

# Set environment for remote operations
export REMOTE_HOST=training-pod
export SSH_KEY_PATH=~/.ssh/id_ed25519
```

### Pod Setup Automation
```bash
# Complete pod setup script
cat > setup_pod.sh << 'EOF'
#!/bin/bash
set -e

echo "Setting up training pod..."

# System packages
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv ninja-build

# Clone repository
cd /root
if [ ! -d "nsa-vibe" ]; then
    git clone https://github.com/seconds-0/nsa-vibe.git
fi
cd nsa-vibe

# Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton packaging ninja
pip install flash-attn --no-build-isolation
pip install numpy hydra-core pydantic pytest hypothesis ruff mypy

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print('FlashAttention OK')"
python -c "from nsa.model.llama_block_nsa import LlamaBlockNSA; print('NSA OK')"

echo "Pod setup complete!"
EOF

# Run setup on remote pod
ssh $REMOTE_HOST 'bash -s' < setup_pod.sh
```

### Remote Training Launch
```bash
# Launch training on remote pod
ssh $REMOTE_HOST << 'EOF'
cd /root/nsa-vibe
source .venv/bin/activate

export CONFIG=configs/m7c_125m_fast_log.yaml
export PYTHONUNBUFFERED=1

# Start training in tmux session
tmux new-session -d -s training
tmux send-keys -t training "python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --fwe-report-docs 500 --loader-timeout 120 --synthetic-on-fail 2>&1 | tee training.log" Enter

# Start watchdog in separate tmux window
tmux new-window -t training -n watchdog
tmux send-keys -t training:watchdog "python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1" Enter

echo "Training started in tmux session 'training'"
EOF
```

### Remote Monitoring
```bash
# Check training status
ssh $REMOTE_HOST "cd /root/nsa-vibe && tail -5 artifacts/train_showcase/heartbeat_rank0.jsonl"

# Get live training logs
ssh $REMOTE_HOST "cd /root/nsa-vibe && tail -f training.log"

# Run health check
ssh $REMOTE_HOST "cd /root/nsa-vibe && python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --heartbeat artifacts/train_showcase/heartbeat_rank0.jsonl"

# Check tmux sessions
ssh $REMOTE_HOST "tmux list-sessions"
```

## Performance Tuning

### Hardware-Specific Optimizations

**A100 (80GB)**:
```bash
export NSA_USE_FA2=1
export NSA_USE_TRITON_SEL=1
export NSA_PREFILL_BATCHED=1
export NSA_ENV_STATIC=1
```

**RTX 4090**:
```bash
export NSA_USE_FA2=1
export NSA_USE_TRITON_SEL=0      # Not supported on Ada
export NSA_USE_SEL_PACK=1
export NSA_PREFILL_BATCHED=1
```

**V100/T4**:
```bash
export NSA_USE_FA2=1
export NSA_USE_TRITON_SEL=1
export NSA_PREFILL_BATCHED=0     # Less memory
export NSA_FORCE_PARITY=0
```

### Sequence Length Optimizations

**Short Sequences (< 2K)**:
```bash
export NSA_USE_SEL_PACK=1
export NSA_PREFILL_BATCHED=1
```

**Long Sequences (> 8K)**:
```bash
export NSA_USE_TRITON_SEL=1      # If available
export NSA_PREFILL_TILE=2048     # Chunk prefill
export NSA_ROPE_SCALE=1.0
```

## Backup and Recovery

### Checkpoint Management
```bash
# Manual checkpoint save
kill -USR2 $(pgrep -f train_showcase)

# List checkpoints
ls -la artifacts/train_showcase/checkpoints/

# Restart from latest checkpoint
python -u scripts/train_showcase.py \
  --dataset fineweb_edu \
  --ddp 0 \
  --resume-from artifacts/train_showcase/checkpoints/latest.pt
```

### Artifact Backup
```bash
# Backup training artifacts
tar -czf training_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  artifacts/train_showcase/ \
  training.log \
  configs/

# Sync to remote storage
rsync -av training_backup_*.tar.gz user@backup-server:/backups/nsa/
```

This runbook provides comprehensive operational procedures for production NSA training systems. Keep it updated as new features and optimizations are added to the system.
### FA‑2 and Selection Defaults (A100)
- Disable FA‑2 by default on A100. SDPA outperforms FA‑2 for our shapes.
  - Set `NSA_USE_FA2=0` or `runtime.fa2_enabled: false`
  - In `configs/profiles/a100.yaml`: `runtime.fa2_min_len_win: -1`, `runtime.fa2_min_len_cmp: -1`.
- Keep selection varlen disabled by default: `NSA_USE_SEL_VARLEN=0`.
- If enabling selection varlen for experiments, rely on masked SDPA fallback (PR40) for correctness and performance.

### Data Loader Settings
- Enable prefetch and tune queue depth to reduce fetch p95 (`NSA_FWE_PREFETCH=1`, `NSA_FWE_Q=8..16`).
- Optional warmup (PR36): `NSA_FWE_WARMUP_BATCHES=32..64`, `NSA_FWE_WARMUP_TIMEOUT=60` to reduce first‑step latency.
- Optional bootstrap: pre‑stage ~5GB JSONL locally (see `scripts/automation/fwe_bootstrap.py`) for strict SLA starts.
