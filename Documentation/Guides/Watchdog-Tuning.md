# Watchdog Tuning Guide

The training watchdog (`scripts/_watchdog.py`) monitors training health and can automatically halt problematic runs. This guide covers configuration and tuning.

## Basic Usage

```bash
# Start watchdog with default settings
python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1

# Monitor without auto-halt (detection only)
python scripts/_watchdog.py --dir artifacts/train_showcase --halt 0
```

## Environment Variables

All watchdog behavior is controlled via environment variables:

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `NSA_WATCH_INTERVAL_S` | `30` | How often to check for anomalies (seconds) |
| `NSA_WATCH_HALT` | `0` | Auto-halt on anomaly (0=disable, 1=enable) |

### Stall Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `NSA_WATCH_HEARTBEAT_STALL_S` | `180` | Max time between heartbeats before stall alert |

### Throughput Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `NSA_WATCH_FLATLINE_N` | `5` | Consecutive zero throughput checks before alert |

### Future Anomaly Detection (Planned)

| Variable | Default | Description |
|----------|---------|-------------|
| `NSA_WATCH_LOSS_FLATLINE_N` | `10` | Consecutive steps with same loss (disabled if 0) |
| `NSA_WATCH_GRAD_ZERO_N` | `3` | Consecutive zero gradient norm alerts |
| `NSA_WATCH_FETCH_STALL_RATIO` | `3.0` | `dt_fetch_s` spike threshold vs baseline |

## Hardware-Specific Tuning

### A100 80GB (High Throughput)
```bash
# Faster monitoring for production workloads
export NSA_WATCH_INTERVAL_S=15
export NSA_WATCH_HEARTBEAT_STALL_S=120
export NSA_WATCH_FLATLINE_N=3
export NSA_WATCH_HALT=1
```

### RTX 4090 (Development)
```bash
# More relaxed for development/debugging
export NSA_WATCH_INTERVAL_S=60
export NSA_WATCH_HEARTBEAT_STALL_S=300
export NSA_WATCH_FLATLINE_N=8
export NSA_WATCH_HALT=0  # Manual intervention preferred
```

### Multi-Node Training
```bash
# Conservative settings for distributed training
export NSA_WATCH_INTERVAL_S=45
export NSA_WATCH_HEARTBEAT_STALL_S=240
export NSA_WATCH_FLATLINE_N=6
export NSA_WATCH_HALT=1
```

## Anomaly Types

The watchdog writes anomaly details to `.anomaly_type` files:

### Heartbeat Stall
- **Trigger**: No heartbeat update within `NSA_WATCH_HEARTBEAT_STALL_S`
- **File content**: `heartbeat_stall`
- **Common causes**: Training hang, data pipeline stall, GPU memory error
- **Action**: Check stack dumps, GPU status, data connectivity

### Throughput Flatline
- **Trigger**: Zero `toks_per_s` for `NSA_WATCH_FLATLINE_N` consecutive checks
- **File content**: `throughput_flatline`
- **Common causes**: Selection branch hang, gradient explosion, data starvation
- **Action**: Check training logs, model parameters, data pipeline health

## Integration with Training

### Trainer Side

The trainer polls for `.HALT` files every training step:

```python
halt_path = out_dir / ".HALT"
if halt_path.exists():
    print("[train] HALT file detected — stopping gracefully.")
    break
```

### Watchdog Side

On anomaly detection, watchdog creates control files:

```python
# Alert files
(out_dir / ".anomaly_type").write_text("heartbeat_stall")

# Halt request (if NSA_WATCH_HALT=1)
(out_dir / ".HALT").touch()
```

## Monitoring Workflows

### Development Workflow
1. Start training: `python scripts/train_showcase.py ...`
2. Start watchdog: `NSA_WATCH_HALT=0 python scripts/_watchdog.py --dir artifacts/train_showcase`
3. Watch alerts in terminal, investigate manually

### Production Workflow
1. Start training: `python scripts/train_showcase.py ...`
2. Start watchdog: `NSA_WATCH_HALT=1 python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1`
3. Monitor for `.HALT` files and anomaly logs
4. Automated restart on graceful halt

### Remote Monitoring
```bash
# Monitor watchdog status remotely
ssh $REMOTE_HOST "tail -f artifacts/train_showcase/.anomaly_type 2>/dev/null || echo 'No anomalies'"

# Check if training was halted
ssh $REMOTE_HOST "test -f artifacts/train_showcase/.HALT && echo 'HALTED' || echo 'RUNNING'"
```

## Troubleshooting

### Watchdog Not Detecting Issues

**Problem**: Training hangs but no stall detected  
**Solution**: Reduce `NSA_WATCH_HEARTBEAT_STALL_S` or check heartbeat file permissions

**Problem**: False positive stall alerts  
**Solution**: Increase `NSA_WATCH_HEARTBEAT_STALL_S` for slower hardware

### Training Not Halting

**Problem**: `.HALT` file exists but training continues  
**Solution**: Check trainer logs for halt polling; ensure latest trainer code

**Problem**: Watchdog creates `.HALT` but training doesn't see it  
**Solution**: Verify `--dir` path matches trainer `out_dir`

### Performance Impact

**Problem**: Watchdog consuming too much CPU  
**Solution**: Increase `NSA_WATCH_INTERVAL_S` to reduce check frequency

**Problem**: Delayed anomaly detection  
**Solution**: Decrease `NSA_WATCH_INTERVAL_S` but monitor system load

## Advanced Configuration

### Custom Anomaly Scripts

You can extend watchdog behavior by monitoring the `.anomaly_type` file:

```bash
#!/bin/bash
# custom_watchdog_handler.sh
while true; do
    if [ -f artifacts/train_showcase/.anomaly_type ]; then
        anomaly=$(cat artifacts/train_showcase/.anomaly_type)
        echo "Detected anomaly: $anomaly"
        
        case "$anomaly" in
            "heartbeat_stall")
                # Custom handling for stalls
                notify-send "Training stalled on $(hostname)"
                ;;
            "throughput_flatline")
                # Custom handling for flatlines
                curl -X POST webhook_url -d "Training flatlined"
                ;;
        esac
        
        # Reset anomaly file after handling
        rm artifacts/train_showcase/.anomaly_type
    fi
    sleep 30
done
```

### Multiple Watchdogs

For multi-GPU training, run one watchdog per rank:

```bash
# Rank 0 watchdog (primary)
NSA_WATCH_HALT=1 python scripts/_watchdog.py --dir artifacts/train_showcase

# Rank 1+ watchdog (monitoring only)  
NSA_WATCH_HALT=0 python scripts/_watchdog.py --dir artifacts/train_showcase
```

### Log Rotation

Prevent log buildup in long runs:

```bash
# Rotate heartbeat logs (run periodically)
find artifacts/train_showcase -name "heartbeat_*.jsonl" -size +100M -exec gzip {} \;
```

## Performance Baselines

Typical monitoring overhead by interval:

| Interval | CPU Usage | Detection Latency | Recommended For |
|----------|-----------|-------------------|-----------------|
| 15s | ~0.1% | 15-30s | Production A100 |
| 30s | ~0.05% | 30-60s | Standard training |
| 60s | ~0.02% | 60-120s | Development |

Choose based on your tolerance for detection delay vs. system overhead.