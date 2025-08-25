# Heartbeat JSONL Schema

The training system writes heartbeat logs to `artifacts/<run_name>/heartbeat_rank<N>.jsonl` with one JSON object per line.

## Standard Fields

Every heartbeat record includes these fields:

| Field | Type | Description |
|-------|------|-------------|
| `ts` | float | Unix timestamp (seconds since epoch) |
| `iso` | string | ISO 8601 timestamp with Z suffix (UTC) |
| `pid` | int | Process ID of the trainer |
| `rank` | int | DDP rank (0 for single-GPU) |
| `step` | int | Training step number |
| `msg` | string | Event type/message |

## GPU Memory Fields

When CUDA is available, these fields are automatically included:

| Field | Type | Description |
|-------|------|-------------|
| `gpu_mem_alloc` | int | Allocated GPU memory in MB |
| `gpu_mem_reserved` | int | Reserved GPU memory in MB |

## Training Metrics (Progress Records)

When `msg="progress"`, additional training metrics are included:

| Field | Type | Description |
|-------|------|-------------|
| `loss` | float | Current training loss (DDP-averaged) |
| `toks_per_s` | float | Global throughput in tokens/second |
| `dt_fetch_s` | float | Data fetch time in seconds (streaming only) |
| `grad_norm` | float | Gradient norm (if `NSA_LOG_GRAD_NORM=1`) |

## Event Types

Common values for the `msg` field:

- `"boot"` - Training initialization
- `"progress"` - Regular training step with metrics
- `"fineweb_loader_ready"` - Streaming data loader initialized
- `"fineweb_loader_error"` - Data loader failed
- `"watchdog_dump"` - Stack dump triggered by watchdog

## Example Records

### Boot Event
```json
{
  "ts": 1692720000.123,
  "iso": "2023-08-22T16:00:00.123Z",
  "pid": 12345,
  "rank": 0,
  "step": 0,
  "msg": "boot",
  "gpu_mem_alloc": 1024,
  "gpu_mem_reserved": 2048,
  "phase": "start"
}
```

### Training Progress
```json
{
  "ts": 1692720060.456,
  "iso": "2023-08-22T16:01:00.456Z", 
  "pid": 12345,
  "rank": 0,
  "step": 100,
  "msg": "progress",
  "gpu_mem_alloc": 5120,
  "gpu_mem_reserved": 6144,
  "loss": 3.2451,
  "toks_per_s": 1245.0,
  "dt_fetch_s": 0.045,
  "grad_norm": 1.23
}
```

### Data Loader Ready
```json
{
  "ts": 1692720005.789,
  "iso": "2023-08-22T16:00:05.789Z",
  "pid": 12345,
  "rank": 0,
  "step": 0,
  "msg": "fineweb_loader_ready",
  "gpu_mem_alloc": 1024,
  "gpu_mem_reserved": 2048,
  "dt": 13.53
}
```

## Monitoring Usage

The watchdog (`scripts/_watchdog.py`) monitors heartbeat files for:

- **Stall detection**: No new records within `NSA_WATCH_HEARTBEAT_STALL_S` (default 180s)
- **Throughput monitoring**: Zero `toks_per_s` for consecutive checks
- **Gradient health**: Zero or non-finite `grad_norm` values

## Reading Heartbeat Files

### Python Example
```python
import json
from pathlib import Path

def read_heartbeat(path: Path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

# Get latest progress record
heartbeat = read_heartbeat(Path("artifacts/train_showcase/heartbeat_rank0.jsonl"))
progress = [r for r in heartbeat if r["msg"] == "progress"]
if progress:
    latest = progress[-1]
    print(f"Step {latest['step']}: loss={latest['loss']:.4f}, toks/s={latest['toks_per_s']:.0f}")
```

### Shell Example
```bash
# Get latest progress record
tail -n 50 artifacts/train_showcase/heartbeat_rank0.jsonl | \
  jq -r 'select(.msg == "progress") | "Step \(.step): loss=\(.loss), toks/s=\(.toks_per_s)"' | \
  tail -1
```

## Schema Evolution

The heartbeat schema is designed to be forward-compatible:
- New fields may be added in future versions
- Existing fields maintain their types and semantics
- Unknown fields should be ignored by parsers
- The `ts`, `rank`, `step`, and `msg` fields are guaranteed to exist