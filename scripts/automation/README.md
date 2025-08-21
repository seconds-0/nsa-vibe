# Prime Intellect M7C Training Automation

## Instant Training Setup

### 🚀 Zero-Effort Training Start
```bash
make train-prime
```
This single command:
- Connects to Prime Intellect automatically
- Sets up the environment
- Clones/updates the repo
- Tests the data loader
- Starts training in tmux
- Gives you the exact commands for monitoring

### 📊 Live Monitoring (Run in New Terminal)
```bash
make monitor-prime
```
This automatically:
- Opens TensorBoard tunnel
- Opens browser to http://localhost:6006
- Shows live loss/throughput graphs

### 📋 Quick Status/Logs
```bash
make status-prime  # Quick status check
make logs-prime    # Tail training logs
```

## How It Works

1. **make train-prime** → Generates remote scripts → SSH executes them → Training starts in tmux
2. **make monitor-prime** → SSH tunnel + TensorBoard → Live graphs in browser
3. **Everything is automated** - no manual SSH, no manual setup, no copy-pasting

## Target Configuration
- Host: ubuntu@216.81.248.82
- Branch: test-plan/m7-training-readiness  
- TensorBoard: http://localhost:6006

## Success Workflow
1. Run `make train-prime` (wait ~2 minutes for setup)
2. Run `make monitor-prime` in new terminal
3. Watch graphs at http://localhost:6006
4. Training runs unattended for hours

Training will auto-resume from checkpoints and handle all GPU memory configurations automatically.