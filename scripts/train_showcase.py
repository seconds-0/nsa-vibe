#!/usr/bin/env python3
import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

from nsa.model.llama_block_nsa import LlamaBlockNSA


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_groups: int,
        d_k: int,
        d_v: int,
        l: int,
        d: int,
        l_sel: int,
        n_sel: int,
        w: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        from nsa.model.llama_block_nsa import RMSNorm
        self.embed = nn.Embedding(vocab, dim)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.blocks = nn.ModuleList(
            [
                LlamaBlockNSA(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_groups=n_kv_groups,
                    d_k=d_k,
                    d_v=d_v,
                    l=l,
                    d=d,
                    l_sel=l_sel,
                    n_sel=n_sel,
                    w=w,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_tok)  # [B,S,dim]
        for blk in self.blocks:
            if self.grad_checkpointing and x.requires_grad:
                import torch.utils.checkpoint as _ckpt
                x = _ckpt.checkpoint(lambda inp: blk(inp), x)
            else:
                x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dump_env_info(out_dir: Path, extra: dict | None = None) -> None:
    try:
        info = {
            "time": datetime.utcnow().isoformat() + "Z",
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "env": {k: v for k, v in os.environ.items() if k.startswith("NSA_") or k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "CONFIG")},
        }
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_capability"] = torch.cuda.get_device_capability(0)
            except Exception:
                pass
        if extra:
            info.update(extra)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "env.json", "w") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass


class Heartbeat:
    def __init__(self, out_dir: Path, rank: int) -> None:
        self.out_dir = out_dir
        self.rank = rank
        self.path = out_dir / f"heartbeat_rank{rank}.jsonl"
        self._lock = threading.Lock()
        self._last_ts = time.time()

    def write(self, step: int, msg: str, extra: dict | None = None) -> None:
        payload = {
            "ts": time.time(),
            "iso": datetime.utcnow().isoformat() + "Z",
            "pid": os.getpid(),
            "rank": self.rank,
            "step": step,
            "msg": msg,
        }
        if torch.cuda.is_available():
            try:
                payload.update(
                    {
                        "gpu_mem_alloc": int(torch.cuda.memory_allocated() // (1024 * 1024)),
                        "gpu_mem_reserved": int(torch.cuda.memory_reserved() // (1024 * 1024)),
                    }
                )
            except Exception:
                pass
        if extra:
            payload.update(extra)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(payload) + "\n")
            self._last_ts = payload["ts"]

    @property
    def last_ts(self) -> float:
        with self._lock:
            return self._last_ts


def _register_signal_dump(out_dir: Path) -> None:
    def dump_stack(signum, frame):
        try:
            dump_path = out_dir / f"stackdump_{int(time.time())}.txt"
            with open(dump_path, "w") as f:
                f.write(f"Signal {signum} at {datetime.utcnow().isoformat()}Z\n")
                for th in threading.enumerate():
                    f.write(f"\n--- Thread: {th.name} ({getattr(th, 'ident', None)}) ---\n")
                    stack = sys._current_frames().get(getattr(th, "ident", None))
                    if stack:
                        f.write("".join(traceback.format_stack(stack)))
        except Exception:
            pass
    try:
        signal.signal(signal.SIGUSR1, dump_stack)
        signal.signal(signal.SIGTERM, dump_stack)
    except Exception:
        # Signals may not be supported on some platforms
        pass


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "fineweb_edu", "fineweb_edu_local"],
        help="Training data source",
    )
    cli.add_argument("--save", type=str, default="artifacts/train_showcase/model.pt", help="Path to save final weights+config (torch.save)")
    cli.add_argument("--ddp", type=int, default=-1, help="Force DDP (1) or single process (0); -1=auto by WORLD_SIZE")
    cli.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from (optional)")
    cli.add_argument("--loader-timeout", type=float, default=60.0, help="Timeout seconds for initial dataset batch fetch")
    cli.add_argument("--fwe-report-docs", type=int, default=1000, help="FineWeb‑Edu progress print frequency (docs)")
    cli.add_argument("--synthetic-on-fail", action="store_true", help="Fallback to synthetic if FineWeb‑Edu stalls")
    cli.add_argument("--local-path", type=str, default="", help="Path for --dataset fineweb_edu_local (text or JSONL)")
    cli_args, _ = cli.parse_known_args()
    cfg_path = os.environ.get("CONFIG", "configs/train_showcase.yaml")
    print(f"[boot] loading config {cfg_path}", flush=True)
    cfg = OmegaConf.load(cfg_path)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    out_dir = Path(cfg.train.out_dir)
    _dump_env_info(out_dir)
    _register_signal_dump(out_dir)

    set_seed(int(cfg.train.seed))

    # DDP init
    want_ddp = cli_args.ddp
    env_ws = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = (env_ws > 1) if want_ddp < 0 else bool(want_ddp)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if ddp and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=os.environ.get("TORCH_BACKEND", "nccl"))
    world_size = torch.distributed.get_world_size() if (ddp and torch.distributed.is_initialized()) else 1
    rank = torch.distributed.get_rank() if (ddp and torch.distributed.is_initialized()) else 0

    # Device
    device = torch.device("cpu")
    if cfg.runtime.device == "cuda" or (cfg.runtime.device == "auto" and torch.cuda.is_available()):
        if ddp:
            torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            device = torch.device("cuda", local_rank % max(1, torch.cuda.device_count()))
        else:
            device = torch.device("cuda")
    dtype = torch.float32
    if str(cfg.runtime.precision).lower() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif str(cfg.runtime.precision).lower() in ("fp16", "float16"):
        dtype = torch.float16

    # Tokenizer/vocab
    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    use_bpe = False
    vocab = 256
    if str(cli_args.dataset).lower() == "fineweb_edu" and str(data_cfg.get("tokenizer", "byte")).lower() == "gpt2":
        try:
            from transformers import GPT2Tokenizer  # type: ignore
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            vocab = int(tok.vocab_size)
            use_bpe = True
        except Exception as e:
            raise RuntimeError("GPT-2 tokenizer requested but transformers not available. Install transformers or switch tokenizer to 'byte'.") from e
    print(f"[train] dataset={cli_args.dataset} tokenizer={'gpt2' if use_bpe else 'byte'}", flush=True)

    model = TinyLM(
        vocab=vocab,
        dim=int(cfg.model.dim),
        n_layers=int(cfg.model.get("n_layers", 1)),
        n_heads=int(cfg.model.n_heads),
        n_kv_groups=int(cfg.model.n_kv_groups),
        d_k=int(cfg.model.d_k),
        d_v=int(cfg.model.d_v),
        l=int(cfg.nsa.l),
        d=int(cfg.nsa.d),
        l_sel=int(cfg.nsa.l_sel),
        n_sel=int(cfg.nsa.n_sel),
        w=int(cfg.nsa.w),
        grad_checkpointing=bool(cfg.runtime.get("gradient_checkpointing", False)),
    ).to(device)
    model.train()
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    # TensorBoard writer (rank 0)
    tb_writer = None
    if SummaryWriter is not None and rank == 0:
        try:
            tb_dir = Path(cfg.train.out_dir) / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))  # type: ignore
            print(f"[train] tensorboard logdir: {tb_dir}", flush=True)
        except Exception as e:
            print(f"[train] tensorboard init failed: {e}")
    if ddp and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

    steps = int(cfg.train.steps)
    S = int(cfg.train.seq_len)
    B_global = int(cfg.train.batch_size)
    if world_size > 1:
        base = B_global // world_size
        extra = B_global % world_size
        B_local = base + (1 if rank < extra else 0)
    else:
        B_local = B_global
    lr = float(cfg.train.lr)
    save_every = int(cfg.train.get("save_every", 0))
    eval_every = int(cfg.train.get("eval_every", 0))
    accum = int(cfg.train.get("accumulate_grad_batches", 1))
    warmup = int(cfg.train.get("warmup_steps", 0))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.train.get("weight_decay", 0.0)))
    total_steps = steps
    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))
        import math
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        progress = min(1.0, max(0.0, progress))
        # Correct cosine schedule without floor
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    # Optional dataset wiring
    use_fwe = (str(cli_args.dataset).lower() == "fineweb_edu")
    use_fwe_local = (str(cli_args.dataset).lower() == "fineweb_edu_local")
    hb = Heartbeat(out_dir, rank)
    hb.write(0, "boot", {"phase": "start"})

    if use_fwe:
        try:
            from scripts.datasets.fineweb_edu_loader import iter_fineweb_edu_batches  # type: ignore
        except Exception as e:
            raise RuntimeError("FineWeb‑Edu loader not available; ensure repo path and optional deps installed (datasets)") from e
        if use_bpe:
            from transformers import GPT2Tokenizer  # type: ignore
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            def encode_bytes(s: str):
                # CRITICAL: Truncate at tokenizer level to prevent warnings and ensure seq_len compliance
                tokens = tok.encode(s)
                if len(tokens) > S - 1:  # Leave room for potential special tokens
                    tokens = tokens[:S-1]
                return tokens
        else:
            def encode_bytes(s: str):
                tokens = list(s.encode("utf-8", errors="ignore"))
                if len(tokens) > S - 1:
                    tokens = tokens[:S-1]
                return tokens
        os.environ["NSA_FWE_REPORT_DOCS"] = str(int(cli_args.fwe_report_docs))
        print("[train] streaming FineWeb‑Edu via datasets (sharded per rank)", flush=True)
        if world_size > 1:
            fwe_train = iter_fineweb_edu_batches(
                encode=encode_bytes, seq_len=S, batch_size=B_global, split_mod=world_size, split_rem=rank
            )
            fwe_val = iter_fineweb_edu_batches(
                encode=encode_bytes, seq_len=S, batch_size=B_global, split_mod=world_size, split_rem=rank
            )
        else:
            fwe_train = iter_fineweb_edu_batches(encode=encode_bytes, seq_len=S, batch_size=B_global, split_mod=100, split_rem=1)
            fwe_val = iter_fineweb_edu_batches(encode=encode_bytes, seq_len=S, batch_size=B_global, split_mod=100, split_rem=0)

        # Smoke test: try to fetch one batch with timeout to detect stalls early
        def _pull_one_batch(result_box: dict) -> None:
            try:
                t0 = time.time()
                result_box["batch"] = next(fwe_train)
                result_box["dt"] = time.time() - t0
                result_box["ok"] = True
            except StopIteration:
                result_box["ok"] = False
                result_box["err"] = "StopIteration"
            except Exception as e:
                result_box["ok"] = False
                result_box["err"] = f"{type(e).__name__}: {e}"

        box: dict = {}
        t = threading.Thread(target=_pull_one_batch, args=(box,), daemon=True)
        t.start()
        t.join(timeout=float(cli_args.loader_timeout))
        if not box.get("ok", False):
            msg = box.get("err", f"timeout waiting for first FineWeb‑Edu batch (≥{cli_args.loader_timeout:.0f}s)")
            print(f"[error] FineWeb‑Edu loader failed: {msg}", flush=True)
            hb.write(0, "fineweb_loader_error", {"error": msg})
            if cli_args.synthetic_on_fail:
                print("[warn] Falling back to synthetic dataset due to loader failure", flush=True)
                use_fwe = False
            else:
                raise RuntimeError(f"FineWeb‑Edu loader stall: {msg}")
        else:
            print(f"[train] first FineWeb‑Edu batch fetched in {box.get('dt', 0.0):.2f}s", flush=True)
            hb.write(0, "fineweb_loader_ready", {"dt": box.get("dt", 0.0)})
            # Put back the consumed batch for training by prepending
            # Re-create an iterator that yields the consumed batch first, then the original iterator
            first_batch = box["batch"]
            def _chain_batches(first, iterator):
                yield first
                yield from iterator
            fwe_train = _chain_batches(first_batch, fwe_train)

    elif use_fwe_local:
        pth = cli_args.local_path
        if not pth:
            raise RuntimeError("--local-path is required when --dataset fineweb_edu_local")
        print(f"[train] reading local dataset: {pth}", flush=True)
        # Build encoder (same semantics as above)
        if use_bpe:
            from transformers import GPT2Tokenizer  # type: ignore
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            def encode_bytes(s: str):
                tokens = tok.encode(s)
                if len(tokens) > S - 1:
                    tokens = tokens[:S-1]
                return tokens
        else:
            def encode_bytes(s: str):
                tokens = list(s.encode("utf-8", errors="ignore"))
                if len(tokens) > S - 1:
                    tokens = tokens[:S-1]
                return tokens
        # Simple local iterator (text or JSONL with {'text': ...})
        def _iter_local_batches(path: str, batch_size: int):
            import json as _json
            buf: list[int] = []
            batch: list[list[int]] = []
            is_jsonl = path.endswith(".jsonl")
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    text = line
                    if is_jsonl:
                        try:
                            obj = _json.loads(line)
                            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                                text = obj["text"]
                        except Exception:
                            pass
                    toks = encode_bytes(text)
                    if not toks:
                        continue
                    buf.extend(toks)
                    while len(buf) >= S:
                        seq = buf[:S]
                        buf = buf[S:]
                        batch.append(seq)
                        if len(batch) >= batch_size:
                            yield batch[:batch_size]
                            batch = batch[batch_size:]
        fwe_train = _iter_local_batches(pth, B_global)
        fwe_val = _iter_local_batches(pth, B_global)

    # Optional resume
    if cli_args.resume:
        try:
            ckpt = torch.load(cli_args.resume, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            if rank == 0:
                print(f"[train] resumed from {cli_args.resume}")
        except Exception as e:
            if rank == 0:
                print(f"[train] resume failed: {e}")

    losses = []
    grad_accum = 0
    t0 = time.time()
    tokens_total = 0
    last_log_time = t0
    last_log_tokens = 0

    # Watchdog to dump stacks if heartbeat stalls > 180s
    def _watchdog():
        while True:
            time.sleep(30.0)
            last = hb.last_ts
            if time.time() - last > 180.0:
                dump_path = out_dir / f"watchdog_stackdump_{int(time.time())}.txt"
                try:
                    with open(dump_path, "w") as f:
                        f.write(f"Watchdog at {datetime.utcnow().isoformat()}Z; last_heartbeat={last}\n")
                        for th in threading.enumerate():
                            f.write(f"\n--- Thread: {th.name} ({getattr(th, 'ident', None)}) ---\n")
                            stack = sys._current_frames().get(getattr(th, "ident", None))
                            if stack:
                                f.write("".join(traceback.format_stack(stack)))
                except Exception:
                    pass
                hb.write(0, "watchdog_dump", {"path": str(dump_path)})
    threading.Thread(target=_watchdog, daemon=True).start()

    # Eval helper
    def maybe_eval(step_idx: int):
        if not eval_every or (step_idx % eval_every != 0):
            return
        try:
            val_losses = []
            for _ in range(10):
                if use_fwe:
                    batch = next(fwe_val)
                    if world_size > 1:
                        start = sum([B_global // world_size + (1 if r < (B_global % world_size) else 0) for r in range(rank)])
                        count = B_local
                        sub = [batch[i] for i in range(start, start + count)] if count > 0 else []
                        xv = torch.tensor(sub, dtype=torch.long, device=device)
                    else:
                        xv = torch.tensor(batch, dtype=torch.long, device=device)
                    yv = xv[:, 1:].contiguous()
                else:
                    xv = torch.randint(low=0, high=vocab, size=(B_local, S), device=device)
                    yv = xv[:, 1:].contiguous()
                with torch.no_grad():
                    out = model(xv)
                    out_trim = out[:, :-1, :].contiguous()
                    lv = loss_fn(out_trim.view(B_local * (S - 1), -1), yv.view(B_local * (S - 1)))
                val_losses.append(float(lv.item()))
            val_loss = sum(val_losses) / max(1, len(val_losses))
            if ddp and world_size > 1:
                t = torch.tensor([val_loss], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                val_loss = float(t.item())
            import math as _m
            ppl = float(_m.exp(val_loss))
            if rank == 0:
                (Path(cfg.train.out_dir) / "val.csv").parent.mkdir(parents=True, exist_ok=True)
                with open(Path(cfg.train.out_dir) / "val.csv", "a") as vf:
                    vf.write(f"{step_idx},{val_loss:.6f},{ppl:.6f}\n")
                try:
                    from torch.utils.tensorboard import SummaryWriter as _SW  # type: ignore
                    # Append to existing writer if created earlier
                except Exception:
                    _SW = None  # type: ignore
        except Exception:
            pass

    for step in range(1, steps + 1):
        # Load batch
        if use_fwe:
            try:
                batch = next(fwe_train)
                if world_size > 1:
                    start = sum([B_global // world_size + (1 if r < (B_global % world_size) else 0) for r in range(rank)])
                    count = B_local
                    sub = [batch[i] for i in range(start, start + count)] if count > 0 else []
                    x = torch.tensor(sub, dtype=torch.long, device=device)
                else:
                    x = torch.tensor(batch, dtype=torch.long, device=device)
                y = x[:, 1:].contiguous()
            except StopIteration:
                x = torch.randint(low=0, high=vocab, size=(B_local, S), device=device)
                y = x[:, 1:].contiguous()
        else:
            x = torch.randint(low=0, high=vocab, size=(B_local, S), device=device)
            y = x[:, 1:].contiguous()

        # CRITICAL: Validate input tensor shape before model forward pass
        expected_shape = (B_local, S)
        if x.shape != expected_shape:
            raise ValueError(f"Input tensor shape mismatch: got {x.shape}, expected {expected_shape}")
        
        if rank == 0 and step <= 5:  # Log first few steps for debugging
            print(f"[debug] step {step}: input shape {x.shape}, seq_len {S}", flush=True)

        logits = model(x)
        logits_trim = logits[:, :-1, :].contiguous()
        loss = loss_fn(logits_trim.view(B_local * (S - 1), vocab), y.view(B_local * (S - 1))) / max(1, accum)

        loss.backward()
        grad_accum += 1
        if grad_accum >= accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.get("grad_clip", 1.0)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            grad_accum = 0

        losses.append(loss.detach().float().item())
        # Throughput accounting (effective tokens for next-token loss = S-1)
        tokens_total += (B_local * max(0, S - 1))
        if step % int(cfg.train.log_every) == 0 or step == 1:
            cur_loss = float(loss.item() * max(1, accum))
            log_loss = cur_loss
            if ddp and world_size > 1:
                t = torch.tensor([log_loss], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                log_loss = float(t.item())
            now = time.time()
            dt = max(1e-6, now - last_log_time)
            toks = tokens_total - last_log_tokens
            toks_per_s_local = toks / dt
            toks_per_s_global = toks_per_s_local * (world_size if ddp else 1)
            if rank == 0:
                # Optional grad-norm logging (enable with NSA_LOG_GRAD_NORM=1)
                gn_val = None
                try:
                    if os.getenv("NSA_LOG_GRAD_NORM", "0").lower() in ("1", "true", "yes"):
                        total = 0.0
                        for p in (model.module.parameters() if hasattr(model, "module") else model.parameters()):
                            if p.grad is not None:
                                # use detach to avoid autograd overhead
                                g = p.grad.detach()
                                total += float(g.norm().item())
                        gn_val = total
                except Exception:
                    gn_val = None
                print(
                    (
                        f"step {step:04d} | loss {log_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"toks/s {toks_per_s_global:.0f}"
                        + (f" | grad_norm {gn_val:.2f}" if gn_val is not None else "")
                    ),
                    flush=True,
                )
                hb_extra = {"loss": log_loss, "toks_per_s": toks_per_s_global}
                if gn_val is not None:
                    hb_extra["grad_norm"] = gn_val
                hb.write(step, "progress", hb_extra)
                (Path(cfg.train.out_dir) / "training.csv").parent.mkdir(parents=True, exist_ok=True)
                with open(Path(cfg.train.out_dir) / "training.csv", "a") as tf:
                    tf.write(f"{step},{log_loss:.6f},{scheduler.get_last_lr()[0]:.6e},{toks_per_s_global:.0f}\n")
                if tb_writer is not None:
                    try:
                        tb_writer.add_scalar("train/loss", log_loss, step)
                        tb_writer.add_scalar("train/toks_per_s", toks_per_s_global, step)
                        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    except Exception:
                        pass
            last_log_time = now
            last_log_tokens = tokens_total
            maybe_eval(step)

        if save_every and (step % save_every == 0) and rank == 0:
            try:
                state = {
                    "state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "step": step,
                }
                pth = Path(cfg.train.out_dir) / f"checkpoint_step{step}.pt"
                torch.save(state, str(pth))
            except Exception:
                pass

    # Save artifacts
    meta = {
        "device": str(device),
        "dtype": str(dtype),
        "steps": steps,
        "seq_len": S,
        "batch_global": B_global,
        "lr": lr,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
    }
    if rank == 0:
        with open(Path(cfg.train.out_dir) / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)
        try:
            state = {
                "state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
            out_path = Path(cli_args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, str(out_path))
        except Exception:
            pass
        print(json.dumps(meta, indent=2), flush=True)
        try:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
        except Exception:
            pass

    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
