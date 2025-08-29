#!/usr/bin/env python3
# NSA M7C Training with FSDP (Fully Sharded Data Parallel) Support
# FSDP Implementation to replace DDP + gradient checkpointing incompatibility

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

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

                x = _ckpt.checkpoint(lambda inp: blk(inp), x, use_reentrant=False)
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


def _dump_env_info(out_dir: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        info = {
            "time": datetime.utcnow().isoformat() + "Z",
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "distributed_backend": "FSDP",  # Mark as FSDP implementation
            "env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith("NSA_") or k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "CONFIG")
            },
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

    def write(self, step: int, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "ts": time.time(),
            "iso": datetime.utcnow().isoformat() + "Z",
            "pid": os.getpid(),
            "rank": self.rank,
            "step": step,
            "msg": msg,
            "backend": "FSDP",  # Mark as FSDP
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
            dump_path = out_dir / f"stackdump_fsdp_{int(time.time())}.txt"
            with open(dump_path, "w") as f:
                f.write(f"Signal {signum} at {datetime.utcnow().isoformat()}Z (FSDP)\n")
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
        pass


def _sdp_kernel_ctx():
    """Select SDPA backend via env flags (same as DDP version)."""
    flash_only = os.getenv("NSA_SDPA_FLASH_ONLY", "0").lower() in ("1", "true", "yes")
    no_flash = os.getenv("NSA_SDPA_NO_FLASH", "0").lower() in ("1", "true", "yes")
    if flash_only:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=False
        )
    if no_flash:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=True, enable_math=True
        )
    return contextlib.nullcontext()


def _dump_mem(out_dir: Path, tag: str) -> None:
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"mem_fsdp_{tag}.txt").write_text(torch.cuda.memory_summary())
        import json as _json

        stats = {k: int(v) for k, v in torch.cuda.memory_stats().items()}
        (out_dir / f"mem_fsdp_{tag}.json").write_text(_json.dumps(stats, indent=2))
    except Exception:
        pass


def _optimizer_state_mb(optim: optim.Optimizer) -> float:
    total = 0
    try:
        for st in optim.state.values():
            if isinstance(st, dict):
                for t in st.values():
                    if torch.is_tensor(t):
                        total += t.numel() * t.element_size()
    except Exception:
        return 0.0
    return total / (1024 * 1024)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "fineweb_edu", "fineweb_edu_local"],
        help="Training data source",
    )
    cli.add_argument(
        "--save",
        type=str,
        default="artifacts/train_showcase_fsdp/model.pt",
        help="Path to save final weights+config",
    )
    cli.add_argument(
        "--ddp", type=int, default=-1, help="Ignored (FSDP always uses distributed when available)"
    )
    cli.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    cli.add_argument(
        "--loader-timeout",
        type=float,
        default=60.0,
        help="Timeout seconds for initial dataset batch fetch",
    )
    cli.add_argument(
        "--fwe-report-docs", type=int, default=1000, help="FineWeb‑Edu progress print frequency"
    )
    cli.add_argument(
        "--synthetic-on-fail",
        action="store_true",
        help="Fallback to synthetic if FineWeb‑Edu stalls",
    )
    cli.add_argument(
        "--local-path", type=str, default="", help="Path for --dataset fineweb_edu_local"
    )
    cli_args, _ = cli.parse_known_args()

    cfg_path = os.environ.get("CONFIG", "configs/train_showcase.yaml")
    print(f"[boot] loading config {cfg_path} (FSDP mode)", flush=True)
    cfg = OmegaConf.load(cfg_path)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    out_dir = Path(cfg.train.out_dir)

    # NSA performance defaults: force vectorized prefill and disable sync-heavy asserts
    # Only set if not already provided by the environment
    os.environ.setdefault("NSA_PREFILL_BATCHED", "1")
    os.environ.setdefault("NSA_STRICT_ASSERTS", "0")
    os.environ.setdefault("NSA_VERIFY_EQ9_MAPPING", "0")
    os.environ.setdefault("NSA_DEBUG_LOG", "0")

    _dump_env_info(out_dir, {"implementation": "FSDP"})
    _register_signal_dump(out_dir)

    set_seed(int(cfg.train.seed))

    # Distributed init (always for FSDP)
    env_ws = int(os.environ.get("WORLD_SIZE", "1"))
    if env_ws > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=os.environ.get("TORCH_BACKEND", "nccl"))

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    # Device
    device = torch.device("cpu")
    if cfg.runtime.device == "cuda" or (cfg.runtime.device == "auto" and torch.cuda.is_available()):
        if world_size > 1:
            torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            device = torch.device("cuda", local_rank % max(1, torch.cuda.device_count()))
        else:
            device = torch.device("cuda")
    # TF32 speedups (safe for training in conjunction with BF16)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    dtype = torch.float32
    if str(cfg.runtime.precision).lower() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif str(cfg.runtime.precision).lower() in ("fp16", "float16"):
        dtype = torch.float16

    # Tokenizer/vocab
    data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    use_bpe = False
    vocab = 256
    if (
        str(cli_args.dataset).lower() == "fineweb_edu"
        and str(data_cfg.get("tokenizer", "byte")).lower() == "gpt2"
    ):
        try:
            from transformers import GPT2Tokenizer  # type: ignore

            tok = GPT2Tokenizer.from_pretrained("gpt2")
            vocab = int(tok.vocab_size)
            use_bpe = True
        except Exception as e:
            raise RuntimeError("GPT-2 tokenizer requested but transformers not available.") from e

    print(
        f"[train] dataset={cli_args.dataset} tokenizer={'gpt2' if use_bpe else 'byte'} backend=FSDP",
        flush=True,
    )

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

    # Startup logging (all ranks)
    print(
        f"[train][fsdp] rank={rank} local_rank={local_rank} world_size={world_size} device={device}",
        flush=True,
    )
    # Print NSA runtime knobs relevant to prefill performance
    print(
        "[train][nsa] prefill_batched="
        f"{os.getenv('NSA_PREFILL_BATCHED', '0')} strict_asserts={os.getenv('NSA_STRICT_ASSERTS', '0')} "
        f"verify_eq9={os.getenv('NSA_VERIFY_EQ9_MAPPING', '0')} debug_log={os.getenv('NSA_DEBUG_LOG', '0')}",
        flush=True,
    )
    if rank == 0:
        print(
            f"[train] gradient_checkpointing={'on' if getattr(model, 'grad_checkpointing', False) else 'off'}",
            flush=True,
        )
        print(f"[train] distributed_backend=FSDP world_size={world_size}", flush=True)
        _dump_mem(out_dir, "boot")

    # Guardrail: prevent single-rank mislaunch on multi-GPU machines unless allowed
    if (
        device.type == "cuda"
        and torch.cuda.device_count() >= 2
        and world_size == 1
        and os.getenv("NSA_ALLOW_SINGLE_RANK", "0").lower() not in ("1", "true", "yes")
    ):
        if os.getenv("CI", "0") == "1":
            logging.warning(
                "[FSDP GUARD] Single-rank launch on multi-GPU in CI; continuing with warning. "
                "Set NSA_ALLOW_SINGLE_RANK=1 to silence or use torchrun --nproc_per_node=2."
            )
        else:
            raise RuntimeError(
                "FSDP detected single-rank launch on a multi-GPU node. "
                "Use: torchrun --standalone --nproc_per_node=2 scripts/train_showcase_fsdp.py ... "
                "(override with NSA_ALLOW_SINGLE_RANK=1 if intentional)."
            )

    # Convert to target dtype
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
        if rank == 0:
            print(f"[train] converted model to {dtype}", flush=True)

    # FSDP Wrapping (if multi-GPU)
    if world_size > 1:
        # Define auto-wrap policy for LlamaBlockNSA (can be disabled for coarse wrap)
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaBlockNSA},
        )

        # Mixed precision policy
        mixed_precision_policy = None
        if dtype == torch.bfloat16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == torch.float16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Sharding strategy can be switched by env (full|grad_op)
        shard_env = os.getenv("NSA_FSDP_SHARDING", "full").strip().lower()
        shard_strategy = (
            ShardingStrategy.FULL_SHARD
            if shard_env in ("full", "fs", "full_shard")
            else ShardingStrategy.SHARD_GRAD_OP
        )

        # Optional toggles via env
        limit_gathers = os.getenv("NSA_FSDP_LIMIT_ALL_GATHERS", "1").lower() in ("1", "true", "yes")
        fwd_prefetch = os.getenv("NSA_FSDP_FORWARD_PREFETCH", "1").lower() in ("1", "true", "yes")
        # Critical perf toggle when using activation checkpointing: keep params gathered through backward
        reshard_after_fwd = os.getenv("NSA_FSDP_RESHARD_AFTER_FWD", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        # Optional: flatten vs. orig params
        use_orig_params = os.getenv("NSA_FSDP_USE_ORIG_PARAMS", "1").lower() in ("1", "true", "yes")
        # Optional: disable auto-wrap to wrap the whole model as a single FSDP unit
        auto_wrap_on = os.getenv("NSA_FSDP_AUTO_WRAP", "1").lower() in ("1", "true", "yes")
        # Optional: backward prefetch policy
        _bp = os.getenv("NSA_FSDP_BACKWARD_PREFETCH", "none").strip().lower()
        if _bp in ("pre", "backward_pre", "pre_allgather"):
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        elif _bp in ("post", "backward_post", "post_allgather"):
            backward_prefetch = BackwardPrefetch.BACKWARD_POST
        else:
            backward_prefetch = None

        model = FSDP(
            model,
            auto_wrap_policy=(auto_wrap_policy if auto_wrap_on else None),
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device() if device.type == "cuda" else None,
            sync_module_states=True,
            sharding_strategy=shard_strategy,
            limit_all_gathers=limit_gathers,
            forward_prefetch=fwd_prefetch,
            # reshard_after_forward not available in PyTorch 2.5
            backward_prefetch=backward_prefetch,
            use_orig_params=use_orig_params,
        )

        if rank == 0:
            extra = []
            extra.append(f"sharding={shard_strategy.name}")
            extra.append(f"limit_all_gathers={limit_gathers}")
            extra.append(f"forward_prefetch={fwd_prefetch}")
            # reshard_after_fwd not available in PyTorch 2.5
            extra.append(f"use_orig_params={use_orig_params}")
            extra.append(f"auto_wrap={auto_wrap_on}")
            if backward_prefetch is not None:
                extra.append(f"backward_prefetch={backward_prefetch.name}")
            print("[train] FSDP wrapped | " + " ".join(extra), flush=True)

    # Dtype audit
    def _dump_dtypes_report(m: nn.Module, out_dir: Path, rank: int) -> None:
        if rank != 0:
            return
        try:
            # For FSDP models, need to access the wrapped module
            mod = getattr(m, "module", m)  # FSDP doesn't use .module but handle both cases
            dcounts: dict[str, int] = {}
            lines: list[str] = []

            # FSDP parameters might need special handling
            for name, p in mod.named_parameters():
                dt = str(p.dtype)
                dcounts[dt] = dcounts.get(dt, 0) + int(p.numel())
                lines.append(f"PARAM {name} {dt} {p.shape}")
            for name, b in mod.named_buffers():
                dt = str(b.dtype)
                dcounts[dt] = dcounts.get(dt, 0) + int(b.numel())
                lines.append(f"BUFFER {name} {dt} {b.shape}")

            total = sum(dcounts.values()) or 1
            summary = ["# DTYPE SUMMARY (elements) - FSDP"] + [
                f"{k}: {v} ({v / total:.2%})" for k, v in sorted(dcounts.items())
            ]
            out = out_dir / "dtypes_report_fsdp.txt"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                f.write("\n".join(summary) + "\n\n")
                f.write("\n".join(lines))
        except Exception as e:
            print(f"[warn] dtype audit failed: {e}")

    _dump_dtypes_report(model, out_dir, rank)

    # TensorBoard writer (rank 0)
    tb_writer = None
    if SummaryWriter is not None and rank == 0:
        try:
            tb_dir = Path(cfg.train.out_dir) / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"[train] tensorboard logdir: {tb_dir}", flush=True)
        except Exception as e:
            print(f"[train] tensorboard init failed: {e}")

    # Training setup
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

    opt = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(cfg.train.get("weight_decay", 0.0))
    )

    total_steps = steps

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))
        import math

        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    # Dataset setup (same as DDP version)
    use_fwe = str(cli_args.dataset).lower() == "fineweb_edu"
    use_fwe_local = str(cli_args.dataset).lower() == "fineweb_edu_local"
    hb = Heartbeat(out_dir, rank)
    hb.write(0, "boot", {"phase": "start", "backend": "FSDP"})

    dt_fetch_last: Optional[float] = None
    halt_path = out_dir / ".HALT"

    if use_fwe:
        try:
            from nsa.data_pipeline import Shard, fineweb_stream_batches  # type: ignore
        except Exception as e:
            raise RuntimeError("FineWeb‑Edu pipeline missing") from e

        if use_bpe:
            from transformers import GPT2Tokenizer  # type: ignore

            tok = GPT2Tokenizer.from_pretrained("gpt2")

            def encode_bytes(s: str):
                tokens = tok.encode(s)
                if len(tokens) > S - 1:
                    tokens = tokens[: S - 1]
                return tokens
        else:

            def encode_bytes(s: str):
                tokens = list(s.encode("utf-8", errors="ignore"))
                if len(tokens) > S - 1:
                    tokens = tokens[: S - 1]
                return tokens

        os.environ["NSA_FWE_REPORT_DOCS"] = str(int(cli_args.fwe_report_docs))
        print("[train] streaming FineWeb‑Edu via datasets (sharded per rank) - FSDP", flush=True)

        if world_size > 1:
            shard = Shard(mod=world_size, rem=rank)
            fwe_train = fineweb_stream_batches(
                encode=encode_bytes,
                seq_len=S,
                batch_size=B_global,
                shard=shard,
                report_docs=int(cli_args.fwe_report_docs),
            )
        else:
            fwe_train = fineweb_stream_batches(
                encode=encode_bytes,
                seq_len=S,
                batch_size=B_global,
                shard=Shard(mod=100, rem=1),
                report_docs=int(cli_args.fwe_report_docs),
            )

        # Smoke test same as DDP version
        def _pull_one_batch(result_box: Dict[str, Any]) -> None:
            try:
                t0 = time.time()
                result_box["batch"] = next(fwe_train)
                result_box["dt"] = time.time() - t0
                result_box["ok"] = True
            except Exception as e:
                result_box["ok"] = False
                result_box["err"] = f"{type(e).__name__}: {e}"

        box: Dict[str, Any] = {}
        t = threading.Thread(target=_pull_one_batch, args=(box,), daemon=True)
        t.start()
        t.join(timeout=float(cli_args.loader_timeout))

        if not box.get("ok", False):
            msg = box.get(
                "err",
                f"timeout waiting for first FineWeb‑Edu batch (≥{cli_args.loader_timeout:.0f}s)",
            )
            print(f"[error] FineWeb‑Edu loader failed: {msg}", flush=True)
            hb.write(0, "fineweb_loader_error", {"error": msg})
            if cli_args.synthetic_on_fail:
                print("[warn] Falling back to synthetic dataset due to loader failure", flush=True)
                use_fwe = False
            else:
                raise RuntimeError(f"FineWeb‑Edu loader stall: {msg}")
        else:
            print(
                f"[train] first FineWeb‑Edu batch fetched in {box.get('dt', 0.0):.2f}s", flush=True
            )
            hb.write(0, "fineweb_loader_ready", {"dt": box.get("dt", 0.0)})
            first_batch = box["batch"]

            # Optional warmup: prefill additional batches to reduce GPU idle
            from scripts.warmup_helper import warmup_loader_with_first_batch
            fwe_train, warmup_stats = warmup_loader_with_first_batch(
                fwe_train,
                first_batch,
                warmup_batches=int(os.getenv("NSA_FWE_WARMUP_BATCHES", "8")),  # Default to 8 batches
                warmup_timeout=float(os.getenv("NSA_FWE_WARMUP_TIMEOUT", "30")),  # Default 30s timeout
                heartbeat_writer=hb
            )
            if warmup_stats.get("enabled") and rank == 0:
                print(f"[train] warmup prefilled {warmup_stats['filled']}/{warmup_stats['requested']} batches in {warmup_stats['wait_ms']:.0f}ms", flush=True)

    # Training loop (same logic as DDP, but FSDP handles synchronization automatically)
    losses = []
    grad_accum = 0
    t0 = time.time()
    tokens_total = 0
    last_log_time = t0
    last_log_tokens = 0

    # Watchdog (same as DDP version)
    def _watchdog():
        while True:
            time.sleep(30.0)
            last = hb.last_ts
            if time.time() - last > 180.0:
                dump_path = out_dir / f"watchdog_stackdump_fsdp_{int(time.time())}.txt"
                try:
                    with open(dump_path, "w") as f:
                        f.write(f"Watchdog at {datetime.utcnow().isoformat()}Z; FSDP backend\n")
                        for th in threading.enumerate():
                            f.write(f"\n--- Thread: {th.name} ({getattr(th, 'ident', None)}) ---\n")
                            stack = sys._current_frames().get(getattr(th, "ident", None))
                            if stack:
                                f.write("".join(traceback.format_stack(stack)))
                except Exception:
                    pass
                hb.write(0, "watchdog_dump", {"path": str(dump_path)})

    threading.Thread(target=_watchdog, daemon=True).start()

    for step in range(1, steps + 1):
        # Load batch (same as DDP version)
        if use_fwe:
            try:
                _t0_fetch = time.time()
                batch = next(fwe_train)
                if world_size > 1:
                    start = sum(
                        [
                            B_global // world_size + (1 if r < (B_global % world_size) else 0)
                            for r in range(rank)
                        ]
                    )
                    count = B_local
                    sub = [batch[i] for i in range(start, start + count)] if count > 0 else []
                    x = torch.tensor(sub, dtype=torch.long, device=device)
                else:
                    x = torch.tensor(batch, dtype=torch.long, device=device)
                y = x[:, 1:].contiguous()
                dt_fetch_last = time.time() - _t0_fetch
            except StopIteration:
                x = torch.randint(low=0, high=vocab, size=(B_local, S), device=device)
                y = x[:, 1:].contiguous()
                dt_fetch_last = None
        else:
            x = torch.randint(low=0, high=vocab, size=(B_local, S), device=device)
            y = x[:, 1:].contiguous()
            dt_fetch_last = None

        # Validate input tensor shape
        expected_shape = (B_local, S)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input tensor shape mismatch: got {x.shape}, expected {expected_shape}"
            )

        if rank == 0 and step <= 5:
            print(f"[debug] step {step}: input shape {x.shape}, seq_len {S} (FSDP)", flush=True)

        # HALT polling
        if halt_path.exists():
            if rank == 0:
                print("[train] HALT file detected — stopping gracefully (FSDP).", flush=True)
            break

        # Forward pass
        with _sdp_kernel_ctx():
            logits = model(x)
        logits_trim = logits[:, :-1, :].contiguous()
        loss = loss_fn(logits_trim.view(B_local * (S - 1), vocab), y.view(B_local * (S - 1))) / max(
            1, accum
        )

        if not torch.isfinite(loss):
            if rank == 0:
                print("[train][FATAL] non-finite loss detected — aborting run (FSDP).", flush=True)
                (out_dir / ".anomaly_type").write_text("nan_loss\n")
                (out_dir / ".HALT").write_text("halt: nan_loss\n")
            break

        # Backward pass - FSDP handles synchronization automatically, no need for no_sync()
        loss.backward()
        grad_accum += 1

        if grad_accum >= accum:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.train.get("grad_clip", 1.0))
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            grad_accum = 0

        losses.append(loss.detach().float().item())
        tokens_total += B_local * max(0, S - 1)

        # Logging (same as DDP version but mark as FSDP)
        if step % int(cfg.train.log_every) == 0 or step == 1:
            cur_loss = float(loss.item() * max(1, accum))
            log_loss = cur_loss

            if world_size > 1:
                t = torch.tensor([log_loss], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                log_loss = float(t.item())

            now = time.time()
            dt = max(1e-6, now - last_log_time)
            toks = tokens_total - last_log_tokens
            toks_per_s_local = toks / dt
            toks_per_s_global = toks_per_s_local * (world_size if world_size > 1 else 1)

            if rank == 0:
                # Grad norm logging
                gn_val = None
                try:
                    if os.getenv("NSA_LOG_GRAD_NORM", "0").lower() in ("1", "true", "yes"):
                        total = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                g = p.grad.detach()
                                total += float(g.norm().item())
                        gn_val = total
                except Exception:
                    gn_val = None

                print(
                    (
                        f"step {step:04d} | loss {log_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"toks/s {toks_per_s_global:.0f} | FSDP"
                        + (f" | grad_norm {gn_val:.2f}" if gn_val is not None else "")
                    ),
                    flush=True,
                )

                hb_extra = {"loss": log_loss, "toks_per_s": toks_per_s_global, "backend": "FSDP"}
                if dt_fetch_last is not None:
                    hb_extra["dt_fetch_s"] = float(dt_fetch_last)
                if gn_val is not None:
                    hb_extra["grad_norm"] = gn_val

                # FSDP-safe diagnostics aggregation across all NSAAttention modules
                try:
                    # Access the wrapped module for traversal
                    root_mod = getattr(model, "_fsdp_wrapped_module", model)
                    from nsa.core.nsa_attention import (
                        NSAAttention,
                    )  # local import to avoid circulars

                    gate_entropies = []
                    gate_entropy_mins = []
                    gate_max_maxes = []
                    gate_collapse_fracs = []
                    branch_shares_acc = []
                    fb_totals = {
                        "selection_triton_fails": 0,
                        "selection_cuda_fails": 0,
                        "selection_pack_fails": 0,
                        "selection_mask_fails": 0,
                        "compressed_fa2_fails": 0,
                        "sliding_fa2_fails": 0,
                        "total_fallbacks": 0,
                    }
                    sel_k_means = []
                    sel_k_maxes = []
                    sel_rows_total = 0
                    sel_pct_at_maxes = []

                    for _, m in root_mod.named_modules():
                        if isinstance(m, NSAAttention):
                            # Gate stats
                            gs = getattr(m, "get_gate_stats", lambda: None)()
                            if gs:
                                gate_entropies.append(gs.get("entropy_mean", 0.0))
                                gate_entropy_mins.append(gs.get("entropy_min", 0.0))
                                gate_max_maxes.append(gs.get("max_gate_max", 0.0))
                                gate_collapse_fracs.append(gs.get("collapse_fraction", 0.0))
                                bs = gs.get("branch_shares") or [0.0, 0.0, 0.0]
                                branch_shares_acc.append(bs)
                            # Fallbacks
                            fb = getattr(m, "get_fallback_counters", lambda: {})()
                            for k in fb_totals:
                                fb_totals[k] += int(fb.get(k, 0))
                            # Selection stats
                            ss = getattr(m, "get_selection_stats", lambda: None)()
                            if ss:
                                if ss.get("rows", 0):
                                    sel_k_means.append(float(ss.get("k_mean", 0.0)))
                                    sel_k_maxes.append(int(ss.get("k_max", 0)))
                                    sel_rows_total += int(ss.get("rows", 0))
                                    sel_pct_at_maxes.append(float(ss.get("pct_at_max", 0.0)))

                    # Aggregate and emit
                    if gate_entropies:
                        import numpy as _np

                        hb_extra.update(
                            {
                                "gate_entropy_mean": float(_np.mean(gate_entropies)),
                                "gate_entropy_min": float(_np.min(gate_entropy_mins)),
                                "gate_max_gate": float(_np.max(gate_max_maxes)),
                                "gate_collapse_frac": float(_np.mean(gate_collapse_fracs)),
                            }
                        )
                        if branch_shares_acc:
                            # Average branch shares across modules
                            bs_arr = _np.array(branch_shares_acc)
                            hb_extra["gate_branch_shares"] = bs_arr.mean(axis=0).round(6).tolist()

                    # Fallback counters CSV (aggregated)
                    fc_path = Path(cfg.train.out_dir) / "fallback_counters_fsdp.csv"
                    if not fc_path.exists():
                        fc_path.write_text(
                            "step,selection_triton_fails,selection_cuda_fails,selection_pack_fails,selection_mask_fails,compressed_fa2_fails,sliding_fa2_fails,total_fallbacks\n"
                        )
                    with open(fc_path, "a") as fcf:
                        fcf.write(
                            f"{step},{fb_totals['selection_triton_fails']},{fb_totals['selection_cuda_fails']},{fb_totals['selection_pack_fails']},{fb_totals['selection_mask_fails']},{fb_totals['compressed_fa2_fails']},{fb_totals['sliding_fa2_fails']},{fb_totals['total_fallbacks']}\n"
                        )
                    hb_extra.update({f"fb_{k}": int(v) for k, v in fb_totals.items()})

                    # Selection stats CSV (averaged/maxed across modules)
                    if sel_k_means or sel_k_maxes:
                        ks_path = Path(cfg.train.out_dir) / "k_stats_fsdp.csv"
                        if not ks_path.exists():
                            with open(ks_path, "w") as kf:
                                kf.write("step,k_mean,k_max,rows,pct_at_max\n")
                        import numpy as _np

                        k_mean = float(_np.mean(sel_k_means)) if sel_k_means else 0.0
                        k_max = int(max(sel_k_maxes)) if sel_k_maxes else 0
                        pct_at_max = float(_np.mean(sel_pct_at_maxes)) if sel_pct_at_maxes else 0.0
                        with open(ks_path, "a") as kf:
                            kf.write(
                                f"{step},{k_mean:.4f},{k_max},{sel_rows_total},{pct_at_max:.4f}\n"
                            )
                        hb_extra.update(
                            {
                                "sel_k_mean": k_mean,
                                "sel_k_max": k_max,
                                "sel_rows": sel_rows_total,
                                "sel_pct_at_max": pct_at_max,
                            }
                        )
                except Exception as e:
                    if step <= 10:
                        print(f"[warn] FSDP diagnostics aggregation failed: {e}", flush=True)

                hb.write(step, "progress", hb_extra)
                (Path(cfg.train.out_dir) / "training_fsdp.csv").parent.mkdir(
                    parents=True, exist_ok=True
                )
                with open(Path(cfg.train.out_dir) / "training_fsdp.csv", "a") as tf:
                    tf.write(
                        f"{step},{log_loss:.6f},{scheduler.get_last_lr()[0]:.6e},{toks_per_s_global:.0f}\n"
                    )

                if tb_writer is not None:
                    try:
                        tb_writer.add_scalar("train/loss", log_loss, step)
                        tb_writer.add_scalar("train/toks_per_s", toks_per_s_global, step)
                        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    except Exception:
                        pass

            last_log_time = now
            last_log_tokens = tokens_total

            # Memory dumps (rank 0)
            if rank == 0:
                try:
                    if step == 1:
                        _dump_mem(out_dir, "step1")
                        mb = _optimizer_state_mb(opt)
                        (out_dir / "opt_state_fsdp_mb.txt").write_text(f"{mb:.2f}\n")
                    dump_every = int(os.getenv("NSA_MEM_DUMP_EVERY", "0") or 0)
                    if dump_every and (step % dump_every == 0):
                        _dump_mem(out_dir, f"step{step}")
                except Exception:
                    pass

        # Checkpointing
        if save_every and (step % save_every == 0) and rank == 0:
            try:
                # For FSDP, save a full (unsharded) state dict on rank 0 only
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    state_dict = model.state_dict()

                state = {
                    "state_dict": state_dict,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "step": step,
                    "backend": "FSDP",
                }
                pth = Path(cfg.train.out_dir) / f"checkpoint_fsdp_step{step}.pt"
                torch.save(state, str(pth))
            except Exception:
                pass

    # Save final artifacts
    meta = {
        "device": str(device),
        "dtype": str(dtype),
        "steps": steps,
        "seq_len": S,
        "batch_global": B_global,
        "lr": lr,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "backend": "FSDP",
    }

    if rank == 0:
        with open(Path(cfg.train.out_dir) / "metrics_fsdp.json", "w") as f:
            json.dump(meta, f, indent=2)

        try:
            # Save a full (unsharded) state dict on rank 0 only
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                state_dict = model.state_dict()
            state = {
                "state_dict": state_dict,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "backend": "FSDP",
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

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
