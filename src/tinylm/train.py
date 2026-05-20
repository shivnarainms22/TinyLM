"""TinyLM training loop.

Entry point: python -m tinylm.train <config.yaml>

WandB is imported lazily inside train() so tests can import the module
without needing wandb installed in the test environment.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from tinylm.data import ShardLoader
from tinylm.model import ModelConfig, TinyLM
from tinylm.muon import Muon, partition_params


_STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    """SLURM sends SIGTERM before SIGKILL at the wall clock. Flag a clean stop."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[train] received signal {signum} — will checkpoint and exit at step end.")


def _should_stop() -> bool:
    return _STOP_REQUESTED


@dataclass
class TrainConfig:
    # Identity
    run_name: str = "run"
    # Optimizer mode: "muon" (matrix->Muon, scalar->AdamW) or "adamw" (all->AdamW)
    optimizer: str = "muon"
    # Model (must match ModelConfig locked dims)
    attention: str = "mla"
    n_layers: int = 18
    d_model: int = 1024
    n_heads: int = 16
    d_latent: int = 512
    d_rope: int = 64
    ffn_hidden: int = 2816
    ctx: int = 2048
    vocab_size: int = 32000
    tie_weights: bool = True
    # Data
    shard_dir: str = "data/shards"
    batch_size: int = 512
    seq_len: int = 2048
    # Training schedule
    total_steps: int = 20000
    warmup_steps: int = 2000
    lr_muon: float = 0.02
    lr_adamw: float = 0.001
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Gradient accumulation (effective_batch = batch_size * grad_accum_steps)
    grad_accum_steps: int = 1
    # BF16 autocast — halves activation memory, enables larger micro-batch on A100
    use_bf16: bool = False
    # Gradient checkpointing — trades compute for memory, enables larger micro-batch
    grad_checkpoint: bool = False
    # Logging / checkpointing
    log_every: int = 10
    save_every: int = 100
    wandb_project: str = "tinylm"
    # Resume
    resume_from: Optional[str] = None
    # Compile (enable on A100, keep False for CPU tests)
    compile: bool = False


def load_config(path: str) -> TrainConfig:
    """Load YAML config and raise ValueError on unknown keys."""
    with open(path) as f:
        d = yaml.safe_load(f)
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(d.keys()) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**d)


def build_optimizers(model: torch.nn.Module, cfg: TrainConfig):
    """Return [(optimizer, lr_max), ...] based on cfg.optimizer.

    "muon"  -> matrix params on Muon, scalar params on AdamW (two optimizers).
    "adamw" -> all params on a single AdamW (runs A and B).
    """
    if cfg.optimizer == "muon":
        matrix_params, scalar_params = partition_params(model)
        muon = Muon(matrix_params, lr=cfg.lr_muon, momentum=0.95)
        adamw = torch.optim.AdamW(
            scalar_params, lr=cfg.lr_adamw,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        return [(muon, cfg.lr_muon), (adamw, cfg.lr_adamw)]
    if cfg.optimizer == "adamw":
        adamw = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr_adamw,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        return [(adamw, cfg.lr_adamw)]
    raise ValueError(
        f"Unknown optimizer mode: {cfg.optimizer!r} (expected 'muon' or 'adamw')"
    )


def cosine_lr(
    step: int,
    warmup: int,
    total: int,
    lr_max: float,
    lr_min_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay to lr_max * lr_min_ratio."""
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_max * (lr_min_ratio + (1.0 - lr_min_ratio) * coeff)


def save_checkpoint(
    path: str,
    step: int,
    model: torch.nn.Module,
    optimizers: list,
    loader: ShardLoader,
    config_dict: dict,
) -> None:
    """Save full training state to a .pt checkpoint file.

    `optimizers` is the list of (optimizer, lr_max) pairs from
    build_optimizers(); their state is stored as an ordered list.
    """
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt, _ in optimizers],
            "loader": loader.state_dict(),
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint from disk. Returns raw dict."""
    return torch.load(path, map_location="cpu", weights_only=True)


def train_step(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    optimizers: list,
    cfg: TrainConfig,
    step: int,
) -> tuple[float, float]:
    """One training step. Returns (loss, grad_norm).

    Updates LR, runs forward/backward, clips gradients, steps every optimizer
    in `optimizers` (list of (optimizer, lr_max) pairs from build_optimizers()).
    Does NOT log to WandB — train() handles that.
    """
    for opt, lr_max in optimizers:
        lr = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, lr_max)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )

    if not loss.isfinite():
        raise RuntimeError(
            f"Loss is {loss.item()} at step {step} — training diverged."
        )

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    for opt, _ in optimizers:
        opt.step()

    return loss.item(), grad_norm.item()


def train(cfg: TrainConfig) -> None:
    """Full training loop with WandB logging and checkpointing."""
    import wandb  # lazy import — tests never call train()

    torch.set_float32_matmul_precision("high")  # enable TF32 tensor cores on A100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = ModelConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_latent=cfg.d_latent, d_rope=cfg.d_rope, ffn_hidden=cfg.ffn_hidden,
        ctx=cfg.ctx, vocab_size=cfg.vocab_size, tie_weights=cfg.tie_weights,
        attention=cfg.attention,
        use_checkpoint=cfg.grad_checkpoint,
    )
    model = TinyLM(model_cfg).to(device)
    if cfg.compile and device == "cuda":
        model = torch.compile(model)

    optimizers = build_optimizers(model, cfg)

    from tinylm.preflight import check_data_sufficiency
    check_data_sufficiency(cfg.shard_dir, cfg.total_steps, cfg.batch_size,
                           cfg.grad_accum_steps, cfg.seq_len, max_epochs=4)
    loader = ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len, max_epochs=4)

    start_step = 0
    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from)
        model.load_state_dict(ckpt["model"])
        for (opt, _), sd in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(sd)
        loader.load_state_dict(ckpt["loader"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {ckpt['step']}")

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=vars(cfg),
        resume="allow",
    )

    os.makedirs("checkpoints", exist_ok=True)
    import signal
    signal.signal(signal.SIGTERM, _handle_sigterm)
    model.train()
    t0 = time.perf_counter()
    tokens_logged = 0

    for step in range(start_step, cfg.total_steps):
        for opt, lr_max in optimizers:
            lr = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, lr_max)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            tokens = loader.next_batch().to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16 and device == "cuda"):
                logits = model(tokens[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    tokens[:, 1:].reshape(-1),
                ) / cfg.grad_accum_steps
            if not loss.isfinite():
                raise RuntimeError(
                    f"Loss is non-finite at step {step} — training diverged."
                )
            loss.backward()
            loss_accum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        for opt, _ in optimizers:
            opt.step()

        loss = loss_accum
        tokens_logged += cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps

        if step % cfg.log_every == 0:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tok_per_sec = tokens_logged / max(elapsed, 1e-9)
            lr_m = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr_muon)
            lr_a = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr_adamw)
            wandb.log(
                {
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/step_time_ms": elapsed / max(step % cfg.log_every, 1) * 1000,
                    "optim/lr_muon": lr_m,
                    "optim/lr_adamw": lr_a,
                },
                step=step,
            )
            print(
                f"step {step:5d} | loss {loss:.4f} | "
                f"tok/s {tok_per_sec:,.0f} | lr_m {lr_m:.2e}"
            )
            t0 = t1
            tokens_logged = 0

        if (step + 1) % cfg.save_every == 0 or step == cfg.total_steps - 1:
            ckpt_path = f"checkpoints/step_{step:05d}.pt"
            save_checkpoint(ckpt_path, step, model, optimizers, loader, vars(cfg))
            save_checkpoint("checkpoints/last.pt", step, model, optimizers, loader, vars(cfg))

        if _should_stop():
            save_checkpoint("checkpoints/last.pt", step, model, optimizers, loader, vars(cfg))
            print(f"[train] checkpointed at step {step} on signal — exiting.")
            break

    wandb.finish()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m tinylm.train <config.yaml>")
        sys.exit(1)
    train(load_config(sys.argv[1]))
