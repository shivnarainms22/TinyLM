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

from tinylm.data import PrefetchLoader, ShardLoader
from tinylm.losses import chunked_cross_entropy
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
    # Keep only the most recent N numbered step_*.pt checkpoints (last.pt always
    # kept). Bounds /scratch usage over a long run. 0 = keep all (no rotation).
    keep_checkpoints: int = 2
    wandb_project: str = "tinylm"
    # Resume
    resume_from: Optional[str] = None
    # Initialize model weights from an existing checkpoint without resuming
    # optimizer, loader, or global step state.
    init_from: Optional[str] = None
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


def apply_env_overrides(cfg: TrainConfig) -> TrainConfig:
    """HPC job scripts inject resume/shard paths via env without editing YAML."""
    if os.environ.get("TINYLM_RESUME"):
        cfg.resume_from = os.environ["TINYLM_RESUME"]
    if os.environ.get("TINYLM_INIT_FROM"):
        cfg.init_from = os.environ["TINYLM_INIT_FROM"]
    if os.environ.get("TINYLM_SHARD_DIR"):
        cfg.shard_dir = os.environ["TINYLM_SHARD_DIR"]
    return cfg


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


def load_model_weights(model: torch.nn.Module, path: str) -> None:
    """Load only model weights from a TinyLM checkpoint.

    Used for continued pretraining phases where the base checkpoint seeds
    weights, but the optimizer, data-loader state, and step count must restart.

    Robust to torch.compile on either side: the checkpoint may have been saved
    from a compiled ('_orig_mod.'-prefixed) or eager model, and `model` here may
    itself be a compiled OptimizedModule. We normalize the checkpoint keys to the
    eager names and always load into the underlying module, so any combination
    of {compiled, eager} x {compiled, eager} matches.
    """
    ckpt = load_checkpoint(path)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    target = getattr(model, "_orig_mod", model)  # unwrap torch.compile wrapper
    target.load_state_dict(state)


def prune_checkpoints(ckpt_dir: str, keep: int) -> None:
    """Delete all but the most recent `keep` numbered step_*.pt checkpoints.

    Never touches last.pt. keep<=0 disables rotation (keeps everything).
    step_*.pt names are zero-padded so lexicographic sort == numeric order.
    """
    if keep <= 0:
        return
    import glob
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    for old in paths[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


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
    loss = chunked_cross_entropy(
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
    base_loader = ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len, max_epochs=4)
    loader = PrefetchLoader(base_loader) if device == "cuda" else base_loader

    start_step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ckpt = load_checkpoint(cfg.resume_from)
        model.load_state_dict(ckpt["model"])
        for (opt, _), sd in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(sd)
        loader.load_state_dict(ckpt["loader"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {ckpt['step']}")
    elif cfg.init_from:
        if not os.path.exists(cfg.init_from):
            raise FileNotFoundError(f"init_from checkpoint not found: {cfg.init_from}")
        load_model_weights(model, cfg.init_from)
        print(f"Initialized model weights from {cfg.init_from}")

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

        # Accumulate loss as a tensor; defer .item()/isfinite to log steps so the
        # GPU pipeline isn't synced grad_accum_steps times per step.
        loss_accum = torch.zeros((), device=device)
        for _ in range(cfg.grad_accum_steps):
            tokens = loader.next_batch().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16 and device == "cuda"):
                logits = model(tokens[:, :-1])
                loss = chunked_cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    tokens[:, 1:].reshape(-1),
                ) / cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt, _ in optimizers:
            opt.step()

        tokens_logged += cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps

        if step % cfg.log_every == 0:
            loss_val = loss_accum.item()       # single sync per log interval
            if not math.isfinite(loss_val):
                raise RuntimeError(f"Loss non-finite at step {step} — diverged.")
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tok_per_sec = tokens_logged / max(elapsed, 1e-9)
            lr_m = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr_muon)
            lr_a = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr_adamw)
            wandb.log(
                {
                    "train/loss": loss_val,
                    "train/grad_norm": grad_norm.item(),
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/step_time_ms": elapsed / max(step % cfg.log_every, 1) * 1000,
                    "optim/lr_muon": lr_m,
                    "optim/lr_adamw": lr_a,
                },
                step=step,
            )
            print(
                f"step {step:5d} | loss {loss_val:.4f} | "
                f"tok/s {tok_per_sec:,.0f} | lr_m {lr_m:.2e}"
            )
            t0 = t1
            tokens_logged = 0

        if (step + 1) % cfg.save_every == 0 or step == cfg.total_steps - 1:
            ckpt_path = f"checkpoints/step_{step:05d}.pt"
            save_checkpoint(ckpt_path, step, model, optimizers, loader, vars(cfg))
            save_checkpoint("checkpoints/last.pt", step, model, optimizers, loader, vars(cfg))
            prune_checkpoints("checkpoints", cfg.keep_checkpoints)

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
    train(apply_env_overrides(load_config(sys.argv[1])))
