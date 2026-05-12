"""Tests for training components: cosine_lr, checkpoint round-trip,
10-step smoke, and checkpoint resume consistency.

All tests run on Windows CPU using a tiny model (2 layers, d_model=64).
WandB is never imported — tests only touch train_step, cosine_lr,
save_checkpoint, and load_checkpoint.
"""

import math
import os

import numpy as np
import pytest
import torch


# ── helpers ────────────────────────────────────────────────────────────────

def _make_shards(tmp_path, n_shards: int = 2, tokens_per_shard: int = 200_000,
                 vocab_size: int = 128):
    shard_dir = str(tmp_path / "shards")
    os.makedirs(shard_dir)
    rng = np.random.default_rng(42)
    for i in range(n_shards):
        data = rng.integers(0, vocab_size, tokens_per_shard, dtype=np.uint16)
        np.save(os.path.join(shard_dir, f"shard_{i:04d}.npy"), data)
    return shard_dir


def _tiny_cfg(shard_dir: str, total_steps: int = 10):
    from tinylm.train import TrainConfig
    return TrainConfig(
        run_name="test",
        attention="mla",
        n_layers=2, d_model=64, n_heads=4,
        d_latent=32, d_rope=8, ffn_hidden=128,
        ctx=32, vocab_size=128, tie_weights=True,
        shard_dir=shard_dir,
        batch_size=2, seq_len=32,
        total_steps=total_steps,
        warmup_steps=2,
        lr_muon=0.02, lr_adamw=0.001,
        weight_decay=0.1, grad_clip=1.0,
        log_every=5, save_every=total_steps,
        wandb_project="tinylm_test",
        resume_from=None,
        compile=False,
    )


def _build_training_components(cfg):
    """Return (model, muon, adamw, loader) for the given TrainConfig."""
    from tinylm.model import ModelConfig, TinyLM
    from tinylm.muon import Muon, partition_params
    from tinylm.data import ShardLoader

    torch.manual_seed(42)
    model_cfg = ModelConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_latent=cfg.d_latent, d_rope=cfg.d_rope, ffn_hidden=cfg.ffn_hidden,
        ctx=cfg.ctx, vocab_size=cfg.vocab_size, tie_weights=cfg.tie_weights,
        attention=cfg.attention,
    )
    model = TinyLM(model_cfg)
    matrix_params, scalar_params = partition_params(model)
    muon = Muon(matrix_params, lr=cfg.lr_muon, momentum=0.95)
    adamw = torch.optim.AdamW(
        scalar_params, lr=cfg.lr_adamw,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    loader = ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len)
    return model, muon, adamw, loader


# ── cosine_lr tests ─────────────────────────────────────────────────────────

def test_cosine_lr_warmup():
    """LR grows linearly from 0 to lr_max during warmup."""
    from tinylm.train import cosine_lr

    lr_max = 0.02
    warmup = 10
    total = 100

    assert cosine_lr(0, warmup, total, lr_max) == pytest.approx(0.0, abs=1e-9)
    assert cosine_lr(5, warmup, total, lr_max) == pytest.approx(lr_max * 0.5, rel=1e-5)
    assert cosine_lr(10, warmup, total, lr_max) == pytest.approx(lr_max, rel=1e-5)


def test_cosine_lr_decay():
    """After warmup, LR decays to lr_max * 0.1 at step == total."""
    from tinylm.train import cosine_lr

    lr_max = 0.02
    warmup = 10
    total = 100

    lr_end = cosine_lr(total, warmup, total, lr_max)
    assert lr_end == pytest.approx(lr_max * 0.1, rel=1e-4)

    # Monotonically decreasing after warmup
    lrs = [cosine_lr(s, warmup, total, lr_max) for s in range(warmup, total + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)), "LR not monotone"
