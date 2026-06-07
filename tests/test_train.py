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
    """Return (model, optimizers, loader) for the given TrainConfig."""
    from tinylm.model import ModelConfig, TinyLM
    from tinylm.train import build_optimizers
    from tinylm.data import ShardLoader

    torch.manual_seed(42)
    model_cfg = ModelConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_latent=cfg.d_latent, d_rope=cfg.d_rope, ffn_hidden=cfg.ffn_hidden,
        ctx=cfg.ctx, vocab_size=cfg.vocab_size, tie_weights=cfg.tie_weights,
        attention=cfg.attention,
    )
    model = TinyLM(model_cfg)
    optimizers = build_optimizers(model, cfg)
    loader = ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len)
    return model, optimizers, loader


def _save_compiled_seed_checkpoint(path, cfg, saved_step):
    """Write a seed checkpoint whose model keys carry the torch.compile
    '_orig_mod.' prefix, mimicking a Run D checkpoint saved with compile=True.

    `saved_step` is deliberately large so an init_from run can prove it does
    NOT inherit the source step counter.
    """
    from tinylm.train import train_step, save_checkpoint

    model, optimizers, loader = _build_training_components(cfg)
    model.train()
    tokens = loader.next_batch()
    train_step(model, tokens, optimizers, cfg, step=0)  # move weights off init

    # save_checkpoint stores model.state_dict() verbatim; wrap so the persisted
    # keys look exactly like a compiled model's ('_orig_mod.<name>').
    class _Prefixed:
        def __init__(self, sd):
            self._sd = {f"_orig_mod.{k}": v for k, v in sd.items()}

        def state_dict(self):
            return self._sd

    save_checkpoint(path, saved_step, _Prefixed(model.state_dict()),
                    optimizers, loader, vars(cfg))


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


# ── integration tests ───────────────────────────────────────────────────────

def test_smoke_10_steps(tmp_path):
    """10 training steps on a tiny model must produce finite, decreasing loss."""
    from tinylm.train import train_step

    shard_dir = _make_shards(tmp_path)
    cfg = _tiny_cfg(shard_dir, total_steps=10)
    model, optimizers, loader = _build_training_components(cfg)
    model.train()

    losses = []
    for step in range(10):
        tokens = loader.next_batch()
        loss, grad_norm = train_step(model, tokens, optimizers, cfg, step)
        losses.append(loss)
        assert math.isfinite(loss), f"NaN/Inf loss at step {step}"
        assert math.isfinite(grad_norm), f"NaN/Inf grad_norm at step {step}"

    assert losses[-1] < losses[0], (
        f"Loss did not decrease after 10 steps: "
        f"{losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_checkpoint_resume_consistency(tmp_path):
    """Steps 5–9 after a checkpoint resume must match a baseline straight run.

    Determinism guarantee (CPU + manual_seed(42)):
      Same model init + same data sequence + same optimizer state
      → identical loss values after resume.
    """
    from tinylm.train import train_step, save_checkpoint, load_checkpoint

    shard_dir = _make_shards(tmp_path)
    cfg = _tiny_cfg(shard_dir, total_steps=10)

    # ── Baseline: 10 straight steps ──────────────────────────────────────
    model_b, optimizers_b, loader_b = _build_training_components(cfg)
    model_b.train()
    baseline_losses = []
    for step in range(10):
        tokens = loader_b.next_batch()
        loss, _ = train_step(model_b, tokens, optimizers_b, cfg, step)
        baseline_losses.append(loss)

    # ── Interrupted: 5 steps, save, reload, 5 more ───────────────────────
    model_i, optimizers_i, loader_i = _build_training_components(cfg)
    model_i.train()
    for step in range(5):
        tokens = loader_i.next_batch()
        train_step(model_i, tokens, optimizers_i, cfg, step)

    ckpt_path = str(tmp_path / "step_00004.pt")
    save_checkpoint(ckpt_path, 4, model_i, optimizers_i, loader_i, vars(cfg))

    # Reload into fresh components
    model_r, optimizers_r, loader_r = _build_training_components(cfg)
    model_r.train()
    ckpt = load_checkpoint(ckpt_path)
    model_r.load_state_dict(ckpt["model"])
    for (opt, _), sd in zip(optimizers_r, ckpt["optimizers"]):
        opt.load_state_dict(sd)
    loader_r.load_state_dict(ckpt["loader"])

    resumed_losses = []
    for step in range(5, 10):
        tokens = loader_r.next_batch()
        loss, _ = train_step(model_r, tokens, optimizers_r, cfg, step)
        resumed_losses.append(loss)

    # Steps 5–9 must match baseline steps 5–9 exactly (CPU is deterministic).
    for i, (bl, rl) in enumerate(zip(baseline_losses[5:], resumed_losses)):
        assert abs(bl - rl) < 1e-4, (
            f"Step {5 + i}: baseline loss {bl:.6f} vs resumed loss {rl:.6f} "
            f"(diff {abs(bl - rl):.2e}) — checkpoint resume is not reproducing "
            f"the training trajectory"
        )


def test_init_checkpoint_loads_weights_without_resume_state(tmp_path):
    """init_from must seed weights only; new phase data/optimizer state stays fresh."""
    from tinylm.train import train_step, save_checkpoint, load_model_weights

    shard_dir = _make_shards(tmp_path)
    cfg = _tiny_cfg(shard_dir, total_steps=3)

    source_model, source_optimizers, source_loader = _build_training_components(cfg)
    source_model.train()
    tokens = source_loader.next_batch()
    train_step(source_model, tokens, source_optimizers, cfg, step=0)
    ckpt_path = str(tmp_path / "run_d_seed.pt")
    save_checkpoint(ckpt_path, 0, source_model, source_optimizers, source_loader, vars(cfg))

    target_model, target_optimizers, target_loader = _build_training_components(cfg)
    initial_loader_state = target_loader.state_dict().copy()

    load_model_weights(target_model, ckpt_path)

    for expected, actual in zip(source_model.parameters(), target_model.parameters()):
        assert torch.allclose(expected, actual)
    assert target_loader.state_dict() == initial_loader_state
    assert all(not opt.state for opt, _ in target_optimizers)


def test_train_init_from_runs_end_to_end(tmp_path, monkeypatch):
    """Stage 0 smoke: the real train() entrypoint loads a compiled-format
    (Run-D-style, '_orig_mod.'-prefixed) seed via init_from, strips the prefix,
    and starts a FRESH phase — step resets, source optimizer/loader not inherited.

    This is the path E1 takes on HPC; the unit tests only cover the helper.
    """
    from tinylm.train import train

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    shard_dir = _make_shards(tmp_path)
    seed_path = str(tmp_path / "run_d_seed.pt")
    _save_compiled_seed_checkpoint(
        seed_path, _tiny_cfg(shard_dir, total_steps=3), saved_step=999
    )

    cfg = _tiny_cfg(shard_dir, total_steps=3)
    cfg.init_from = seed_path
    cfg.log_every = 1

    train(cfg)  # must not raise: '_orig_mod.' strip + init_from branch + full loop

    last = tmp_path / "checkpoints" / "last.pt"
    assert last.exists(), "init_from run wrote no checkpoint"
    ckpt = torch.load(str(last), map_location="cpu", weights_only=True)
    assert ckpt["step"] == 2, (
        f"init_from inherited step state (got {ckpt['step']}, expected 2) — "
        f"continued pretraining must reset the step counter, not resume from "
        f"the seed's saved step (999)"
    )


def test_train_resume_after_init_continues(tmp_path, monkeypatch):
    """After an init_from phase writes a checkpoint, train() resumes via
    resume_from and advances the step counter — the HPC requeue pattern E1 uses.
    """
    from tinylm.train import train

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    shard_dir = _make_shards(tmp_path)
    seed_path = str(tmp_path / "run_d_seed.pt")
    _save_compiled_seed_checkpoint(
        seed_path, _tiny_cfg(shard_dir, total_steps=3), saved_step=999
    )

    cfg = _tiny_cfg(shard_dir, total_steps=3)
    cfg.init_from = seed_path
    train(cfg)  # phase 1: steps 0..2, checkpoint at step 2

    cfg2 = _tiny_cfg(shard_dir, total_steps=5)
    cfg2.resume_from = str(tmp_path / "checkpoints" / "last.pt")
    train(cfg2)  # phase 2: resume at step 3, run to step 4

    ckpt = torch.load(
        str(tmp_path / "checkpoints" / "last.pt"), map_location="cpu", weights_only=True
    )
    assert ckpt["step"] == 4, (
        f"resume after init did not continue the trajectory "
        f"(got step {ckpt['step']}, expected 4)"
    )


def test_prune_checkpoints_keeps_last_n(tmp_path):
    """prune_checkpoints keeps the most recent N step_*.pt, never touches last.pt."""
    from tinylm.train import prune_checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for s in (499, 999, 1499, 1999):
        (ckpt_dir / f"step_{s:05d}.pt").write_text("x")
    (ckpt_dir / "last.pt").write_text("x")

    prune_checkpoints(str(ckpt_dir), keep=2)

    remaining = sorted(p.name for p in ckpt_dir.glob("*.pt"))
    assert remaining == ["last.pt", "step_01499.pt", "step_01999.pt"]


def test_prune_checkpoints_keep_zero_is_noop(tmp_path):
    """keep=0 disables rotation (keeps everything)."""
    from tinylm.train import prune_checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for s in (499, 999):
        (ckpt_dir / f"step_{s:05d}.pt").write_text("x")
    prune_checkpoints(str(ckpt_dir), keep=0)
    assert len(list(ckpt_dir.glob("step_*.pt"))) == 2


def test_env_overrides_apply(tmp_path, monkeypatch):
    """TINYLM_RESUME / TINYLM_INIT_FROM / TINYLM_SHARD_DIR override config fields."""
    from tinylm.train import load_config, apply_env_overrides
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "run_name: t\nshard_dir: data/shards\nresume_from: null\ninit_from: null\n"
        "attention: mla\noptimizer: muon\n"
    )
    cfg = load_config(str(cfg_path))
    monkeypatch.setenv("TINYLM_RESUME", "checkpoints/last.pt")
    monkeypatch.setenv("TINYLM_INIT_FROM", "/scratch/me/run_D/last.pt")
    monkeypatch.setenv("TINYLM_SHARD_DIR", "/scratch/me/data")
    cfg = apply_env_overrides(cfg)
    assert cfg.resume_from == "checkpoints/last.pt"
    assert cfg.init_from == "/scratch/me/run_D/last.pt"
    assert cfg.shard_dir == "/scratch/me/data"
