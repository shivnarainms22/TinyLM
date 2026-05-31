# TinyLM HPC Re-run — Full 4-Run Ablation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the full A/B/C/D ablation on Northeastern Explorer (SLURM, 8h cap, 40GB-resilient) on honest data (8B unique FineWeb-Edu tokens, ~3 epochs), with a hardened training stack, preserving the v1 Run D as a contrast artifact.

**Architecture:** Reuse the existing tested model/optimizer/data code. Add (1) a `muon|adamw` optimizer switch so all four ablation arms run from config, (2) data-safety guards that make the v1 data-repetition bug impossible, (3) an 8h-cap survival layer (SIGTERM checkpoint + `last.pt` + SLURM auto-rechain ported from D:/DiffMamba), and (4) throughput/40GB fixes shared identically by all runs so the ablation stays valid.

**Tech Stack:** Python 3.10+, PyTorch 2.x (cu128 on Explorer), NumPy, PyYAML, SLURM, conda, lm-eval, HuggingFace Hub, WandB. Tests run on Windows CPU via `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-20-tinylm-hpc-rerun-design.md`

**Branch:** `feat/hpc-rerun-ablation` (already created; main untouched).

---

## File Structure

**New modules (focused, independently testable):**
- `src/tinylm/losses.py` — `chunked_cross_entropy()` (memory-bounded CE for 40GB).
- `src/tinylm/preflight.py` — `check_data_sufficiency()` (the arithmetic guard).

**Modified modules:**
- `src/tinylm/train.py` — `optimizer` config field; `build_optimizers()`; refactor `train_step`/loop/`save_checkpoint`/`load_checkpoint` to an optimizer **list**; SIGTERM handler + `_should_stop`; `last.pt` on save; chunked-CE in loss; `TINYLM_RESUME`/`TINYLM_SHARD_DIR` env overrides; preflight call.
- `src/tinylm/data.py` — `ShardLoader` epoch counter + `max_epochs` guard; `PrefetchLoader` wrapper (background thread, pinned memory, exact resume state).

**Modified tests / new tests:**
- `tests/test_train.py` — migrate to optimizer-list API (keeps 30 green).
- `tests/test_optimizer_switch.py` (new), `tests/test_losses.py` (new), `tests/test_preflight.py` (new), `tests/test_prefetch.py` (new), `tests/test_sigterm.py` (new), `tests/test_data.py` — add epoch-guard tests.

**Configs:**
- Create `configs/run_A_mha_adamw.yaml`, `configs/run_B_mla_adamw.yaml`, `configs/run_C_mha_muon.yaml`, `configs/smoke_gate.yaml`.
- Modify `configs/run_D_mla_muon.yaml` (8B/23k steps, `optimizer` field, 40GB batch).

**Scripts:**
- Create `scripts/setup_hpc.sh`, `scripts/hpc_job.sh`, `scripts/submit_hpc.sh`.
- Modify `scripts/upload_checkpoints.py` (add `--prefix` for per-run subfolder).
- `scripts/tokenize_shards.py` — no code change; invoked with `--max-shards 80`.

---

## Chunk 1: Optimizer switch (enables runs A & B)

### Task 1: Add `optimizer` field + `build_optimizers()`

**Files:**
- Modify: `src/tinylm/train.py` (`TrainConfig` ~line 26-65; optimizer construction ~line 182-187)
- Test: `tests/test_optimizer_switch.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimizer_switch.py
"""Tests for the optimizer mode switch (muon vs adamw-only)."""
import torch


def _tiny_model():
    from tinylm.model import ModelConfig, TinyLM
    torch.manual_seed(0)
    return TinyLM(ModelConfig(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, tie_weights=True, attention="mla",
    ))


def _cfg(optimizer):
    from tinylm.train import TrainConfig
    return TrainConfig(optimizer=optimizer, lr_muon=0.02, lr_adamw=0.001, weight_decay=0.1)


def test_adamw_mode_puts_all_params_on_one_adamw():
    from tinylm.train import build_optimizers
    model = _tiny_model()
    opts = build_optimizers(model, _cfg("adamw"))
    assert len(opts) == 1
    opt, lr_max = opts[0]
    assert isinstance(opt, torch.optim.AdamW)
    assert lr_max == 0.001
    n_in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_model = sum(p.numel() for p in model.parameters())
    assert n_in_opt == n_model


def test_muon_mode_splits_matrix_and_scalar_without_overlap():
    from tinylm.train import build_optimizers
    from tinylm.muon import Muon
    model = _tiny_model()
    opts = build_optimizers(model, _cfg("muon"))
    assert len(opts) == 2
    kinds = {type(o).__name__ for o, _ in opts}
    assert kinds == {"Muon", "AdamW"}
    ids = [id(p) for o, _ in opts for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "param appears in more than one optimizer"
    assert len(ids) == len(list(model.parameters())), "not all params covered"


def test_unknown_optimizer_raises():
    import pytest
    from tinylm.train import build_optimizers
    with pytest.raises(ValueError):
        build_optimizers(_tiny_model(), _cfg("sgd"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer_switch.py -v`
Expected: FAIL — `TrainConfig` has no `optimizer` field / `build_optimizers` not defined.

- [ ] **Step 3: Implement**

In `src/tinylm/train.py`, add to `TrainConfig` (near the optimizer hyperparameters, after `attention`):

```python
    # Optimizer mode: "muon" (matrix->Muon, scalar->AdamW) or "adamw" (all->AdamW)
    optimizer: str = "muon"
```

Add a module-level function (after `load_config`):

```python
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
    raise ValueError(f"Unknown optimizer mode: {cfg.optimizer!r} (expected 'muon' or 'adamw')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer_switch.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/train.py tests/test_optimizer_switch.py
git commit -m "feat: add muon|adamw optimizer mode switch for ablation runs A/B"
```

### Task 2: Refactor train loop, `train_step`, and checkpoint I/O to optimizer list

**Files:**
- Modify: `src/tinylm/train.py` (`train_step` 122-161, `save_checkpoint` 94-114, `load_checkpoint` 117-119, `train()` loop 182-242)
- Modify: `tests/test_train.py` (helper + call sites)

- [ ] **Step 1: Update the existing tests to the new API (these are the failing tests)**

In `tests/test_train.py`, replace `_build_training_components` and the `train_step`/`save_checkpoint` call sites:

```python
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
```

Update `test_smoke_10_steps`: `model, optimizers, loader = _build_training_components(cfg)` and `loss, grad_norm = train_step(model, tokens, optimizers, cfg, step)`.

Update `test_checkpoint_resume_consistency`: unpack `model, optimizers, loader`; call `train_step(model, tokens, optimizers, cfg, step)`; save with `save_checkpoint(ckpt_path, 4, model_i, optimizers_i, loader_i, vars(cfg))`; restore with:
```python
    for (opt, _), sd in zip(optimizers_r, ckpt["optimizers"]):
        opt.load_state_dict(sd)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `train_step`/`save_checkpoint` still expect `muon, adamw`.

- [ ] **Step 3: Implement the refactor in `src/tinylm/train.py`**

`train_step` signature → `train_step(model, tokens, optimizers, cfg, step)`. Replace the per-optimizer LR/zero_grad/step blocks:

```python
    for opt, lr_max in optimizers:
        lr = cosine_lr(step, cfg.warmup_steps, cfg.total_steps, lr_max)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
    # ... forward / loss / backward / clip unchanged ...
    for opt, _ in optimizers:
        opt.step()
```
(Return `(loss.item(), grad_norm.item())` as before.)

`save_checkpoint` signature → `save_checkpoint(path, step, model, optimizers, loader, config_dict)`; store:
```python
        "optimizers": [opt.state_dict() for opt, _ in optimizers],
```
(remove the separate `"muon"`/`"adamw"` keys).

In `train()`:
- Replace lines 182-187 with `optimizers = build_optimizers(model, cfg)`.
- Replace resume block (192-199): `for (opt, _), sd in zip(optimizers, ckpt["optimizers"]): opt.load_state_dict(sd)`.
- Replace the inline LR/zero_grad/step blocks (214-242) with the per-optimizer loop above.
- Update the save call (273): `save_checkpoint(ckpt_path, step, model, optimizers, loader, vars(cfg))`.

- [ ] **Step 4: Run full suite to verify green**

Run: `pytest tests/ -v`
Expected: PASS (all prior tests + Task 1 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/train.py tests/test_train.py
git commit -m "refactor: drive training loop and checkpoints from optimizer list"
```

---

## Chunk 2: Data-safety guards (kills the v1 repetition bug)

### Task 3: `ShardLoader` epoch counter + `max_epochs` guard

**Files:**
- Modify: `src/tinylm/data.py` (`ShardLoader.__init__` 24-32, advancement 44-62, state_dict 64-72)
- Test: `tests/test_data.py` (add tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_data.py`)

```python
def test_epoch_increments_on_wrap(tmp_path):
    from tinylm.data import ShardLoader
    # 1 shard, 90 tokens, batch 2 x (seq 8 +1) = 18 tokens -> 5 batches per epoch
    shard_dir = _make_shards(tmp_path, n_shards=1, tokens_per_shard=90)
    loader = ShardLoader(shard_dir, batch_size=2, seq_len=8, max_epochs=10)
    assert loader.epoch == 0
    for _ in range(5):
        loader.next_batch()
    assert loader.epoch == 1, f"expected epoch 1 after one pass, got {loader.epoch}"


def test_raises_when_max_epochs_exceeded(tmp_path):
    import pytest
    from tinylm.data import ShardLoader
    shard_dir = _make_shards(tmp_path, n_shards=1, tokens_per_shard=90)
    loader = ShardLoader(shard_dir, batch_size=2, seq_len=8, max_epochs=1)
    with pytest.raises(RuntimeError, match="max_epochs"):
        for _ in range(20):  # would loop well past 1 epoch
            loader.next_batch()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py -k "epoch" -v`
Expected: FAIL — `ShardLoader` has no `max_epochs`/`epoch`.

- [ ] **Step 3: Implement** in `src/tinylm/data.py`

Add `max_epochs: int | None = None` to `__init__`, set `self.max_epochs = max_epochs`, `self.epoch = 0`. Add a helper and route both wrap sites through it:

```python
    def _advance_shard(self) -> None:
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)
        if self.shard_idx == 0:
            self.epoch += 1
            print(f"[ShardLoader] completed epoch {self.epoch}")
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                raise RuntimeError(
                    f"max_epochs={self.max_epochs} reached — data would loop further. "
                    f"This guards against the v1 silent-repetition bug. "
                    f"Tokenize more shards or lower total_steps."
                )
        self._tokens = self._load(self.shard_idx)
        self.token_pos = 0
```
Replace the two inline wrap blocks (current lines 46-49 and 57-60) with `self._advance_shard()`. Add `"epoch": self.epoch` to `state_dict()` and restore it in `load_state_dict()` (default 0 for old states).

- [ ] **Step 4: Run to verify pass** (and full data suite for no regressions)

Run: `pytest tests/test_data.py -v`
Expected: PASS (existing 3 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/data.py tests/test_data.py
git commit -m "feat: ShardLoader epoch counter + max_epochs guard against silent repetition"
```

### Task 4: Pre-flight data-sufficiency arithmetic guard

**Files:**
- Create: `src/tinylm/preflight.py`
- Test: `tests/test_preflight.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preflight.py
import os
import numpy as np
import pytest


def _make_shards(tmp_path, n_shards, tokens_per_shard):
    d = str(tmp_path / "shards")
    os.makedirs(d)
    for i in range(n_shards):
        np.save(os.path.join(d, f"shard_{i:04d}.npy"),
                np.zeros(tokens_per_shard, dtype=np.uint16))
    return d


def test_passes_when_data_sufficient(tmp_path):
    from tinylm.preflight import check_data_sufficiency
    d = _make_shards(tmp_path, n_shards=4, tokens_per_shard=1000)  # 4000 unique
    # processed = steps*batch*accum*seq = 10*2*2*100 = 4000; max_epochs=2 -> need 2000
    check_data_sufficiency(d, total_steps=10, batch_size=2, grad_accum_steps=2,
                           seq_len=100, max_epochs=2)  # 4000*2 >= 4000 -> OK


def test_raises_when_underprovisioned(tmp_path):
    from tinylm.preflight import check_data_sufficiency
    d = _make_shards(tmp_path, n_shards=1, tokens_per_shard=500)  # 500 unique
    with pytest.raises(ValueError, match="insufficient"):
        # processed = 100*2*2*100 = 40000; max_epochs=4 -> 500*4=2000 < 40000
        check_data_sufficiency(d, total_steps=100, batch_size=2, grad_accum_steps=2,
                               seq_len=100, max_epochs=4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_preflight.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `src/tinylm/preflight.py`

```python
"""Pre-flight data-sufficiency check.

Prevents the v1 bug where the config implied far more tokens than were
tokenized, causing silent data repetition. Counts real shard sizes (cheap,
via mmap headers) and asserts the data can supply the run within max_epochs.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def count_shard_tokens(shard_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npy in {shard_dir!r}")
    return sum(int(np.load(p, mmap_mode="r").shape[0]) for p in paths)


def check_data_sufficiency(shard_dir: str, total_steps: int, batch_size: int,
                           grad_accum_steps: int, seq_len: int,
                           max_epochs: int) -> None:
    unique = count_shard_tokens(shard_dir)
    processed = total_steps * batch_size * grad_accum_steps * seq_len
    available = unique * max_epochs
    print(f"[preflight] unique={unique:,} tokens | processed={processed:,} | "
          f"max_epochs={max_epochs} | available={available:,}")
    if available < processed:
        raise ValueError(
            f"Data insufficient: need {processed:,} processed tokens but only "
            f"{unique:,} unique x {max_epochs} epochs = {available:,} available. "
            f"Tokenize more shards or lower total_steps."
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_preflight.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire into `train()` (call before the loop) and commit**

In `train()` after the loader is built, add:
```python
    from tinylm.preflight import check_data_sufficiency
    check_data_sufficiency(cfg.shard_dir, cfg.total_steps, cfg.batch_size,
                           cfg.grad_accum_steps, cfg.seq_len, max_epochs=4)
```
and construct the loader with `max_epochs=4`: `ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len, max_epochs=4)`.

```bash
git add src/tinylm/preflight.py tests/test_preflight.py src/tinylm/train.py
git commit -m "feat: pre-flight data-sufficiency guard wired into train()"
```

---

## Chunk 3: 8h-cap survival (SIGTERM + last.pt + resume override)

### Task 5: SIGTERM handler + `last.pt` on save

**Files:**
- Modify: `src/tinylm/train.py` (top-level handler; save block 271-273; loop)
- Test: `tests/test_sigterm.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sigterm.py
def test_sigterm_sets_stop_flag():
    import tinylm.train as T
    T._STOP_REQUESTED = False
    T._handle_sigterm(15, None)  # 15 == SIGTERM
    assert T._should_stop() is True


def test_should_stop_false_by_default():
    import tinylm.train as T
    T._STOP_REQUESTED = False
    assert T._should_stop() is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sigterm.py -v`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement** in `src/tinylm/train.py`

Module level (after imports):
```python
_STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    """SLURM sends SIGTERM before SIGKILL at the wall clock. Flag a clean stop."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[train] received signal {signum} — will checkpoint and exit at step end.")


def _should_stop() -> bool:
    return _STOP_REQUESTED
```

In `train()`, register the handler before the loop:
```python
    import signal
    signal.signal(signal.SIGTERM, _handle_sigterm)
```

Add a `_save_all(step)` helper inside the save path that writes both files:
```python
            ckpt_path = f"checkpoints/step_{step:05d}.pt"
            save_checkpoint(ckpt_path, step, model, optimizers, loader, vars(cfg))
            save_checkpoint("checkpoints/last.pt", step, model, optimizers, loader, vars(cfg))
```

At the end of each step (after the save block), honor the stop request:
```python
        if _should_stop():
            save_checkpoint("checkpoints/last.pt", step, model, optimizers, loader, vars(cfg))
            print(f"[train] checkpointed at step {step} on signal — exiting.")
            break
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_sigterm.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/train.py tests/test_sigterm.py
git commit -m "feat: SIGTERM checkpoint-on-kill + last.pt for 8h-cap rechaining"
```

### Task 6: `TINYLM_RESUME` / `TINYLM_SHARD_DIR` env overrides

**Files:**
- Modify: `src/tinylm/train.py` (`__main__` block 278-284)
- Test: `tests/test_train.py` (add)

- [ ] **Step 1: Write the failing test** (append to `tests/test_train.py`)

```python
def test_env_overrides_apply(tmp_path, monkeypatch):
    """TINYLM_RESUME / TINYLM_SHARD_DIR override config fields."""
    from tinylm.train import load_config, apply_env_overrides
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "run_name: t\nshard_dir: data/shards\nresume_from: null\n"
        "attention: mla\noptimizer: muon\n"
    )
    cfg = load_config(str(cfg_path))
    monkeypatch.setenv("TINYLM_RESUME", "checkpoints/last.pt")
    monkeypatch.setenv("TINYLM_SHARD_DIR", "/scratch/me/data")
    cfg = apply_env_overrides(cfg)
    assert cfg.resume_from == "checkpoints/last.pt"
    assert cfg.shard_dir == "/scratch/me/data"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train.py::test_env_overrides_apply -v`
Expected: FAIL — `apply_env_overrides` not defined.

- [ ] **Step 3: Implement** in `src/tinylm/train.py`

```python
def apply_env_overrides(cfg: TrainConfig) -> TrainConfig:
    """HPC job scripts inject resume/shard paths via env without editing YAML."""
    if os.environ.get("TINYLM_RESUME"):
        cfg.resume_from = os.environ["TINYLM_RESUME"]
    if os.environ.get("TINYLM_SHARD_DIR"):
        cfg.shard_dir = os.environ["TINYLM_SHARD_DIR"]
    return cfg
```
In `__main__`: `train(apply_env_overrides(load_config(sys.argv[1])))`.
Also make resume tolerant of a missing file (first segment): in `train()`, guard `if cfg.resume_from and os.path.exists(cfg.resume_from):` before loading.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/ -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/train.py tests/test_train.py
git commit -m "feat: TINYLM_RESUME/TINYLM_SHARD_DIR env overrides for HPC rechain"
```

---

## Chunk 4: Throughput / 40GB fixes (identical across all 4 runs)

### Task 7: Hoist CUDA syncs out of the grad-accum loop

**Files:**
- Modify: `src/tinylm/train.py` (loop 224-244)

- [ ] **Step 1: Implement** — accumulate the loss as a tensor; only sync on log steps.

Replace the grad-accum loop body so `loss_accum` is a detached tensor sum and `.item()` / `.isfinite()` are called **once per step**, not per micro-batch:
```python
        loss_accum = torch.zeros((), device=device)
        for _ in range(cfg.grad_accum_steps):
            tokens = loader.next_batch().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=cfg.use_bf16 and device == "cuda"):
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
        if step % cfg.log_every == 0:
            loss_val = loss_accum.item()      # single sync
            if not math.isfinite(loss_val):
                raise RuntimeError(f"Loss non-finite at step {step} — diverged.")
```
(Move the existing `wandb.log`/print block to use `loss_val` and `grad_norm.item()`.)

> Note: this changes when divergence is detected from every micro-batch to every `log_every` steps — acceptable; `grad_clip` still runs every step. `chunked_cross_entropy` is added in Task 8; until then keep `F.cross_entropy`.

- [ ] **Step 2: Verify no regression**

Run: `pytest tests/test_train.py -v`
Expected: PASS (smoke + resume determinism still hold on CPU).

- [ ] **Step 3: Commit**

```bash
git add src/tinylm/train.py
git commit -m "perf: hoist loss .item()/isfinite syncs out of grad-accum loop"
```

### Task 8: Chunked cross-entropy (40GB memory headroom)

**Files:**
- Create: `src/tinylm/losses.py`
- Test: `tests/test_losses.py` (create)
- Modify: `src/tinylm/train.py` (import + use in loop and `train_step`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch
import torch.nn.functional as F


def test_chunked_ce_matches_reference():
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(0)
    logits = torch.randn(257, 1000)        # N not divisible by chunk
    targets = torch.randint(0, 1000, (257,))
    ref = F.cross_entropy(logits, targets)
    got = chunked_cross_entropy(logits, targets, chunk_size=64)
    assert torch.allclose(got, ref, atol=1e-5), f"{got} vs {ref}"


def test_chunked_ce_single_chunk_equals_reference():
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(1)
    logits = torch.randn(32, 50)
    targets = torch.randint(0, 50, (32,))
    assert torch.allclose(
        chunked_cross_entropy(logits, targets, chunk_size=999),
        F.cross_entropy(logits, targets), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** `src/tinylm/losses.py`

```python
"""Memory-bounded cross-entropy.

F.cross_entropy upcasts logits to fp32 internally; on a (B*T, 32000) tensor
that transient fp32 copy is multiple GB and OOMs a 40GB A100. Chunking the
upcast bounds peak memory to chunk_size * vocab * 4 bytes while producing a
result numerically equal to F.cross_entropy (token-mean reduction).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          chunk_size: int = 4096) -> torch.Tensor:
    """Token-mean CE over (N, V) logits and (N,) targets, computed in chunks."""
    n = logits.shape[0]
    total = logits.new_zeros(())
    for i in range(0, n, chunk_size):
        total = total + F.cross_entropy(
            logits[i:i + chunk_size].float(),
            targets[i:i + chunk_size],
            reduction="sum",
        )
    return total / n
```

- [ ] **Step 4: Run to verify pass; wire into train.py**

Run: `pytest tests/test_losses.py -v` → PASS.
Add `from tinylm.losses import chunked_cross_entropy` to `train.py` and use it in both `train_step` and the `train()` loop (replacing `F.cross_entropy`). Run `pytest tests/ -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tinylm/losses.py tests/test_losses.py src/tinylm/train.py
git commit -m "perf: chunked cross-entropy to bound fp32 logits memory on 40GB"
```

### Task 9: `PrefetchLoader` — background thread, pinned memory, exact resume

**Files:**
- Modify: `src/tinylm/data.py` (add class)
- Test: `tests/test_prefetch.py` (create)
- Modify: `src/tinylm/train.py` (wrap loader when CUDA)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prefetch.py
import os
import numpy as np
import torch


def _make_shards(tmp_path, n_shards=2, tokens_per_shard=5000):
    d = str(tmp_path / "shards")
    os.makedirs(d)
    rng = np.random.default_rng(7)
    for i in range(n_shards):
        np.save(os.path.join(d, f"shard_{i:04d}.npy"),
                rng.integers(0, 128, tokens_per_shard, dtype=np.uint16))
    return d


def test_prefetch_yields_same_sequence_as_sync(tmp_path):
    from tinylm.data import ShardLoader, PrefetchLoader
    d = _make_shards(tmp_path)
    sync = ShardLoader(d, batch_size=2, seq_len=16)
    pf = PrefetchLoader(ShardLoader(d, batch_size=2, seq_len=16))
    try:
        for _ in range(20):
            assert torch.equal(sync.next_batch(), pf.next_batch())
    finally:
        pf.close()


def test_prefetch_state_matches_consumed_position(tmp_path):
    from tinylm.data import ShardLoader, PrefetchLoader
    d = _make_shards(tmp_path)
    sync = ShardLoader(d, batch_size=2, seq_len=16)
    pf = PrefetchLoader(ShardLoader(d, batch_size=2, seq_len=16))
    try:
        for _ in range(7):
            sync.next_batch(); pf.next_batch()
        assert pf.state_dict() == sync.state_dict()
    finally:
        pf.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_prefetch.py -v`
Expected: FAIL — `PrefetchLoader` not defined.

- [ ] **Step 3: Implement** in `src/tinylm/data.py`

```python
import queue
import threading


class PrefetchLoader:
    """Background-thread wrapper around ShardLoader.

    Removes the GPU stall caused by synchronous np.load + tensor assembly on the
    training thread. Yields pinned CPU tensors (caller does .to(device,
    non_blocking=True)). Resume is exact: state_dict() returns the underlying
    loader position AFTER the last batch the consumer actually received, so the
    discarded look-ahead batches are regenerated identically on resume.
    """

    def __init__(self, loader: "ShardLoader", depth: int = 2):
        self._loader = loader
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._last_state = loader.state_dict()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            batch = self._loader.next_batch()
            state = self._loader.state_dict()  # position AFTER this batch
            try:
                batch = batch.pin_memory()
            except RuntimeError:
                pass  # no CUDA / pinning unavailable (CPU tests)
            try:
                self._q.put((batch, state), timeout=0.5)
            except queue.Full:
                continue

    def next_batch(self) -> torch.Tensor:
        batch, state = self._q.get()
        self._last_state = state
        return batch

    def state_dict(self) -> dict:
        return self._last_state

    def load_state_dict(self, state: dict) -> None:
        self.close()
        self._loader.load_state_dict(state)
        self._last_state = self._loader.state_dict()
        self._stop = threading.Event()
        self._q = queue.Queue(maxsize=self._q.maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_prefetch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire into `train()` and commit**

In `train()`, wrap only on CUDA so CPU tests keep using the bare loader:
```python
    base_loader = ShardLoader(cfg.shard_dir, cfg.batch_size, cfg.seq_len, max_epochs=4)
    loader = PrefetchLoader(base_loader) if device == "cuda" else base_loader
```
`PrefetchLoader.state_dict`/`load_state_dict` already match the checkpoint calls.

```bash
git add src/tinylm/data.py tests/test_prefetch.py src/tinylm/train.py
git commit -m "perf: PrefetchLoader (bg thread + pinned mem) with exact-resume state"
```

---

## Chunk 5: Configs

### Task 10: Create A/B/C configs, smoke config, update D

**Files:**
- Create: `configs/run_A_mha_adamw.yaml`, `configs/run_B_mla_adamw.yaml`, `configs/run_C_mha_muon.yaml`, `configs/smoke_gate.yaml`
- Modify: `configs/run_D_mla_muon.yaml`

- [ ] **Step 1: Update D** (`configs/run_D_mla_muon.yaml`): set the honest-data recipe.

```yaml
run_name: run_D_mla_muon_v2     # distinct from v1 "run_D_full" (no WandB collision)
attention: mla
optimizer: muon
n_layers: 18
d_model: 1024
n_heads: 16
d_latent: 512
d_rope: 64
ffn_hidden: 2816
ctx: 2048
vocab_size: 32000
tie_weights: true
shard_dir: data/shards          # overridden by TINYLM_SHARD_DIR on HPC
batch_size: 16                  # 40GB starting micro-batch; confirm in smoke run
grad_accum_steps: 32            # effective batch = 16 * 32 = 512
seq_len: 2048
total_steps: 23000              # ~24B processed / 1.05M per step
warmup_steps: 2000
lr_muon: 0.02
lr_adamw: 0.001
weight_decay: 0.1
grad_clip: 1.0
log_every: 10
save_every: 500
wandb_project: tinylm
resume_from: null
compile: true
use_bf16: true
grad_checkpoint: false
```

- [ ] **Step 2: Create A/B/C** as copies differing only in `run_name`, `attention`, `optimizer`:

| File | run_name | attention | optimizer |
|---|---|---|---|
| `run_A_mha_adamw.yaml` | `run_A_mha_adamw` | `mha` | `adamw` |
| `run_B_mla_adamw.yaml` | `run_B_mla_adamw` | `mla` | `adamw` |
| `run_C_mha_muon.yaml`  | `run_C_mha_muon`  | `mha` | `muon`  |

All other fields identical to D's recipe above.

- [ ] **Step 3: Create `smoke_gate.yaml`** — copy of D with `run_name: smoke_gate`, `total_steps: 800`, `warmup_steps: 100`, `save_every: 200`.

- [ ] **Step 4: Verify all five configs load**

Run (PowerShell):
```powershell
foreach ($c in "run_A_mha_adamw","run_B_mla_adamw","run_C_mha_muon","run_D_mla_muon","smoke_gate") {
  python -c "from tinylm.train import load_config; print('$c', load_config('configs/$c.yaml').optimizer)"
}
```
Expected: prints each name + its optimizer mode, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "feat: ablation configs A/B/C + smoke gate; D updated to 8B/23k honest-data recipe"
```

---

## Chunk 6: HPC scripts (ported from D:/DiffMamba)

### Task 11: `setup_hpc.sh`

**Files:** Create `scripts/setup_hpc.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# One-time setup for TinyLM on Northeastern Explorer.
#   bash scripts/setup_hpc.sh
# No CUDA-extension compiles needed (MLA + Muon are pure PyTorch).
set -euo pipefail
SCRATCH="/scratch/${USER}"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^tinylm "; then
    echo "conda env 'tinylm' exists — skipping create."
else
    conda create -n tinylm python=3.11 -y
fi
conda activate tinylm

# Explorer driver 570.x caps at CUDA 12.8 — a cu130 wheel loads but is_available()==False.
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers datasets wandb huggingface_hub pyyaml numpy "lm-eval>=0.4.9" pytest
pip install -q -e "${HOME}/TinyLM"

mkdir -p "${SCRATCH}/tinylm/data" "${SCRATCH}/tinylm/runs" "${SCRATCH}/tinylm/logs" "${SCRATCH}/wandb"

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
echo "Add to ~/.bashrc: export HF_TOKEN=... WANDB_API_KEY=... HF_HUB_REPO_ID=..."
```

- [ ] **Step 2: Lint locally**

Run: `bash -n scripts/setup_hpc.sh`
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_hpc.sh
git commit -m "feat: Explorer one-time setup script (conda + torch cu128, no CUDA builds)"
```

### Task 12: `hpc_job.sh` (8h segment + pre-emptive rechain)

**Files:** Create `scripts/hpc_job.sh`

- [ ] **Step 1: Write the script** (adapted from DiffMamba `scripts/hpc_job.sh`)

```bash
#!/usr/bin/env bash
# 8h SLURM segment for one TinyLM run. Submit via scripts/submit_hpc.sh.
# Pre-emptively chains the next segment; resumes from checkpoints/last.pt.
# Env injected by submit: RUN_NAME, CONFIG, TOTAL_STEPS
#SBATCH --partition=gpu
#SBATCH --time=7:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:SIGTERM@120
set -euo pipefail
USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
RUN_DIR="${SCRATCH}/tinylm/runs/${RUN_NAME}"; LOG_DIR="${SCRATCH}/tinylm/logs"
CKPT="${RUN_DIR}/checkpoints/last.pt"
mkdir -p "${RUN_DIR}/checkpoints" "${LOG_DIR}"

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="${SCRATCH}/wandb"
export TINYLM_SHARD_DIR="${SCRATCH}/tinylm/data"

# Early-exit if this run already reached TOTAL_STEPS.
if [[ -f "${CKPT}" ]]; then
    STEP=$(python -c "import torch;print(torch.load('${CKPT}',map_location='cpu',weights_only=True).get('step',0))")
    echo "Resuming ${RUN_NAME} from step ${STEP}/${TOTAL_STEPS}"
    if [[ "${STEP}" -ge "$((TOTAL_STEPS-1))" ]]; then echo "Already complete."; exit 0; fi
    export TINYLM_RESUME="${CKPT}"
else
    echo "No checkpoint — starting ${RUN_NAME} from scratch."
fi

# Pre-emptively queue the next segment (survives a hard SIGKILL at the wall).
NEXT=$(sbatch --dependency=afterany:"${SLURM_JOB_ID}" --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG="${CONFIG}",TOTAL_STEPS="${TOTAL_STEPS}" \
    "${REPO}/scripts/hpc_job.sh" | awk '{print $NF}')
echo "Next segment queued as ${NEXT}."

cd "${RUN_DIR}"   # checkpoints/ is written here, relative to cwd
python -m tinylm.train "${REPO}/${CONFIG}" \
    || echo "Training exited non-zero (SIGTERM save is normal)."
echo "=== ${RUN_NAME} job ${SLURM_JOB_ID} done $(date) ==="
```

> Note: `train.py` writes `checkpoints/` relative to cwd, so we `cd "${RUN_DIR}"`. Each run gets its own `checkpoints/last.pt`.

- [ ] **Step 2: Lint**

Run: `bash -n scripts/hpc_job.sh`
Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_job.sh
git commit -m "feat: 8h SLURM segment with pre-emptive rechain + SIGTERM resume"
```

### Task 13: `submit_hpc.sh`

**Files:** Create `scripts/submit_hpc.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Submit one TinyLM run chain to Explorer.
#   bash scripts/submit_hpc.sh <run_name> <config_rel_path> <total_steps>
# e.g. bash scripts/submit_hpc.sh run_A_mha_adamw configs/run_A_mha_adamw.yaml 23000
set -euo pipefail
RUN_NAME="${1:?run_name}"; CONFIG="${2:?config path}"; TOTAL_STEPS="${3:?total_steps}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"; LOG_DIR="${SCRATCH}/tinylm/logs"
mkdir -p "${LOG_DIR}"
for V in HF_TOKEN WANDB_API_KEY HF_HUB_REPO_ID; do
    [[ -z "${!V:-}" ]] && echo "WARNING: ${V} not set."
done
JID=$(sbatch --job-name="${RUN_NAME}" --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG="${CONFIG}",TOTAL_STEPS="${TOTAL_STEPS}" \
    "${REPO}/scripts/hpc_job.sh" | awk '{print $NF}')
echo "Submitted ${RUN_NAME} as ${JID}. Monitor: squeue -u ${USER} ; tail -f ${LOG_DIR}/${RUN_NAME}_${JID}.log"
```

- [ ] **Step 2: Lint + commit**

```bash
bash -n scripts/submit_hpc.sh
git add scripts/submit_hpc.sh
git commit -m "feat: submit_hpc.sh to launch one run chain per ablation arm"
```

### Task 14: `upload_checkpoints.py` per-run subfolder

**Files:** Modify `scripts/upload_checkpoints.py`

- [ ] **Step 1: Add `--prefix`** — add the arg in `parse_args`:
```python
    p.add_argument("--prefix", default="", help="Path prefix in repo, e.g. run_A")
```
and set `path_in_repo=os.path.join(args.prefix, filename) if args.prefix else filename` in the `upload_file` call.

- [ ] **Step 2: Sanity-check it imports**

Run: `python scripts/upload_checkpoints.py --help`
Expected: usage shows `--prefix`.

- [ ] **Step 3: Commit**

```bash
git add scripts/upload_checkpoints.py
git commit -m "feat: per-run subfolder prefix for checkpoint uploads"
```

---

## Chunk 7: Operational runbook (Explorer — not TDD; gated checkpoints)

> These steps run on the cluster. Do them in order; do not launch the four full runs until every smoke gate is green.

### Task 15: Push branch, set up Explorer, tokenize 8B tokens

- [ ] Push the branch and pull on Explorer:
```bash
git push -u origin feat/hpc-rerun-ablation
# on Explorer login node:
git clone https://github.com/shivnarainms22/TinyLM.git ~/TinyLM || (cd ~/TinyLM && git fetch && git checkout feat/hpc-rerun-ablation && git pull)
```
- [ ] `bash scripts/setup_hpc.sh` ; export `HF_TOKEN`, `WANDB_API_KEY`, `HF_HUB_REPO_ID`.
- [ ] Tokenize 8B unique tokens to scratch (interactive GPU/CPU session; ~hours):
```bash
python scripts/tokenize_shards.py --split sample-100BT --out-dir /scratch/$USER/tinylm/data --max-shards 80
```
- [ ] **Verify token count gate:**
```bash
python -c "from tinylm.preflight import count_shard_tokens as c; print(c('/scratch/$USER/tinylm/data'))"
```
Expected: ≈8.0e9. Confirm `≥ 8_000_000_000` before proceeding.

### Task 16: Smoke gate (capped run) — must pass all gates

- [ ] Submit: `bash scripts/submit_hpc.sh smoke_gate configs/smoke_gate.yaml 800`
- [ ] Watch the log and confirm **all** gates (kill after):
  - [ ] tokens/sec printed (record it; expect ~50–65k on 40GB)
  - [ ] step-0 loss ≈ ln(32000) ≈ 10.4 (init sane)
  - [ ] loss clearly decreasing by step 100; < 5.0 by step 500
  - [ ] zero NaN/Inf
  - [ ] no `torch.compile` graph-break/recompile spam
  - [ ] **OOM check:** if OOM, lower `batch_size` (16→8, raise `grad_accum_steps` 32→64) in all configs and re-smoke. If still OOM, set `compile: false` then try `grad_checkpoint: true` (note the corrupted-grad warning — verify resume/loss sanity).
- [ ] **Resume gate:** note loss at the last smoke checkpoint, resubmit the same command (it resumes from `last.pt`), confirm loss continues from the saved value (not a jump).
- [ ] Record final smoke `batch_size`/`grad_accum_steps`; if changed, update all five configs + commit.

### Task 17: Launch the four runs

- [ ] Start the per-run checkpoint uploaders (one per run, e.g. in tmux), each pointed at its run dir with its prefix:
```bash
python scripts/upload_checkpoints.py --repo $HF_HUB_REPO_ID --ckpt-dir /scratch/$USER/tinylm/runs/run_A_mha_adamw/checkpoints --prefix run_A &
# ...repeat for run_B/run_C/run_D...
```
- [ ] Submit all four chains:
```bash
bash scripts/submit_hpc.sh run_A_mha_adamw configs/run_A_mha_adamw.yaml 23000
bash scripts/submit_hpc.sh run_B_mla_adamw configs/run_B_mla_adamw.yaml 23000
bash scripts/submit_hpc.sh run_C_mha_muon  configs/run_C_mha_muon.yaml  23000
bash scripts/submit_hpc.sh run_D_mla_muon_v2 configs/run_D_mla_muon.yaml 23000
```
- [ ] Monitor `squeue -u $USER`; each run auto-rechains across ~14 segments. Record per-run tokens/sec (expect MHA runs A/C faster than MLA runs B/D).

### Task 18: Eval each run

- [ ] After a run reaches step 22999, eval its final checkpoint:
```bash
python scripts/eval_tinylm.py --checkpoint /scratch/$USER/tinylm/runs/<run>/checkpoints/step_22999.pt \
    --tokenizer meta-llama/Llama-2-7b-hf --output results/<run>_eval.json
```
- [ ] Repeat for all four runs → `results/run_{A,B,C,D}_eval.json`.

### Task 19: Preserve v1, write ablation results, update README

- [ ] **Preserve v1 first (no overwrite):**
```bash
mkdir -p results/v1_repeated_data
git mv results/run_D_eval.json results/v1_repeated_data/run_D_eval.json
git mv results/baseline_comparison.md results/v1_repeated_data/baseline_comparison.md
```
- [ ] Add new `results/run_{A,B,C,D}_eval.json` and a new `results/ablation_comparison.md` with the 4×4 table (rows = runs A–D, cols = HellaSwag/ARC-Easy/LAMBADA/Winogrande) + per-run tokens/sec + a row for v1 Run D and the TinyLlama-1.1B baseline (from `baseline_results.json`).
- [ ] Update `README.md`: **leave the pinned hypothesis verbatim**; relabel the existing Run D table as "Run D v1 (data-repeated)"; add the new ablation table below; update the ablation status table (A/B/C/D → Complete).
- [ ] Commit:
```bash
git add results/ README.md
git commit -m "docs: add 4-run ablation results; archive v1 Run D as contrast"
```

### Task 20: Finalize

- [ ] Run `pytest tests/ -v` one final time on the branch (CPU) — confirm all green.
- [ ] Use the `finishing-a-development-branch` skill to decide merge/PR into `main`.
- [ ] Update HF model card + memory files (`project_tinylm_state`, `project_tinylm_run_d_results`, `project_tinylm_lessons`) with the new ablation outcome.

---

## Verification summary (run before claiming completion)

- [ ] `pytest tests/ -v` → all green (30 existing + new optimizer/epoch/preflight/sigterm/losses/prefetch tests).
- [ ] All five configs load and report the right optimizer mode.
- [ ] Smoke gates all green on a 40GB node (tokens/sec, loss curve, NaN-free, resume continuity).
- [ ] Token count ≥ 8e9 confirmed before launch.
- [ ] Four eval JSONs produced; ablation table built; v1 preserved under `results/v1_repeated_data/`; hypothesis unchanged.
