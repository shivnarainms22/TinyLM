---
date: 2026-05-12
topic: tinylm-phase2
status: approved
reference: 250M_SLM_Implementation_Plan_revised.pdf (Phase 2, Phase 3 partial)
predecessor: 2026-05-12-tinylm-phase1-design.md
---

# TinyLM — Phase 2 Design Lock-in

## Project context

Phase 1 delivered the full model architecture (`model.py`, `muon.py`, 12 tests green,
274,642,944 parameters). Phase 2 builds the training stack and runs the "Toy Run —
Proof of Life": 1B tokens on a RunPod A100-80GB using Run D config (MLA + Muon).
All four go/no-go gates must be green before committing to Phase 4's ~$180 spend.

## Decisions locked in brainstorming (2026-05-12)

| Decision | Value | Reasoning |
|---|---|---|
| Data loading | Pre-tokenized `.npy` shards | Only way the 80k tok/sec gate is meaningful; streaming bottleneck hides real throughput. |
| GPU target | RunPod A100-80GB | User has RunPod credits. Zero code difference vs. Vast.ai — only setup instructions change. |
| Logging | WandB | Real-time loss curves, permanent record for gate verification, free tier sufficient. |
| Phase scope | Phase 2 only — FineWeb-Edu loader + training loop | Annealing mix (NuminaMath + OpenHermes) deferred to Phase 3, built before Phase 4. |
| Code split | `data.py` + `train.py` (Option 1) | `ShardLoader` tested independently; shard-resume logic is the highest-risk component. |

## Phase 2 deliverables

| # | Artifact | Path | Notes |
|---|---|---|---|
| 1 | FineWeb-Edu shard loader | `src/tinylm/data.py` | `ShardLoader` class, state-dict for resume |
| 2 | Training loop | `src/tinylm/train.py` | Dual optimizer, cosine LR, WandB, checkpoint |
| 3 | Pre-tokenizer script | `scripts/tokenize_shards.py` | Runs on RunPod CPU; writes `shard_XXXX.npy` |
| 4 | RunPod setup script | `scripts/setup_runpod.sh` | pip installs, WandB/HF login, runs tokenizer |
| 5 | Toy run config | `configs/run_D_mla_muon.yaml` | 1000 steps ≈ 1B tokens, Run D hyperparams |
| 6 | Data loader tests | `tests/test_data.py` | 3 tests — shape, shard wrap, state-dict round-trip |
| 7 | Training smoke tests | `tests/test_train.py` | 2 tests — 10-step smoke, checkpoint resume consistency |
| — | CLAUDE.md update | `CLAUDE.md` | Add Phase 2 run commands |

## Component design

### `src/tinylm/data.py` — ShardLoader

Reads pre-tokenized `.npy` shard files (dtype `uint16`, 100M tokens each). Yields
`(B, T+1)` token tensors — caller slices `[:, :-1]` as input and `[:, 1:]` as target.

```python
class ShardLoader:
    def __init__(self, shard_dir: str, batch_size: int, seq_len: int): ...
    def next_batch(self) -> torch.Tensor:  # (B, T+1) int64
    def state_dict(self) -> dict:           # {'shard_idx': int, 'token_pos': int}
    def load_state_dict(self, state: dict): ...
```

After exhausting all shards the loader wraps to shard 0. This is a no-op for the
toy run (1B tokens, shards cover exactly that), but required for Phase 4's 20B run
where two passes over the 10B FineWeb-Edu sample are needed.

Shard files are sorted lexicographically so `shard_0000.npy` … `shard_0009.npy` is
the canonical order.

### `src/tinylm/train.py` — training loop

**Entry point:** `python -m tinylm.train configs/run_D_mla_muon.yaml`

**Config loading:** YAML → flat dict → `TrainConfig` dataclass. Unknown keys raise
`ValueError` at startup (no silent misconfigs).

**Optimizer setup** (uses `partition_params` from `muon.py`):
```
matrix_params → Muon(lr=lr_muon, momentum=0.95, ns_steps=5)
scalar_params → AdamW(lr=lr_adamw, weight_decay=weight_decay, betas=(0.9, 0.95))
```

**LR schedule:** linear warmup from 0 to `lr_max` over `warmup_steps`, then cosine
decay to `lr_max × 0.1`. Same ratio applied to both optimizers (different `lr_max`).
Updated each step before the optimizer step.

**Training loop (per step):**
```
tokens = loader.next_batch()            # (B, T+1) on device
logits = model(tokens[:, :-1])          # (B, T, vocab)
loss = F.cross_entropy(logits.view(-1, vocab_size), tokens[:, 1:].reshape(-1))
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
[update LR in both param groups]
muon.step(); adamw.step()
[zero_grad(set_to_none=True)]
```

**Checkpoint format** (saves `checkpoints/step_{N:05d}.pt`):
```python
{
    'step': int,
    'model': model.state_dict(),
    'muon': muon.state_dict(),
    'adamw': adamw.state_dict(),
    'loader': loader.state_dict(),   # shard_idx + token_pos
    'config': config_dict,
}
```

Resume: set `resume_from: checkpoints/step_00500.pt` in YAML.

**WandB logging** (every `log_every` steps):
- `train/loss`, `train/grad_norm`
- `perf/tokens_per_sec`, `perf/step_time_ms`
- `optim/lr_muon`, `optim/lr_adamw`

**Red flags (print warning + continue; never silently ignore):**
- Loss spike > 0.5 units in a single step → log warning
- Any NaN/Inf in loss → raise immediately with step number

### `scripts/tokenize_shards.py`

CLI: `python scripts/tokenize_shards.py --split sample-10BT --out-dir data/shards --max-shards 10`

- Tokenizer: `meta-llama/Llama-2-7b-hf` (locked in Phase 0; `vocab_size=32000`)
- Appends `eos_token_id` after each document
- Shard size: 100M tokens → `shard_0000.npy` (uint16, ~200MB each)
- `--max-shards 10` gives 1B tokens for the toy run; omit for Phase 4's full dataset

### `scripts/setup_runpod.sh`

```bash
#!/bin/bash
set -e
pip install torch transformers datasets wandb pyyaml --quiet
pip install -e /workspace/tinylm --quiet
wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN
mkdir -p /workspace/tinylm/data/shards
mkdir -p /workspace/tinylm/checkpoints
python /workspace/tinylm/scripts/tokenize_shards.py \
    --split sample-10BT \
    --out-dir /workspace/tinylm/data/shards \
    --max-shards 10
echo "Setup complete. Run: python -m tinylm.train configs/run_D_mla_muon.yaml"
```

### `configs/run_D_mla_muon.yaml`

```yaml
# TinyLM — Run D (MLA + Muon) — Toy run (1B tokens)
# Upgrade to full run: total_steps: 20000, warmup_steps: 2000
run_name: run_D_toy
attention: mla

# Model (locked — do not change)
n_layers: 18
d_model: 1024
n_heads: 16
d_latent: 512
d_rope: 64
ffn_hidden: 2816
ctx: 2048
vocab_size: 32000
tie_weights: true

# Data
shard_dir: data/shards
batch_size: 512
seq_len: 2048

# Training
total_steps: 1000
warmup_steps: 100
lr_muon: 0.02
lr_adamw: 0.001
weight_decay: 0.1
grad_clip: 1.0

# Logging
log_every: 10
save_every: 100
wandb_project: tinylm

# Resume (set to checkpoint path to resume)
resume_from: null
```

## Testing strategy

Tests run on **Windows CPU, no GPU**. A tiny model (`n_layers=2, d_model=64,
n_heads=4, d_latent=32, d_rope=8, ffn_hidden=128, vocab_size=128`) is used for all
smoke tests. Synthetic `.npy` shards are created in `tmp` directories.

### `tests/test_data.py` (3 tests)

| Test | Asserts |
|---|---|
| `test_batch_shape` | `next_batch()` returns `(batch_size, seq_len+1)` int64 tensor |
| `test_shard_wrapping` | Exhausting 2 synthetic shards wraps to shard 0; token at position 0 of wrap matches position 0 of first shard |
| `test_state_dict_round_trip` | After 3 batches, `state_dict()` captures correct position; `load_state_dict()` + `next_batch()` yields the same tensor as the original sequence |

### `tests/test_train.py` (2 tests)

| Test | Asserts |
|---|---|
| `test_smoke_10_steps` | 10 steps on tiny model + synthetic shards: loss is finite at every step, and loss at step 10 < loss at step 1 |
| `test_checkpoint_resume_consistency` | Run 5 steps baseline; separately run 3 steps, save checkpoint, load, run 2 more steps — loss at step 5 matches the baseline |

## RunPod execution workflow

```
1. Launch A100-80GB instance on RunPod
2. Clone repo: git clone <repo> /workspace/tinylm
3. bash /workspace/tinylm/scripts/setup_runpod.sh
   (sets up deps, tokenizes 1B tokens — ~30 min)
4. python -m tinylm.train configs/run_D_mla_muon.yaml
5. [At step ~500] Ctrl+C to kill
6. Set resume_from: checkpoints/step_00500.pt in YAML
7. python -m tinylm.train configs/run_D_mla_muon.yaml
8. Verify WandB shows loss continues from step 500 value
```

## Go/no-go gate (from ablation_plan.md)

All four must be green before Phase 4:

| Gate | Target |
|---|---|
| tokens/sec | ≥ 80,000 on A100-80GB |
| Loss at step 100 | clearly decreasing |
| Loss at step 500 | < 5.0 |
| NaN/Inf events | zero |
| Checkpoint resume | loss continues from saved value (not reset) |

## Out of scope for Phase 2

- Annealing mix (NuminaMath-CoT + OpenHermes-2.5) — Phase 3
- Full 20B token tokenization — Phase 3
- Runs A, B, C configs — Phase 4
- `eval_wrapper.py` (HF PreTrainedModel adapter for lm-eval) — Phase 5
- `inference.py` (KV-cache memory measurement) — Phase 5

## Open issues flagged

- **`torch.compile` on A100:** Training will benefit from `torch.compile(model)`.
  Phase 1 unit tests run in eager mode (Windows CPU). Enable compile in `train.py`
  behind a `--compile` flag so CPU tests keep working.
- **`pyyaml` dependency:** Add to `pyproject.toml` dev deps or main deps for
  RunPod. Since `train.py` is the runtime entry point, it belongs in main deps.
- **HuggingFace auth for Llama-2 tokenizer:** `meta-llama/Llama-2-7b-hf` is
  gated. User needs HuggingFace token with Llama-2 access. `setup_runpod.sh`
  handles the login step; user must supply `$HF_TOKEN`.
