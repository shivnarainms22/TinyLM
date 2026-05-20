# TinyLM HPC Re-run — Full 4-Run Ablation (Design Spec)

**Date:** 2026-05-20
**Status:** Approved (design phase)
**Supersedes operationally:** the single buggy Run D (1B tokens repeated ~21×).
**Does NOT supersede:** the pinned hypothesis, the TinyLlama-1.1B baseline, or
the v1 Run D as a documented contrast artifact.

---

## 1. Motivation

The shipped portfolio has one run (D = MLA+Muon) with two structural problems,
both caused by compute cost on rented RunPod A100s, not by the science:

1. **Data bug (Mistake 1):** trained on ~1B unique FineWeb-Edu tokens repeated
   ~21× over 20k steps, not 20B unique. `ShardLoader` wrapped silently. This
   depressed the long-range-coherence benchmarks (HellaSwag 32.4%,
   LAMBADA 29.2%).
2. **No ablation (Mistake 3):** only 1 of 4 runs completed, so MLA-vs-MHA and
   Muon-vs-AdamW cannot be isolated. The portfolio's headline weakness.

University HPC (Northeastern Explorer) removes the cost constraint that caused
both. This spec re-runs the **full A/B/C/D ablation** on honest data, on
free compute, reusing the already-built-and-tested model/optimizer/training
code.

## 2. Goals & non-goals

**Goals**
- Train all four ablation arms (A/B/C/D) on identical data and identical
  optimized code, so only attention/optimizer differ.
- Train on **8B unique FineWeb-Edu tokens, ~3 epochs (~24B processed,
  ~23k steps)** — ~4× past Chinchilla, in the regime where the benchmarks
  become meaningful for a 275M model.
- Run reliably on **40GB A100 nodes** under Explorer's **8-hour job cap**,
  without assuming an 80GB node is available.
- Preserve all prior-run artifacts as a labeled "v1" contrast.

**Non-goals**
- Annealing data mix (Mistake 4) — explicitly **deferred**. All four runs use
  pure FineWeb-Edu for a clean ablation. May be added later as a single
  "Run D+" enhancement.
- Hitting the 80k tok/s gate on 40GB hardware (see §7 — honest expectation is
  ~62k on 40GB; 80k is only realistic on an 80GB node, which we do not assume).
- Multi-GPU / distributed training (single A100 per job, as on RunPod).

## 3. Locked constants (unchanged from the project spec)

- Baseline: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Tokenizer: `meta-llama/Llama-2-7b-hf`, `vocab_size=32000`
- Architecture: `n_layers=18, d_model=1024, n_heads=16, d_latent=512,
  d_rope=64, ffn_hidden=2816, ctx=2048, tie_weights=true` (274.6M params)
- Benchmark suite: HellaSwag, ARC-Easy, LAMBADA, Winogrande
- Effective batch: 512 × 2048 ≈ 1.05M tokens/step
- Pinned hypothesis in README.md: **not edited** to match results.

## 4. Ablation matrix + the one required code change

| Run | `attention` | `optimizer` |
|-----|-------------|-------------|
| A   | `mha`       | `adamw`     |
| B   | `mla`       | `adamw`     |
| C   | `mha`       | `muon`      |
| D   | `mla`       | `muon`      |

All four configs are identical except those two fields.

**Code change (the only training-logic change):** `train.py` currently
hardcodes Muon(matrix params) + AdamW(scalar params). Add an
`optimizer: muon|adamw` config field:
- `muon` → today's behavior (matrix params → Muon, scalar params → AdamW).
- `adamw` → **all** params → AdamW, no Muon instance created.

LR handling: in `adamw` mode, `lr_adamw` governs all params and `lr_muon` is
unused. Cosine schedule + warmup unchanged. All other hyperparameters stay
locked across the four runs.

## 5. Data pipeline (closes the Phase-3 gap)

- Tokenize **8B unique** FineWeb-Edu tokens → ~80 shards × ~100M tokens,
  `uint16` `.npy`, written to `/scratch/$USER/tinylm/data`. Extend the existing
  `scripts/tokenize_shards.py` (parametrize shard count / source subset; use
  FineWeb-Edu `sample-100BT`).
- **Hard pre-flight arithmetic guard** (permanently kills Mistake 1 & 6):
  - Pre-flight assertion before training:
    `num_shards × shard_size ≥ processed_tokens / max_epochs`.
  - `ShardLoader` gains an **epoch counter** and a `max_epochs` ceiling
    (default 4). It logs each epoch boundary and **raises** instead of silently
    wrapping past `max_epochs` (current silent wrap: `data.py:46-49`).

## 6. SLURM auto-rechain (ported from D:/DiffMamba)

Port three scripts, adapted from DiffMamba's Lightning+Hydra stack to TinyLM's
plain `python -m tinylm.train` + YAML + `step_NNNNN.pt` stack.

- **`scripts/setup_hpc.sh`** — conda env (Python 3.10+), `module load
  anaconda3/2024.06 cuda/12.8.0`, **torch cu128** (Explorer driver 570.x caps
  at CUDA 12.8; a cu130 build silently reports `cuda.is_available()==False`),
  plus `lm-eval transformers datasets wandb huggingface_hub pytest`.
  **No CUDA-extension compiles** — MLA and Muon are pure PyTorch, so we skip
  DiffMamba's `causal-conv1d`/`mamba-ssm`/`flash-attn` source builds entirely.
- **`scripts/hpc_job.sh`** — one 8h job (`#SBATCH --time=7:50:00`,
  `--gres=gpu:a100:1`, `--cpus-per-task=8`, `--mem=64G`). Pattern lifted from
  DiffMamba `hpc_job.sh`:
  1. Pre-emptively `sbatch --dependency=afterany:$SLURM_JOB_ID` the next segment
     **before** training starts (survives a hard SIGKILL at wall time).
  2. Early-exit if the latest checkpoint's step ≥ `total_steps`.
  3. Otherwise resume from `checkpoints/last.pt` and train.
- **`scripts/submit_hpc.sh`** — submit one independent chain per run
  (A/B/C/D queue concurrently; the scheduler runs them as GPUs free up).

**Resume adaptation:** DiffMamba resumes from a fixed `last.ckpt`; TinyLM saves
`step_NNNNN.pt`. On every save, also write/overwrite `checkpoints/last.pt`
(deterministic resume target). Add an **env/CLI override** so `hpc_job.sh` can
set `resume_from` → `last.pt` without editing YAML.

## 7. SIGTERM checkpoint-on-kill (new, mandatory under the 8h cap)

SLURM sends SIGTERM before SIGKILL at the wall. Add a signal handler in
`train.py`:
- On SIGTERM: set a flag; at the end of the current step, save checkpoint +
  `last.pt`, then exit cleanly.
- Pair with `#SBATCH --signal=B:SIGTERM@120` (≈2-minute warning).

Cadence math: ~1.05M tokens/step ÷ ~62k tok/s ≈ 17s/step → ~1,700 steps per 8h
segment → ~14 segments/run, ~56 segments across 4 runs. With the SIGTERM
handler, a wall-time kill loses ≈0 progress; `save_every=500` (~2.3h) remains
as a coarse backup.

## 8. Throughput / 40GB fixes (shared by all 4 runs → ablation stays valid)

Measured RunPod throughput was 62.6k tok/s (~33% MFU on 80GB A100) vs the
≥80k gate. Root causes identified in code:

1. **Async-prefetch dataloader** + `pin_memory` + `.to(device,
   non_blocking=True)`. Fixes GPU starvation from the synchronous, single-thread
   loader (`train.py:226`, `data.py`). **Numerics-identical.**
2. **Hoist CUDA syncs** out of the grad-accum loop — remove per-micro-batch
   `.item()`/`.isfinite()` (`train.py:233,238`), which force a device sync 16×
   per step; sync only on `log_every`. **Numerics-identical.**
3. **Chunked/fused cross-entropy** — avoid materializing the full
   `(B,T,32000)` fp32 logits (`train.py:228-232`). Primary lever for 40GB
   memory headroom and a speedup. Tiny numerics delta, **applied identically to
   all 4 runs**.
4. **40GB micro-batch:** the v1 config used micro-batch 32 (~50GB peak) on an
   80GB card (`run_D_mla_muon.yaml:21,46`). On 40GB this OOMs. Smoke-test the
   largest micro-batch that fits (likely 8–16), set `grad_accum` to hold
   effective batch 512. Document the chosen value.

**Honest expectation:** ~62k tok/s on 40GB after fixes (MHA runs A/C faster than
MLA runs B/D, since MLA up-projects latent→K/V and concatenates rope dims).
80k is only realistic on an 80GB node, which this design does not assume. The
fixes mainly buy back what the 40GB micro-batch penalty costs.

**Ablation validity:** because all four runs share the optimized code, the
comparison stays clean. The smoke run validates the loss curve is unaffected
(step-0 loss ≈ ln(32000) ≈ 10.4).

## 9. Smoke-gate run first (separate, capped — the toy run Lessons demanded)

A capped config (`total_steps≈800`) run on Explorer 40GB **before** the four
full runs, to verify go/no-go gates:
- tokens/sec measured
- loss clearly decreasing at step 100; < 5.0 at step 500
- zero NaN/Inf
- checkpoint resume continues loss from the saved value
- SIGTERM handler saves and exits cleanly
- rechain submits the next segment and resumes correctly
- `torch.compile` has no graph breaks / recompiles on Explorer's torch

Kill after gates pass. Only then launch A/B/C/D.

## 10. Eval, HF, results

- **Eval:** after each run reaches ~23k steps, run `scripts/eval_tinylm.py` on
  the 4 locked benchmarks → `results/run_{A,B,C,D}_eval.json`.
- **HF:** `scripts/upload_checkpoints.py` polls `/scratch` checkpoints and
  uploads to **one repo with per-run subfolders** (`run_A/`…`run_D/`). The v1
  `step_19999.pt` is retained as the v1 artifact (see §11).
- **Deliverable:** a 4-row ablation table (4 benchmarks) + MHA-vs-MLA KV-cache
  comparison + per-run tokens/sec. Update `README.md` and `results/`.

## 11. Preservation & versioning (answers "will this overwrite prior work?")

Nothing prior is destroyed. Concretely:

| Asset | Handling |
|-------|----------|
| GitHub `main` | All work on `feat/hpc-rerun-ablation`; `origin/main` untouched until a deliberate merge. v1 stays in history. |
| `results/run_D_eval.json`, `results/baseline_comparison.md` | **Archived** to `results/v1_repeated_data/` before new results are written as `results/run_{A,B,C,D}_*`. |
| `baseline_results.json`, `results/baseline/` | **Untouched** — still the comparison target. |
| `README.md` pinned hypothesis | **Verbatim, not edited** (non-negotiable). Old Run D row relabeled "v1 (data-repeated)"; new ablation rows added below. |
| `configs/run_D_mla_muon.yaml` | Modified for the new recipe; old recipe preserved in git history + memory file. New `run_A/B/C` configs added. |
| HF `Shiv-22/tinylm-checkpoints` (`step_19999.pt`) | Kept as the v1 artifact; new runs go in per-run subfolders. |
| WandB `dig7xsqf` (`run_D_full`) | Preserved; new runs use distinct `run_name`s to avoid `resume="allow"` collision. |

The v1 Run D is retained deliberately as a **contrast data point**: "1B tokens
× 21 epochs" vs "8B unique × 3 epochs" is itself a finding about data
repetition.

## 12. Testing (TDD — failing test first; all 30 existing tests stay green)

New tests required before implementation code:
- **Optimizer switch:** `adamw` mode puts all params on AdamW; `muon` mode
  partitions matrix→Muon, scalar→AdamW (both verified by param-group contents).
- **ShardLoader epoch guard:** epoch counter increments on wrap; raises when
  `max_epochs` exceeded.
- **Pre-flight arithmetic guard:** catches an under-provisioned shard set.
- **SIGTERM handler:** signal sets the flag; checkpoint + `last.pt` saved on
  the simulated signal.
- **Async dataloader:** yields batches identical to the synchronous loader
  (determinism), and the `pin_memory` path works.
- **Chunked CE:** matches `F.cross_entropy` within tolerance.
- **Resume override:** env/CLI override of `resume_from` works.

## 13. Risks / open items

- **40GB OOM even at micro-batch 8** → fall back to gradient checkpointing, but
  the v1 config warns that **compile + checkpoint corrupts gradients**
  (`run_D_mla_muon.yaml:47`). May force disabling `compile` on 40GB. Resolved
  empirically in the smoke run.
- **Queue waits** between 8h segments stretch wall-clock beyond pure compute
  (~4.4 days/run of compute; calendar time depends on cluster contention).
- **`torch.compile`** behavior on Explorer's torch version — verified in smoke
  run before committing to 4 full runs.
- **Stale `worktree-agent-*` branches** exist locally; out of scope, left as-is.

## 14. Execution order (for the plan)

1. Code changes + tests (optimizer switch, epoch guard, pre-flight guard,
   SIGTERM handler, async loader, chunked CE, resume override).
2. A/B/C configs + updated D config (8B tokens, ~23k steps, optimizer field,
   40GB batch).
3. HPC scripts (setup/job/submit) ported + adapted.
4. Tokenize 8B unique tokens on Explorer.
5. Smoke-gate run; verify all gates.
6. Launch A/B/C/D chains.
7. Eval + HF upload + ablation table + README/results update (preserving v1).
