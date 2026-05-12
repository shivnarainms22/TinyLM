---
date: 2026-05-12
topic: tinylm-phase0
status: approved
reference: 250M_SLM_Implementation_Plan_revised.pdf
---

# TinyLM — Phase 0 Design Lock-in

## Project context

Train a ~275M parameter small language model using Multi-head Latent Attention
(MLA, from DeepSeek-V2) and the Muon optimizer (Newton-Schulz orthogonalization)
on 20B tokens of FineWeb-Edu. Deliverable is a four-row ablation table that
isolates the contribution of MLA, Muon, and their combination versus a standard
MHA + AdamW baseline at fixed compute. Reference plan: PDF in repo root.

## Decisions locked in brainstorming

| Decision | Value | Source / Reasoning |
|---|---|---|
| Baseline model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | User choice; modern arch, RoPE-native, popular comparison point. |
| Tokenizer | `meta-llama/Llama-2-7b-hf` | Follows from baseline; locks `vocab_size = 32000`. |
| Local compute (Phase 0/1) | Google Colab / Kaggle free tier for baseline eval; local Windows CPU for unit tests | User has no local GPU. |
| Training compute (Phase 2/4) | Rented A100-80GB on Vast.ai (spot) | Per PDF. |
| Codebase base | Fork of `https://github.com/KellerJordan/modded-nanogpt` | PDF Phase 1 Step 1; ships working Muon + FineWeb dataloader. |
| Budget commitment | Plan all 4 ablation runs; Phase 2 toy run is the real go/no-go gate | Keeps options open without scoping down prematurely. |
| Hypothesis style | Loose, direction-only; refine numbers post-Phase 5 | User preference; honors PDF Phase 0 Step 1 (must pin *something*) without committing to specific percentages pre-data. |

## Phase 0 deliverables (this spec → 5 artifacts)

| # | Artifact | Path |
|---|---|---|
| 1 | Pinned hypothesis | `README.md` (top section) |
| 2 | Baseline eval JSON for TinyLlama-1.1B on the 4 benchmarks | `baseline_results.json` |
| 3 | Locked ablation table (4 rows: A/B/C/D) | `docs/ablation_plan.md` |
| 4 | Locked benchmark suite | `docs/eval_suite.md` |
| 5 | Per-project rules for subagents | `CLAUDE.md` |

### Hypothesis text (verbatim, to be pinned in `README.md`)

> A 275M parameter model trained with MLA + Muon on 20B tokens of FineWeb-Edu
> will achieve materially-better-than-random performance on HellaSwag,
> ARC-Easy, LAMBADA, and Winogrande, while demonstrating a measurable
> KV-cache memory reduction at inference versus an equivalent MHA baseline
> (Run A). Exact percentage targets are filled in post-Phase 5.

### Benchmark suite (locked — no additions after Phase 0)

- HellaSwag (commonsense reasoning)
- ARC-Easy (factual recall / QA)
- LAMBADA (long-range LM coherence)
- Winogrande (coreference resolution)

Same `lm_eval` command for all 4 ablation checkpoints + the baseline.

### Ablation table (locked — one variable changes per row)

| Run | Attention | Optimizer | Purpose |
|---|---|---|---|
| A | Standard MHA | AdamW | Baseline — everything else compares to this |
| B | MLA | AdamW | Isolates MLA contribution |
| C | Standard MHA | Muon | Isolates Muon contribution |
| D | MLA | Muon | Full system — the pitch model |

All four runs share: data shards, tokenizer, schedule (warmup 2000 / total 20000),
grad clip 1.0, batch size 512 × seq 2048, annealing switch at step 18000.

## Repo structure (created during Phase 0, populated through Phase 5)

```
D:\TinyLM\
├── README.md
├── CLAUDE.md
├── .gitignore
├── pyproject.toml             # or requirements.txt — pinned deps
├── baseline_results.json
├── docs/
│   ├── ablation_plan.md
│   ├── eval_suite.md
│   └── superpowers/{specs,plans}/
├── modded-nanogpt/            # forked / vendored
├── src/tinylm/
│   ├── model.py               # MHAttention, MLAttention, transformer block
│   ├── muon.py                # newton_schulz + Muon optimizer
│   ├── data.py                # FineWeb-Edu loader + annealing mix
│   ├── train.py               # training loop + WandB
│   ├── eval_wrapper.py        # HF PreTrainedModel wrapper for lm-eval
│   └── inference.py           # forward_with_cache + KV measurement
├── tests/
│   ├── test_mla.py            # 3 non-negotiable MLA tests (PDF Phase 1 Step 3)
│   ├── test_muon.py           # Newton-Schulz convergence checks
│   └── test_inference_cache.py
├── configs/
│   ├── run_A_mha_adamw.yaml
│   ├── run_B_mla_adamw.yaml
│   ├── run_C_mha_muon.yaml
│   └── run_D_mla_muon.yaml
├── scripts/
│   ├── eval_baseline.py       # Colab/Kaggle-runnable
│   └── tokenize_shards.py
└── results/                   # per-run lm-eval JSON outputs
```

## Phase 0 execution order

1. `git init` at `D:\TinyLM`, write `.gitignore`.
2. Write `README.md` with hypothesis pinned at top.
3. Write `CLAUDE.md` (tech stack, test/lint/run commands, per global rules).
4. Write `docs/eval_suite.md` and `docs/ablation_plan.md`.
5. Write `scripts/eval_baseline.py` — a notebook-friendly script the user can
   paste into Colab/Kaggle. It pip-installs `lm-eval`, runs the 4 tasks on
   TinyLlama-1.1B, dumps JSON.
6. **User runs the script on Colab/Kaggle**, downloads `baseline_results.json`,
   places it in repo root, commits.
7. Phase 0 complete. Hand to writing-plans for the Phase 1 (architecture +
   unit tests) execution plan.

## Out of scope for Phase 0

- Any model code (MLA, MHA, transformer block) — Phase 1.
- Muon optimizer code — Phase 1.
- Dataloader / tokenizer / shard generation — Phase 3.
- Any training runs — Phase 2 toy run is the first.

## Open issues flagged for later phases (not Phase 0 blockers)

- **modded-nanogpt is Linux-first and uses `torch.compile` heavily.** Phase 1
  unit tests on Windows CPU must use eager mode + `--no-compile`. Real training
  runs on Linux A100, so this only affects local dev ergonomics.
- **`lm-eval` cannot load a raw `nn.Module`.** Per PDF Phase 5 Step 1 callout,
  we must wrap the trained model as a HF `PreTrainedModel` subclass with a
  `config.json` before invoking `lm_eval --model hf`. Plan accordingly in Phase 5.
- **MLA inference path must thread per-layer KV cache through
  `forward_with_cache()`.** PDF Phase 1 Step 2b. Otherwise the 3.5× KV-cache
  claim does not materialize and Phase 5 Step 2's empirical measurement will
  show no reduction.

## Success criteria for Phase 0

- All 5 artifacts exist on disk.
- `baseline_results.json` contains accuracy numbers for TinyLlama-1.1B on
  all 4 benchmarks.
- README hypothesis is pinned and committed (timestamp visible in `git log`).
- No code in `src/tinylm/` yet — that's a Phase 1 gate.
