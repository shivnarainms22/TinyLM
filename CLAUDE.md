# TinyLM — Per-Project Rules

This file is loaded into every Claude Code conversation in this repo.
Global rules in `~/.claude/CLAUDE.md` still apply; this file adds
project-specific context.

## Project

275M parameter SLM training portfolio. Reference plan:
`250M_SLM_Implementation_Plan_revised.pdf`. Current phase tracked in
`README.md`.

## Tech Stack

- Python 3.10+
- PyTorch 2.x (installed in Phase 1)
- HuggingFace `datasets`, `transformers`, `lm-eval` (installed as
  needed per phase)
- WandB for training logs (Phase 2+)
- Training: rented A100-80GB via Vast.ai
- Local dev: Windows 11 CPU only (Phase 0–1 work)

## Locked design constants

Do not change without updating the spec doc first.

- Baseline: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Tokenizer: `meta-llama/Llama-2-7b-hf`
- `vocab_size = 32000`
- Benchmark suite: HellaSwag, ARC-Easy, LAMBADA, Winogrande (no
  additions after Phase 0)
- Ablation rows: A (MHA+AdamW), B (MLA+AdamW), C (MHA+Muon),
  D (MLA+Muon)
- Codebase base: `https://github.com/KellerJordan/modded-nanogpt`

## How to run things

Phase 0 (current):
- All artifacts are docs or wrapper scripts. No `pip install`
  required locally.
- `scripts/eval_baseline.py` runs on Colab/Kaggle, not on this
  Windows machine.

Phase 1+ commands will be added here as the tooling lands.

## Conventions

- Specs live in `docs/superpowers/specs/`
- Plans live in `docs/superpowers/plans/`
- Per-run configs in `configs/run_{A,B,C,D}_*.yaml`
- Tests in `tests/` mirror the `src/tinylm/` tree
- Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`,
  `test:`, `chore:`).

## Non-negotiables

- No code is written before its failing test (TDD per global rules).
- The four MLA unit tests in PDF Phase 1 Step 3 are mandatory before
  any training run.
- Checkpoint resume is tested in Phase 2 before any Phase 4 run.
- The hypothesis in README.md is not edited to match results.
