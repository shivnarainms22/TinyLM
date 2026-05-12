# TinyLM — Phase 0 Execution Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the five Phase 0 lock-in artifacts (hypothesis, baseline eval JSON, ablation table, eval-suite doc, per-project CLAUDE.md) plus a Colab/Kaggle-runnable script that produces `baseline_results.json` for TinyLlama-1.1B on HellaSwag, ARC-Easy, LAMBADA, and Winogrande.

**Architecture:** Documentation-only phase. No `src/tinylm/` code. One Python script (`scripts/eval_baseline.py`) that is wrapper-only over the `lm-eval` CLI — runs externally on a free GPU (Colab/Kaggle), result JSON is committed back. Local validation is limited to: file presence, syntax parsing, and a `--dry-run` smoke test.

**Tech Stack:** Markdown for all docs. Python 3.10+ for the eval script. `lm-eval-harness` (pip-installed inside Colab). No project dependencies installed locally — Phase 0 deliberately keeps the local Windows environment empty.

**Spec reference:** `docs/superpowers/specs/2026-05-12-tinylm-phase0-design.md` (commit `84c515b`).

**Out of scope (defer to later phases):**
- Any `src/tinylm/` Python code (Phase 1)
- Model architecture, Muon, dataloader (Phases 1, 3)
- Training runs (Phases 2, 4)
- `pyproject.toml` / requirements (Phase 1 — when we first import torch)

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `README.md` | Create | Project intro + pinned hypothesis (PDF Phase 0 Step 1 artifact). |
| `CLAUDE.md` | Create | Per-project rules for subagents per global CLAUDE.md requirement. |
| `docs/eval_suite.md` | Create | Locked benchmark choices + exact `lm_eval` invocation. |
| `docs/ablation_plan.md` | Create | The four ablation rows + locked schedule. |
| `scripts/eval_baseline.py` | Create | Colab/Kaggle-runnable wrapper that invokes `lm_eval` and dumps `baseline_results.json`. |
| `baseline_results.json` | Create externally | Produced by user running the script on Colab/Kaggle, then committed locally. |

---

## Chunk 1: Documentation lock-in

### Task 1: Pin hypothesis in README.md

**Files:**
- Create: `D:\TinyLM\README.md`

- [ ] **Step 1: Write README.md**

```markdown
# TinyLM

A 275M parameter small language model trained with Multi-head Latent
Attention (MLA) and the Muon optimizer (Newton-Schulz orthogonalization),
benchmarked against TinyLlama-1.1B via a four-run ablation table.

Reference plan: `250M_SLM_Implementation_Plan_revised.pdf` (repo root).

---

## Pinned Hypothesis

> A 275M parameter model trained with MLA + Muon on 20B tokens of
> FineWeb-Edu will achieve materially-better-than-random performance on
> HellaSwag, ARC-Easy, LAMBADA, and Winogrande, while demonstrating a
> measurable KV-cache memory reduction at inference versus an equivalent
> MHA baseline (Run A). Exact percentage targets are filled in
> post-Phase 5.

**Hypothesis pinned: 2026-05-12.** This is a falsifiable, direction-only
claim per PDF Phase 0 Step 1. Numbers (parity %, KV-reduction %) are
deliberately left open and will be filled in based on actual Phase 5
results, **not** edited to match results post-hoc.

---

## Baseline

- Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Tokenizer: `meta-llama/Llama-2-7b-hf` (vocab 32000)
- Baseline benchmark numbers: see `baseline_results.json`

---

## Ablation Table (locked)

| Run | Attention | Optimizer | Purpose |
|---|---|---|---|
| A | Standard MHA | AdamW | Baseline |
| B | MLA | AdamW | Isolates MLA |
| C | Standard MHA | Muon | Isolates Muon |
| D | MLA + Muon | Muon | Full system — the pitch model |

See `docs/ablation_plan.md` and `docs/eval_suite.md` for the locked
schedule and benchmark suite.

---

## Status

- [x] Phase 0 — Design lock-in (this commit set)
- [ ] Phase 1 — Architecture + unit tests
- [ ] Phase 2 — Toy run (1B tokens, go/no-go gate)
- [ ] Phase 3 — Data pipeline
- [ ] Phase 4 — Full training run + 4 ablations
- [ ] Phase 5 — Eval + interview narrative
```

- [ ] **Step 2: Commit**

```bash
git -C D:/TinyLM add README.md
git -C D:/TinyLM commit -m "docs: pin Phase 0 hypothesis in README"
```

Expected: 1 file changed, ~60 insertions.

---

### Task 2: Per-project CLAUDE.md

**Files:**
- Create: `D:\TinyLM\CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git -C D:/TinyLM add CLAUDE.md
git -C D:/TinyLM commit -m "docs: add per-project CLAUDE.md with locked constants"
```

Expected: 1 file changed.

---

### Task 3: Eval suite lock-in doc

**Files:**
- Create: `D:\TinyLM\docs\eval_suite.md`

- [ ] **Step 1: Write docs/eval_suite.md**

````markdown
# Evaluation Suite (locked 2026-05-12)

Four benchmarks chosen before any training. **No additions after
results are in** — that is cherry-picking and breaks the experiment.

| Benchmark | Task name (lm-eval) | What it measures |
|---|---|---|
| HellaSwag | `hellaswag` | Commonsense reasoning (4-way multiple choice) |
| ARC-Easy | `arc_easy` | Factual QA |
| LAMBADA (OpenAI split) | `lambada_openai` | Long-range LM coherence (last-word prediction) |
| Winogrande | `winogrande` | Coreference resolution |

## Invocation

The exact same command must be used for the baseline and all four
ablation checkpoints. The only thing that changes is the `pretrained=`
arg.

```bash
lm_eval --model hf \
  --model_args pretrained=<MODEL_PATH_OR_HF_ID> \
  --tasks hellaswag,arc_easy,lambada_openai,winogrande \
  --device cuda --batch_size 16 \
  --output_path ./results/<run_name>_eval.json
```

## Reported metric per benchmark

| Benchmark | Metric (per lm-eval default) |
|---|---|
| HellaSwag | `acc_norm` (length-normalized accuracy) |
| ARC-Easy | `acc_norm` |
| LAMBADA OpenAI | `acc` (exact-match) and `perplexity` |
| Winogrande | `acc` |

When comparing runs, use the same metric column. For
"average benchmark score" claims, average the headline accuracy
(`acc_norm` for HS/ARC-E, `acc` for LAMBADA/Winogrande).

## Baseline results

Produced by `scripts/eval_baseline.py` on Colab/Kaggle. Output lives
at `baseline_results.json` in the repo root.
````

- [ ] **Step 2: Commit**

```bash
git -C D:/TinyLM add docs/eval_suite.md
git -C D:/TinyLM commit -m "docs: lock benchmark suite (4 tasks, no additions later)"
```

---

### Task 4: Ablation plan doc

**Files:**
- Create: `D:\TinyLM\docs\ablation_plan.md`

- [ ] **Step 1: Write docs/ablation_plan.md**

```markdown
# Ablation Plan (locked 2026-05-12)

Four runs, one variable changes per row. Without rows A–C, row D
proves nothing — it is just a demo, not an experiment.

## Rows

| Run | Attention | Optimizer (matrix) | Optimizer (scalar/embed) | Purpose | Est. cost |
|---|---|---|---|---|---|
| A | Standard MHA | AdamW | AdamW | Baseline | ~$45 |
| B | MLA | AdamW | AdamW | Isolates MLA contribution vs. MHA | ~$48 |
| C | Standard MHA | Muon | AdamW | Isolates Muon contribution vs. AdamW | ~$45 |
| D | MLA | Muon | AdamW | Full system — the pitch model | ~$48 |

## Shared invariants (do NOT vary between runs)

- Total tokens: 20B
- Steps: 20000 (warmup 2000)
- Batch size: 512 sequences × 2048 seq_len = ~1M tokens/step
- Grad clip: 1.0
- LR schedule: cosine with linear warmup
- Annealing switch: step 18000 (pure FineWeb-Edu → 50/25/25 mix
  with NuminaMath-CoT + OpenHermes-2.5)
- Tokenizer: `meta-llama/Llama-2-7b-hf` (vocab 32000)
- Data shards: identical for all runs
- Model dims: `n_layers=18, d_model=1024, n_heads=16, d_latent=512,
  d_rope=64, ffn_hidden=2816, ctx=2048, tie_weights=True`

## Per-row-only differences

| Setting | A | B | C | D |
|---|---|---|---|---|
| Attention class | MHA | MLA | MHA | MLA |
| Matrix optimizer | AdamW | AdamW | Muon | Muon |
| lr_max (matrix) | 0.001 | 0.001 | 0.02 | 0.02 |

Scalar/embed/LM-head/LN params always use AdamW (`lr=0.001`,
`wd=0.1`). This is non-negotiable per PDF — Muon on the vocab
embedding destroys learned token geometry.

## Phase 2 gate (go/no-go for Phase 4)

The Phase 2 toy run uses Run D config on 1B tokens (~3hr, ~$5) to
validate:
1. tokens/sec ≥ 80,000 on A100
2. Loss at step 1000 clearly decreasing and below 5.0
3. Zero NaN/Inf events
4. Checkpoint resume works (kill at step 2000, resume, verify loss
   continues from same value)

Only after all four green do we commit to Phase 4 spend.
```

- [ ] **Step 2: Commit**

```bash
git -C D:/TinyLM add docs/ablation_plan.md
git -C D:/TinyLM commit -m "docs: lock 4-row ablation plan and shared invariants"
```

---

## Chunk 2: Baseline eval script

### Task 5: Create scripts/eval_baseline.py

**Files:**
- Create: `D:\TinyLM\scripts\eval_baseline.py`

This script is *not* run locally on the Windows CPU box. It is designed
to be pasted into a Colab or Kaggle notebook cell, or run via
`!python scripts/eval_baseline.py` from such a notebook. Local validation
is limited to syntax parsing.

- [ ] **Step 1: Write scripts/eval_baseline.py**

```python
"""Baseline eval runner for TinyLM Phase 0.

Runs `lm-eval-harness` on TinyLlama-1.1B against the locked four-task
suite (HellaSwag, ARC-Easy, LAMBADA OpenAI, Winogrande) and writes
results to baseline_results.json at the repo root.

USAGE (Colab/Kaggle notebook, GPU runtime selected):

    !pip install -q lm-eval==0.4.* transformers accelerate
    !python scripts/eval_baseline.py

USAGE (dry-run, validates the command without invoking lm-eval):

    python scripts/eval_baseline.py --dry-run

The script is intentionally a thin wrapper. All eval logic lives in
`lm-eval` itself; this file just pins the exact command so the same
invocation is reproducible in Phase 5 for the four ablation
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASELINE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
TASKS = ["hellaswag", "arc_easy", "lambada_openai", "winogrande"]
OUTPUT_PATH = "baseline_results.json"


def build_command(output_dir: Path, batch_size: int) -> list[str]:
    """Return the exact argv that will be invoked.

    Phase 5 uses an identical command shape against each ablation
    checkpoint, with only `pretrained=` swapped.
    """
    return [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={BASELINE_MODEL}",
        "--tasks", ",".join(TASKS),
        "--device", "cuda",
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
    ]


def find_results_json(output_dir: Path) -> Path:
    """lm-eval writes a nested results JSON; find the canonical one."""
    candidates = sorted(output_dir.rglob("results_*.json"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results JSON produced in {output_dir}. "
            "Check the lm-eval stderr above."
        )
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="lm-eval batch size (16 fits a free Colab T4; bump on A100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline"),
        help="lm-eval working dir (results JSON written here).",
    )
    parser.add_argument(
        "--final-path",
        type=Path,
        default=Path(OUTPUT_PATH),
        help="Destination for the canonical baseline_results.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without invoking lm-eval.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args.output_dir, args.batch_size)

    print("Command:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        print("[dry-run] Skipping lm-eval invocation.", flush=True)
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"lm-eval exited {result.returncode}", file=sys.stderr)
        return result.returncode

    src = find_results_json(args.output_dir)
    payload = json.loads(src.read_text())
    args.final_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote canonical baseline JSON to {args.final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Validate the script parses (syntax check)**

Run: `python -c "import ast; ast.parse(open(r'D:/TinyLM/scripts/eval_baseline.py').read()); print('parse ok')"`

Expected output: `parse ok`

- [ ] **Step 3: Validate the dry-run path works locally (no GPU, no lm-eval needed)**

Run: `python D:/TinyLM/scripts/eval_baseline.py --dry-run`

Expected: prints a `Command:` line containing
`lm_eval --model hf --model_args pretrained=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --tasks hellaswag,arc_easy,lambada_openai,winogrande --device cuda --batch_size 16 --output_path results/baseline`
followed by `[dry-run] Skipping lm-eval invocation.` Exit code 0.

If this fails on Windows due to `pathlib.Path` handling, fix and re-run before committing.

- [ ] **Step 4: Commit**

```bash
git -C D:/TinyLM add scripts/eval_baseline.py
git -C D:/TinyLM commit -m "feat: add Colab/Kaggle baseline eval script for TinyLlama-1.1B"
```

---

## Chunk 3: External execution + close-out

### Task 6: Run baseline eval on Colab/Kaggle (USER ACTION)

**This task is performed by the user, not by an agent.** Phase 0 closes out
when `baseline_results.json` is committed.

- [ ] **Step 1: Open a fresh Colab or Kaggle notebook with GPU runtime**

Colab: Runtime → Change runtime type → T4 GPU (free).
Kaggle: New Notebook → Settings → Accelerator → GPU T4 x2.

- [ ] **Step 2: Upload `scripts/eval_baseline.py` or clone the repo**

Either drag the file into the notebook file pane, or:

```python
!git clone <your-repo-url> tinylm && cd tinylm
```

- [ ] **Step 3: Install lm-eval**

```python
!pip install -q lm-eval==0.4.* transformers accelerate
```

- [ ] **Step 4: Run the script**

```python
!python scripts/eval_baseline.py --batch-size 16
```

Expected: lm-eval downloads TinyLlama-1.1B (~2.2 GB), then runs four
tasks sequentially. Total wall time ~30–60 min on a free T4. Output
ends with `Wrote canonical baseline JSON to baseline_results.json`.

- [ ] **Step 5: Download `baseline_results.json` from the notebook**

In Colab: right-click the file in the Files pane → Download.
In Kaggle: it appears under "Output" once the notebook completes.

- [ ] **Step 6: Place file in repo root and commit**

Copy `baseline_results.json` to `D:\TinyLM\baseline_results.json`, then:

```bash
git -C D:/TinyLM add baseline_results.json
git -C D:/TinyLM commit -m "data: add TinyLlama-1.1B baseline eval results"
```

---

### Task 7: Phase 0 close-out

- [ ] **Step 1: Verify all 5 artifacts exist and are committed**

Run:
```bash
git -C D:/TinyLM ls-files | findstr /R "README.md CLAUDE.md docs/eval_suite.md docs/ablation_plan.md baseline_results.json scripts/eval_baseline.py"
```

Expected: all 6 paths listed (5 artifacts + the eval script).

- [ ] **Step 2: Tag the Phase 0 boundary**

```bash
git -C D:/TinyLM tag -a phase0-complete -m "Phase 0 design lock-in complete"
```

- [ ] **Step 3: Update README status checklist**

Edit `README.md`, replace `- [x] Phase 0` line is already checked; no
change needed unless any artifact slipped.

- [ ] **Step 4: Hand off to next brainstorming session for Phase 1**

Phase 1 (architecture + MLA/Muon unit tests) is its own brainstorm →
plan → execute cycle per global CLAUDE.md. Start a fresh conversation
with: *"Begin Phase 1 of TinyLM — architecture + unit tests per
docs/superpowers/specs/2026-05-12-tinylm-phase0-design.md."*

---

## Verification checklist (before declaring Phase 0 done)

- [ ] `README.md` exists, hypothesis pinned verbatim, `git log` shows commit date 2026-05-12.
- [ ] `CLAUDE.md` exists with locked constants section.
- [ ] `docs/eval_suite.md` lists exactly 4 tasks; no more.
- [ ] `docs/ablation_plan.md` lists exactly 4 rows A/B/C/D with shared invariants section.
- [ ] `scripts/eval_baseline.py` passes `--dry-run` on Windows with no external deps.
- [ ] `baseline_results.json` contains accuracy numbers for all 4 tasks against TinyLlama-1.1B.
- [ ] `git tag phase0-complete` exists.
- [ ] **No file under `src/tinylm/` exists.** Phase 1 gate.

## Notes for the executing agent

- All file paths are Windows-style with backslashes for `Write`, but
  git/bash commands use forward slashes. This is intentional — Windows
  git accepts both.
- The eval script's `--dry-run` is the only local validation possible
  without a GPU and without installing lm-eval. Do not try to invoke
  `lm_eval` from the Windows host.
- Do not add `pyproject.toml` or `requirements.txt` in Phase 0. The
  first dependency lands in Phase 1 when we import torch.
- Commit after each task. Do not batch commits across tasks — frequent
  commits are a hard rule per global CLAUDE.md.
