# HPC Re-run Ablation — A/B/C/D vs TinyLlama-1.1B

Full 4-run ablation on Northeastern Explorer (A100-40GB), fixing the v1 data bug.
This file is filled in as each run's eval lands.

**Setup:** 275M params · 8B **unique** FineWeb-Edu tokens (~3 epochs, ~24B
processed) · 23,000 steps (warmup 2,000) · effective batch 512 × 2048 ·
pure FineWeb-Edu (no annealing mix) · final ckpt `step_22999.pt`.
**Eval:** `scripts/eval_tinylm.py`, lm-eval, 0-shot, on the 4 locked benchmarks.

> Supersedes the 1B-token / annealing / cost assumptions in
> `docs/ablation_plan.md` (those were the v1 RunPod-era plan). The ablation
> *structure* (one variable per row) is unchanged.

## Ablation arms

| Run | Attention | Matrix optimizer | Purpose |
|---|---|---|---|
| A | MHA | AdamW | Baseline (no MLA, no Muon) |
| B | MLA | AdamW | Isolates MLA vs MHA |
| C | MHA | Muon | Isolates Muon vs AdamW |
| D | MLA | Muon | Full system — the pitch model |

Scalar/embed/LM-head/LN params always use AdamW (`lr=0.001`, `wd=0.1`).

## Results — headline metric per task

HellaSwag & ARC-Easy reported as **acc_norm** (field convention; length-bias
removed), LAMBADA & Winogrande as **acc** (no norm variant). All numbers are
metric-matched to the baseline.

| Benchmark (metric) | A (MHA+AdamW) | B (MLA+AdamW) | C (MHA+Muon) | D (MLA+Muon) | TinyLlama-1.1B |
|:---|:---:|:---:|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 39.46% | 40.55% | 41.23% | 59.1% |
| ARC-Easy (acc_norm)  | 51.85% | 50.08% | 51.05% | 51.22% | 55.7% |
| LAMBADA (acc)        | 34.17% | 34.33% | 35.47% | 36.81% | 58.9% |
| Winogrande (acc)     | 49.72% | 52.57% | 51.46% | 51.30% | 58.9% |
| **Average**          | **43.62%** | **44.11%** | **44.63%** | **45.14%** | **58.2%** |

(Run D was submitted on Explorer as `run_D_mla_muon_v2` — same ablation arm,
just a relaunch naming artifact.)

### Deltas vs baseline (metric-matched)

| Benchmark | A | B | C | D | Baseline | Δ D |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 39.46% | 40.55% | 41.23% | 59.1% | −17.9 |
| ARC-Easy (acc_norm)  | 51.85% | 50.08% | 51.05% | 51.22% | 55.7% | −4.5 |
| LAMBADA (acc)        | 34.17% | 34.33% | 35.47% | 36.81% | 58.9% | −22.1 |
| Winogrande (acc)     | 49.72% | 52.57% | 51.46% | 51.30% | 58.9% | −7.6 |
| **Average**          | 43.62% | 44.11% | 44.63% | 45.14% | 58.2% | **−13.1** |

Average ordering: **A < B < C < D**, monotonic — each "upgrade" (MLA, Muon,
both) adds something. D is the closest arm to the 1.1B baseline on average.

Raw `acc` for the normed tasks (secondary): A → HellaSwag 32.70%, ARC-Easy
57.91%; B → 33.20%, 55.64%; C → 33.71%, 58.04%; D → 34.00%, 57.53%.

### Full 2×2 ablation (avg, headline metric)

|  | AdamW | Muon | Δ (Muon − AdamW) |
|:---|:---:|:---:|:---:|
| **MHA** | A 43.62 | C 44.63 | **+1.02** |
| **MLA** | B 44.11 | D 45.14 | **+1.03** |
| **Δ (MLA − MHA)** | **+0.49** | **+0.50** | — |

**Effects are remarkably consistent and additive:** Muon contributes ~+1.0 avg
regardless of attention; MLA contributes ~+0.5 avg regardless of optimizer; sum
of individual effects = 1.52, observed A→D = +1.52 — interaction ≈ 0 within
rounding. *Caveat:* single-seed eval with stderrs ~0.5–1.0%, so an interaction
below the noise floor can't be detected — only that there is no *large* one.

### Ablation comparisons (3 of 4 arms in; B still needed)

#### A → C: Muon vs AdamW, MHA fixed (optimizer effect with MHA)

| Benchmark | A (AdamW) | C (Muon) | Δ (C − A) |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 40.55% | **+1.82** (~2.6σ) |
| ARC-Easy (acc_norm)  | 51.85% | 51.05% | −0.80 (tie) |
| LAMBADA (acc)        | 34.17% | 35.47% | +1.30 |
| Winogrande (acc)     | 49.72% | 51.46% | +1.74 (both ≈ chance) |
| **Average**          | 43.62% | 44.63% | **+1.02** |

#### C → D: MLA vs MHA, Muon fixed (MLA effect with Muon)

| Benchmark | C (MHA+Muon) | D (MLA+Muon) | Δ (D − C) |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 40.55% | 41.23% | +0.68 (~1σ) |
| ARC-Easy (acc_norm)  | 51.05% | 51.22% | +0.17 (tie) |
| LAMBADA (acc)        | 35.47% | 36.81% | +1.34 |
| Winogrande (acc)     | 51.46% | 51.30% | −0.16 (tie) |
| **Average**          | 44.63% | 45.14% | **+0.50** |

#### A → D: Full system vs control (headline portfolio comparison)

| Benchmark | A (MHA+AdamW) | D (MLA+Muon) | Δ (D − A) |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 41.23% | **+2.50** (~3.6σ) |
| ARC-Easy (acc_norm)  | 51.85% | 51.22% | −0.63 (tie) |
| LAMBADA (acc)        | 34.17% | 36.81% | +2.64 |
| Winogrande (acc)     | 49.72% | 51.30% | +1.58 (both ≈ chance) |
| **Average**          | 43.62% | 45.14% | **+1.52** |

#### A → B: MLA vs MHA, AdamW fixed (MLA effect with AdamW)

| Benchmark | A (MHA+AdamW) | B (MLA+AdamW) | Δ (B − A) |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 39.46% | +0.73 (~1σ) |
| ARC-Easy (acc_norm)  | 51.85% | 50.08% | −1.77 (within stderr) |
| LAMBADA (acc)        | 34.17% | 34.33% | +0.16 |
| Winogrande (acc)     | 49.72% | 52.57% | +2.85 (both ≈ chance) |
| **Average**          | 43.62% | 44.11% | **+0.49** |

#### B → D: Muon vs AdamW, MLA fixed (optimizer effect with MLA)

| Benchmark | B (MLA+AdamW) | D (MLA+Muon) | Δ (D − B) |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 39.46% | 41.23% | **+1.77** (~2.6σ) |
| ARC-Easy (acc_norm)  | 50.08% | 51.22% | +1.14 |
| LAMBADA (acc)        | 34.33% | 36.81% | +2.48 |
| Winogrande (acc)     | 52.57% | 51.30% | −1.27 (both ≈ chance) |
| **Average**          | 44.11% | 45.14% | **+1.03** |

**Final read on the 2×2:**
- Optimizer effect is consistent across attention: A→C +1.02, B→D +1.03.
- Architecture effect is consistent across optimizer: A→B +0.49, C→D +0.50.
- HellaSwag is the cleanest single-task signal — Muon +1.8 (~2.6σ) on both
  A→C and B→D; MLA +0.7 (~1σ) on both A→B and C→D.
- ARC-Easy and Winogrande are pure noise (means flip between arms, all within
  stderr of each other). LAMBADA shows consistent direction but moderate stderr.
- A < B < C < D ordering on average is monotonic — each upgrade adds something.

### Data fix — new Run D vs v1 buggy Run D (same MLA+Muon arm)

| Benchmark | v1 D (1B×21) | new D (8B unique) | Δ |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 37.1% | 41.23% | +4.13 |
| ARC-Easy (acc_norm)  | 48.4% | 51.22% | +2.82 |
| LAMBADA (acc)        | 29.2% | 36.81% | **+7.61** |
| Winogrande (acc)     | 50.0% | 51.30% | +1.30 |
| **Average**          | 41.18% | 45.14% | **+3.97** |

**The data fix is the largest single delta in the project** — ~2.6× the headline
ablation gain (+3.97 vs +1.52). LAMBADA — the most data-hungry, long-range task —
jumped +7.61 once we stopped looping. Portfolio framing:
*"architecture choices contributed +1.5 pts; fixing the training data
contributed +4 pts."*

## Notes

- **Metric discipline:** always compare the *same* metric on both sides. ARC-Easy's
  58.04% is plain `acc`; the baseline's 55.7% is `acc_norm` — comparing those two
  is invalid and does **not** show Run C beating the baseline.
- **ARC-Easy is the strongest relative arm:** within ~4.7 pts (acc_norm) of a
  4× larger model — competitive, but does not beat it.
- **HellaSwag & LAMBADA are the weak spots** — both reward long-range coherence,
  hardest for a 275M model. **Winogrande 51.46% ≈ chance (50%)** — minimal signal.
- **vs the buggy v1 Run D** (MLA+Muon, 1B tokens ×21, in `baseline_comparison.md`):
  Run C is higher on every metric incl. acc_norm (HellaSwag 40.55 vs 37.1, ARC-Easy
  51.05 vs 48.4, LAMBADA acc 35.47 vs 29.2). Signal that the 8B-unique-token fix
  helps — but confounded (C=MHA vs v1 D=MLA); the clean data-fix comparison is
  v1 D vs the new Run D (both MLA+Muon).
- Real ablation conclusions (MLA vs MHA, Muon vs AdamW) require all four arms.
