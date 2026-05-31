# TinyLM

A 275M parameter small language model trained with Multi-head Latent
Attention (MLA) and the Muon optimizer (Newton-Schulz orthogonalization),
benchmarked against TinyLlama-1.1B.

**Model on HuggingFace:** [Shiv-22/tinylm](https://huggingface.co/Shiv-22/tinylm)

Reference plan: `250M_SLM_Implementation_Plan_revised.pdf` (repo root).

---

## Pinned Hypothesis

> A 275M parameter model trained with MLA + Muon on 1B tokens of
> FineWeb-Edu (20k steps, data repeated) will achieve materially-better-than-random performance on
> HellaSwag, ARC-Easy, LAMBADA, and Winogrande, while demonstrating a
> measurable KV-cache memory reduction at inference versus an equivalent
> MHA baseline (Run A). Exact percentage targets are filled in
> post-Phase 5.

**Hypothesis pinned: 2026-05-12.** This is a falsifiable, direction-only
claim per PDF Phase 0 Step 1. Numbers (parity %, KV-reduction %) are
deliberately left open and will be filled in based on actual Phase 5
results, **not** edited to match results post-hoc.

---

## HPC Re-run — Full Ablation (8B unique tokens, 23k steps)

Following v1, the full 4-arm ablation was re-run on Northeastern Explorer
(A100-40GB) with the v1 data bug fixed: **8B unique** FineWeb-Edu tokens
(~3 epochs, ~24B processed) replacing v1's 1B tokens repeated ~21×.

### Headline finding

**Training data quality dominates architecture choices at this scale.** The
data fix alone (v1 buggy Run D → new Run D, same MLA+Muon arm) is worth
**+3.97 pts** average — roughly **2.6× larger** than the full
architecture-and-optimizer ablation gain of **+1.52 pts** (control A → full
system D).

### Full 2×2 ablation (avg of headline metrics)

|  | AdamW | Muon | Δ (Muon − AdamW) |
|:---|:---:|:---:|:---:|
| **MHA** | A 43.62 | C 44.64 | **+1.02** |
| **MLA** | B 44.11 | D 45.14 | **+1.03** |
| **Δ (MLA − MHA)** | **+0.49** | **+0.50** | — |

**Effects are consistent and additive.** Muon contributes ~+1.0 pt regardless
of attention type; MLA contributes ~+0.5 pt regardless of optimizer; the two
add (+1.52 observed vs +1.51 sum). Single-seed eval, so any interaction below
the ~1% noise floor is not detectable.

### Per-benchmark (metric-matched to baseline)

| Benchmark | A (MHA+AdamW) | B (MLA+AdamW) | C (MHA+Muon) | **D (MLA+Muon)** | TinyLlama-1.1B |
|:---|:---:|:---:|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 38.73% | 39.46% | 40.55% | **41.23%** | 59.1% |
| ARC-Easy (acc_norm) | 51.85% | 50.08% | 51.05% | **51.22%** | 55.7% |
| LAMBADA (acc) | 34.17% | 34.33% | 35.47% | **36.81%** | 58.9% |
| Winogrande (acc) | 49.72% | 52.57% | 51.46% | **51.30%** | 58.9% |
| **Average** | 43.62% | 44.11% | 44.64% | **45.14%** | 58.2% |

Average ordering is monotonic **A < B < C < D**. HellaSwag and LAMBADA give
clean monotonic per-arm ordering with the largest signal-to-noise; ARC-Easy
and Winogrande are within stderr of each other across all four arms (noise).

### Data-fix detail (v1 buggy D → new D, same MLA+Muon arm)

| Benchmark | v1 D (1B×21) | new D (8B unique) | Δ |
|:---|:---:|:---:|:---:|
| HellaSwag (acc_norm) | 37.1% | 41.23% | +4.13 |
| ARC-Easy (acc_norm) | 48.4% | 51.22% | +2.82 |
| LAMBADA (acc) | 29.2% | 36.81% | **+7.61** |
| Winogrande (acc) | 50.0% | 51.30% | +1.30 |
| **Average** | 41.18% | 45.14% | **+3.97** |

LAMBADA — the most data-hungry, long-range task — gained the most (+7.61)
once data stopped looping. ARC-Easy and Winogrande, less sensitive to
training-data diversity, moved less.

Full breakdown: `results/hpc_rerun_ablation.md`  
HPC re-run checkpoints: `Shiv-22/tinylm-checkpoints-v2` on HuggingFace *(per-arm
folders: `run_A/`, `run_B/`, `run_C/`, `run_D/`)*. Run D (the pitch model)
also published at `Shiv-22/tinylm` with model card.

---

## Results — v1 Run D (MLA + Muon, 1B×21 tokens, historical)

> Preserved for contrast vs the HPC re-run above. The v1 effort trained on
> 1B unique tokens repeated ~21× on RunPod A100-80GB; the HPC re-run uses
> 8B unique tokens and adds the full 4-arm ablation.

| Benchmark | Run D (275M) | TinyLlama-1.1B | Delta |
|:---|:---:|:---:|:---:|
| HellaSwag | 32.4% | 59.1% | -26.7% |
| ARC-Easy | **53.8%** | 55.7% | **-1.9%** |
| LAMBADA | 29.2% | 58.9% | -29.7% |
| Winogrande | 50.0% | 58.9% | -8.9% |
| **Average** | **41.3%** | **58.2%** | **-16.9%** |

Full results: `results/run_D_eval.json` · `results/baseline_comparison.md`  
Checkpoint: `Shiv-22/tinylm-checkpoints` on HuggingFace (step_19999.pt)  
Training run: [WandB](https://wandb.ai/shivnarainms22-northeastern-university/tinylm/runs/dig7xsqf)

---

## Baseline

- Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Tokenizer: `meta-llama/Llama-2-7b-hf` (vocab 32000)
- Baseline benchmark numbers: see `baseline_results.json`

---

## Ablation Table (HPC re-run, all complete)

| Run | Attention | Optimizer | Status | Avg (headline) | Δ vs A |
|---|---|---|---|:---:|:---:|
| A | Standard MHA | AdamW | ✅ Complete | 43.62% | — |
| B | MLA | AdamW | ✅ Complete | 44.11% | +0.49 |
| C | Standard MHA | Muon | ✅ Complete | 44.64% | +1.02 |
| D | MLA | Muon | ✅ Complete (pitch) | **45.14%** | **+1.52** |

(v1 trained Run D only — 1B×21 tokens — and is preserved as the historical
contrast above.)

---

## Status

- [x] Phase 0 — Design lock-in
- [x] Phase 1 — Architecture + unit tests (30/30 green, 274.6M params)
- [x] Phase 2 — Training stack (tests green)
- [x] v1 — Run D complete (20k steps, 1B×21 tokens, final loss 2.22, RunPod A100)
- [x] v1 — Eval complete (Run D benchmarked, results in `results/baseline_comparison.md`)
- [x] HPC Re-run — All 4 arms trained (23k steps each, 8B unique tokens, Explorer A100-40GB)
- [x] HPC Re-run — All 4 arms evaled (full 2×2 ablation in `results/hpc_rerun_ablation.md`)
- [x] HF model cards published — main, ablation checkpoints, v1 historical
