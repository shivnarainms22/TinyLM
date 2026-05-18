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

## Results — Run D (MLA + Muon)

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

## Ablation Table

| Run | Attention | Optimizer | Status |
|---|---|---|---|
| A | Standard MHA | AdamW | Not run |
| B | MLA | AdamW | Not run |
| C | Standard MHA | Muon | Not run |
| D | MLA + Muon | Muon | **Complete** |

---

## Status

- [x] Phase 0 — Design lock-in
- [x] Phase 1 — Architecture + unit tests (30/30 green, 274.6M params)
- [x] Phase 2 — Training stack (tests green)
- [x] Phase 4 — Run D complete (20k steps, final loss 2.22)
- [x] Phase 5 — Eval complete (Run D benchmarked, results saved)
- [ ] Phase 5 — Interview narrative + HF model card
