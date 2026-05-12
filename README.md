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

- [x] Phase 0 — Design lock-in
- [x] Phase 1 — Architecture + unit tests (12/12 green, 274.6M params)
- [ ] Phase 2 — Toy run (1B tokens, go/no-go gate)
- [ ] Phase 3 — Data pipeline
- [ ] Phase 4 — Full training run + 4 ablations
- [ ] Phase 5 — Eval + interview narrative
