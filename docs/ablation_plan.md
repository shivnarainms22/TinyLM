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

- Total tokens: 1B unique (20k steps, data loops ~21×)
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
