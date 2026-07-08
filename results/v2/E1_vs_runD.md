# E1 (fresh FineWeb-Edu continuation) vs Run D

**Probe:** continued pretraining from Run D weights (`init_from`, optimizer/loader/step
reset) on **2.1B fresh, provably non-overlapping** FineWeb-Edu tokens.
- Data: `sample-100BT`, skipped Run D's 8B prefix exactly (`tokens_skipped = 8,000,001,108`
  — the straddling document was dropped, so zero token overlap with Run D's training set).
- Schedule: 2000 steps, ~1.05M tokens/step, `lr_muon = 0.006` (conservative continuation LR).
- Architecture: identical 275M MLA+Muon Run D model.

## Locked-suite results (headline metrics)

| Benchmark | Metric | Run D | E1 | Δ | ±1σ | Beats noise? |
|-----------|--------|-------|-----|------|-----|:---:|
| HellaSwag | acc_norm | 0.4123 | 0.4105 | −0.0018 | ±0.0049 | ❌ |
| HellaSwag | acc | 0.3400 | 0.3402 | +0.0002 | ±0.0047 | ❌ |
| ARC-Easy | acc | 0.5753 | 0.5800 | +0.0046 | ±0.0101 | ❌ |
| ARC-Easy | acc_norm | 0.5122 | 0.5114 | −0.0009 | ±0.0103 | ❌ |
| LAMBADA | acc | 0.3681 | 0.3664 | −0.0018 | ±0.0067 | ❌ |
| LAMBADA | perplexity | 26.54 | 26.89 | +0.34 (worse) | ±0.94 | ❌ |
| Winogrande | acc | 0.5130 | 0.5201 | +0.0071 | ±0.0140 | ❌ |

## Verdict: no measurable change (true no-op)

Every delta is within 1 standard error. The two most sensitive signals settle it:
- **HellaSwag `acc_norm`** (the headline metric) moved slightly *down*.
- **LAMBADA perplexity** — a continuous metric that would expose hidden progress —
  rose 26.54 → 26.89 (worse, within noise).

So this is **not** a "quietly improving, just below the accuracy threshold" case. There is
no signal of improvement, and no regression either: the `init_from` continued-pretraining
mechanics and the conservative LR worked exactly as intended (no catastrophic forgetting).

## What this establishes

- **At this probe budget, volume of same-distribution data is not the lever.** A model that
  already saw 8B FineWeb-Edu tokens gains nothing measurable from 2.1B more. The provably
  disjoint control makes this a clean negative result, not a confound.
- **It does not yet distinguish** "same-data specifically doesn't help" from "2.1B at low LR
  is too small to move any benchmark." **E2 (same budget, broader web/code/math mixture)** is
  the diagnostic that separates them:
  - E2 moves → data *composition* is the lever → continue to E3 / full run.
  - E2 also flat → the probe budget itself is too small → the decision becomes "scale up the
    continuation," not "try more recipes."

## Artifacts
- `results/v2/run_E1_eval.json` — full lm-eval output.
- Run D baseline: `results/run_D_eval.json`.
