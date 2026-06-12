# E2 (broader web/code/math mixture continuation) vs Run D

**Probe:** continued pretraining from Run D weights (`init_from`, optimizer/loader/step
reset) on **2.1B tokens** of a broader curated mixture.
- Mixture (achieved token proportions, from shard manifest): **55.0% FineWeb-Edu /
  20.0% general web / 15.0% code / 10.0% math** — `fineweb-edu` (sample-100BT, skipping
  Run D's exact 8B prefix → provably disjoint), `fineweb` (web), `codeparrot/codeparrot-clean`
  (code), `HuggingFaceTB/finemath` (math). Document-interleaved so every shard is a
  representative mix (no mid-run distribution shift).
- Schedule: 2000 steps, ~1.05M tokens/step, `lr_muon = 0.006` (same conservative
  continuation LR as E1). Final train loss ~2.1–2.4, no NaN/Inf, clean run to step 1999.
- Architecture: identical 275M MLA+Muon Run D model.

This is the direct A/B partner to **E1** (same base weights, same 2.1B budget, same LR,
same step count) — **the only variable is data composition.**

## Locked-suite results (Δ vs per-metric eval stderr, same convention as E1)

| Benchmark | Metric | Run D | E2 | Δ | ±1σ | Past noise? |
|-----------|--------|-------|-----|------|-----|:---:|
| HellaSwag | acc_norm | 0.4123 | 0.4077 | −0.0046 | ±0.0049 | ❌ (down) |
| HellaSwag | acc | 0.3400 | 0.3405 | +0.0005 | ±0.0047 | ❌ |
| ARC-Easy | acc | 0.5753 | 0.5800 | +0.0047 | ±0.0101 | ❌ |
| ARC-Easy | acc_norm | 0.5122 | 0.5059 | −0.0063 | ±0.0103 | ❌ (down) |
| LAMBADA | acc | 0.3681 | 0.3749 | +0.0068 | ±0.0067 | ⚠️ ~1.0σ |
| **LAMBADA** | **perplexity** | **26.54** | **25.08** | **−1.46 (better)** | ±0.86 | ✅ ~1.7σ |
| Winogrande | acc | 0.5130 | 0.5146 | +0.0016 | ±0.0140 | ❌ |

## The clean test: E1 vs E2 head-to-head (only data composition differs)

| Metric | E1 (pure FineWeb-Edu) | E2 (mixture) | E2 − E1 |
|--------|----------------------:|-------------:|--------:|
| LAMBADA perplexity | 26.89 (worse than D) | **25.08 (better than D)** | **−1.81** |
| LAMBADA acc | 0.3664 | 0.3749 | +0.0085 |
| HellaSwag acc_norm | 0.4105 | 0.4077 | −0.0028 |
| Winogrande acc | 0.5201 | 0.5146 | −0.0055 |
| ARC-Easy acc | 0.5800 | 0.5800 | 0.0000 |

## Verdict: composition is a real but narrow lever — a language-modeling gain, not (yet) a reasoning gain

1. **Composition moves LAMBADA where same-distribution volume could not.** E1 nudged
   LAMBADA perplexity the *wrong* way (26.54 → 26.89); E2 moves it the *right* way
   (26.54 → 25.08). Identical base, budget, and LR — the only difference is the data.
   The E1→E2 perplexity swing (−1.81) is the strongest, cleanest signal of the v2 phase
   so far.
2. **The lift is confined to language modeling (LAMBADA), not commonsense reasoning.**
   HellaSwag / ARC / Winogrande are flat; HellaSwag acc_norm and Winogrande are even
   marginally *below* E1 (within noise) — consistent with diluting FineWeb-Edu from 100%
   to 55%. The mixture makes a slightly better language model, not a better reasoner.
3. **The "probe budget is too small" hypothesis (left open by E1) is refuted.** 2.1B was
   enough to resolve a composition effect E1 could not produce. The probe methodology works.

## Honest caveats (do not over-read this result)

- **Significance is carried by the contrast, not the single delta.** Against a rigorous
  two-sample stderr (combining Run D's and E2's), the E2-vs-D LAMBADA-ppl gain is only
  ~1.1–1.2σ — suggestive, not conclusive. It is the **E1-vs-E2** directional flip that
  makes the composition effect credible, because it cancels shared base-model variance.
- **The locked suite is blind to 25% of E2's data.** Code (15%) and math (10%) made up a
  quarter of the budget, but HellaSwag / ARC-Easy / LAMBADA / Winogrande contain **no code
  and no math task.** E2's true benefit on those capabilities is **unmeasured** — "reasoning
  flat" means *commonsense-MCQ* reasoning only. A held-out perplexity diagnostic on
  non-FineWeb text (code / math / general web) would expose any hidden gain. The locked
  headline suite is non-negotiable and stays as-is; any such probe lives under
  `results/v2/dev/` and is reported as a diagnostic, never folded into the locked suite.

## Decision → next step: E3 (distillation), not a full E2 scale-up

Data composition alone improved LM quality but left commonsense reasoning flat, and reasoning
(HellaSwag / ARC) is precisely the gap to the TinyLlama-1.1B baseline. A full mixed-data run
would most likely compound the LAMBADA gain while reasoning stays flat — lower expected value.
**E3** adds teacher-distilled explanation/QA text, which targets the reasoning capability that
data-composition alone did not move. Run E3 as the next 2.1B probe before any full-scale spend.

## Artifacts
- `results/v2/run_E2_eval.json` — full lm-eval output (on Explorer scratch).
- E1 comparison: `results/v2/E1_vs_runD.md`.
- Run D baseline: `results/run_D_eval.json`.
- Mixture manifest: `/scratch/$USER/tinylm/data_v2/mixed_web_code_math/manifest.json`.
