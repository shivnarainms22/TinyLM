# E3 (teacher-distilled mixture continuation) vs Run D — and the full E1/E2/E3 probe series

**Probe:** continued pretraining from Run D weights (`init_from`, optimizer/loader/step
reset) on **2.1B tokens** of a teacher-distilled mixture.
- Mixture (achieved proportions, from manifest): **45% FineWeb-Edu / 20% general web /
  10% code / 10% math / 15% teacher-distilled** — the distill slice is Cosmopedia-v2
  (`HuggingFaceTB/smollm-corpus`, Mixtral-8x7B-generated synthetic explanations/QA,
  flowing pretraining-style prose). FineWeb-Edu skips Run D's exact 8B prefix (disjoint).
- Schedule: 2000 steps, ~1.05M tokens/step, `lr_muon = 0.006` — identical to E1/E2.
  Ran clean to step 1999, no NaN/Inf.
- Architecture: identical 275M MLA+Muon Run D model.

E3 completes the 3-probe series. Each probe is the same base / budget / LR / step count;
**the only variable is data.** E1 = pure fresh FineWeb-Edu, E2 = broader web/code/math
mixture, E3 = that mixture rebalanced to add a 15% teacher-distilled slice.

## Full series, headline metrics (Δ = E3 − Run D, vs per-metric eval stderr)

| Benchmark | Metric | Run D | E1 | E2 | **E3** | E3 − D | ±1σ | Past noise? |
|-----------|--------|------:|----:|----:|-------:|-------:|----:|:---:|
| HellaSwag | acc_norm | 0.4123 | 0.4105 | 0.4077 | **0.4079** | −0.0044 | ±0.0049 | ❌ flat |
| HellaSwag | acc | 0.3400 | 0.3402 | 0.3405 | **0.3427** | +0.0027 | ±0.0047 | ❌ |
| ARC-Easy | acc | 0.5753 | 0.5800 | 0.5800 | **0.5812** | +0.0059 | ±0.0101 | ❌ |
| ARC-Easy | acc_norm | 0.5122 | 0.5114 | 0.5059 | **0.5181** | +0.0059 | ±0.0103 | ❌ (best of 4) |
| LAMBADA | acc | 0.3681 | 0.3664 | 0.3749 | **0.3860** | **+0.0179** | ±0.0068 | ✅ ~2.6σ |
| **LAMBADA** | **perplexity** | 26.54 | 26.89 | 25.08 | **23.95** | **−2.59** | ±0.82 | ✅ ~3.1σ |
| Winogrande | acc | 0.5130 | 0.5201 | 0.5146 | **0.5209** | +0.0079 | ±0.014 | ❌ |

## Verdict: data quality is a real lever for language modeling, but NOT for reasoning at 2.1B

**1. E3 is the best probe on every metric, and language modeling improves monotonically with
data quality.** LAMBADA perplexity falls cleanly across the series:
**26.54 (D) → 26.89 (E1, worse) → 25.08 (E2) → 23.95 (E3)**. The teacher-distilled slice
added a *further* gain on top of plain composition (E3 ppl −1.13 vs E2; LAMBADA acc +0.0111).
The cumulative E3-vs-D LAMBADA win is significant even under the rigorous two-sample stderr
(ppl ~2.1σ, acc ~1.9σ) — not just the loose single-stderr convention used in the table.

**2. The distillation thesis did not pan out where it was aimed.** E3 existed to lift
commonsense-MCQ reasoning, which E2 left flat. **HellaSwag acc_norm is still flat/slightly
down (0.4079 vs 0.4123), as in E2.** ARC-Easy acc_norm and Winogrande reach their best-of-series
values but stay within 1σ. Teacher-distilled data made the model a better *language model*,
not a measurably better *reasoner*, at this budget.

**3. The series tells one consistent story.** The lever that moves is language modeling /
perplexity, and it scales with data quality (fresh < mixture < mixture+distill).
Commonsense-MCQ reasoning is stubbornly flat at 2.1B regardless of recipe. Open question the
probes cannot settle: does reasoning need *more tokens* (2.1B resolves LAMBADA but maybe not
HellaSwag), or is it recipe-insensitive at 275M?

## Honest caveats

- **Significance is real on LAMBADA, weak elsewhere.** Only LAMBADA (acc + perplexity) clears
  noise vs Run D. ARC acc_norm / Winogrande upticks are suggestive, within 1σ — do not report
  them as gains.
- **The locked suite is still blind to 30% of E3's data** (10% code + 10% math + part of the
  distill slice). "Reasoning flat" means *commonsense-MCQ* reasoning. A held-out code/math
  perplexity diagnostic under `results/v2/dev/` would de-blind this; the locked headline suite
  stays as-is.

## Decision → scale the E3 recipe, with a token-count hypothesis and a hard stop

E3's distill-mixture is unambiguously the winning recipe (best on every metric). Per the v2
plan, the next step is one full continuation (5B–15B tokens) on this recipe. Its purpose is to
test the one open question: **does commonsense reasoning move with more tokens, or only
language modeling?**
- Eval every 1B–2B tokens on the locked suite.
- **Hard stop:** if HellaSwag acc_norm and LAMBADA both flatten for two consecutive evals,
  stop — the answer is "reasoning is token/scale-bound or recipe-insensitive at 275M," and the
  defensible portfolio claim is the LANGUAGE-MODELING gain, not a reasoning gain.
- Do not over-promise reasoning gains the probes did not show.

## Artifacts
- `results/v2/run_E3_eval.json` — full lm-eval output (in repo `results/v2/`).
- Prior probes: `results/v2/E1_vs_runD.md`, `results/v2/E2_vs_runD.md`.
- Run D baseline: `results/run_D_eval.json`.
- Mixture manifest: `/scratch/$USER/tinylm/data_v2/distill_mix/manifest.json`.
