# TinyLM v2 — Data & Distillation Phase: Findings Summary

**One-line takeaway:** Starting from the best ablation model (Run D, 275M MLA+Muon), we
tested whether *better data* improves it via continued pretraining. Across three controlled
2.1B-token probes **and** one full 7.3B-token run, **better data is a real, significant lever
for language modeling (perplexity), but does not move commonsense-reasoning benchmarks — even
at Run D's own ~7B-token budget.** The full E3 run drove LAMBADA perplexity to **23.20** (the
project's best, −3.34 vs Run D, ~4σ) while HellaSwag acc_norm ended at **0.4125 — statistically
identical to the Run D starting point (0.4123).** Reasoning at 275M is capacity/scale-bound, not
token-bound; the defensible claim is the **language-modeling gain, not a reasoning gain.**

---

## What we did

All v2 work is **continued pretraining** from the published Run D checkpoint: load its weights,
**reset** the optimizer / data-loader / step counter (`init_from`, not a resume), and train on
new data. Reported separately from the locked A/B/C/D ablation — never folded into it.

Three probes, each **identical** except for the data (same Run D base, same 2.1B tokens, same
LR=0.006, same 2000 steps). This isolates **data composition** as the only variable:

| Probe | Data recipe | Question it answers |
|---|---|---|
| **E1** | 100% fresh FineWeb-Edu (provably disjoint from Run D's 8B) | Does *more of the same* data help? |
| **E2** | 55% FWE / 20% web / 15% code / 10% math | Does a *broader mixture* help? |
| **E3** | 45% FWE / 20% web / 10% code / 10% math / **15% teacher-distilled** (Cosmopedia-v2) | Does *teacher-distilled* data help, esp. reasoning? |

---

## The numbers (all four locked benchmarks, all probes vs Run D base)

| Benchmark | Metric | **Run D** (base) | E1 | E2 | **E3** (best) | E3 − base | Significant? |
|---|---|---:|---:|---:|---:|---:|:--:|
| HellaSwag | acc_norm | 0.4123 | 0.4105 | 0.4077 | 0.4079 | −0.0044 | ❌ flat |
| ARC-Easy | acc_norm | 0.5122 | 0.5114 | 0.5059 | **0.5181** | +0.0059 | ❌ (best of 4, in noise) |
| ARC-Easy | acc | 0.5753 | 0.5800 | 0.5800 | **0.5812** | +0.0059 | ❌ |
| LAMBADA | acc | 0.3681 | 0.3664 | 0.3749 | **0.3860** | **+0.0179** | ✅ ~2.6σ |
| **LAMBADA** | **perplexity** ↓ | 26.54 | 26.89 | 25.08 | **23.95** | **−2.59** | ✅ ~3.1σ |
| Winogrande | acc | 0.5130 | 0.5201 | 0.5146 | **0.5209** | +0.0079 | ❌ |

(Significance uses the per-metric eval stderr; the LAMBADA win holds even under the stricter
two-sample test — perplexity ~2.1σ, acc ~1.9σ.)

---

## The three findings, in plain language

**1. More of the same data does nothing (E1).** A model that already saw 8B FineWeb-Edu tokens
gained *zero* measurable benefit from 2.1B more — every metric within noise, and LAMBADA
perplexity actually got slightly worse. Volume of same-distribution data is not the lever.

**2. Data composition is a real lever — but only for language modeling (E2, E3).** Broadening
the mixture (E2), then adding teacher-distilled text (E3), drove a clean, monotonic improvement
in LAMBADA perplexity: **26.54 → 26.89 → 25.08 → 23.95**. The E3 gain over the base is
statistically significant on both LAMBADA metrics. Each step up in data *quality* bought a real
LM improvement.

**3. Reasoning did not move — and that was the surprise.** The whole point of the E3 distillation
slice was to lift commonsense reasoning (HellaSwag / ARC / Winogrande). It didn't. HellaSwag
acc_norm stayed flat/slightly down across *all three* recipes; ARC and Winogrande nudged to
their best-of-series values but stayed within noise. At a 2.1B continuation budget, reasoning
benchmarks are insensitive to data recipe.

---

## Honest caveats

- **Only LAMBADA clears noise.** The ARC / Winogrande upticks are suggestive, not significant —
  not reported as gains.
- **The locked suite is blind to ~30% of E3's data.** It has no code or math benchmark, so any
  capability the code/math/distill slices built is currently invisible. "Reasoning flat" means
  *commonsense-MCQ* reasoning specifically. (A held-out code/math perplexity diagnostic under
  `results/v2/dev/` would de-blind this; not yet run.)
- **Probes are 2.1B continuations, not from-scratch runs.** They bound what *continued*
  pretraining buys, not what the data would do from scratch.

---

## The full E3 run: the token-count verdict (DONE)

E3's distill-mixture was the unambiguous probe winner, so we scaled it: **7.34B tokens (7000
steps), init_from Run D, same recipe and LR**, evaluated at ~1.8B/3.7B/5.5B/7.3B tokens. It
tested the one question the probes left open. **The answer: only language modeling moves.**

| Metric | Run D (base) | E3 probe (2.1B) | **E3-full (7.3B)** | Δ vs base | Significant? |
|---|---:|---:|---:|---:|:--:|
| LAMBADA perplexity ↓ | 26.54 | 23.95 | **23.20** | **−3.34** | ✅ ~4σ (project best) |
| LAMBADA acc | 0.3681 | 0.3860 | **0.3901** | **+0.0220** | ✅ ~3σ |
| HellaSwag acc_norm | 0.4123 | 0.4079 | **0.4125** | +0.0002 | ❌ flat |
| ARC-Easy acc_norm | 0.5122 | 0.5181 | **0.5080** | −0.0042 | ❌ flat |
| Winogrande | 0.5130 | 0.5209 | **0.5146** | +0.0016 | ❌ flat |

- **LM improves monotonically with tokens** (ppl 25.23 → 24.18 → 23.24 → 23.20), plateauing by
  ~5.5B — clear diminishing returns.
- **Reasoning is flat even at 7.3B.** HellaSwag acc_norm ended statistically identical to where
  Run D began. Reasoning at 275M is **capacity/scale-bound, not token-bound**.
- Per the pre-committed **hard stop**, the defensible claim is the language-modeling gain, not a
  reasoning gain. The pinned hypothesis is never edited to match results.
- Full detail: `results/v2/E3full_vs_runD.md`.

## Per-probe / per-run detail
- `results/v2/E1_vs_runD.md`, `E2_vs_runD.md`, `E3_vs_runD.md`, `E3full_vs_runD.md`
- Raw eval JSONs: `results/v2/run_E{1,2,3}_eval.json`, `run_E3full_step{01749,03499,05249,06999}_eval.json`; base: `results/run_D_eval.json`
