# E3-FULL (scaled distill-mixture continuation) vs Run D — the token-count verdict

**Run:** continued pretraining from Run D weights (`init_from`, optimizer/loader/step reset)
on **7.34B tokens** (7000 steps) of the winning E3 teacher-distilled mixture.
- Recipe (unchanged from the E3 probe): **45% FineWeb-Edu / 20% general web / 10% code /
  10% math / 15% teacher-distilled** (Cosmopedia-v2, Mixtral-8x7B synthetic prose).
- Schedule: 7000 steps, ~1.05M tokens/step, `lr_muon = 0.006`, cosine decay over the full 7000.
- Architecture: identical 275M MLA+Muon Run D model. Config `configs/v2/run_E3_distill_mix_full.yaml`.
- Checkpointed every 875 steps; evaluated at steps 1749 / 3499 / 5249 / 6999 on the locked suite.

This is the run the E3 probe pointed to. Its sole purpose: settle the one question three 2.1B
probes could not — **does commonsense reasoning move with more tokens, or only language
modeling?** The token budget (~7.3B) matches Run D's own ~8B, so this is a fair "same scale,
better data" test of continued pretraining.

## Trajectory across the run (each checkpoint vs the locked suite)

| Checkpoint | Tokens | LAMBADA ppl ↓ | LAMBADA acc ↑ | HellaSwag acc_norm | ARC-E acc_norm | Winogrande |
|-----------|:------:|:------:|:------:|:------:|:------:|:------:|
| **Run D** (base) | — | 26.54 | 0.3681 | 0.4123 | 0.5122 | 0.5130 |
| E3 probe | 2.1B | 23.95 | 0.3860 | 0.4079 | 0.5181 | 0.5209 |
| step 1749 | 1.84B | 25.23 | 0.3796 | 0.4072 | 0.5143 | 0.5170 |
| step 3499 | 3.67B | 24.18 | 0.3788 | 0.4086 | 0.5139 | 0.5233 |
| step 5249 | 5.51B | 23.24 | 0.3918 | 0.4104 | 0.5072 | 0.5193 |
| **step 6999** | **7.34B** | **23.20** | **0.3901** | **0.4125** | **0.5080** | **0.5146** |

## Final (step 6999) vs Run D — significance (per-metric eval ±1σ)

| Benchmark | Metric | Run D | **E3-full** | Δ | ±1σ | Past noise? |
|-----------|--------|------:|------------:|------:|----:|:---:|
| LAMBADA | perplexity ↓ | 26.54 | **23.20** | **−3.34** | ±0.82 | ✅ ~4.1σ |
| LAMBADA | acc | 0.3681 | **0.3901** | **+0.0220** | ±0.0068 | ✅ ~3.2σ |
| HellaSwag | acc_norm | 0.4123 | **0.4125** | +0.0002 | ±0.0049 | ❌ flat |
| ARC-Easy | acc_norm | 0.5122 | **0.5080** | −0.0042 | ±0.0103 | ❌ flat |
| Winogrande | acc | 0.5130 | **0.5146** | +0.0016 | ±0.014 | ❌ flat |

Headline average (HellaSwag acc_norm, ARC-E acc_norm, LAMBADA acc, Winogrande): **45.63%** vs
Run D **45.14%** — a +0.49 pt move that is **entirely LAMBADA-driven**; the other three are flat.

## Verdict: reasoning is NOT token-bound at 275M — only language modeling scales

**1. Language modeling improves cleanly and monotonically with tokens.** LAMBADA perplexity
falls **25.23 → 24.18 → 23.24 → 23.20**, ending at the lowest value in the entire project —
**−3.34 vs Run D (~4σ)**, with LAMBADA acc +0.0220 (~3σ). This extends the probe series' data-
quality lever into a data-quality-**and-scale** lever, and it is the strongest, most robust
result of v2.

**2. LM gains plateau by ~5.5B tokens.** The last 1.84B tokens moved perplexity only 23.24 →
23.20 (−0.04) — clear diminishing returns. Most of the achievable LM gain from this recipe is
banked by ~5B tokens.

**3. Commonsense reasoning stayed flat — even at 7.3B tokens.** HellaSwag acc_norm ended at
**0.4125, statistically identical to the Run D starting point 0.4123** (Δ +0.0002, well inside
±0.0049). The tiny monotonic creep (0.4072 → 0.4125) is sub-noise. ARC-Easy acc_norm and
Winogrande are flat/within noise throughout. **Pouring in Run D's full token budget of higher-
quality, teacher-distilled data did not move commonsense-MCQ reasoning.**

**Answer to the driving question:** reasoning at this scale is **not token-bound** — it is
**capacity/scale-bound or recipe-insensitive at 275M**. Per the pre-committed hard-stop rule
in `E3_vs_runD.md`, the defensible portfolio claim is the **language-modeling gain, not a
reasoning gain.** This is a clean, honest negative result: at a fixed 275M scale, continued
pretraining on better/distilled data buys perplexity, not commonsense reasoning.

## Honest caveats

- **Probe vs full-run steps are not directly comparable.** The probe's 23.95 ppl at 2.1B vs the
  full run's 25.23 at 1.84B is a **schedule artifact**: the probe's 2000-step cosine decayed LR
  to ~0 by its end, while the full run at step 1749 is only ~25% through a 7000-step decay (LR
  still high). The shard sets also differ (disjoint FWE prefixes). The valid comparisons are
  **endpoint-vs-endpoint** and **each vs Run D**, not step-matched cross-run.
- **Only LAMBADA clears noise.** The +0.49 headline-average uptick is LAMBADA-carried; do not
  report HellaSwag/ARC/Winogrande as gains.
- **The locked suite is still blind to ~30% of the data** (code + math + part of the distill
  slice). "Reasoning flat" means *commonsense-MCQ* reasoning. A held-out code/math perplexity
  diagnostic under `results/v2/dev/` would de-blind this; not yet run.

## Artifacts
- `results/v2/run_E3full_step{01749,03499,05249,06999}_eval.json` — full lm-eval output.
- Probe series: `results/v2/E1_vs_runD.md`, `E2_vs_runD.md`, `E3_vs_runD.md`; summary `SUMMARY.md`.
- Run D baseline: `results/run_D_eval.json`. Config: `configs/v2/run_E3_distill_mix_full.yaml`.
