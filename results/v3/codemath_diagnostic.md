# Held-out code/math perplexity — de-blinding the v2 data-quality result

**Deliverable 2 of the v3 post-training phase.** v2 concluded on a negative result:
continued pretraining on a better/distilled data mixture moved language modeling
(LAMBADA ppl 26.54 → 23.20, ~4σ) but left commonsense reasoning flat (HellaSwag
acc_norm 0.4123 → 0.4125). Deliverable 1 then showed that null is not a zero-shot
artifact — it survives at 5-shot too.

But every one of those measurements shares a **blind spot**: the locked suite
(HellaSwag, ARC-Easy, LAMBADA, Winogrande) is scored entirely on English commonsense
text. It cannot see the ~30% of E3-full's training mixture that is **code, math, and
distilled QA**. So "better data did nothing" was only ever a statement about the part
of the model the benchmarks can measure. This diagnostic measures the rest.

## Setup

E3-full (`step_06999`) **is** Run D (`step_22999`) continued-pretrained for 7.34B tokens
on the E3 mixture — **45% FineWeb-Edu / 20% web / 10% code / 10% math / 15%
Cosmopedia-v2** (Mixtral-distilled explanations/QA). Run D itself saw **FineWeb-Edu
only** — essentially no code and little math. So a before/after of the same model
lineage isolates exactly what the mixed-data continuation bought on the data the locked
suite is blind to.

Both checkpoints are scored on two held-out shards — **~10M tokens each** of code and of
math — built disjoint from the training data (`build_heldout_codemath.py`, 1B-token skip),
so a low number is generalization within-distribution, **not** memorization. Metric is
mean per-token negative log-likelihood (→ perplexity) at `ctx=2048`, via
`scripts/eval_perplexity.py`.

## Result — the mixture bought a large, previously-invisible modeling gain

| Held-out set | Run D (FWE-only) | E3-full (mix+distill) | ppl ratio | bits/token saved |
|--------------|-----------------:|----------------------:|:---------:|:----------------:|
| **code** | 13.45 | **3.17** | **4.24×** | **2.08** |
| **math** | 12.92 | **6.43** | **2.01×** | **1.01** |

*(Both over 9,993,454 tokens. "bits/token saved" = the NLL reduction in bits: code
3.750 → 1.665, math 3.692 → 2.685.)*

**On code, E3-full's perplexity is 4.2× lower than Run D's** — it needs ~2.1 fewer bits
to predict each code token. On math it is **2.0× lower**, ~1.0 bit/token. These are not
marginal, within-noise deltas like the reasoning benchmarks; they are the largest
capability differences measured anywhere in the v2/v3 work, and they are exactly where
the locked suite has no instrument.

For scale: v2's headline language-modeling gain was LAMBADA ppl 26.54 → 23.20, a ~13%
relative reduction. The code perplexity gain here is a **76% reduction** and the math a
**50% reduction**. The data-quality lever was never small — the benchmarks were just
pointed the wrong way.

## Interpretation

**The v2 verdict was correct but under-stated by its own instruments.** "Data quality
scales language modeling, not commonsense reasoning" is right — but the *magnitude* of
the language-modeling gain was hidden, because LAMBADA (English commonsense cloze) only
lightly reflects a mixture whose real content is code and math. Measure the model on the
distribution it was actually trained on and the gain is enormous.

This also resolves the apparent paradox of the whole v2 phase: it looked like 7.34B extra
tokens of carefully-built mixture "did almost nothing" (flat reasoning, modest LAMBADA).
They did a great deal — they taught the model to model code and math far better — but
that entire axis of improvement fell in the suite's blind spot. Reasoning stayed
capacity-bound at 275M; **language modeling of the new domains scaled sharply**, on the
same clean negative-for-reasoning, positive-for-LM story v2 told, now with the LM side
properly quantified.

## Verdict

De-blinding the ~30% of the data the locked suite cannot see confirms and sharpens v2:
the distill-mixture continuation delivered a **4.2× (code) / 2.0× (math) perplexity
reduction** over the FineWeb-Edu-only baseline. Better pretraining data is a large,
real lever for language modeling — including on domains the commonsense benchmarks are
structurally unable to detect — and a null lever for commonsense reasoning at 275M. The
gain was there all along; this is the first measurement aimed at it.

## Honest caveats

- **Perplexity, not task accuracy.** This is held-out language-modeling loss, not
  HumanEval/GSM8K. It shows the mixture taught the model to *model* code/math text far
  better; it is not a claim of downstream code-generation or math-solving skill.
- **In-distribution by construction.** E3-full saw code/math during training and Run D
  did not, so lower ppl for E3-full is the *expected* consequence of the mixture — that
  is the point (does the mix "take"?), but it is not a surprise and not an
  architecture/optimizer effect. The shards are held-out (disjoint 1B-skip), so it is
  generalization within-distribution, not memorization.
- **Absolute ppl values are domain-relative.** Code (3.17) reads lower than math (6.43)
  partly because tokenized source code is more repetitive/predictable than mixed
  natural-language math; cross-domain absolute comparison is not meaningful, the
  within-domain D-vs-E3-full contrast is.

## Artifacts

- `results/v3/ppl_D_code.json`, `results/v3/ppl_D_math.json`,
  `results/v3/ppl_E3full_code.json`, `results/v3/ppl_E3full_math.json` — raw NLL/ppl/token-count.
- Held-out builder: `scripts/build_heldout_codemath.py` (+`.sh`). Eval:
  `scripts/eval_perplexity.py`, job `scripts/eval_v3_ppl_job.sh`. Perplexity core:
  `tinylm/perplexity.py`.

## Appendix — complete raw metrics

| Model | Set | n_tokens | mean NLL (nats) | bits/token | perplexity |
|-------|-----|---------:|----------------:|-----------:|-----------:|
| Run D | code | 9,993,454 | 2.5991 | 3.750 | 13.4513 |
| Run D | math | 9,993,454 | 2.5590 | 3.692 | 12.9226 |
| E3-full | code | 9,993,454 | 1.1539 | 1.665 | 3.1705 |
| E3-full | math | 9,993,454 | 1.8611 | 2.685 | 6.4310 |
