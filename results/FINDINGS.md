# TinyLM — What This Project Found

*A single-page synthesis of all four tracks. Each section links the detailed writeup.*

TinyLM trained a **275M-parameter** language model (Multi-head Latent Attention + the Muon
optimizer) from scratch on FineWeb-Edu, benchmarked it against **TinyLlama-1.1B**, then spent
three further tracks trying to make its *reasoning* better. The architecture work succeeded.
The reasoning work produced a **negative result**, established three independent ways.

That negative result is the most interesting thing here, so it is stated up front:

> **At 275M parameters, commonsense reasoning did not respond to any intervention tried —
> more data, better data, or a larger teacher's supervision — while every one of those
> interventions measurably improved language modeling. Reasoning at this scale appears to be
> capacity-bound, not data-bound.**

Three interventions of genuinely different *kinds* all failed the same way, and all three
succeeded on the language-modeling channel. That convergence is what makes the claim worth
making; a single flat run would not be.

---

## The evidence in one table

Every row is a controlled intervention on the same 275M model. "Reasoning" = HellaSwag /
ARC-Easy / Winogrande (commonsense MCQ). "Language modeling" = LAMBADA perplexity, plus
held-out perplexity where the benchmarks were blind.

| # | Intervention | Track | Reasoning | Language modeling |
|:--|:---|:---|:---|:---|
| 1 | Architecture: MLA + Muon vs MHA + AdamW | v1 | **+1.5 avg** ✅ | improves |
| 2 | Fix the data (8B unique vs 1B repeated ×21) | v1 | **+4.0 avg** ✅ | LAMBADA acc +7.6 |
| 3 | +2.1B *more* same-distribution tokens (E1) | v2 | flat ❌ | flat (ppl 26.54→26.89) |
| 4 | +2.1B *broader* mixture tokens (E2) | v2 | flat ❌ | ppl → 25.08 ✅ |
| 5 | + distilled-text slice (E3) | v2 | flat ❌ | ppl → 23.95 ✅ |
| 6 | Scale that recipe to 7.3B tokens (E3-full) | v2 | flat ❌ | ppl → **23.20** ✅ (~4σ) |
| 7 | Show it 5 examples (few-shot) | v3 | flat ❌ * | — |
| 8 | Measure the blind spot (held-out code/math ppl) | v3 | — | **4.2× / 2.0×** lower ✅ |
| 9 | Instruction-tune it (SmolTalk SFT) | v3 | ARC-E −2.5 ⚠️ | preserved (23.20→22.97) |
| 10 | Distill a 4× teacher's logits into it (KD) | v4 | flat/down ❌ | **worse** (→28.85) ⚠️ |

\* *Few-shot does lift ARC-Easy ~+11 pts — but identically in both models compared, so it is
answer-**format** learning, not reasoning. See D1.*

**Rows 1–2 moved reasoning. Rows 3–10 did not.** What separates them: rows 1–2 changed the
model's *capacity to learn* (architecture) or fixed a *broken* training signal. Everything
after that added more or better supervision to an already-healthy 275M model — and it had
nowhere to go.

---

## Track 1 — Architecture ablation (the thing that worked)

Full 2×2, one variable per arm, 8B unique tokens × 23k steps each, all four arms trained and
evaluated on Explorer A100s.

|  | AdamW | Muon | Δ (Muon − AdamW) |
|:---|:---:|:---:|:---:|
| **MHA** | A 43.62 | C 44.63 | **+1.02** |
| **MLA** | B 44.11 | D 45.14 | **+1.03** |
| **Δ (MLA − MHA)** | **+0.49** | **+0.50** | — |

**The effects are consistent and additive.** Muon contributes ~+1.0 average regardless of
attention type; MLA contributes ~+0.5 regardless of optimizer; the individual effects sum to
1.52 and the observed A→D gap is +1.52 — no detectable interaction. Ordering is monotonic
(A < B < C < D). HellaSwag is the cleanest single signal (Muon +1.8, ~2.6σ, on both A→C and
B→D).

**The bigger lever was not architecture — it was data hygiene.** Fixing the v1 bug (1B tokens
repeated ~21× → 8B unique) was worth **+3.97 average** on the identical MLA+Muon arm, ~2.6×
the entire architecture gain. Honest framing: *architecture bought +1.5 points; not looping
the data bought +4.0.*

**Both clauses of the pinned hypothesis are closed.** Benchmark parity: Run D lands within
**1.9 pts** of TinyLlama-1.1B on ARC-Easy — a model with 4× the parameters and ~150× the
unique tokens. KV cache: MLA caches `d_latent + d_rope = 576` values per token per layer vs
an equivalent MHA's `2·d_model = 2048` — a **3.56× reduction (71.9%)**, 144.0 → 40.5 MiB at
2048-token context, verified by unit test, not just derived.

→ `results/hpc_rerun_ablation.md`, `results/kv_cache_reduction.md`

## Track 2 — Can better *data* fix reasoning? (v2)

Four continued-pretraining runs off Run D, each changing only the data:

| Probe | Data | LAMBADA ppl ↓ | Reasoning |
|:---|:---|:---:|:---|
| Run D | 8B FineWeb-Edu (base) | 26.54 | — |
| E1 | +2.1B *fresh, provably disjoint* FWE | 26.89 | flat |
| E2 | +2.1B web/code/math mixture | 25.08 | flat |
| E3 | + 15% Cosmopedia-v2 distilled text | 23.95 | flat |
| E3-full | that recipe at **7.34B** tokens | **23.20** | flat |

**Language modeling scales cleanly with data quality *and* volume** — monotonic, ~4σ, and it
plateaus around 5.5B tokens. **Commonsense reasoning never moves**, not even at a token
budget matching Run D's entire original training run.

E1 is the load-bearing control: provably zero token overlap with Run D's training set, so
"more of the same data does nothing" is a clean negative, not a confound.

→ `results/v2/SUMMARY.md`, `results/v2/E{1,2,3}_vs_runD.md`, `results/v2/E3full_vs_runD.md`

## Track 3 — Is the negative result an artifact? (v3)

Three attacks on the v2 conclusion, because a flat result deserves adversarial scrutiny:

**D1 — Is it a zero-shot artifact?** No. At 5-shot every reasoning metric stays inside noise,
exactly as at 0-shot. The models *do* in-context-learn — ARC-Easy jumps ~+11 pts — but by an
identical amount in both models compared, so it is answer-*format* learning, a property of
the 275M scale rather than of any data recipe.

**D2 — Were the benchmarks even looking?** Partly not. The locked suite is blind to the ~30%
of E3's mixture that is code and math. Held-out perplexity (~10M tokens each) shows what it
missed:

| Held-out set | Run D | E3-full | Reduction |
|:---|:---:|:---:|:---:|
| code | 13.45 | **3.17** | **4.24× (−76%)** |
| math | 12.92 | **6.43** | **2.01× (−50%)** |

The data lever was never small — **the instrument was pointed the wrong way.** This is the
result that keeps the v2 negative honest: better data did a great deal, just not to the thing
the benchmarks measure.

**D3 — Does post-training help?** SmolTalk SFT produces a working instruction-following model
at the expected **alignment tax**: language modeling preserved (ppl 23.20 → 22.97), commonsense
flat, ARC-Easy −2.5 pts (the model shifts toward *answering* rather than *ranking* canned
options). The capability SFT added is real and qualitative — and structurally invisible to
this suite.

→ `results/v3/fewshot_vs_0shot.md`, `results/v3/codemath_diagnostic.md`, `results/v3/sft_vs_base.md`

## Track 4 — Can a better *teacher* fix reasoning? (v4)

The last escape hatch. If data can't inject reasoning, maybe supervision from a larger model
can. Logit-level KD from **TinyLlama-1.1B** (4× larger; the project's own baseline; identical
Llama-2 vocab) into Run D — designed as a **single-variable contrast against E1**: same init,
same shards, same 2.1B-token budget, same LR, **only the loss differs**.

| Metric | Run D | E1 (CE control) | **KD** | Δ vs D |
|:---|:---:|:---:|:---:|:---|
| HellaSwag acc_norm | 0.4123 | 0.4105 | **0.4070** | flat |
| ARC-Easy acc_norm | 0.5122 | 0.5114 | **0.4924** | −0.0198 (~1.4σ) ⚠️ |
| Winogrande acc | 0.5130 | 0.5201 | **0.5146** | flat |
| LAMBADA ppl ↓ | 26.54 | 26.89 | **28.85** | **+2.30 worse** (~1.7σ) ⚠️ |

**No reasoning transfer, and a real cost to language modeling.** Because E1 held perplexity
flat on identical tokens with plain cross-entropy, the ~2-point regression is attributable to
the **distillation objective itself**, not the data, LR, or continuation mechanics. A 4×
teacher's output distribution could not inject what the student cannot represent — and made
it pay for the attempt.

→ `results/v4/KD_vs_E1.md`

---

## What I'd tell another engineer

1. **Data hygiene beat architecture by 2.6×.** The single largest gain in the project came
   from noticing the training data was looping — not from MLA, not from Muon. Check the data
   before tuning the model.
2. **Controls are what make a negative result publishable.** E1 (provably disjoint data, plain
   CE) is cited by both the v2 and v4 conclusions. Without it, "KD made things worse" would be
   an anecdote instead of an attribution.
3. **A benchmark suite is an instrument, and instruments have blind spots.** D2 found a 4.2×
   perplexity improvement that four benchmarks scored as zero. Lock your suite early for
   discipline — then deliberately measure outside it.
4. **Pre-register the verdict.** The v4 read ("reasoning moves = ceiling broken; reasoning
   flat = decisive negative") was written before the run finished. That is the difference
   between a finding and a rationalization.
5. **Small models fail in a specific direction.** Everything tried improved *language
   modeling* and nothing improved *commonsense reasoning*. At 275M the bottleneck is capacity,
   and no amount of supervision substitutes for it.

## Honest limits on all of the above

- **Single seed per arm.** Stderrs are ~0.5–1.0 pts; the ablation can rule out a *large*
  MLA×Muon interaction, not a small one.
- **One suite, four tasks.** HellaSwag, ARC-Easy, LAMBADA, Winogrande — locked at Phase 0 for
  discipline. Winogrande sits near chance (~50%) for a model this size and carries little signal.
- **"Reasoning" here means commonsense MCQ**, the standard proxy — not mathematical or
  multi-step reasoning, which were never measured on these benchmarks.
- **One recipe per intervention.** One KD config (forward-KL, α = 0.5, T = 2, top-k = 64), one
  SFT dataset and schedule, one continuation LR. These are single well-controlled points, not
  sweeps — a different KD configuration might well avoid the LM regression.
- **The capacity claim is an inference, not a measurement.** Nothing here trained a 500M or 1B
  variant to confirm the bound is capacity rather than something else common to all ten
  interventions. That is the experiment this project points at next.

---

## Map of the work

| Track | What it asked | Detail |
|:---|:---|:---|
| v1 | Do MLA + Muon help at 275M? | `results/hpc_rerun_ablation.md` |
| — | Was the KV-cache claim real? | `results/kv_cache_reduction.md` |
| v2 | Can better data fix reasoning? | `results/v2/SUMMARY.md` |
| v3 | Is that negative result an artifact? | `results/v3/*.md` |
| v4 | Can a better teacher fix reasoning? | `results/v4/KD_vs_E1.md` |

**Training logs (W&B):**
[v1 ablation](https://wandb.ai/shivnarainms22-northeastern-university/tinylm) ·
[v2 probes](https://wandb.ai/shivnarainms22-northeastern-university/tinylm-v2) ·
[v3 SFT](https://wandb.ai/shivnarainms22-northeastern-university/tinylm-v3) ·
[v4 KD](https://wandb.ai/shivnarainms22-northeastern-university/tinylm-v4)

**Models:** [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) (Run D base) ·
[`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) (SFT) ·
[`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) (all 4 ablation arms)
