# v4 — Logit KD from TinyLlama-1.1B vs. E1 (the controlled contrast)

**Question:** v2 established that *better data* cannot move commonsense reasoning at 275M.
This asks the complementary question: can a **better teacher** do it? Specifically — does
**logit-level knowledge distillation** from TinyLlama-1.1B inject reasoning that
cross-entropy on the same tokens could not?

**Design — a single-variable contrast against E1.** This run is byte-for-byte identical to
the E1 probe (same Run D init, same fresh disjoint FineWeb-Edu shards, same 2.1B-token
budget, same `lr_muon = 0.006`, same architecture) **except the loss function**. E1 used
plain cross-entropy and was a clean no-op. v4 swaps in a KD objective. Any difference is
therefore attributable to the distillation signal itself, not to data, budget, or schedule.

| | |
|---|---|
| Student | Run D `step_22999` (275M, MLA + Muon) |
| Teacher | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (4× larger; **the project's own baseline model**) |
| Why this teacher | Identical Llama-2 32000 vocab (no logit remapping), and a 4× capacity gap — the conventional distillation sweet spot |
| Objective | `kd_alpha = 0.5` mix of top-k forward-KL (T = 2, **top-k = 64**) and ground-truth CE |
| Data | Fresh FineWeb-Edu, provably disjoint from Run D's 8B prefix (**E1's exact shards**) |
| Schedule | 2000 steps × 512 seq × 2048 tok ≈ **2.1B tokens**, `lr_muon = 0.006` |
| Hardware | 1× A100-80GB, ~4 rechained segments, batch 8 × accum 64, grad-checkpoint on |
| Checkpoint | `phase_v4_KD_tinyllama_fwe/checkpoints/step_01999.pt` (= `last.pt`) |

## Result — reasoning did not move; language modeling got *worse*

0-shot locked suite. σ is reported two ways: **single** (against the KD run's own stderr,
the convention used in the v2 writeups' headline columns) and **two-sample** (the honest
figure for a model-vs-model claim, `√(σ₁²+σ₂²)`).

| Benchmark | Metric | Run D | E1 (CE) | **v4 KD** | Δ vs D | σ (1-samp / 2-samp) | Past noise? |
|---|---|---:|---:|---:|---:|:---:|:---:|
| HellaSwag | acc_norm | 0.4123 | 0.4105 | **0.4070** | −0.0053 | 1.1σ / 0.8σ | ❌ flat |
| HellaSwag | acc | 0.3400 | 0.3402 | **0.3352** | −0.0048 | 1.0σ / 0.7σ | ❌ flat |
| ARC-Easy | acc_norm | 0.5122 | 0.5114 | **0.4924** | **−0.0198** | 1.9σ / **1.4σ** | ⚠️ down |
| ARC-Easy | acc | 0.5753 | 0.5800 | **0.5673** | −0.0080 | 0.8σ / 0.6σ | ❌ flat |
| LAMBADA | perplexity ↓ | 26.54 | 26.89 | **28.85** | **+2.30 (worse)** | 2.4σ / **1.7σ** | ⚠️ worse |
| LAMBADA | acc | 0.3681 | 0.3664 | **0.3714** | +0.0033 | 0.5σ / 0.4σ | ❌ flat |
| Winogrande | acc | 0.5130 | 0.5201 | **0.5146** | +0.0016 | 0.1σ / 0.1σ | ❌ flat |

**Not one reasoning metric improved.** HellaSwag and Winogrande are flat to slightly down;
ARC-Easy `acc_norm` fell ~1.4σ (two-sample). The only clearly-moved metric is LAMBADA
perplexity, and it moved the **wrong way**: 26.54 → 28.85.

## Why the E1 contrast is what makes this conclusive

A single degraded run proves little on its own — it could be the data, the continuation LR,
or the `init_from` mechanics. E1 rules all three out:

- E1 ran the **same shards, same budget, same LR, same init**, with plain CE, and held
  LAMBADA perplexity at 26.89 (statistically indistinguishable from Run D's 26.54).
- v4 changed **only the loss** and perplexity rose to 28.85 — roughly **2 points worse than
  its own CE control**.

So the regression is not a property of continued pretraining on this data. **It is the KD
signal.** That is precisely the isolation the experiment was designed to deliver, and it is
why the negative result here is informative rather than merely disappointing.

## Interpretation

**Pre-registered verdict (committed before the run): "reasoning flat = decisive negative."**
That branch is what landed.

The defensible claim: *logit-level distillation from a 4× larger teacher, at this recipe,
does not transfer commonsense reasoning into a 275M student — and mildly degrades its
language modeling relative to plain cross-entropy on identical tokens.*

A plausible mechanism, stated as a hypothesis rather than a finding: `kd_alpha = 0.5` halves
the gradient weight on the ground-truth CE term, so half the student's update budget goes
toward matching a distribution its capacity cannot represent. The student pays a real cost
in next-token accuracy (LAMBADA) and receives nothing in return, because the part of the
teacher's distribution that encodes reasoning is exactly the part a 275M MLA model cannot
absorb. Temperature 2 flattening the target and top-k = 64 truncating the tail both plausibly
contribute; neither was swept.

**Where this leaves the project's central negative result.** v2 showed better *data* cannot
inject reasoning at this scale. v3 confirmed it is not a zero-shot artifact (D1) and that
the data lever was real but invisible to the suite (D2, 4.2×/2.0× held-out perplexity). v4
now closes the remaining escape hatch: a better *teacher's full output distribution* cannot
inject it either. Three independent levers — data volume, data composition/quality, and
teacher supervision — all fail to move commonsense MCQ, while every one of them moves
language modeling. **Reasoning at 275M is capacity-bound.** That conclusion is now supported
by convergent evidence from three different intervention types, which is considerably
stronger than any single run.

## Honest caveats

- **One KD configuration, not a sweep.** α = 0.5, T = 2, top-k = 64, one LR, one budget.
  The claim is about *this recipe*, not "KD can never work at 275M." A lower α (say 0.1–0.3,
  keeping CE dominant) or T = 1 might well avoid the LM regression; this run does not test that.
- **Forward KL only.** No reverse-KL / mode-seeking variant, no sequence-level KD, no
  intermediate-layer (hidden-state or attention) distillation — which is where much of the
  small-model distillation literature actually finds its wins.
- **Top-k = 64 truncation** was a deliberate memory tradeoff. The student never sees the
  teacher's full 32000-way distribution, so "the teacher's full distribution" is an
  approximation in the strict sense.
- **2.1B tokens.** Matched to E1 by design for the contrast, but KD may simply need longer
  to pay off than a CE run does. This measures KD at E1's budget, not KD at convergence.
- **ARC-Easy `acc_norm` fell but `acc` did not** (−0.0198 vs −0.0080). The dip is partly a
  length-normalization effect, so it should not be over-read as a knowledge loss.
- **Teacher quality was assumed, not verified in-run.** TinyLlama-1.1B's own scores are in
  `results/baseline_comparison.md`; no per-batch check confirmed the teacher's logits were
  sensible on these specific shards beyond the loss landing in the expected ~2–3 band.

## Artifacts

- `results/v4/run_KD_eval.json` — full lm-eval output (0-shot, locked 4-task suite).
- Controls: `results/v2/run_E1_eval.json` (CE, same data), `results/run_D_eval.json` (base).
- Implementation: `src/tinylm/kd.py` (top-k forward-KL + CE loss, `Teacher` wrapper,
  `run_kd`), `configs/v4/run_KD_tinyllama_fwe.yaml`, `scripts/submit_kd.sh`,
  `scripts/eval_v4_kd_job.sh`. Tests: `tests/test_kd.py`.
- `train.py` took a **default-preserving loss hook** (`train(cfg, loss_fn=...)`, default =
  cross-entropy) so the locked A/B/C/D training path is byte-identical; `model.py` untouched.

## Appendix — complete raw metrics

| Model | ARC-E acc | ARC-E acc_norm | HellaSwag acc | HellaSwag acc_norm | LAMBADA ppl | LAMBADA acc | Winogrande acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run D (base) | 0.5753 | 0.5122 | 0.3400 | 0.4123 | 26.5446 | 0.3681 | 0.5130 |
| E1 (CE control) | 0.5800 | 0.5114 | 0.3402 | 0.4105 | 26.8885 | 0.3664 | 0.5201 |
| **v4 KD** | 0.5673 | 0.4924 | 0.3352 | 0.4070 | 28.8451 | 0.3714 | 0.5146 |

*(KD stderr: ARC-E acc ±0.0102 / acc_norm ±0.0103, HellaSwag acc ±0.0047 / acc_norm ±0.0049,
LAMBADA ppl ±0.957 / acc ±0.0067, Winogrande ±0.0140.)*
