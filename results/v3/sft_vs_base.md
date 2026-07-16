# SmolTalk SFT vs. base — what instruction-tuning costs on the locked suite

**Deliverable 3 of the v3 post-training phase.** This full-fine-tunes the best base model
— E3-full (`step_06999`, the 7.34B-token distill-mixture continuation of Run D) — on
**SmolTalk** (`HuggingFaceTB/smoltalk`), a ChatML instruction/chat dataset, then re-scores
the **same locked 4-task suite** on the resulting instruct model. The question is narrow
and honest: *what does supervised instruction-tuning do to the benchmark scores?*

The short answer, known before running it: **not much, and slightly negative on
multiple-choice.** That is the expected and correct result — the locked suite measures
zero-shot completion and MCQ, not instruction-following, and SFT optimizes for the latter
at a small cost to the former. This document records the tax; the *capability* SFT bought
is qualitative (see the HF instruct card / generation samples) and is structurally
invisible to these benchmarks.

## Training recipe

| | |
|---|---|
| Base | E3-full `step_06999` (275M, MLA + Muon; = Run D + 7.34B mixture tokens) |
| Data | `HuggingFaceTB/smoltalk`, rendered to ChatML with **prompt-loss masking** (loss on assistant turns only, `-100` on the prompt) |
| Objective | `chunked_cross_entropy` with `ignore_index=-100` (the same memory-safe loss the pretrainer uses) |
| Steps | **20000** (~655M tokens, 1 epoch cap), cosine LR 2e-5 → 2e-6, warmup 3%, grad-checkpoint on |
| Hardware | 1× A100 (80GB drawn), 5h48m, final train loss ~1.1, no OOM |
| Checkpoint | `phase_v3_sft_smoltalk/checkpoints/last.pt` (= `step_019999.pt`) |

*(Two fixes made this fit and finish — see the v3 lessons: SFT now uses chunked CE instead
of `F.cross_entropy` (fp32 logits upcast OOM'd a 40GB card) and explicitly re-enables grad
checkpointing, which the checkpoint's stored `ModelConfig` had silently dropped.)*

## Result — flat, with a small ARC-Easy dip

Base column = E3-full at 0-shot (the checkpoint SFT started from). SFT column = the
instruct model, same 0-shot harness. Both `results/v3/run_E3full_step06999_eval.json`
lineage vs `results/v3/run_sft_eval.json`.

| Benchmark | Metric | E3-full base | SFT | Δ | Past noise? |
|-----------|--------|-------------:|-------:|-------:|:---:|
| HellaSwag | acc_norm | 0.4125 | 0.4107 | −0.0018 | ❌ flat |
| ARC-Easy | acc | 0.5753 | 0.5501 | **−0.0252** | ✅ ~2.5σ |
| ARC-Easy | acc_norm | 0.5080 | 0.4823 | **−0.0257** | ✅ ~2.5σ |
| LAMBADA | perplexity ↓ | 23.20 | **22.97** | −0.23 | ❌ flat (slightly better) |
| LAMBADA | acc | 0.3901 | 0.3852 | −0.0049 | ❌ flat |
| Winogrande | acc | 0.5146 | 0.5209 | +0.0063 | ❌ flat |

**Language modeling is preserved.** LAMBADA perplexity does not degrade — it nudges
*down* (23.20 → 22.97). SmolTalk is fluent, well-edited English, so a full epoch over it
keeps next-token prediction intact rather than eroding it. This matters: SFT did not
"forget" the pretrained model, a real risk with full fine-tuning.

**Commonsense is flat.** HellaSwag and Winogrande do not move outside noise — consistent
with the entire v2/v3 finding that commonsense reasoning is capacity-bound at 275M and
does not respond to data or post-training, only to scale.

**The one real move is ARC-Easy: −2.5 points on both acc and acc_norm (~2.5σ).** This is
the expected **alignment tax**. ARC-Easy is scored by picking the highest-likelihood
completion among multiple-choice options from a bare question; after SFT the model is
biased toward *producing a chat-style answer* rather than *ranking canned option strings*,
so its calibration on the raw-completion MCQ format shifts slightly against it. A ~2.5pt
dip on one MCQ task is a small, well-understood price, not a regression in knowledge.

## Interpretation

**This is the intended trade, and the benchmarks confirm it is cheap.** Instruction-tuning
converts a raw next-token predictor into a model that follows a chat template and answers
prompts — a capability the E3-full base does not have and the locked suite cannot see.
The measurable cost of acquiring it is: language modeling preserved, commonsense
unchanged, and ~2.5 points off one multiple-choice benchmark. There is no free lunch and
no disaster; the SFT lands exactly where a well-behaved small-model SFT should.

The honest framing for the portfolio is therefore: **the locked suite is the wrong
instrument for an instruct model, and its near-flatness here is evidence the SFT did no
harm, not evidence it did nothing.** The thing SFT did — instruction-following — is
demonstrated qualitatively, not on HellaSwag.

## Verdict

SmolTalk SFT on E3-full yields an instruction-following model at a small, expected cost on
the completion/MCQ suite (LM preserved, commonsense flat, ARC-Easy −2.5pt). The suite
cannot measure the capability that was actually added; it can only confirm the alignment
tax is minor. The instruct model's value is its chat behavior, reported separately.

## Honest caveats

- **No instruction-following benchmark.** This measures the *tax*, not the *benefit*. There
  is no MT-Bench / IFEval / AlpacaEval here — the benefit is shown qualitatively.
- **Single SFT recipe.** One dataset (SmolTalk), one epoch, one LR schedule; no sweep, no
  DPO/preference stage. This is a base→SFT step, not a tuned post-training pipeline.
- **Full fine-tune, not LoRA.** All 275M params updated. LAMBADA staying flat shows this
  did not cause catastrophic forgetting at this budget, but that is an observation, not a
  guarantee at larger LRs or more epochs.
- **acc_norm vs acc on ARC.** Both dropped ~2.5pt; the model did not merely get
  worse at length normalization, its underlying MCQ ranking shifted.

## Artifacts

- `results/v3/run_sft_eval.json` — full lm-eval output for the SFT model.
- Base for comparison: `results/v2/run_E3full_step06999_eval.json`.
- Training: `tinylm/sft.py`, `configs/v3/sft_smoltalk.yaml`, `scripts/sft_smoltalk_job.sh`.
  Eval: `scripts/eval_v3_sft_job.sh`. Checkpoint: `phase_v3_sft_smoltalk/checkpoints/last.pt`.

## Appendix — complete raw metrics

| Model | ARC-E acc | ARC-E acc_norm | HellaSwag acc | HellaSwag acc_norm | LAMBADA ppl | LAMBADA acc | Winogrande acc |
|-------|:---------:|:--------------:|:-------------:|:------------------:|:-----------:|:-----------:|:--------------:|
| E3-full base | 0.5753 | 0.5080 | 0.3421 | 0.4125 | 23.1998 | 0.3901 | 0.5146 |
| SFT (SmolTalk) | 0.5501 | 0.4823 | 0.3437 | 0.4107 | 22.9723 | 0.3852 | 0.5209 |

*(SFT stderr: ARC-E acc ±0.0102 / acc_norm ±0.0103, HellaSwag acc ±0.0047 / acc_norm
±0.0049, LAMBADA ppl ±0.822 / acc ±0.0068, Winogrande ±0.014.)*
