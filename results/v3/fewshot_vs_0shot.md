# 5-shot vs 0-shot (Run D and E3-full) — is v2's flat reasoning a zero-shot artifact?

**Deliverable 1 of the v3 post-training phase.** v2 closed on a negative result: continued
pretraining on better/distilled data moved language modeling (LAMBADA ppl 26.54 → 23.20, ~4σ)
but left commonsense reasoning flat (HellaSwag acc_norm 0.4123 → 0.4125). One live objection to
that conclusion was **measurement, not capability**: the locked suite is scored 0-shot, and a
275M base model may simply fail to infer the task format from a bare prompt. If so, the "flat
reasoning" finding would be an artifact of the harness rather than a fact about the model.

This re-runs the **same locked 4-task suite at 5-shot** on both endpoints — Run D (the 2×2
ablation winner) and E3-full (`step_06999`, the 7.34B-token distill-mixture continuation) —
using `scripts/eval_tinylm.py --num-fewshot 5`. Nothing else changes.

## E3-full vs Run D, at 5 shots

| Benchmark | Metric | Run D | E3-full | Δ | Past noise? |
|-----------|--------|------:|--------:|------:|:---:|
| HellaSwag | acc_norm | 0.4100 | 0.4148 | +0.0048 | ❌ 0.7σ |
| ARC-Easy | acc_norm | 0.6204 | 0.6216 | +0.0013 | ❌ 0.1σ |
| Winogrande | acc | 0.5012 | 0.5178 | +0.0166 | ❌ 0.8σ |
| LAMBADA | acc | 0.2793 | 0.2859 | +0.0066 | ❌ 0.7σ |
| LAMBADA | perplexity ↓ | 45.09 | 40.89 | **−4.20** | ✅ ~2.0σ |

**The v2 verdict survives.** Every reasoning metric stays inside noise at 5-shot, exactly as it
did at 0-shot. The only metric that separates the two models is LAMBADA perplexity — the same
language-modeling channel v2 identified, and nothing else. Few-shot prompting does not surface a
hidden reasoning gain from the distill-mixture recipe, because there isn't one to surface.

(Winogrande's +0.0166 is the largest reasoning delta and it is still only 0.8σ on a 1267-item
task where chance is 0.50. It is not a result.)

## The model *does* do in-context learning — just not where reasoning lives

The more interesting half of this experiment is the within-model 0-shot → 5-shot effect, which
rules out the trivial explanation that 275M is simply too small to use examples at all:

| Benchmark | Metric | Run D Δ | E3-full Δ |
|-----------|--------|--------:|----------:|
| ARC-Easy | acc_norm | **+0.1082** | **+0.1136** |
| HellaSwag | acc_norm | −0.0023 | +0.0023 |
| Winogrande | acc | −0.0118 | +0.0032 |
| LAMBADA | acc | −0.0889 | −0.1042 |
| LAMBADA | perplexity ↓ | +18.55 | +17.69 |

**ARC-Easy jumps ~11 points in both models** (0.512 → 0.620 and 0.508 → 0.622). That is a large,
unambiguous in-context learning effect: shown five worked multiple-choice items, the model gets
substantially better at multiple-choice. So the 0-shot scores were, in part, understating it —
but only on the one task whose *format* the examples teach.

**HellaSwag and Winogrande do not move at all** (|Δ| ≤ 0.012). These are the two tasks that
actually probe commonsense reasoning rather than answer-format compliance, and five examples buy
nothing on either. **LAMBADA degrades sharply** (ppl 26.5 → 45.1), which is expected and not a
defect: prepending five unrelated passages to a cloze-completion task actively interferes with
the continuation being scored.

Both models gain the same ~11 ARC points. The ICL ability is therefore a property of the **275M
scale and the task format**, not of the data recipe — E3-full's 7.34B extra distilled tokens
bought no additional in-context learning ability over Run D.

## Verdict

**v2's flat reasoning is not a zero-shot artifact.** Given five in-context examples the model
demonstrably learns the *format* of a task (+11 pts on ARC-Easy) while gaining nothing on the
tasks that require *commonsense* (HellaSwag, Winogrande) — and the gap between Run D and E3-full
stays inside noise on every reasoning metric at both 0 and 5 shots.

This strengthens rather than revises the v2 conclusion: at 275M, better pretraining data is a
real lever for language modeling and a null lever for commonsense reasoning, and the null is a
fact about the model, not about how it was prompted.

## Honest caveats

- **5 shots only.** A single few-shot setting; no 1/2/10-shot sweep, so this rules out "0-shot
  was hiding the gain" but does not map the full ICL curve.
- **ARC-Easy's +11 is format-learning, not necessarily reasoning.** acc and acc_norm converge at
  5-shot (0.6195/0.6204), consistent with the model getting better at picking among options
  rather than at answering questions.
- **Same blind spot as v2**: the locked suite still cannot see the code/math/distill portion of
  the training data. That is what Deliverable 2 (held-out code/math perplexity) is for.

## Artifacts

- `results/v3/run_D_5shot_eval.json`, `results/v3/run_E3full_5shot_eval.json` — full lm-eval output.
- Job: `scripts/eval_v3_fewshot_job.sh`. 0-shot baselines: `results/run_D_eval.json`,
  `results/v2/run_E3full_step06999_eval.json`.

## Appendix — complete raw metrics

| Model | Shots | ARC-E acc | ARC-E acc_norm | HellaSwag acc | HellaSwag acc_norm | LAMBADA ppl | LAMBADA acc | Winogrande acc |
|-------|:-----:|:---------:|:--------------:|:-------------:|:------------------:|:-----------:|:-----------:|:--------------:|
| Run D | 0 | 0.5753 | 0.5122 | 0.3400 | 0.4123 | 26.5446 | 0.3681 | 0.5130 |
| Run D | 5 | 0.6195 | 0.6204 | 0.3402 | 0.4100 | 45.0940 | 0.2793 | 0.5012 |
| E3-full | 0 | 0.5753 | 0.5080 | 0.3421 | 0.4125 | 23.1998 | 0.3901 | 0.5146 |
| E3-full | 5 | 0.6221 | 0.6216 | 0.3447 | 0.4148 | 40.8932 | 0.2859 | 0.5178 |
