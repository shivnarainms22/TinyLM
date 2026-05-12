# Evaluation Suite (locked 2026-05-12)

Four benchmarks chosen before any training. **No additions after
results are in** — that is cherry-picking and breaks the experiment.

| Benchmark | Task name (lm-eval) | What it measures |
|---|---|---|
| HellaSwag | `hellaswag` | Commonsense reasoning (4-way multiple choice) |
| ARC-Easy | `arc_easy` | Factual QA |
| LAMBADA (OpenAI split) | `lambada_openai` | Long-range LM coherence (last-word prediction) |
| Winogrande | `winogrande` | Coreference resolution |

## Invocation

The exact same command must be used for the baseline and all four
ablation checkpoints. The only thing that changes is the `pretrained=`
arg.

```bash
lm_eval --model hf \
  --model_args pretrained=<MODEL_PATH_OR_HF_ID> \
  --tasks hellaswag,arc_easy,lambada_openai,winogrande \
  --device cuda --batch_size 16 \
  --output_path ./results/<run_name>_eval.json
```

## Reported metric per benchmark

| Benchmark | Metric (per lm-eval default) |
|---|---|
| HellaSwag | `acc_norm` (length-normalized accuracy) |
| ARC-Easy | `acc_norm` |
| LAMBADA OpenAI | `acc` (exact-match) and `perplexity` |
| Winogrande | `acc` |

When comparing runs, use the same metric column. For
"average benchmark score" claims, average the headline accuracy
(`acc_norm` for HS/ARC-E, `acc` for LAMBADA/Winogrande).

## Baseline results

Produced by `scripts/eval_baseline.py` on Colab/Kaggle. Output lives
at `baseline_results.json` in the repo root.
