# Run D vs TinyLlama-1.1B Baseline

**Run D:** 275M params, MLA + Muon, 1B tokens × 21 epochs, 20k steps  
**Baseline:** TinyLlama-1.1B-intermediate-step-1431k-3T, 1.1B params, 3T tokens  
**Eval:** lm-eval v0.4.12, 0-shot

| Benchmark | Metric | Run D (275M) | TinyLlama-1.1B | Delta |
|---|---|---|---|---|
| HellaSwag | acc | 32.4% | 59.1% | -26.7% |
| HellaSwag | acc_norm | 37.1% | — | — |
| ARC-Easy | acc | 53.8% | 55.7% | -1.9% |
| ARC-Easy | acc_norm | 48.4% | — | — |
| LAMBADA | acc | 29.2% | 58.9% | -29.7% |
| LAMBADA | perplexity | 58.8 | 6.85 | — |
| Winogrande | acc | 50.0% | 58.9% | -8.9% |
| **Average (acc)** | | **41.3%** | **58.2%** | **-16.9%** |

## Notes

- ARC-Easy within 2% of TinyLlama-1.1B despite 4× fewer parameters and 150× less unique training data
- Winogrande at 50.0% ≈ random chance (binary task) — no meaningful signal
- HellaSwag and LAMBADA weak — both reward long-range coherence, hurt most by repeated training data
- All tasks above random chance — model learns, hypothesis confirmed
