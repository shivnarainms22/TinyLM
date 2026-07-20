---
license: apache-2.0
library_name: pytorch
tags:
  - causal-lm
  - mla
  - muon
  - historical
---

# TinyLM v1 Checkpoint (historical)

Single MLA+Muon training run from the v1 TinyLM effort (RunPod A100-80GB,
May 2026). Trained on **1B unique** FineWeb-Edu tokens **repeated ~21×**
over 20k steps — the data bug the HPC re-run later fixed.

**Preserved here for historical contrast — not the recommended model.**

- **Recommended model:** [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) (Run D from the HPC re-run, 8B unique tokens, +3.97 avg pts above this v1 on the same architecture)
- **Full 4-arm ablation checkpoints:** [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2)
- **Source code:** [github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM)
- **📄 What the project found (all four tracks, one page):** [`results/FINDINGS.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/FINDINGS.md)

## TinyLM model family

| Repo | What it is |
|---|---|
| [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) | **Base 275M** — Run D (MLA + Muon), ablation winner; the model for downstream use |
| [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) | **Instruct** — the base SmolTalk-SFT'd for chat (ChatML) |
| [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) | **All 4 ablation arms** (A/B/C/D) from the HPC re-run |
| [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints) **← this repo** | **v1 historical** checkpoint (1B×21 tokens, pre data-fix) |

Source & full results: [github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM)

## v1 eval (0-shot)

| Benchmark | Metric | v1 D |
|---|---|:---:|
| HellaSwag | acc_norm | 37.1% |
| ARC-Easy | acc_norm | 48.4% |
| LAMBADA | acc | 29.2% |
| Winogrande | acc | 50.0% |
| **Average** | | **41.18%** |

The notably weak LAMBADA (long-range coherence) was the main signal that
repeated data was hurting; the HPC re-run with 8B unique tokens lifted
LAMBADA to 36.81% (+7.61) on the same arm.

## License

Apache 2.0.
