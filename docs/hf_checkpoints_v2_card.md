---
license: apache-2.0
library_name: pytorch
tags:
  - causal-lm
  - mla
  - muon
  - ablation
  - checkpoints
datasets:
  - HuggingFaceFW/fineweb-edu
---

# TinyLM Checkpoints — A/B/C/D Ablation + E3-full continued pretraining

Weight storage for the TinyLM 275M project. Two sets of checkpoints live here:

1. **`run_A`–`run_D`** — all four arms of the architecture ablation, re-run on
   Northeastern Explorer HPC (A100-40GB). Each arm trained 23k steps on
   **8B unique** FineWeb-Edu tokens (~3 epochs, ~24B processed).
2. **`e3_full`** — the v2 continued-pretraining track: Run D further trained on
   a 7.34B-token distill mixture. `e3_full/step_06999.pt` is the base model
   behind [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct).

**For the model card, full eval results, and recommended usage → [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm)** (Run D — the best-performing arm).

## TinyLM model family

| Repo | What it is |
|---|---|
| [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) | **Base 275M** — Run D (MLA + Muon), ablation winner; the model for downstream use |
| [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) | **Instruct** — the base SmolTalk-SFT'd for chat (ChatML) |
| [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) **← this repo** | **All 4 ablation arms** (A/B/C/D) + the **E3-full** continued-pretraining checkpoints |
| [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints) | **v1 historical** checkpoint (1B×21 tokens, pre data-fix) |

Source & full results: [github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM)

## Ablation matrix

| Arm | Attention | Optimizer | File | Headline avg |
|-----|-----------|-----------|------|:---:|
| A | Standard MHA | AdamW | `run_A/step_22999.pt` | 43.62% |
| B | MLA | AdamW | `run_B/step_22999.pt` | 44.11% |
| C | Standard MHA | Muon | `run_C/step_22999.pt` | 44.64% |
| **D** | **MLA** | **Muon** | **`run_D/step_22999.pt`** | **45.14%** |

All four arms differ only in attention class (MHA vs MLA) and matrix optimizer
(AdamW vs Muon). All other settings — data, schedule, batch size, model
dimensions, tokenizer — are identical.

**Full breakdown:** https://github.com/shivnarainms22/TinyLM/blob/main/results/hpc_rerun_ablation.md

## E3-full — continued pretraining (v2 track)

Run D continued on a 7.34B-token distill mixture (45% FineWeb-Edu / 20% general
web / 10% code / 10% math / 15% teacher-distilled), 7000 steps, checkpointed
every 875. Same 275M MLA+Muon architecture — this is a *data* intervention, not
an architecture one.

| File | Tokens | LAMBADA ppl ↓ | HellaSwag acc_norm |
|------|:------:|:------:|:------:|
| `run_D/step_22999.pt` (base) | — | 26.54 | 0.4123 |
| `e3_full/step_01749.pt` | 1.84B | 25.23 | 0.4072 |
| `e3_full/step_03499.pt` | 3.67B | 24.18 | 0.4086 |
| `e3_full/step_05249.pt` | 5.51B | 23.24 | 0.4104 |
| **`e3_full/step_06999.pt`** (final) | **7.34B** | **23.20** | **0.4125** |

`step_06999.pt` is the **base model behind
[`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct)**.
The intermediate checkpoints are kept because the *trajectory* is the result:
language modeling improves monotonically (−3.34 ppl, ~4σ) while commonsense
reasoning stays flat within noise across the whole run.

**Full breakdown:** [`results/v2/E3full_vs_runD.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/v2/E3full_vs_runD.md)

**📄 What the project found (all four tracks, one page):** [`results/FINDINGS.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/FINDINGS.md)

**Training logs:** all four arms are logged side by side on Weights & Biases —
[`tinylm`](https://wandb.ai/shivnarainms22-northeastern-university/tinylm)
(`run_A_mha_adamw`, `run_B_mla_adamw`, `run_C_mha_muon`, `run_D_mla_muon_v2`).

## Loading a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download
from tinylm.model import TinyLM, ModelConfig

# Ablation arms: run_A, run_B, run_C, run_D  (all at step_22999.pt)
# E3-full:       e3_full/step_{01749,03499,05249,06999}.pt
filename = "run_D/step_22999.pt"

ckpt_path = hf_hub_download(
    repo_id="Shiv-22/tinylm-checkpoints-v2",
    filename=filename,
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

model = TinyLM(ModelConfig(**ckpt["config"]))
state = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state):
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
```

The `TinyLM` model class lives in the source repo:
[github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM).

## v1 contrast

The earlier RunPod-era TinyLM (1B unique tokens looped ~21×, single arm) is
preserved at [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints).
The data fix (1B×21 → 8B unique, this re-run) was worth **+3.97 avg pts** over
that v1 on the same MLA+Muon arm — roughly 2.6× the architecture+optimizer
ablation gain.

## License

Apache 2.0. Inherits the permissive terms of
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) for the
codebase and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By) for the training data.
