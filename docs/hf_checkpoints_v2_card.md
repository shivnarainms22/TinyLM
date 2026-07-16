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

# TinyLM Checkpoints — Full A/B/C/D Ablation (HPC re-run)

All four trained checkpoints from the TinyLM 275M architecture ablation,
re-run on Northeastern Explorer HPC (A100-40GB). Each arm trained 23k steps
on **8B unique** FineWeb-Edu tokens (~3 epochs, ~24B processed).

**For the model card, full eval results, and recommended usage → [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm)** (Run D — the best-performing arm).

## TinyLM model family

| Repo | What it is |
|---|---|
| [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) | **Base 275M** — Run D (MLA + Muon), ablation winner; the model for downstream use |
| [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) | **Instruct** — the base SmolTalk-SFT'd for chat (ChatML) |
| [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) **← this repo** | **All 4 ablation arms** (A/B/C/D) from the HPC re-run |
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

## Loading a specific arm

```python
import torch
from huggingface_hub import hf_hub_download
from tinylm.model import TinyLM, ModelConfig

arm = "run_D"   # or run_A, run_B, run_C

ckpt_path = hf_hub_download(
    repo_id="Shiv-22/tinylm-checkpoints-v2",
    filename=f"{arm}/step_22999.pt",
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
