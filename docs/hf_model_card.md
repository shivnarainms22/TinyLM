---
license: apache-2.0
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - causal-lm
  - small-language-model
  - mla
  - multi-head-latent-attention
  - muon
  - pretrained
datasets:
  - HuggingFaceFW/fineweb-edu
---

# TinyLM 275M (MLA + Muon)

A **275M parameter** small language model trained from scratch with
**Multi-head Latent Attention (MLA, DeepSeek-V2 style)** and the **Muon
optimizer**, on **8B unique tokens** of FineWeb-Edu. Benchmarked against
TinyLlama-1.1B as part of a 4-arm architecture ablation.

- **Source code:** https://github.com/shivnarainms22/TinyLM
- **Full ablation results:** see [`results/hpc_rerun_ablation.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/hpc_rerun_ablation.md) in the repo
- **All four ablation checkpoints:** [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2)

This repo holds the **Run D** arm of the ablation (MLA + Muon) — the
best-performing of the four and the model intended for downstream use.

## TinyLM model family

| Repo | What it is |
|---|---|
| [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) **← this repo** | **Base 275M** — Run D (MLA + Muon), ablation winner; the model for downstream use |
| [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) | **Instruct** — this base SmolTalk-SFT'd for chat (ChatML) |
| [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) | **All 4 ablation arms** (A/B/C/D) from the HPC re-run |
| [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints) | **v1 historical** checkpoint (1B×21 tokens, pre data-fix) |

Source & full results: [github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM)

---

## Model details

| | |
|---|---|
| Parameters | 274.6M |
| Architecture | TinyLlama-style decoder-only Transformer with MLA |
| Layers | 18 |
| Hidden size | 1024 |
| Attention heads | 16 |
| MLA latent dim | 512 (decoupled RoPE 64) |
| FFN hidden | 2816 (SwiGLU) |
| Context length | 2048 |
| Vocab | 32,000 |
| Tokenizer | [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Tied embeddings | Yes |
| Precision | bfloat16 |

## Training

| | |
|---|---|
| Dataset | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (8B unique tokens) |
| Tokens processed | ~24B (~3 epochs) |
| Steps | 23,000 (warmup 2,000) |
| Effective batch | 512 sequences × 2048 tokens ≈ 1.05M tokens/step |
| Optimizer | **Muon** for matrix params (lr 0.02) + **AdamW** for scalar/embed/LM-head/LN (lr 0.001, wd 0.1) |
| LR schedule | Cosine with linear warmup |
| Grad clip | 1.0 |
| Hardware | A100-40GB (Northeastern Explorer HPC) |
| Codebase base | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |

Pure FineWeb-Edu throughout (no annealing mix, no instruction tuning).

## Evaluation

0-shot eval via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). HellaSwag and ARC-Easy reported as `acc_norm` (length-normalized accuracy); LAMBADA and Winogrande as `acc`.

| Benchmark | Metric | TinyLM 275M (this model) | TinyLlama-1.1B baseline | Δ |
|---|---|---:|---:|---:|
| HellaSwag | acc_norm | 41.23% | 59.1% | −17.9 |
| ARC-Easy | acc_norm | 51.22% | 55.7% | −4.5 |
| LAMBADA | acc | 36.81% | 58.9% | −22.1 |
| Winogrande | acc | 51.30% | 58.9% | −7.6 |
| **Average** | | **45.14%** | 58.2% | **−13.1** |

Baseline = [`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T).

### Ablation (full 2×2)

The four arms differ only in attention class and matrix optimizer; all other
training settings are identical.

|  | AdamW | Muon | Δ (Muon − AdamW) |
|:---|:---:|:---:|:---:|
| **MHA** | Run A: 43.62% | Run C: 44.64% | +1.02 |
| **MLA** | Run B: 44.11% | **Run D: 45.14%** *(this model)* | +1.03 |
| **Δ (MLA − MHA)** | +0.49 | +0.50 | — |

**Findings:**

- **Muon contributes ~+1.0 pt avg, consistent across attention type** (+1.02 with MHA, +1.03 with MLA).
- **MLA contributes ~+0.5 pt avg, consistent across optimizer** (+0.49 with AdamW, +0.50 with Muon).
- **Effects are roughly additive** — sum of individual effects = +1.51, observed Run A → Run D = +1.52. Single-seed eval, so interactions below the ~1% noise floor are not detectable.

HellaSwag and LAMBADA are the cleanest signals (monotonic A < B < C < D);
ARC-Easy and Winogrande sit within stderr across all four arms.

---

## Usage

This model uses a custom architecture (MLA with decoupled RoPE) that is not
in HuggingFace `transformers`. To load it, install the source repo:

```bash
git clone https://github.com/shivnarainms22/TinyLM
cd TinyLM
pip install torch transformers huggingface_hub
```

Then:

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tinylm.model import TinyLM, ModelConfig

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="Shiv-22/tinylm", filename="step_22999.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# Build and load model
model = TinyLM(ModelConfig(**ckpt["config"]))
state = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state):
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
model.load_state_dict(state)
model.eval().to("cuda").to(torch.bfloat16)

# Tokenize and generate
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
prompt_ids = tok.encode("The capital of France is", return_tensors="pt").to("cuda")

with torch.no_grad():
    for _ in range(20):
        logits = model(prompt_ids)
        next_id = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        prompt_ids = torch.cat([prompt_ids, next_id], dim=1)

print(tok.decode(prompt_ids[0]))
```

## Limitations

- **Small training budget for a base model.** 8B unique tokens / ~24B processed is well below modern SLMs (TinyLlama-1.1B saw 3T). Absolute benchmark numbers reflect that.
- **Below the 1.1B baseline on all four tasks.** The headline portfolio finding is the *architecture* comparison (MLA+Muon vs MHA+AdamW), not raw absolute capability.
- **Pretrain-only.** No instruction tuning, no RLHF, no safety filtering beyond what FineWeb-Edu already applies upstream.
- **Winogrande at ~51% is essentially chance** (50% binary task) — this benchmark is not a meaningful capability signal at 275M scale.
- **Single-seed evals.** Stderrs of ~0.5–1.0% on `acc_norm` metrics; differences smaller than that should be read as noise.
- **Custom architecture.** Not compatible with `transformers.AutoModel.from_pretrained` out of the box.

## Project history

- **v1** (2026-05, RunPod A100-80GB): single MLA+Muon training run on 1B unique tokens repeated ~21×. Established the training pipeline but the data looping hurt long-range coherence (LAMBADA acc 29.2%). v1 weights preserved at [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints) for contrast.
- **HPC re-run** (2026-05, Northeastern Explorer A100-40GB): full 4-arm ablation on 8B unique tokens. The weights in this repo are from the **Run D** arm of that re-run.

Re-running the same MLA+Muon arm with the data fix (1B×21 → 8B unique) was
worth **+3.97 pts** average — roughly 2.6× the architecture-and-optimizer
ablation gain. Data quality dominates architecture at this scale.

## Citation

```bibtex
@misc{tinylm-275m,
  author       = {Shivnarain},
  title        = {TinyLM 275M: A small language model with MLA and Muon},
  year         = {2026},
  publisher    = {HuggingFace},
  url          = {https://huggingface.co/Shiv-22/tinylm},
}
```

## License

Apache 2.0. Inherits the permissive terms of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) for the codebase and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By) for the training data.
