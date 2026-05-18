---
license: apache-2.0
language:
- en
tags:
- pytorch
- language-model
- causal-lm
- mla
- muon
- from-scratch
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
model-index:
- name: TinyLM
  results:
  - task:
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
    metrics:
    - type: acc
      value: 0.324
      name: acc (0-shot)
  - task:
      type: text-generation
    dataset:
      name: ARC-Easy
      type: arc_easy
    metrics:
    - type: acc
      value: 0.538
      name: acc (0-shot)
  - task:
      type: text-generation
    dataset:
      name: LAMBADA
      type: lambada_openai
    metrics:
    - type: acc
      value: 0.292
      name: acc (0-shot)
  - task:
      type: text-generation
    dataset:
      name: Winogrande
      type: winogrande
    metrics:
    - type: acc
      value: 0.500
      name: acc (0-shot)
---

# TinyLM — 275M Parameter SLM

A 275M parameter causal language model trained **from scratch** using
**Multi-head Latent Attention (MLA)** ([DeepSeek-V2](https://arxiv.org/abs/2405.04434))
and the **[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)**,
trained on 1B tokens of FineWeb-Edu.

Built as a research portfolio piece to study the effect of modern KV-compression
(MLA) and second-order optimizer improvements (Muon) at 275M scale.

---

## Model Details

| Property | Value |
|---|---|
| Parameters | ~275M |
| Architecture | Transformer (MLA) |
| Layers | 18 |
| d\_model | 1024 |
| Attention heads | 16 |
| KV latent dim (MLA) | 512 |
| Decoupled RoPE dim | 64 |
| FFN hidden | 2816 (SwiGLU) |
| Context length | 2048 |
| Vocab size | 32,000 |
| Tokenizer | [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

**Attention — MLA:** KV cache stores `d_latent + d_rope = 576` values per token
per layer instead of `n_heads × head_dim × 2 = 2048`, giving a **3.6× KV cache
reduction** at inference time. Positional information is carried only by the
`d_rope = 64` decoupled RoPE branch; the full latent path has no positional bias.

**Optimizer — Muon:** Newton-Schulz orthogonalization is applied to weight gradients
before the AdamW update step, providing approximate second-order curvature correction
without the cost of a full Hessian.

**Other:** RMSNorm, SwiGLU activations, tied input/output embeddings, BF16 training.

---

## Training

| Property | Value |
|---|---|
| Dataset | FineWeb-Edu (1B unique tokens, 10 shards) |
| Steps | 20,000 |
| Effective epochs | ~21 (data repeated) |
| Batch size | 512 sequences × 2048 tokens |
| LR schedule | Cosine with warmup |
| Precision | BF16 |
| Hardware | 1× A100-80GB |
| Final loss | 2.22 |
| Gradient norm (final) | 0.094 |

Training logs: [WandB run](https://wandb.ai/shivnarainms22-northeastern-university/tinylm/runs/dig7xsqf)

> **Note:** Training uses 1B unique tokens repeated ~21× over 20k steps (no 
> annealing mix). This limits long-range coherence; benchmarks that reward it
> (HellaSwag, LAMBADA) are most affected.

---

## Benchmark Results (0-shot)

Evaluated with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.12.
Baseline is `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (3T tokens, 4× larger).

| Benchmark | TinyLM 275M | TinyLlama 1.1B | Δ |
|---|---|---|---|
| HellaSwag (acc) | 32.4% | 59.1% | −26.7% |
| ARC-Easy (acc) | **53.8%** | 55.7% | −1.9% |
| LAMBADA (acc) | 29.2% | 58.9% | −29.7% |
| Winogrande (acc) | 50.0% | 58.9% | −8.9% |
| **Average** | **41.3%** | **58.2%** | **−16.9%** |

ARC-Easy is within 2% of a model trained on 150× more unique tokens with 4× more parameters.
HellaSwag and LAMBADA are weak, which is expected from a model trained on heavily repeated data
— both tasks heavily reward long-range coherence.

---

## Checkpoint

The model checkpoint (`step_19999.pt`, 2.33 GB) is in a separate repo:

**[Shiv-22/tinylm-checkpoints](https://huggingface.co/Shiv-22/tinylm-checkpoints)**

`generate.py` downloads it automatically on first run.

---

## Usage

### Requirements

```bash
pip install torch transformers huggingface_hub
```

### Quick start

```bash
git clone https://huggingface.co/Shiv-22/tinylm
cd tinylm
python generate.py --prompt "The theory of relativity states that"
```

Interactive mode:

```bash
python generate.py
```

Greedy decoding:

```bash
python generate.py --prompt "Once upon a time" --temperature 0
```

### Programmatic loading

```python
import torch
from huggingface_hub import hf_hub_download
from tinylm.model import ModelConfig, TinyLM

ckpt_path = hf_hub_download("Shiv-22/tinylm-checkpoints", "step_19999.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
c = ckpt["config"]

model = TinyLM(ModelConfig(
    n_layers=c["n_layers"], d_model=c["d_model"], n_heads=c["n_heads"],
    d_latent=c["d_latent"], d_rope=c["d_rope"], ffn_hidden=c["ffn_hidden"],
    ctx=c["ctx"], vocab_size=c["vocab_size"], tie_weights=c["tie_weights"],
    attention=c["attention"],
))
state = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in state):
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
```

---

## Limitations

- **Base model only** — no instruction tuning or RLHF. Outputs raw continuations.
- **English educational text** — trained on FineWeb-Edu; other domains/languages will be poor.
- **Repeated data** — 1B unique tokens × 21 epochs limits long-range coherence.
- **Single run** — only the MLA+Muon configuration (Run D) was trained to completion;
  no ablation comparison is available.

---

## Acknowledgments

Architecture is based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan. MLA adapted from the
[DeepSeek-V2 HuggingFace implementation](https://huggingface.co/deepseek-ai/DeepSeek-V2)
(MIT licensed). Muon optimizer by Keller Jordan.
