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
  - sft
  - instruct
  - chat
datasets:
  - HuggingFaceTB/smoltalk
base_model: Shiv-22/tinylm
---

# TinyLM 275M — Instruct (SmolTalk SFT)

An **instruction-tuned** 275M small language model. This is the **E3-full** base
(Run D — MLA + Muon — continued-pretrained on a 7.34B-token distill-mixture)
supervised-fine-tuned for one epoch on **[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)**
with a ChatML-style template and prompt-loss masking.

- **Source code:** https://github.com/shivnarainms22/TinyLM
- **Base model:** the E3-full checkpoint from the v2 continued-pretraining track
  (see [`results/v2/E3full_vs_runD.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/v2/E3full_vs_runD.md))
- **SFT details & eval:** [`results/v3/sft_vs_base.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/v3/sft_vs_base.md)

## TinyLM model family

| Repo | What it is |
|---|---|
| [`Shiv-22/tinylm`](https://huggingface.co/Shiv-22/tinylm) | **Base 275M** — Run D (MLA + Muon), ablation winner; the model for downstream use |
| [`Shiv-22/tinylm-instruct`](https://huggingface.co/Shiv-22/tinylm-instruct) **← this repo** | **Instruct** — the base SmolTalk-SFT'd for chat (ChatML) |
| [`Shiv-22/tinylm-checkpoints-v2`](https://huggingface.co/Shiv-22/tinylm-checkpoints-v2) | **All 4 ablation arms** (A/B/C/D) from the HPC re-run |
| [`Shiv-22/tinylm-checkpoints`](https://huggingface.co/Shiv-22/tinylm-checkpoints) | **v1 historical** checkpoint (1B×21 tokens, pre data-fix) |

Source & full results: [github.com/shivnarainms22/TinyLM](https://github.com/shivnarainms22/TinyLM)

> **Scope.** This is a 275M research model, part of a training-methods portfolio.
> It follows instructions and holds a chat format, but it is **not** a reliable
> assistant — expect factual slips, weak multi-step reasoning, and occasional
> repetition. See **Limitations**.

---

## Prompt format

The model was trained on this exact text template (no new special tokens; vocab
stays 32,000). Encode **without** a BOS token; the model closes its own turn with EOS.

```
<|user|>
{your message}
<|assistant|>
```

An optional system turn may precede it: `<|system|>\n{system}\n`. Generation
continues after the final `<|assistant|>\n` and stops at EOS. The repo's
`scripts/generate_sft_samples.py` builds this priming string exactly.

## Usage

This is a **custom PyTorch model** (not a `transformers` architecture), so load it with
the repo's code rather than `AutoModelForCausalLM`:

```bash
git clone https://github.com/shivnarainms22/TinyLM && cd TinyLM
pip install torch transformers huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import sys; sys.path.insert(0, "src")
from tinylm.loader import load_from_checkpoint
from tinylm.sft import render_chat, _ROLE_PREFIX

ckpt = hf_hub_download("Shiv-22/tinylm-instruct", "tinylm_sft_smoltalk.pt")
model = load_from_checkpoint(ckpt, device="cpu").eval()

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
enc = lambda s: tok.encode(s, add_special_tokens=False)  # no BOS
ids, _ = render_chat([{"role": "user", "content": "Explain photosynthesis simply."}],
                     enc, tok.eos_token_id)
ids += enc(_ROLE_PREFIX["assistant"])   # prime the assistant turn
# ... greedy-decode from `ids`, stop at eos — see scripts/generate_sft_samples.py
```

`scripts/generate_sft_samples.py --checkpoint <path>` does exactly this end to end.

## Training

| | |
|---|---|
| Base | E3-full `step_06999` (275M, MLA + Muon) |
| Data | [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), ChatML render, **loss on assistant turns only** |
| Objective | Memory-safe chunked cross-entropy, `ignore_index=-100` |
| Steps | 20,000 (~655M tokens, 1 epoch cap) |
| LR | cosine 2e-5 → 2e-6, 3% warmup |
| Hardware | 1× A100, 5h48m, final train loss ~1.1 |

## Evaluation — the alignment tax

Instruction-tuning is evaluated honestly for its **cost** on the locked
completion/MCQ suite (it cannot measure instruction-following, only what SFT
trades away). Base = E3-full at 0-shot; both scored with the same harness.

| Benchmark | Metric | E3-full base | Instruct (SFT) | Δ |
|-----------|--------|-------------:|---------------:|------:|
| LAMBADA | perplexity ↓ | 23.20 | **22.97** | −0.23 (preserved) |
| HellaSwag | acc_norm | 0.4125 | 0.4107 | −0.0018 (flat) |
| Winogrande | acc | 0.5146 | 0.5209 | +0.0063 (flat) |
| ARC-Easy | acc | 0.5753 | 0.5501 | **−0.0252** (~2.5σ) |

Language modeling is **preserved** (LAMBADA perplexity even nudges down — no
catastrophic forgetting), commonsense is flat, and the one real cost is ~2.5
points on ARC-Easy multiple-choice — the expected tax of shifting the model from
ranking canned options toward producing chat answers.

## Sample outputs

Greedy decoding, unedited. Full set (8 prompts):
[`results/v3/sft_samples.md`](https://github.com/shivnarainms22/TinyLM/blob/main/results/v3/sft_samples.md).

**Explain photosynthesis in one simple paragraph.**
> Photosynthesis is the process by which plants convert light energy into
> chemical energy, which is used to produce food for the plant. The process
> involves the absorption of light energy by chlorophyll in the leaves of a
> plant, which then absorbs the energy and uses it to produce glucose, a type
> of sugar. […]

**Rewrite this sentence to be more polite: 'Send me the file now.'**
> I'll send you the file now.

**Give me two tips for staying focused while studying.**
> First, establish a study routine that works for you. Set aside dedicated time
> each day to study… Another key tip is to create a dedicated study space… turn
> off any distractions like phones or TVs.

The model reliably adopts the assistant role, stays on topic, and produces
coherent, on-format answers — that is the instruction-following the benchmark
table above cannot see.

## Limitations

These are real and visible in the same sample set — shown, not hidden:

- **Weak multi-step reasoning / arithmetic.** Asked a train's average speed
  (60 km / 2 h), it answered "120 km/h" then "≈1.25" — wrong. Consistent with
  the v2/v3 finding that reasoning at 275M is **capacity-bound**, and SFT does
  not add reasoning the base model lacks.
- **Factual errors.** "good morning" → Spanish came back as *"¿Dónde está el
  día?"* (incorrect). It will state wrong facts confidently.
- **Repetition on open-ended generation.** Under greedy decoding, creative
  prompts (e.g. a poem) can collapse into a repeated line. Sampling
  (`temperature 0.7`, `top_p 0.9`) mitigates this but trades determinism.
- **No safety tuning.** Only SmolTalk SFT — no preference optimization (DPO/RLHF),
  no safety alignment. Do not deploy in user-facing or high-stakes settings.

## Intended use

Research and education: studying small-model post-training, prompt-template
effects, and the gap between benchmark scores and instruction-following. Not
intended for production assistance.
