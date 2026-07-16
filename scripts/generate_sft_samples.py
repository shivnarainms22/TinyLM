#!/usr/bin/env python3
"""Generate qualitative chat samples from the v3 SmolTalk-SFT model.

The locked benchmark suite cannot see instruction-following (see
results/v3/sft_vs_base.md), so this produces the direct evidence: real
prompt -> chat responses, primed with the exact ChatML template the model
was fine-tuned on. Writes a markdown table to results/v3/sft_samples.md and
echoes each sample as it is produced.

Run on an A100 against the SFT checkpoint:

    python scripts/generate_sft_samples.py \
        --checkpoint /scratch/$USER/tinylm/runs/phase_v3_sft_smoltalk/checkpoints/last.pt

Greedy by default (temperature 0) so the samples are reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from tinylm.loader import load_from_checkpoint
    from tinylm.sft import _ROLE_PREFIX, render_chat
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from tinylm.loader import load_from_checkpoint
    from tinylm.sft import _ROLE_PREFIX, render_chat

TOKENIZER = "meta-llama/Llama-2-7b-hf"

# A spread that exercises different instruction types: explanation, generation,
# procedure, factual QA, rewriting, and light reasoning.
PROMPTS: list[str] = [
    "Explain photosynthesis in one simple paragraph.",
    "Write a short four-line poem about the ocean.",
    "List three steps to make a cup of tea.",
    "What is the capital of France?",
    "Rewrite this sentence to be more polite: 'Send me the file now.'",
    "If a train travels 60 km in 2 hours, what is its average speed?",
    "Give me two tips for staying focused while studying.",
    "Translate 'good morning' into Spanish.",
]


def build_prompt_ids(
    prompt: str,
    encode: Callable[[str], list[int]],
    eos_id: int,
    *,
    system: Optional[str] = None,
) -> list[int]:
    """Token ids that prime the model to answer, matching SFT's template exactly.

    Renders the user (and optional system) turn with render_chat, then appends
    the assistant-open marker so generation continues as the assistant turn.
    No BOS and no EOS: SFT encoded with add_special_tokens=False and only
    closes the assistant turn *after* the content the model must produce.
    """
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    input_ids, _ = render_chat(messages, encode, eos_id)
    return input_ids + encode(_ROLE_PREFIX["assistant"])


@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    eos_id: int,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 0.9,
    device: str = "cpu",
) -> list[int]:
    """Decode a continuation from prompt_ids; stops at eos. Returns new ids only."""
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    ctx = model.cfg.ctx
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -ctx:])[:, -1, :].float()
        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_id = sorted_ids.gather(1, torch.multinomial(sorted_probs, 1))
        if next_id.item() == eos_id:
            break
        tokens = torch.cat([tokens, next_id], dim=1)
    return tokens[0, len(prompt_ids):].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the SFT .pt")
    parser.add_argument("--out", default="results/v3/sft_samples.md")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy (reproducible)"
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer ({TOKENIZER}) ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    eos_id = tok.eos_token_id

    print(f"Loading SFT model {args.checkpoint} ...")
    model = load_from_checkpoint(args.checkpoint, device=args.device).eval()

    rows = []
    for prompt in PROMPTS:
        prompt_ids = build_prompt_ids(prompt, encode, eos_id)
        new_ids = generate(
            model,
            prompt_ids,
            eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        response = tok.decode(new_ids, skip_special_tokens=True).strip()
        print(f"\n>>> {prompt}\n{response}")
        rows.append((prompt, response))

    decode = "greedy" if args.temperature == 0.0 else f"temp={args.temperature}, top_p={args.top_p}"
    lines = [
        "# TinyLM v3 SFT — qualitative chat samples",
        "",
        f"Model: SmolTalk-SFT of E3-full (`{Path(args.checkpoint).name}`, 275M). "
        f"Decoding: {decode}, max_new_tokens={args.max_new_tokens}. "
        "Prompts primed with the training ChatML template "
        "(`<|user|>\\n...\\n<|assistant|>\\n`).",
        "",
        "These demonstrate the instruction-following the locked benchmark suite "
        "cannot measure (see `sft_vs_base.md`). Unedited model output.",
        "",
    ]
    for prompt, response in rows:
        lines.append(f"### {prompt}")
        lines.append("")
        lines.append("> " + response.replace("\n", "\n> "))
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {len(rows)} samples to {out}")


if __name__ == "__main__":
    main()
