#!/usr/bin/env python3
"""Generate text from a TinyLM checkpoint.

Usage:
    # Download checkpoint from HF automatically:
    python scripts/generate.py --prompt "The theory of relativity states that"

    # Interactive mode:
    python scripts/generate.py

    # Local checkpoint:
    python scripts/generate.py --checkpoint checkpoints/step_19999.pt

    # Greedy decoding:
    python scripts/generate.py --prompt "Once upon a time" --temperature 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from tinylm.model import ModelConfig, TinyLM
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from tinylm.model import ModelConfig, TinyLM

HF_CHECKPOINT_REPO = "Shiv-22/tinylm-checkpoints"
HF_CHECKPOINT_FILE = "step_19999.pt"
TOKENIZER = "meta-llama/Llama-2-7b-hf"


def load_model(checkpoint: str | None, device: str) -> TinyLM:
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        print(f"Downloading checkpoint from {HF_CHECKPOINT_REPO}...")
        checkpoint = hf_hub_download(
            repo_id=HF_CHECKPOINT_REPO, filename=HF_CHECKPOINT_FILE
        )
    print(f"Loading {checkpoint} ...")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
    return model.to(device).eval()


@torch.no_grad()
def generate(
    model: TinyLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids = bos + tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        inp = tokens[:, -model.cfg.ctx:]
        logits = model(inp)[:, -1, :].float()  # (1, vocab)

        if temperature == 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
            cumsum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            sample_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_ids.gather(1, sample_idx)

        tokens = torch.cat([tokens, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break

    generated = tokens[0, len(ids):].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to local .pt checkpoint (default: download from HF)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt text (omit for interactive mode)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling probability threshold")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer ({TOKENIZER}) ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = load_model(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Ready — {n_params / 1e6:.0f}M params on {args.device}\n")

    def run(prompt: str) -> None:
        out = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        print(f"[prompt]  {prompt}")
        print(f"[output]  {out}\n")

    if args.prompt:
        run(args.prompt)
    else:
        print("Interactive mode — enter a prompt and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(">>> ").strip()
                if prompt:
                    run(prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


if __name__ == "__main__":
    main()
