"""TinyLM supervised fine-tuning (SFT) module.

Entry point:  python -m tinylm.sft <configs/v3/sft_smoltalk.yaml>

WandB and datasets are imported lazily inside run_sft() so tests never
trigger those imports.

ChatML-style text template (no new special tokens — vocab_size stays 32000):

  system turn:    "<|system|>\\n" + content + "\\n"
  user turn:      "<|user|>\\n"   + content + "\\n"
  assistant turn: "<|assistant|>\\n" + content  [then eos_id token appended]

Labels: -100 everywhere except assistant content tokens and the terminating
eos_id of each assistant turn.  Callers apply the causal shift
(logits[:, :-1] vs labels[:, 1:]) with ignore_index=-100.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from typing import Callable, Generator, Optional

import torch
import torch.nn.functional as F
import yaml

from tinylm.loader import load_from_checkpoint
from tinylm.model import ModelConfig


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_ROLE_PREFIX = {
    "system":    "<|system|>\n",
    "user":      "<|user|>\n",
    "assistant": "<|assistant|>\n",
}
_ROLE_SUFFIX = {
    "system":    "\n",
    "user":      "\n",
    "assistant": "",   # no trailing newline; eos_id token is appended instead
}


def render_chat(
    messages: list[dict],
    encode: Callable[[str], list[int]],
    eos_id: int,
    *,
    roles: Optional[dict] = None,
) -> tuple[list[int], list[int]]:
    """Render a chat as token ids with prompt-loss masking.

    Args:
        messages: list of {"role": str, "content": str} dicts.
        encode:   str -> list[int] tokenizer callable.
        eos_id:   integer token id appended after each assistant turn.
        roles:    optional override dict for role prefix strings (unused by
                  default; reserved for callers that supply custom templates).

    Returns:
        (input_ids, labels) — equal-length lists; labels[i] == -100 for
        all non-assistant-content positions.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        is_assistant = (role == "assistant")

        prefix = _ROLE_PREFIX[role]
        suffix = _ROLE_SUFFIX[role]

        # Role-marker prefix — always masked
        prefix_ids = encode(prefix)
        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))

        # Content tokens
        content_ids = encode(content)
        input_ids.extend(content_ids)
        if is_assistant:
            labels.extend(content_ids)          # unmasked — train on this
        else:
            labels.extend([-100] * len(content_ids))

        # Suffix (newline for system/user) — always masked
        if suffix:
            suffix_ids = encode(suffix)
            input_ids.extend(suffix_ids)
            labels.extend([-100] * len(suffix_ids))

        # EOS token terminates assistant turns — unmasked
        if is_assistant:
            input_ids.append(eos_id)
            labels.append(eos_id)

    return input_ids, labels


# ---------------------------------------------------------------------------
# Dataset iterator
# ---------------------------------------------------------------------------

def iter_sft_examples(
    rows: object,
    encode: Callable[[str], list[int]],
    eos_id: int,
    max_len: int,
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Yield (input_ids, labels) for each row whose length <= max_len.

    Args:
        rows:    iterable of dicts with a "messages" key.
        encode:  str -> list[int] tokenizer callable.
        eos_id:  EOS token id.
        max_len: skip (don't truncate) examples longer than this.
    """
    for row in rows:
        input_ids, labels = render_chat(row["messages"], encode, eos_id)
        if len(input_ids) > max_len:
            continue
        yield input_ids, labels


# ---------------------------------------------------------------------------
# Loss helper (pure, unit-testable)
# ---------------------------------------------------------------------------

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal SFT cross-entropy loss with prompt masking.

    Args:
        logits: (B, T, V) model output.
        labels: (B, T) integer targets; -100 positions are ignored.

    The causal shift (predict token t+1 from position t) is applied here:
        logits[:, :-1]  vs  labels[:, 1:]
    """
    shift_logits = logits[:, :-1].contiguous().reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


# ---------------------------------------------------------------------------
# SFT config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    init_from: str = "checkpoints/last.pt"
    dataset: str = "HuggingFaceTB/smoltalk"
    lr: float = 2e-5
    weight_decay: float = 0.1
    epochs: int = 3
    batch_size: int = 16
    max_len: int = 2048
    warmup_ratio: float = 0.03
    save_every: int = 500
    out_dir: str = "outputs/sft"
    wandb_project: str = "tinylm-v3"


def load_sft_config(path: str) -> SFTConfig:
    """Load YAML into SFTConfig; raise ValueError on unknown keys."""
    with open(path) as f:
        d = yaml.safe_load(f)
    valid = {f.name for f in fields(SFTConfig)}
    unknown = set(d.keys()) - valid
    if unknown:
        raise ValueError(f"Unknown SFT config keys: {sorted(unknown)}")
    return SFTConfig(**d)


def apply_env_overrides(cfg: SFTConfig) -> SFTConfig:
    """Let the batch script inject cluster paths without editing YAML.

    Mirrors train.py's env-override pattern (TINYLM_INIT_FROM etc.).
    """
    if os.environ.get("TINYLM_SFT_INIT_FROM"):
        cfg.init_from = os.environ["TINYLM_SFT_INIT_FROM"]
    if os.environ.get("TINYLM_SFT_OUT"):
        cfg.out_dir = os.environ["TINYLM_SFT_OUT"]
    return cfg


# ---------------------------------------------------------------------------
# Cosine LR with warmup (mirrored from train.py)
# ---------------------------------------------------------------------------

def _cosine_lr(
    step: int,
    warmup: int,
    total: int,
    lr_max: float,
    lr_min_ratio: float = 0.1,
) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_max * (lr_min_ratio + (1.0 - lr_min_ratio) * coeff)


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (input_ids, labels) to the batch max length.

    input_ids padded with pad_id; labels padded with -100.
    """
    max_len = max(len(ids) for ids, _ in examples)
    padded_ids = []
    padded_labels = []
    for ids, labs in examples:
        pad = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad)
        padded_labels.append(labs + [-100] * pad)
    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_sft(cfg: SFTConfig) -> None:
    """Full SFT training loop.

    SmolTalk is loaded lazily so this import never runs in tests.
    """
    # Lazy imports — not available in test environments
    import datasets as hf_datasets  # noqa: F401 (lazy)
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"

    model = load_from_checkpoint(cfg.init_from, device)
    model.train()
    V = model.cfg.vocab_size

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Load SmolTalk
    ds = hf_datasets.load_dataset(cfg.dataset, "all", split="train")

    # Determine total steps for LR schedule
    # We do a full pass through the dataset per epoch; compute lazily via len
    try:
        n_rows = len(ds)
    except Exception:
        n_rows = 0  # streaming — unknown

    # Rough estimate: assume ~80% of rows pass the max_len filter
    approx_steps_per_epoch = max(1, int(n_rows * 0.8) // cfg.batch_size)
    total_steps = approx_steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    # WandB (guarded)
    _wandb = None
    try:
        import wandb as _wandb_mod
        _wandb = _wandb_mod
        _wandb.init(project=cfg.wandb_project, config=vars(cfg))
    except Exception:
        pass

    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Real Llama-2 tokenizer — the base model's embeddings are tied to this
    # locked 32000-token vocab, so SFT MUST tokenize with it (NOT byte-level).
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    EOS = tok.eos_token_id

    step = 0
    for epoch in range(cfg.epochs):
        batch: list[tuple[list[int], list[int]]] = []
        for row in ds:
            example = list(
                iter_sft_examples([row], encode, EOS, cfg.max_len)
            )
            if not example:
                continue
            batch.append(example[0])
            if len(batch) < cfg.batch_size:
                continue

            # --- training step ---
            input_ids, labels = _pad_batch(batch)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            batch = []

            lr = _cosine_lr(step, warmup_steps, total_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids)
                loss = sft_loss(logits, labels)

            if not loss.isfinite():
                raise RuntimeError(
                    f"SFT loss non-finite at step {step} — training diverged."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if _wandb is not None:
                _wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

            print(f"epoch {epoch} step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}")

            if (step + 1) % cfg.save_every == 0:
                _save_sft_checkpoint(
                    os.path.join(ckpt_dir, f"step_{step:06d}.pt"),
                    step, model,
                )

            step += 1

        # Save at epoch end
        _save_sft_checkpoint(
            os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt"),
            step, model,
        )

    # Final checkpoint
    _save_sft_checkpoint(os.path.join(ckpt_dir, "last.pt"), step, model)

    if _wandb is not None:
        _wandb.finish()


def _save_sft_checkpoint(path: str, step: int, model: object) -> None:
    """Save an SFT checkpoint compatible with load_from_checkpoint."""
    cfg = model.cfg
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "config": {
                "n_layers":    cfg.n_layers,
                "d_model":     cfg.d_model,
                "n_heads":     cfg.n_heads,
                "d_latent":    cfg.d_latent,
                "d_rope":      cfg.d_rope,
                "ffn_hidden":  cfg.ffn_hidden,
                "ctx":         cfg.ctx,
                "vocab_size":  cfg.vocab_size,
                "tie_weights": cfg.tie_weights,
                "attention":   cfg.attention,
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m tinylm.sft <config.yaml>")
        sys.exit(1)
    run_sft(apply_env_overrides(load_sft_config(sys.argv[1])))
