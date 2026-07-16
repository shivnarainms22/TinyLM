"""Token-level negative log-likelihood and perplexity.

Kept as a pure, dependency-light module (torch only) so it is unit-testable
independently of model loading or lm-eval. `scripts/eval_perplexity.py` is the
thin CLI that feeds real model logits through these functions.

Both models compared in the v3 code/math diagnostic share the Llama-2 tokenizer,
so per-token perplexity is directly comparable — no bits-per-byte normalization
needed (that only matters across differing tokenizers).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def nll_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[float, int]:
    """Sum of token-level NLL (in nats) and the count of scored tokens.

    Args:
        logits: shape ``[..., vocab]`` (any leading dims are flattened).
        targets: shape ``[...]`` matching ``logits`` minus the vocab dim.
        ignore_index: target value whose positions are excluded from the loss.

    Returns:
        ``(total_nll, n_tokens)`` where ``n_tokens`` counts only unmasked targets.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    total_nll = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum"
    ).item()
    n_tokens = int((flat_targets != ignore_index).sum().item())
    return total_nll, n_tokens


def perplexity(total_nll: float, n_tokens: int) -> float:
    """Perplexity = exp(mean NLL in nats). Raises if no tokens were scored."""
    if n_tokens <= 0:
        raise ValueError("perplexity undefined: zero scored tokens")
    return math.exp(total_nll / n_tokens)
