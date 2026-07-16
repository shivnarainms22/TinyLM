"""Memory-bounded cross-entropy.

F.cross_entropy upcasts logits to fp32 internally; on a (B*T, 32000) tensor
that transient fp32 copy is multiple GB and OOMs a 40GB A100. Chunking the
upcast bounds peak memory to chunk_size * vocab * 4 bytes while producing a
result numerically equal to F.cross_entropy (token-mean reduction).

``ignore_index`` mirrors F.cross_entropy: masked targets contribute no loss and
no gradient, and are excluded from the token-mean denominator. SFT needs this —
prompt tokens are masked with -100, and a plain sum/N average over all rows
would silently divide by the padded length.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          chunk_size: int = 4096,
                          ignore_index: int = -100) -> torch.Tensor:
    """Token-mean CE over (N, V) logits and (N,) targets, computed in chunks.

    Returns a finite 0.0 (not NaN) when every target is masked, so a fully
    masked batch cannot poison training.
    """
    n = logits.shape[0]
    total = logits.new_zeros(())
    valid = (targets != ignore_index).sum()
    for i in range(0, n, chunk_size):
        total = total + F.cross_entropy(
            logits[i:i + chunk_size].float(),
            targets[i:i + chunk_size],
            ignore_index=ignore_index,
            reduction="sum",
        )
    return total / valid.clamp(min=1)
