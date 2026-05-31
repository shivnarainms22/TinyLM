"""Memory-bounded cross-entropy.

F.cross_entropy upcasts logits to fp32 internally; on a (B*T, 32000) tensor
that transient fp32 copy is multiple GB and OOMs a 40GB A100. Chunking the
upcast bounds peak memory to chunk_size * vocab * 4 bytes while producing a
result numerically equal to F.cross_entropy (token-mean reduction).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          chunk_size: int = 4096) -> torch.Tensor:
    """Token-mean CE over (N, V) logits and (N,) targets, computed in chunks."""
    n = logits.shape[0]
    total = logits.new_zeros(())
    for i in range(0, n, chunk_size):
        total = total + F.cross_entropy(
            logits[i:i + chunk_size].float(),
            targets[i:i + chunk_size],
            reduction="sum",
        )
    return total / n
