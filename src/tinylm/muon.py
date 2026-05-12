"""Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Ported from Keller Jordan's reference implementation in
github.com/KellerJordan/modded-nanogpt (MIT licensed). The quintic
coefficients (3.4445, -4.7750, 2.0315) are tuned to maximize the slope
near zero so small singular values converge quickly.
"""

from __future__ import annotations

import torch


def newton_schulz(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration.

    Returns a matrix with the same shape as X whose singular values are
    pushed toward 1 (i.e. approximately orthogonal). Runs in bf16 per
    reference impl; output is cast back to float32.
    """
    assert X.ndim >= 2, f"expected ndim>=2, got shape {tuple(X.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    Y = X.bfloat16()
    transposed = False
    if Y.size(-2) > Y.size(-1):
        Y = Y.mT
        transposed = True
    Y = Y / (Y.norm() + 1e-7)
    for _ in range(steps):
        A = Y @ Y.mT
        Y = (a * Y) + (b * (A @ Y)) + (c * (A @ A @ Y))
    if transposed:
        Y = Y.mT
    return Y.to(torch.float32)
