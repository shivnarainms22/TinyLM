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


def partition_params(model: torch.nn.Module):
    """Split parameters into (muon_group, adamw_group).

    Muon group: 2D+ matrix weights in core transformer blocks (Q/K/V/O
    projections, FFN). AdamW group: everything else — embeddings, LM
    head, all norms, all biases.

    Returns (matrix_params, scalar_params) as lists.
    """
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        lname = name.lower()
        is_excluded = (
            p.ndim < 2
            or "embed" in lname
            or "lm_head" in lname
            or "norm" in lname
            or lname.endswith(".bias")
        )
        if is_excluded:
            scalar_params.append(p)
        else:
            matrix_params.append(p)
    return matrix_params, scalar_params


class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.

    Applies Nesterov-style momentum BEFORE orthogonalization. Only
    matrix-shaped parameters (ndim >= 2) are orthogonalized; the caller
    is responsible for filtering out embeddings, LM head, layer norms,
    and biases (use `partition_params()` from `tinylm.model`).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mu = group["momentum"]
            ns_steps = group["ns_steps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # Update momentum buffer first: buf = mu*buf + g
                buf.mul_(mu).add_(g)
                # Nesterov look-ahead: v = mu*buf_orth + g
                # Orthogonalize the momentum buffer for numerical stability
                if buf.ndim >= 2:
                    buf_orth = newton_schulz(buf, steps=ns_steps)
                    m, n = p.shape[-2], p.shape[-1]
                    buf_orth = buf_orth * max(
                        1.0, (max(m, n) / min(m, n)) ** 0.5
                    )
                    nesterov = mu * buf_orth + g
                else:
                    # For vectors/scalars, skip orthogonalization
                    nesterov = buf
                p.add_(nesterov, alpha=-lr)
