"""Tests for Muon optimizer and Newton-Schulz orthogonalization."""

import torch

from tinylm.muon import newton_schulz


def test_newton_schulz_orthogonalizes():
    """NS pushes singular values toward 1 and dramatically reduces sv spread.

    For a random (256, 128) matrix, singular values start in [4.7, 27.7].
    After 5 NS steps they should land in roughly [0.5, 1.5] — not perfect
    orthogonality, but sufficient for the Muon optimizer use case. The off-
    diagonal elements of Y.T @ Y (cross-column correlations) should be small.
    """
    torch.manual_seed(0)
    X = torch.randn(256, 128)
    Y = newton_schulz(X, steps=5)
    assert Y.shape == X.shape

    svs = torch.linalg.svdvals(Y.float())
    assert svs.min().item() > 0.5, (
        f"Smallest singular value {svs.min():.3f} < 0.5 — NS failed to "
        f"push small SVs toward 1"
    )
    assert svs.max().item() < 1.5, (
        f"Largest singular value {svs.max():.3f} > 1.5 — NS failed to "
        f"push large SVs toward 1"
    )
    # Verify off-diagonal columns are uncorrelated (orthogonal subspace).
    gram = Y.T @ Y
    off_diag = gram - gram.diag().diag()
    assert off_diag.abs().max().item() < 0.15, (
        f"Off-diagonal max {off_diag.abs().max():.3f} — columns not orthogonal"
    )


def test_non_square_transpose_trick():
    """NS works for both tall (m>n) and wide (m<n) matrices."""
    torch.manual_seed(1)
    tall = torch.randn(512, 64)
    wide = torch.randn(64, 512)

    out_tall = newton_schulz(tall, steps=5)
    out_wide = newton_schulz(wide, steps=5)

    assert out_tall.shape == tall.shape, (
        f"tall: expected {tall.shape}, got {out_tall.shape}"
    )
    assert out_wide.shape == wide.shape, (
        f"wide: expected {wide.shape}, got {out_wide.shape}"
    )
    # Both should cluster singular values toward 1
    svs_tall = torch.linalg.svdvals(out_tall.float())
    svs_wide = torch.linalg.svdvals(out_wide.float())
    assert svs_tall.min().item() > 0.5 and svs_tall.max().item() < 1.5
    assert svs_wide.min().item() > 0.5 and svs_wide.max().item() < 1.5


def test_nesterov_momentum_applied():
    """Muon's step 2 direction incorporates the step-1 gradient via momentum.

    NS normalizes its input direction, so a FIXED gradient gives the same
    update at every step regardless of momentum scaling. To expose the
    Nesterov effect we use two DIFFERENT gradients: the step-2 update must
    mix in the step-1 gradient direction (through mu²*g1 in the Nesterov
    direction mu*buf + g2), producing a different result from a fresh
    single-step optimizer that has no accumulated momentum.
    """
    from tinylm.muon import Muon

    torch.manual_seed(2)
    g1 = torch.randn(8, 8)
    g2 = torch.randn(8, 8)  # different direction from g1

    # Train for 2 steps with different gradients.
    p = torch.nn.Parameter(torch.zeros(8, 8))
    opt = Muon([p], lr=0.02, momentum=0.95)
    p.grad = g1.clone()
    opt.step()
    snap = p.detach().clone()
    p.grad = g2.clone()
    opt.step()
    delta_with_momentum = p.detach() - snap  # step-2 update

    # Fresh single-step optimizer on only g2 (no accumulated momentum).
    p_fresh = torch.nn.Parameter(snap.clone())
    opt_fresh = Muon([p_fresh], lr=0.02, momentum=0.95)
    p_fresh.grad = g2.clone()
    opt_fresh.step()
    delta_no_momentum = p_fresh.detach() - snap

    # The two updates must differ because buf in the trained opt carries g1.
    assert delta_with_momentum.abs().max().item() > 1e-4, "Update is zero"
    assert not torch.allclose(delta_with_momentum, delta_no_momentum, atol=1e-4), (
        "Step-2 update is identical to a fresh single-step update — "
        "momentum buffer is not influencing the Nesterov direction."
    )


def test_param_filter_excludes_embed_lm_head_norm_bias():
    """`partition_params` must route 1D / embedding / LM head / norm
    weights into the AdamW group, not the Muon group.

    Muon on the vocab embedding destroys learned token geometry (per
    PDF Phase 1 Step 4). This filter is the guardrail.
    """
    from tinylm.muon import partition_params

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embed = torch.nn.Embedding(100, 16)
            self.attn_q = torch.nn.Linear(16, 16, bias=False)
            self.norm = torch.nn.LayerNorm(16)
            self.lm_head = torch.nn.Linear(16, 100, bias=True)

    model = Toy()
    matrix_group, scalar_group = partition_params(model)

    matrix_ids = {id(p) for p in matrix_group}
    scalar_ids = {id(p) for p in scalar_group}

    # Only attn_q.weight should be in the Muon group
    assert id(model.attn_q.weight) in matrix_ids
    # Everything else must NOT be in the Muon group
    assert id(model.tok_embed.weight) in scalar_ids
    assert id(model.lm_head.weight) in scalar_ids
    assert id(model.lm_head.bias) in scalar_ids
    assert id(model.norm.weight) in scalar_ids
    assert id(model.norm.bias) in scalar_ids
    # No overlap, no leaks
    assert matrix_ids.isdisjoint(scalar_ids)
    total_params = sum(1 for _ in model.parameters())
    assert len(matrix_ids) + len(scalar_ids) == total_params
