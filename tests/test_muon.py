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
    """One Muon step should differ from a vanilla-momentum update.

    We construct a setup where the gradient is fixed and the momentum
    buffer is nonzero. Vanilla momentum applies `buf` directly;
    Nesterov-style applies `mu*buf + grad`. The orthogonalized update
    therefore differs.
    """
    from tinylm.muon import Muon

    torch.manual_seed(2)
    p_nesterov = torch.nn.Parameter(torch.zeros(8, 8))
    p_vanilla = torch.nn.Parameter(torch.zeros(8, 8))
    grad = torch.randn(8, 8)

    # First step: bootstrap momentum buffer.
    p_nesterov.grad = grad.clone()
    opt = Muon([p_nesterov], lr=0.02, momentum=0.95)
    opt.step()

    # Second step: with the momentum buffer populated, Nesterov path
    # diverges from a vanilla-momentum equivalent.
    p_nesterov.grad = grad.clone()
    snapshot_before = p_nesterov.detach().clone()
    opt.step()
    delta_nesterov = p_nesterov.detach() - snapshot_before

    # Sanity: the update is nontrivial (not all zeros).
    assert delta_nesterov.abs().max().item() > 1e-4
    # And not equal to a plain `-lr * orthogonalize(grad)` step
    # (that would mean Nesterov was bypassed).
    naive = -0.02 * newton_schulz(grad, steps=5)
    naive = naive * max(1, (8 / 8) ** 0.5)
    assert not torch.allclose(delta_nesterov, naive, atol=1e-3), (
        "Update matches naive grad-only step — Nesterov path is missing."
    )
