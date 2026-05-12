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
