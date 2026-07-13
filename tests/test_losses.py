import torch
import torch.nn.functional as F


def test_chunked_ce_matches_reference():
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(0)
    logits = torch.randn(257, 1000)        # N not divisible by chunk
    targets = torch.randint(0, 1000, (257,))
    ref = F.cross_entropy(logits, targets)
    got = chunked_cross_entropy(logits, targets, chunk_size=64)
    assert torch.allclose(got, ref, atol=1e-5), f"{got} vs {ref}"


def test_chunked_ce_single_chunk_equals_reference():
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(1)
    logits = torch.randn(32, 50)
    targets = torch.randint(0, 50, (32,))
    assert torch.allclose(
        chunked_cross_entropy(logits, targets, chunk_size=999),
        F.cross_entropy(logits, targets), atol=1e-6)


def test_chunked_ce_ignores_masked_targets():
    """SFT masks prompt tokens with -100; those must not contribute to the loss
    and must not inflate the denominator."""
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(2)
    logits = torch.randn(300, 1000)
    targets = torch.randint(0, 1000, (300,))
    targets[:120] = -100                    # mask a chunk-straddling prefix
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    got = chunked_cross_entropy(logits, targets, chunk_size=64, ignore_index=-100)
    assert torch.allclose(got, ref, atol=1e-5), f"{got} vs {ref}"


def test_chunked_ce_fully_masked_chunk_does_not_poison_loss():
    """A chunk in which every target is masked must contribute nothing — not NaN."""
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(3)
    logits = torch.randn(128, 500)
    targets = torch.randint(0, 500, (128,))
    targets[:64] = -100                     # first full chunk entirely masked
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    got = chunked_cross_entropy(logits, targets, chunk_size=64, ignore_index=-100)
    assert torch.isfinite(got), "fully-masked chunk produced a non-finite loss"
    assert torch.allclose(got, ref, atol=1e-5), f"{got} vs {ref}"


def test_chunked_ce_all_masked_returns_zero():
    """All targets masked -> a finite zero, so training never sees a NaN loss."""
    from tinylm.losses import chunked_cross_entropy
    logits = torch.randn(32, 50)
    targets = torch.full((32,), -100)
    got = chunked_cross_entropy(logits, targets, chunk_size=16, ignore_index=-100)
    assert torch.isfinite(got) and got.item() == 0.0, f"expected finite 0.0, got {got}"


def test_chunked_ce_backward_flows_through_unmasked_only():
    """Gradient reaches unmasked rows and is exactly zero on masked rows."""
    from tinylm.losses import chunked_cross_entropy
    torch.manual_seed(4)
    logits = torch.randn(8, 20, requires_grad=True)
    targets = torch.randint(0, 20, (8,))
    targets[:5] = -100
    chunked_cross_entropy(logits, targets, chunk_size=4, ignore_index=-100).backward()
    assert logits.grad[:5].abs().sum().item() == 0.0, "masked rows received gradient"
    assert logits.grad[5:].abs().sum().item() > 0.0, "unmasked rows got no gradient"
