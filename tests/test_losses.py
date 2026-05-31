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
