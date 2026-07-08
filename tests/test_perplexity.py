"""TDD for tinylm.perplexity — the correctness-critical NLL/perplexity math.

Fixtures use hand-computable values: for a uniform distribution over V tokens,
the per-token NLL is ln(V), so perplexity == V exactly.
"""
import math

import torch

from tinylm.perplexity import nll_sum, perplexity


def test_uniform_vocab2_perplexity_is_2():
    # Zero logits over vocab=2 -> softmax 0.5 each -> NLL per token = ln 2.
    logits = torch.zeros(2, 2)          # [n_tokens=2, vocab=2]
    targets = torch.tensor([0, 1])
    total_nll, n = nll_sum(logits, targets)
    assert n == 2
    assert math.isclose(total_nll, 2 * math.log(2), rel_tol=1e-6)
    assert math.isclose(perplexity(total_nll, n), 2.0, rel_tol=1e-6)


def test_uniform_vocab5_perplexity_is_5():
    logits = torch.zeros(3, 5)
    targets = torch.tensor([0, 2, 4])
    total_nll, n = nll_sum(logits, targets)
    assert n == 3
    assert math.isclose(perplexity(total_nll, n), 5.0, rel_tol=1e-6)


def test_ignore_index_excludes_masked_tokens():
    logits = torch.zeros(2, 2)
    targets = torch.tensor([0, -100])   # second token masked out
    total_nll, n = nll_sum(logits, targets, ignore_index=-100)
    assert n == 1
    assert math.isclose(total_nll, math.log(2), rel_tol=1e-6)
    assert math.isclose(perplexity(total_nll, n), 2.0, rel_tol=1e-6)


def test_accepts_batched_shape():
    # [batch, seq, vocab] and [batch, seq] should flatten correctly.
    logits = torch.zeros(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)
    total_nll, n = nll_sum(logits, targets)
    assert n == 6
    assert math.isclose(perplexity(total_nll, n), 4.0, rel_tol=1e-6)


def test_perplexity_zero_tokens_raises():
    import pytest
    with pytest.raises(ValueError):
        perplexity(0.0, 0)
