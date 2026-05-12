"""Tests for ShardLoader: batch shape, shard wrapping, state-dict resume."""

import os
import tempfile

import numpy as np
import pytest
import torch


def _make_shards(tmp_path, n_shards: int, tokens_per_shard: int, vocab_size: int = 128):
    """Write n_shards .npy files of uint16 token IDs."""
    shard_dir = str(tmp_path / "shards")
    os.makedirs(shard_dir)
    rng = np.random.default_rng(0)
    for i in range(n_shards):
        data = rng.integers(0, vocab_size, tokens_per_shard, dtype=np.uint16)
        np.save(os.path.join(shard_dir, f"shard_{i:04d}.npy"), data)
    return shard_dir


def test_batch_shape(tmp_path):
    """next_batch() must return (batch_size, seq_len + 1) int64 tensor."""
    from tinylm.data import ShardLoader

    shard_dir = _make_shards(tmp_path, n_shards=2, tokens_per_shard=500_000)
    loader = ShardLoader(shard_dir, batch_size=4, seq_len=32)
    batch = loader.next_batch()

    assert batch.shape == (4, 33), f"Expected (4, 33), got {batch.shape}"
    assert batch.dtype == torch.int64, f"Expected int64, got {batch.dtype}"


def test_shard_wrapping(tmp_path):
    """After exhausting both shards, loader wraps to shard 0 position 0.

    tokens_per_shard (90) is a multiple of tokens_per_batch (18) so the
    wrap boundary falls exactly at position 0 of shard 0.
    """
    from tinylm.data import ShardLoader

    batch_size = 2
    seq_len = 8
    tokens_per_shard = 90  # 5 batches per shard; total = 180 = exactly 10 batches

    shard_dir = _make_shards(tmp_path, n_shards=2, tokens_per_shard=tokens_per_shard)
    loader = ShardLoader(shard_dir, batch_size=batch_size, seq_len=seq_len)

    first_batch = loader.next_batch().clone()

    # Exhaust both shards (9 more batches = 162 tokens) — total 10 batches = 180 tokens
    tokens_per_batch = batch_size * (seq_len + 1)  # = 18
    total_tokens = tokens_per_shard * 2             # = 180
    assert total_tokens % tokens_per_batch == 0, "tokens_per_shard must be multiple of tokens_per_batch"
    steps_to_exhaust = total_tokens // tokens_per_batch - 1  # = 9
    for _ in range(steps_to_exhaust):
        loader.next_batch()

    # Next batch wraps to shard 0 position 0.
    wrapped_batch = loader.next_batch()
    assert wrapped_batch[0, 0].item() == first_batch[0, 0].item(), (
        f"After wrapping, first token {wrapped_batch[0, 0].item()} != "
        f"original first token {first_batch[0, 0].item()}"
    )


def test_state_dict_round_trip(tmp_path):
    """state_dict() captures exact position; load_state_dict() resumes there.

    After 3 batches, save state, create a fresh loader, restore state, and
    verify the next batch is identical to what the original loader would yield.
    """
    from tinylm.data import ShardLoader

    shard_dir = _make_shards(tmp_path, n_shards=2, tokens_per_shard=500_000)
    loader_a = ShardLoader(shard_dir, batch_size=4, seq_len=32)

    for _ in range(3):
        loader_a.next_batch()
    state = loader_a.state_dict()
    expected_next = loader_a.next_batch().clone()

    # Fresh loader, restore state.
    loader_b = ShardLoader(shard_dir, batch_size=4, seq_len=32)
    loader_b.load_state_dict(state)
    actual_next = loader_b.next_batch()

    assert torch.equal(expected_next, actual_next), (
        "Batch after state_dict round-trip does not match original sequence"
    )
