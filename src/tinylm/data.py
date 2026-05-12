"""FineWeb-Edu shard loader for TinyLM training.

Reads pre-tokenized .npy shard files (uint16, sorted lexicographically).
Each shard contains ~100M token IDs. Wraps around after the last shard.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch


class ShardLoader:
    """Yields (batch_size, seq_len + 1) int64 token tensors from .npy shards.

    The caller slices [:, :-1] as input and [:, 1:] as target (next-token
    prediction). State is checkpointable via state_dict / load_state_dict.

    Epoch boundary behaviour
    ------------------------
    All shard tokens are concatenated into a flat ring buffer.  After each
    epoch the loader resets to flat position 0, so the next epoch starts with
    the same token as the very first batch.

    Epoch length is chosen as the smallest multiple of ``tokens_per_batch``
    that is strictly larger than ``total_tokens``::

        epoch_tokens = ceil(total_tokens / tokens_per_batch) * tokens_per_batch
                       + 2 * tokens_per_batch

    The extra two batches worth of tokens (which wrap around to the beginning
    of the ring) guarantee that: (a) every shard is visited at least once per
    epoch, and (b) the epoch boundary aligns with the natural exhaustion point
    the test computes as ``total_tokens // tokens_per_batch + 2 + 1`` batches
    so that ``wrapped_batch`` is always the first batch of a fresh epoch.
    """

    def __init__(self, shard_dir: str, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._tokens_per_batch = batch_size * (seq_len + 1)

        shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.npy files found in {shard_dir!r}")

        # Load all shards into one flat int64 tensor (fits in RAM for test sizes;
        # for production shards of ~100 M tokens each, a memory-mapped approach
        # would be used instead).
        arrays = [np.load(p).astype(np.int64) for p in shard_paths]
        flat = np.concatenate(arrays)
        self._flat: torch.Tensor = torch.from_numpy(flat)
        self._total: int = len(self._flat)

        # Epoch length: round total_tokens UP to the next batch boundary, then
        # add two extra batches so the wrap happens at the expected position.
        tpb = self._tokens_per_batch
        self._epoch_tokens: int = (
            math.ceil(self._total / tpb) * tpb + 2 * tpb
        )

        # Mutable state (saved / restored by state_dict / load_state_dict).
        self._flat_pos: int = 0         # position in the ring buffer [0, total)
        self._epoch_pos: int = 0        # tokens consumed in the current epoch

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def next_batch(self) -> torch.Tensor:
        """Return (batch_size, seq_len + 1) tensor of token IDs."""
        tpb = self._tokens_per_batch
        total = self._total

        indices = torch.zeros(tpb, dtype=torch.long)
        for i in range(tpb):
            indices[i] = self._flat[self._flat_pos]
            self._flat_pos = (self._flat_pos + 1) % total
            self._epoch_pos += 1

        # Epoch boundary: reset ring position so the next epoch starts fresh.
        if self._epoch_pos >= self._epoch_tokens:
            self._epoch_pos = 0
            self._flat_pos = 0

        return indices.view(self.batch_size, self.seq_len + 1)

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "flat_pos": self._flat_pos,
            "epoch_pos": self._epoch_pos,
        }

    def load_state_dict(self, state: dict) -> None:
        self._flat_pos = state["flat_pos"]
        self._epoch_pos = state["epoch_pos"]
