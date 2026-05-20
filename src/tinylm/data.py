"""FineWeb-Edu shard loader for TinyLM training.

Reads pre-tokenized .npy shard files (uint16, sorted lexicographically).
Each shard contains ~100M token IDs. Only one shard is in memory at a time.
Wraps around after the last shard.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class ShardLoader:
    """Yields (batch_size, seq_len + 1) int64 token tensors from .npy shards.

    Only one shard is in memory at a time. State is checkpointable via
    state_dict / load_state_dict.
    """

    def __init__(self, shard_dir: str, batch_size: int, seq_len: int,
                 max_epochs: int | None = None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_epochs = max_epochs
        self.shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.npy files found in {shard_dir!r}")
        self.shard_idx = 0
        self.token_pos = 0
        self.epoch = 0
        self._tokens: torch.Tensor = self._load(0)

    def _load(self, idx: int) -> torch.Tensor:
        """Load shard idx from disk as int64 tensor."""
        data = np.load(self.shards[idx])
        return torch.from_numpy(data.astype(np.int64))

    def _advance_shard(self) -> None:
        """Move to the next shard, counting epochs and guarding against the
        v1 silent-repetition bug (data looping far past what was intended)."""
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)
        if self.shard_idx == 0:
            self.epoch += 1
            print(f"[ShardLoader] completed epoch {self.epoch}")
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                raise RuntimeError(
                    f"max_epochs={self.max_epochs} reached — data would loop "
                    f"further. This guards against the v1 silent-repetition bug. "
                    f"Tokenize more shards or lower total_steps."
                )
        self._tokens = self._load(self.shard_idx)
        self.token_pos = 0

    def next_batch(self) -> torch.Tensor:
        """Return (batch_size, seq_len + 1) tensor of token IDs."""
        needed = self.batch_size * (self.seq_len + 1)
        chunks: list[torch.Tensor] = []
        remaining = needed

        while remaining > 0:
            if self.token_pos >= len(self._tokens):
                self._advance_shard()

            take = min(len(self._tokens) - self.token_pos, remaining)
            chunks.append(self._tokens[self.token_pos : self.token_pos + take])
            self.token_pos += take
            remaining -= take

        # Pre-advance so the next next_batch() call always starts at a valid shard.
        if self.token_pos >= len(self._tokens):
            self._advance_shard()

        return torch.cat(chunks).view(self.batch_size, self.seq_len + 1)

    def state_dict(self) -> dict:
        """Return serializable loader position checkpoint."""
        return {
            "shard_idx": self.shard_idx,
            "token_pos": self.token_pos,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore loader position from a state_dict checkpoint."""
        self.shard_idx = state["shard_idx"]
        self.token_pos = state["token_pos"]
        self.epoch = state.get("epoch", 0)
        self._tokens = self._load(self.shard_idx)
