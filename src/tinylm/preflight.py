"""Pre-flight data-sufficiency check.

Prevents the v1 bug where the config implied far more tokens than were
tokenized, causing silent data repetition. Counts real shard sizes (cheap,
via mmap headers) and asserts the data can supply the run within max_epochs.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def count_shard_tokens(shard_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npy in {shard_dir!r}")
    return sum(int(np.load(p, mmap_mode="r").shape[0]) for p in paths)


def check_data_sufficiency(shard_dir: str, total_steps: int, batch_size: int,
                           grad_accum_steps: int, seq_len: int,
                           max_epochs: int) -> None:
    unique = count_shard_tokens(shard_dir)
    processed = total_steps * batch_size * grad_accum_steps * seq_len
    available = unique * max_epochs
    print(f"[preflight] unique={unique:,} tokens | processed={processed:,} | "
          f"max_epochs={max_epochs} | available={available:,}")
    if available < processed:
        raise ValueError(
            f"Data insufficient: need {processed:,} processed tokens but only "
            f"{unique:,} unique x {max_epochs} epochs = {available:,} available. "
            f"Tokenize more shards or lower total_steps."
        )
