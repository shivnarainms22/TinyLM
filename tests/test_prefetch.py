import os

import numpy as np
import torch


def _make_shards(tmp_path, n_shards=2, tokens_per_shard=5000):
    d = str(tmp_path / "shards")
    os.makedirs(d)
    rng = np.random.default_rng(7)
    for i in range(n_shards):
        np.save(os.path.join(d, f"shard_{i:04d}.npy"),
                rng.integers(0, 128, tokens_per_shard, dtype=np.uint16))
    return d


def test_prefetch_yields_same_sequence_as_sync(tmp_path):
    from tinylm.data import ShardLoader, PrefetchLoader
    d = _make_shards(tmp_path)
    sync = ShardLoader(d, batch_size=2, seq_len=16)
    pf = PrefetchLoader(ShardLoader(d, batch_size=2, seq_len=16))
    try:
        for _ in range(20):
            assert torch.equal(sync.next_batch(), pf.next_batch())
    finally:
        pf.close()


def test_prefetch_state_matches_consumed_position(tmp_path):
    from tinylm.data import ShardLoader, PrefetchLoader
    d = _make_shards(tmp_path)
    sync = ShardLoader(d, batch_size=2, seq_len=16)
    pf = PrefetchLoader(ShardLoader(d, batch_size=2, seq_len=16))
    try:
        for _ in range(7):
            sync.next_batch(); pf.next_batch()
        assert pf.state_dict() == sync.state_dict()
    finally:
        pf.close()
