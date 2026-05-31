import os

import numpy as np
import pytest


def _make_shards(tmp_path, n_shards, tokens_per_shard):
    d = str(tmp_path / "shards")
    os.makedirs(d)
    for i in range(n_shards):
        np.save(os.path.join(d, f"shard_{i:04d}.npy"),
                np.zeros(tokens_per_shard, dtype=np.uint16))
    return d


def test_passes_when_data_sufficient(tmp_path):
    from tinylm.preflight import check_data_sufficiency
    d = _make_shards(tmp_path, n_shards=4, tokens_per_shard=1000)  # 4000 unique
    # processed = steps*batch*accum*seq = 10*2*2*100 = 4000; max_epochs=2 -> need 2000
    check_data_sufficiency(d, total_steps=10, batch_size=2, grad_accum_steps=2,
                           seq_len=100, max_epochs=2)  # 4000*2 >= 4000 -> OK


def test_raises_when_underprovisioned(tmp_path):
    from tinylm.preflight import check_data_sufficiency
    d = _make_shards(tmp_path, n_shards=1, tokens_per_shard=500)  # 500 unique
    with pytest.raises(ValueError, match="insufficient"):
        # processed = 100*2*2*100 = 40000; max_epochs=4 -> 500*4=2000 < 40000
        check_data_sufficiency(d, total_steps=100, batch_size=2, grad_accum_steps=2,
                               seq_len=100, max_epochs=4)
