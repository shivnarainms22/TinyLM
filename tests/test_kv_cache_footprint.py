"""Tests for the KV-cache footprint computation.

This pins the numbers behind the Phase 5 KV-reduction hypothesis claim so the
figure in results/kv_cache_reduction.md and the model cards can't silently drift
from the architecture. The cache *width* is separately verified end-to-end by
test_mla.py::test_kv_cache_shape_incremental; this checks the arithmetic that
turns that width into the headline ratio and memory footprint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tinylm.model import ModelConfig  # noqa: E402
from kv_cache_footprint import (  # noqa: E402
    per_token_per_layer_floats,
    kv_cache_bytes,
    reduction_ratio,
)


def test_per_token_widths_match_the_architecture():
    """MHA caches full K+V (2*d_model); MLA caches latent + rope (d_latent+d_rope)."""
    cfg = ModelConfig()  # locked defaults: d_model 1024, heads 16, latent 512, rope 64
    mha, mla = per_token_per_layer_floats(cfg)
    assert mha == 2048          # 2 * 16 * 64
    assert mla == 576           # 512 + 64


def test_reduction_ratio_is_3_56x():
    cfg = ModelConfig()
    assert reduction_ratio(cfg) == 2048 / 576
    assert round(reduction_ratio(cfg), 2) == 3.56


def test_full_context_footprint_bf16():
    """B=1, full 2048-ctx, 18 layers, bf16 (2 bytes): 144.0 MiB -> 40.5 MiB."""
    cfg = ModelConfig()
    mha_bytes, mla_bytes = kv_cache_bytes(cfg, seq_len=2048, batch=1, dtype_bytes=2)
    assert mha_bytes == 2048 * 18 * 2048 * 2   # 150,994,944
    assert mla_bytes == 576 * 18 * 2048 * 2    # 42,467,328
    assert round(mha_bytes / 1024 / 1024, 1) == 144.0
    assert round(mla_bytes / 1024 / 1024, 1) == 40.5


def test_ratio_is_config_driven_not_hardcoded():
    """A config with no compression (latent=d_model, rope=head_dim) approaches 1x-ish,
    proving the number tracks the architecture rather than a baked-in constant."""
    cfg = ModelConfig(d_latent=1024, d_rope=64)
    mha, mla = per_token_per_layer_floats(cfg)
    assert mla == 1088
    assert reduction_ratio(cfg) == 2048 / 1088
