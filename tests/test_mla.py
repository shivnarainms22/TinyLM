"""MLA test suite.

Three PDF-mandatory tests (KV-compressed, output-shape, causal-mask)
plus five defensive tests (RoPE decoupling, param count, gradient
flow, MLA≈MHA equivalence at identity, KV-cache shape during
incremental inference). All eight are blocking gates before any
training run."""

import torch

from tinylm.model import ModelConfig, MLAttention, build_rope_cache


def _small_cfg(**overrides) -> ModelConfig:
    """Small config that keeps MLA semantics but runs fast on CPU."""
    base = dict(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_kv_latent_compressed():
    """PDF Test 1: kv_down output dim equals d_latent, not n_heads*d_head."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    kv_latent = mla.kv_down(x)
    assert kv_latent.shape[-1] == cfg.d_latent
    assert kv_latent.shape[-1] != cfg.n_heads * (cfg.d_model // cfg.n_heads)
