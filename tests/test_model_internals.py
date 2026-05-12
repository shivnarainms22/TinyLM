"""Internals tests — RMSNorm, RoPE, FFN. Not part of canonical 12 but
useful while building model.py."""

import torch

from tinylm.model import ModelConfig, RMSNorm


def test_model_config_defaults_lock():
    cfg = ModelConfig()
    assert cfg.n_layers == 18
    assert cfg.d_model == 1024
    assert cfg.n_heads == 16
    assert cfg.d_latent == 512
    assert cfg.d_rope == 64
    assert cfg.ffn_hidden == 2816
    assert cfg.ctx == 2048
    assert cfg.vocab_size == 32000
    assert cfg.tie_weights is True
    assert cfg.attention in {"mha", "mla"}


def test_rmsnorm_unit_norm():
    """RMSNorm with weight=1 should produce activations with RMS ≈ 1."""
    torch.manual_seed(0)
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 3.7
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
