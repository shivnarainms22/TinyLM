"""Internals tests — RMSNorm, RoPE, FFN. Not part of canonical 12 but
useful while building model.py."""

import torch

from tinylm.model import (
    ModelConfig,
    RMSNorm,
    apply_rope,
    build_rope_cache,
    SwiGLUFFN,
)


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


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=128, head_dim=64, base=10000.0)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_apply_rope_preserves_norm():
    """RoPE is a rotation: it must preserve the L2 norm of each token."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 64)  # (B, H, T, head_dim)
    cos, sin = build_rope_cache(seq_len=8, head_dim=64, base=10000.0)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(
        x.pow(2).sum(dim=-1), y.pow(2).sum(dim=-1), atol=1e-5
    )


def test_apply_rope_position_zero_is_identity():
    """At position 0, cos=1 and sin=0, so RoPE should be a no-op."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, 32)
    cos, sin = build_rope_cache(seq_len=1, head_dim=32, base=10000.0)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x, y, atol=1e-6)


def test_swiglu_ffn_shape_and_no_bias():
    ffn = SwiGLUFFN(d_model=64, ffn_hidden=176)
    x = torch.randn(2, 8, 64)
    y = ffn(x)
    assert y.shape == (2, 8, 64)
    for name, p in ffn.named_parameters():
        assert "bias" not in name, f"unexpected bias param: {name}"
