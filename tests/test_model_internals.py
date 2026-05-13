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


from tinylm.model import MHAttention


def test_mha_shape_and_causal():
    """MHA preserves shape and respects causal masking."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=4, ctx=16)
    attn = MHAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=16, head_dim=cfg.d_model // cfg.n_heads, base=cfg.rope_base
    )
    x = torch.randn(2, 16, 64)
    out = attn(x, cos, sin)
    assert out.shape == (2, 16, 64)

    # Causal: perturb position 5; positions 0..4 unchanged.
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    out2 = attn(x2, cos, sin)
    assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-5)


from tinylm import TinyLM


def test_tinylm_forward_smoke():
    """TinyLM produces logits of shape (B, T, vocab_size)."""
    cfg = ModelConfig(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    model = TinyLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)


import math
import torch.nn.functional as F


def _small_cfg(attention: str) -> ModelConfig:
    return ModelConfig(
        n_layers=4, d_model=128, n_heads=4, d_latent=64, d_rope=16,
        ffn_hidden=256, ctx=64, vocab_size=1024, attention=attention,
    )


def _init_loss(cfg: ModelConfig, seed: int = 0) -> float:
    """CE loss of an untrained model on random tokens with INDEPENDENT
    random labels. Independent labels are required: with tied embeddings,
    using input tokens as their own targets lets the residual stream
    leak an identity bias and the test under-reports loss.
    """
    torch.manual_seed(seed)
    model = TinyLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (4, 32))
    labels = torch.randint(0, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        logits = model(tokens)
    return F.cross_entropy(
        logits.view(-1, cfg.vocab_size), labels.view(-1)
    ).item()


def test_init_loss_near_uniform_mla():
    """Untrained MLA model on random tokens with random labels must give
    CE ≈ ln(vocab_size). Step 0 loss should match a uniform-output prior.
    Caught the N(0,1) embedding default + missing residual scaling.
    """
    cfg = _small_cfg("mla")
    loss = _init_loss(cfg)
    expected = math.log(cfg.vocab_size)  # ≈ 6.93 for vocab=1024
    assert abs(loss - expected) < 0.5, (
        f"init loss {loss:.3f} far from ln(vocab)={expected:.3f}; "
        f"check model init (embedding/linear std, residual scaling)"
    )


def test_init_loss_near_uniform_mha():
    """Same check for the MHA variant (covers run_A / run_C ablations)."""
    cfg = _small_cfg("mha")
    loss = _init_loss(cfg)
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.5, (
        f"init loss {loss:.3f} far from ln(vocab)={expected:.3f}"
    )


def test_init_logits_bounded():
    """Initial logits must have O(1) std, not O(100).

    Catches the case where embeddings or LM-head weights have unit-scale
    init: logits = h @ embed.T then has std ~ sqrt(d_model), softmax is
    razor-sharp on a random token, and CE blows up.
    """
    torch.manual_seed(0)
    cfg = _small_cfg("mla")
    model = TinyLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.std().item() < 3.0, (
        f"initial logits std {logits.std().item():.2f} too large; "
        f"embedding/LM-head init is likely unit-scale instead of ~0.02"
    )
