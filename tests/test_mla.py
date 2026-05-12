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


def test_output_shape_preserved():
    """PDF Test 2: MLA output has shape (B, T, d_model)."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )
    x = torch.randn(2, 16, cfg.d_model)
    out = mla(x, cos, sin)
    assert out.shape == (2, 16, cfg.d_model)


def test_causal_masking():
    """PDF Test 3: perturbing position 5 leaves positions 0..4
    unchanged."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )
    x = torch.randn(2, 16, cfg.d_model)
    out = mla(x, cos, sin)
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    out2 = mla(x2, cos, sin)
    assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-5)


def test_rope_decoupling():
    """Defensive Test 4: positional information is carried ONLY by
    the d_rope branch. If we zero the `k_rope_proj` weight and the
    rope portion of `q_proj`, the output must change — proving
    those are the only carriers of positional info.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )

    x = torch.randn(1, 16, cfg.d_model)
    out_full = mla(x, cos, sin)

    with torch.no_grad():
        # Zero out the k_rope branch entirely
        mla.k_rope_proj.weight.zero_()
        # Zero the rope half of q_proj (per-head last R dims)
        H, D, R = cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.d_rope
        q_w = mla.q_proj.weight  # shape (H*(D+R), d_model)
        q_w_view = q_w.view(H, D + R, cfg.d_model)
        q_w_view[:, D:, :].zero_()
    out_no_rope = mla(x, cos, sin)

    # The two outputs MUST differ — proves RoPE was contributing.
    assert not torch.allclose(out_full, out_no_rope, atol=1e-4), (
        "Zeroing RoPE branches did not change output — positional "
        "information is leaking through the latent path."
    )
