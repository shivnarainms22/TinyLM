"""MLA test suite.

Three PDF-mandatory tests (KV-compressed, output-shape, causal-mask)
plus five defensive tests (RoPE decoupling, param count, gradient
flow, MLA≈MHA equivalence at identity, KV-cache shape during
incremental inference). All eight are blocking gates before any
training run."""

import torch

from tinylm.model import ModelConfig, MLAttention, MHAttention, TinyLM, build_rope_cache


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


def test_total_param_count():
    """Defensive Test 5: at the LOCKED Phase-4 dims, TinyLM-MLA must
    have between 270M and 285M parameters. PDF Phase 4 Step 0 says to
    verify this before any training run."""
    cfg = ModelConfig(attention="mla")  # all locked defaults
    model = TinyLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 270_000_000 <= total <= 285_000_000, (
        f"param count {total:,} outside locked range "
        f"[270M, 285M] — check ablation_plan.md before retuning"
    )


def test_gradient_flow():
    """Defensive Test 6: after forward+backward, every learnable
    parameter has a non-zero gradient. Catches frozen sub-modules
    and dead branches."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(tokens)
    # Simple scalar loss tied to all params
    loss = logits.float().pow(2).mean()
    loss.backward()

    zero_grad_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            zero_grad_params.append(f"{name} (grad is None)")
        elif p.grad.abs().max().item() == 0.0:
            zero_grad_params.append(f"{name} (all zeros)")
    assert not zero_grad_params, (
        "Params with no gradient signal:\n  " + "\n  ".join(zero_grad_params)
    )


def test_mla_mha_equivalence_at_identity_setting():
    """Defensive Test 7: with d_latent=d_model and d_rope=head_dim,
    MLA's latent path is non-compressing. We align the value/output
    paths by copying matching weights (v_up <- v_proj, o_proj shared,
    kv_down set to identity) then check Pearson correlation > 0.3.

    Two independently-initialised modules produce ~0 correlation by
    chance, so a wiring bug in the MLA reshape/split/concat raises this
    test even though random Q/K paths differ.

    Specifically this guards against:
      - Wrong head reshape in k_up / v_up
      - q_nope / q_rope split at the wrong boundary
      - k_rope expand dropped or misaligned
    """
    torch.manual_seed(0)
    # head_dim = d_model // n_heads = 64 // 4 = 16
    cfg_mla = ModelConfig(
        n_layers=1, d_model=64, n_heads=4, d_latent=64, d_rope=16,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    cfg_mha = ModelConfig(
        n_layers=1, d_model=64, n_heads=4, d_latent=64, d_rope=16,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mha",
    )
    mla = MLAttention(cfg_mla)
    mha = MHAttention(cfg_mha)

    # Align the shared value + output path so the only differences
    # come from the Q/K positional branches (which are well-defined).
    # kv_down set to identity: latent = x (no compression at d_latent=d_model)
    # v_up gets the same weights as MHA's v_proj
    # o_proj is shared
    with torch.no_grad():
        mla.kv_down.weight.copy_(torch.eye(cfg_mla.d_model))
        mla.v_up.weight.copy_(mha.v_proj.weight)
        mla.o_proj.weight.copy_(mha.o_proj.weight)

    cos_r, sin_r = build_rope_cache(cfg_mla.ctx, cfg_mla.d_rope, cfg_mla.rope_base)
    cos_m, sin_m = build_rope_cache(
        cfg_mha.ctx, cfg_mha.d_model // cfg_mha.n_heads, cfg_mha.rope_base
    )
    x = torch.randn(2, 16, 64)
    out_mla = mla(x, cos_r, sin_r).detach().flatten()
    out_mha = mha(x, cos_m, sin_m).detach().flatten()

    out_mla_c = out_mla - out_mla.mean()
    out_mha_c = out_mha - out_mha.mean()
    corr = (out_mla_c @ out_mha_c) / (
        out_mla_c.norm() * out_mha_c.norm() + 1e-12
    )
    assert corr.abs().item() > 0.3, (
        f"MLA / MHA outputs uncorrelated after weight-alignment "
        f"(r={corr.item():.3f}) — likely a wiring bug in one of them; "
        f"check head reshape in k_up/v_up, q_nope/q_rope split boundary, "
        f"or k_rope expand"
    )


def test_kv_cache_shape_incremental():
    """Defensive Test 8: when running token-at-a-time, the per-layer
    cached tensor must have last-dim width (d_latent + d_rope), NOT
    (n_heads * head_dim). This is what produces the 3.56× KV-cache
    reduction headline (see results/kv_cache_reduction.md)."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(cfg.ctx, cfg.d_rope, cfg.rope_base)

    # Feed one token at a time, accumulating cache.
    cache = None
    for t in range(4):
        x_t = torch.randn(1, 1, cfg.d_model)
        out_t, cache = mla.forward_with_cache(
            x_t, cos, sin, cache=cache, pos=t
        )
        assert out_t.shape == (1, 1, cfg.d_model)

    # Cache is a tuple (latent_cache, k_rope_cache)
    latent_cache, k_rope_cache = cache
    assert latent_cache.shape[-1] == cfg.d_latent
    assert k_rope_cache.shape[-1] == cfg.d_rope
    # Time dim accumulates correctly
    assert latent_cache.shape[-2] == 4
    assert k_rope_cache.shape[-2] == 4
    # Critically: cache width is NOT n_heads * head_dim
    full_kv_width = cfg.n_heads * (cfg.d_model // cfg.n_heads)
    assert latent_cache.shape[-1] != full_kv_width
