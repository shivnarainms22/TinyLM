"""TinyLM model: MHA and MLA variants in one file.

Dims locked in docs/ablation_plan.md:
  n_layers=18, d_model=1024, n_heads=16, d_latent=512, d_rope=64,
  ffn_hidden=2816, ctx=2048, tie_weights=True, vocab_size=32000.

MLAttention is adapted (and aggressively simplified) from the
DeepSeek-V2 modeling code in HuggingFace `transformers`, MIT licensed.
We drop expert routing, YARN scaling, and any feature unused by a
275M dense LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layers: int = 18
    d_model: int = 1024
    n_heads: int = 16
    d_latent: int = 512        # MLA only
    d_rope: int = 64           # MLA only — decoupled RoPE projection size
    ffn_hidden: int = 2816
    ctx: int = 2048
    vocab_size: int = 32000
    tie_weights: bool = True
    rope_base: float = 10000.0
    attention: Literal["mha", "mla"] = "mla"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embedding.

    Returns (cos, sin) each of shape (seq_len, head_dim).
    Uses split-half layout to match apply_rope below.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (T, half)
    cos = freqs.cos().repeat(1, 2)              # (T, head_dim)
    sin = freqs.sin().repeat(1, 2)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to x. x shape is (..., T, head_dim); cos/sin are
    (T, head_dim) and broadcast over leading dims."""
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN as used in Llama/TinyLlama: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MHAttention(nn.Module):
    """Standard causal multi-head attention with RoPE on Q and K.

    No biases, no dropout. Uses scaled_dot_product_attention with
    is_causal=True for the causal mask.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Slice rope cache to current T
        cos = rope_cos[:T]
        sin = rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)


class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, simplified dense).

    Ported from the DeepSeek-V2 reference (HuggingFace transformers,
    MIT licensed). Aggressively stripped: no expert routing, no YARN
    scaling, no q-LoRA. Decoupled RoPE: positional info is carried
    only by a small `d_rope`-wide projection, NOT by the full latent
    KV path. Getting this wrong yields a model that trains but has
    broken position encoding (PDF Phase 1 Step 2 callout).

    KV path:
        x -> kv_down (d_model -> d_latent) -> [LATENT, no RoPE]
        x -> k_rope_proj (d_model -> d_rope) -> [RoPE applied here]

    Q path:
        x -> q_proj (d_model -> n_heads*(d_head + d_rope))
            split into [q_nope (no RoPE), q_rope (RoPE applied)]

    Inference cache stores per-token (LATENT, K_ROPE) of total width
    `d_latent + d_rope`, NOT `n_heads * d_head`. That is the source
    of the KV-compression ratio claimed in the Phase 5 headline.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.d_latent = cfg.d_latent
        self.d_rope = cfg.d_rope

        # Q: split per head into (head_dim no-rope) + (d_rope rope)
        self.q_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * (self.head_dim + cfg.d_rope),
            bias=False,
        )
        # KV down-projection to the latent
        self.kv_down = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)
        # Latent norm (DeepSeek-V2 applies RMSNorm on the latent)
        self.kv_norm = RMSNorm(cfg.d_latent)
        # K-rope projection (single-head, broadcast across heads)
        self.k_rope_proj = nn.Linear(cfg.d_model, cfg.d_rope, bias=False)
        # Up-projections from latent to per-head K_nope and V
        self.k_up = nn.Linear(
            cfg.d_latent, cfg.n_heads * self.head_dim, bias=False
        )
        self.v_up = nn.Linear(
            cfg.d_latent, cfg.n_heads * self.head_dim, bias=False
        )
        # Output projection
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, R = self.n_heads, self.head_dim, self.d_rope

        # Q: (B, T, H*(D+R)) -> split
        q = self.q_proj(x).view(B, T, H, D + R)
        q_nope, q_rope = q.split([D, R], dim=-1)
        # (B, H, T, D) and (B, H, T, R)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)

        # KV latent + rope branch
        latent = self.kv_norm(self.kv_down(x))          # (B, T, d_latent)
        k_nope = self.k_up(latent).view(B, T, H, D).transpose(1, 2)
        v = self.v_up(latent).view(B, T, H, D).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, T, 1, R).transpose(1, 2)
        # broadcast k_rope to all heads
        k_rope = k_rope.expand(B, H, T, R)

        # Apply RoPE only on the rope branches — slice to d_rope width
        cos_r = rope_cos[:T, :R]
        sin_r = rope_sin[:T, :R]
        q_rope = apply_rope(q_rope, cos_r, sin_r)
        k_rope = apply_rope(k_rope, cos_r, sin_r)

        # Concatenate to form full Q and K of width D+R
        q_full = torch.cat([q_nope, q_rope], dim=-1)    # (B, H, T, D+R)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward for one or more new tokens.

        cache: (latent_cache, k_rope_cache) from previous step, each
            shaped (B, T_past, d_latent) and (B, T_past, d_rope), or
            None for the first call.
        pos: absolute position of the first new token.

        Returns (output, new_cache). Cache width is (d_latent + d_rope)
        per token — the source of MLA's KV-cache memory advantage.
        """
        B, T_new, _ = x.shape
        H, D, R = self.n_heads, self.head_dim, self.d_rope

        # Compact representations for new tokens
        new_latent = self.kv_norm(self.kv_down(x))   # (B, T_new, d_latent)
        new_k_rope_pre = self.k_rope_proj(x)          # (B, T_new, d_rope)

        # Append to cache
        if cache is None:
            latent_cache = new_latent
            k_rope_cache = new_k_rope_pre
        else:
            latent_prev, k_rope_prev = cache
            latent_cache = torch.cat([latent_prev, new_latent], dim=1)
            k_rope_cache = torch.cat([k_rope_prev, new_k_rope_pre], dim=1)
        T_total = latent_cache.shape[1]

        # Expand cache to per-head K_nope and V
        k_nope = self.k_up(latent_cache).view(B, T_total, H, D).transpose(1, 2)
        v = self.v_up(latent_cache).view(B, T_total, H, D).transpose(1, 2)

        # Apply RoPE to full k_rope cache at absolute positions
        cos_r = rope_cos[:T_total, :R]
        sin_r = rope_sin[:T_total, :R]
        k_rope = k_rope_cache.view(B, T_total, 1, R).transpose(1, 2)
        k_rope = k_rope.expand(B, H, T_total, R)
        k_rope = apply_rope(k_rope, cos_r, sin_r)

        # Q for new tokens only
        q = self.q_proj(x).view(B, T_new, H, D + R)
        q_nope, q_rope = q.split([D, R], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        cos_q = rope_cos[pos:pos + T_new, :R]
        sin_q = rope_sin[pos:pos + T_new, :R]
        q_rope = apply_rope(q_rope, cos_q, sin_q)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        # Causal mask: new token at abs pos `pos+i` can attend to cache
        # positions [0, pos+i] inclusive.
        attn_mask = torch.ones(
            T_new, T_total, dtype=torch.bool, device=x.device
        )
        for i in range(T_new):
            attn_mask[i, pos + i + 1:] = False

        out = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=attn_mask
        )
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        return self.o_proj(out), (latent_cache, k_rope_cache)


class Block(nn.Module):
    """Transformer block: Attn(RMSNorm(x)) + FFN(RMSNorm(x)), residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        if cfg.attention == "mla":
            self.attn = MLAttention(cfg)
        else:
            self.attn = MHAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """The full 275M model: token embed + N blocks + final norm + LM head.

    With `tie_weights=True`, the LM head shares weight with `tok_embed`."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        # RoPE cache: MLA uses d_rope, MHA uses head_dim
        rope_head_dim = (
            cfg.d_rope
            if cfg.attention == "mla"
            else cfg.d_model // cfg.n_heads
        )
        cos, sin = build_rope_cache(cfg.ctx, rope_head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        if not cfg.tie_weights:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> logits (B, T, vocab_size)."""
        h = self.tok_embed(tokens)
        for blk in self.blocks:
            h = blk(h, self.rope_cos, self.rope_sin)
        h = self.final_norm(h)
        if self.cfg.tie_weights:
            return h @ self.tok_embed.weight.T
        return self.lm_head(h)
