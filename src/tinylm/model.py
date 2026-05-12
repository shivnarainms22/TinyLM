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
