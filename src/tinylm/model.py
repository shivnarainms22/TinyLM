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
