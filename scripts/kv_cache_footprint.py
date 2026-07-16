#!/usr/bin/env python3
"""Compute the MLA-vs-MHA KV-cache footprint for a TinyLM config.

Closes the KV-reduction half of the pinned hypothesis with a reproducible number
(see results/kv_cache_reduction.md). MLA caches a compressed latent
(`d_latent + d_rope`) per token per layer; an equivalent MHA (Run A) caches full
K and V (`2 * n_heads * head_dim = 2 * d_model`). The ratio is exact and
deterministic from the config — this just does the arithmetic and prints it.

    python scripts/kv_cache_footprint.py                 # locked defaults
    python scripts/kv_cache_footprint.py --seq-len 4096  # a different context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tinylm.model import ModelConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from tinylm.model import ModelConfig


def per_token_per_layer_floats(cfg: ModelConfig) -> tuple[int, int]:
    """(MHA, MLA) cached floats per token per layer.

    MHA stores full K and V: 2 * n_heads * head_dim (= 2 * d_model).
    MLA stores the compressed latent plus the decoupled-RoPE key: d_latent + d_rope.
    """
    head_dim = cfg.d_model // cfg.n_heads
    mha = 2 * cfg.n_heads * head_dim
    mla = cfg.d_latent + cfg.d_rope
    return mha, mla


def kv_cache_bytes(
    cfg: ModelConfig, seq_len: int, batch: int = 1, dtype_bytes: int = 2
) -> tuple[int, int]:
    """(MHA, MLA) total KV-cache bytes for a full seq_len context."""
    mha_ptpl, mla_ptpl = per_token_per_layer_floats(cfg)
    scale = batch * seq_len * cfg.n_layers * dtype_bytes
    return mha_ptpl * scale, mla_ptpl * scale


def reduction_ratio(cfg: ModelConfig) -> float:
    """How many times smaller the MLA cache is than the equivalent MHA cache."""
    mha, mla = per_token_per_layer_floats(cfg)
    return mha / mla


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-len", type=int, default=None, help="default: cfg.ctx")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype-bytes", type=int, default=2, help="2 = bf16/fp16")
    args = p.parse_args()

    cfg = ModelConfig()
    seq_len = args.seq_len if args.seq_len is not None else cfg.ctx
    mha_ptpl, mla_ptpl = per_token_per_layer_floats(cfg)
    mha_b, mla_b = kv_cache_bytes(cfg, seq_len, args.batch, args.dtype_bytes)
    mib = 1024 * 1024

    print(f"Config: d_model={cfg.d_model} heads={cfg.n_heads} "
          f"d_latent={cfg.d_latent} d_rope={cfg.d_rope} layers={cfg.n_layers}")
    print(f"Per token per layer:  MHA {mha_ptpl} floats   MLA {mla_ptpl} floats")
    print(f"Reduction ratio:      {reduction_ratio(cfg):.2f}x  "
          f"({100 * (1 - mla_ptpl / mha_ptpl):.1f}% smaller)")
    print(f"Full cache (B={args.batch}, seq_len={seq_len}, "
          f"{args.dtype_bytes}B/elem):")
    print(f"    MHA {mha_b / mib:8.1f} MiB")
    print(f"    MLA {mla_b / mib:8.1f} MiB")
    print(f"    saved {(mha_b - mla_b) / mib:.1f} MiB")


if __name__ == "__main__":
    main()
