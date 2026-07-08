"""Build a TinyLM model from a raw training checkpoint.

Shared by the eval scripts (lm-eval wrapper and the perplexity diagnostic) so the
checkpoint-load path lives in one place. torch-only — no lm-eval / HF deps — so it
imports cleanly wherever the package is available.
"""
from __future__ import annotations

from typing import Mapping

import torch

from tinylm.model import ModelConfig, TinyLM


def strip_compile_prefix(state: Mapping[str, torch.Tensor]) -> dict:
    """Drop the ``_orig_mod.`` prefix torch.compile adds to state-dict keys.

    Returns the mapping unchanged (as a dict) when no such prefix is present.
    """
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return dict(state)


def load_from_checkpoint(ckpt_path: str, device: str = "cpu") -> TinyLM:
    """Load a ``.pt`` checkpoint into an eval-mode ``TinyLM`` on ``device``.

    The architecture is reconstructed from the ``config`` embedded in the
    checkpoint, so no external YAML is needed.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    c = ckpt["config"]
    model = TinyLM(
        ModelConfig(
            n_layers=c["n_layers"],
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            d_latent=c["d_latent"],
            d_rope=c["d_rope"],
            ffn_hidden=c["ffn_hidden"],
            ctx=c["ctx"],
            vocab_size=c["vocab_size"],
            tie_weights=c["tie_weights"],
            attention=c["attention"],
        )
    )
    model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    return model.to(device).eval()
