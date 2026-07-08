"""TDD for tinylm.loader.strip_compile_prefix (the pure, testable part of the
checkpoint loader). Building a full model needs a real checkpoint, exercised on
the cluster; the prefix stripping is where the subtle bug lives, so it is tested."""
import torch

from tinylm.loader import strip_compile_prefix


def test_strips_orig_mod_prefix():
    state = {"_orig_mod.tok_embed.weight": torch.zeros(2), "_orig_mod.lm_head.bias": torch.zeros(2)}
    out = strip_compile_prefix(state)
    assert set(out.keys()) == {"tok_embed.weight", "lm_head.bias"}


def test_leaves_unprefixed_state_untouched():
    state = {"tok_embed.weight": torch.zeros(2), "layers.0.mlp.gate.weight": torch.zeros(2)}
    out = strip_compile_prefix(state)
    assert set(out.keys()) == set(state.keys())


def test_returns_a_dict_copy_not_the_original_mapping():
    state = {"tok_embed.weight": torch.zeros(2)}
    out = strip_compile_prefix(state)
    assert isinstance(out, dict)
    assert out is not state
