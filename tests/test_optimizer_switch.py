"""Tests for the optimizer mode switch (muon vs adamw-only)."""
import torch


def _tiny_model():
    from tinylm.model import ModelConfig, TinyLM
    torch.manual_seed(0)
    return TinyLM(ModelConfig(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, tie_weights=True, attention="mla",
    ))


def _cfg(optimizer):
    from tinylm.train import TrainConfig
    return TrainConfig(optimizer=optimizer, lr_muon=0.02, lr_adamw=0.001, weight_decay=0.1)


def test_adamw_mode_puts_all_params_on_one_adamw():
    from tinylm.train import build_optimizers
    model = _tiny_model()
    opts = build_optimizers(model, _cfg("adamw"))
    assert len(opts) == 1
    opt, lr_max = opts[0]
    assert isinstance(opt, torch.optim.AdamW)
    assert lr_max == 0.001
    n_in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_model = sum(p.numel() for p in model.parameters())
    assert n_in_opt == n_model


def test_muon_mode_splits_matrix_and_scalar_without_overlap():
    from tinylm.train import build_optimizers
    model = _tiny_model()
    opts = build_optimizers(model, _cfg("muon"))
    assert len(opts) == 2
    kinds = {type(o).__name__ for o, _ in opts}
    assert kinds == {"Muon", "AdamW"}
    ids = [id(p) for o, _ in opts for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "param appears in more than one optimizer"
    assert len(ids) == len(list(model.parameters())), "not all params covered"


def test_unknown_optimizer_raises():
    import pytest
    from tinylm.train import build_optimizers
    with pytest.raises(ValueError):
        build_optimizers(_tiny_model(), _cfg("sgd"))
