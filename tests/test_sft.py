"""TDD tests for tinylm.sft — SFT module: render_chat, iter_sft_examples, sft_loss.

All tests use fake in-memory callables and data — no network, no transformers,
no datasets, no GPU required.

Encoding convention: one token id per character (ord(c)) so token sequences
are predictable and the masking logic can be verified exactly.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tinylm.sft import (
    SFTConfig,
    apply_env_overrides,
    apply_grad_checkpoint,
    iter_sft_examples,
    load_sft_config,
    render_chat,
    sft_loss,
)


# ---------------------------------------------------------------------------
# Memory fit — SFT must run on a 40GB A100, not just an 80GB one.
#
# The first full-run attempt OOMed on a 40GB card: sft_loss called plain
# F.cross_entropy (whose internal fp32 upcast of a (B*T, 32000) tensor is
# multiple GB — the exact failure chunked_cross_entropy exists to prevent),
# and the model ran without gradient checkpointing, so every block's
# activations for a 16x2048 batch stayed resident.
# ---------------------------------------------------------------------------

def test_sft_loss_uses_chunked_cross_entropy(monkeypatch):
    """sft_loss must route through the memory-bounded CE, not F.cross_entropy."""
    import tinylm.sft as sft_mod

    calls = []
    real = sft_mod.chunked_cross_entropy

    def spy(logits, targets, *args, **kwargs):
        calls.append(kwargs.get("ignore_index"))
        return real(logits, targets, *args, **kwargs)

    monkeypatch.setattr(sft_mod, "chunked_cross_entropy", spy)

    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    loss = sft_loss(logits, labels)

    assert calls, "sft_loss did not call chunked_cross_entropy"
    assert calls[0] == -100, "chunked CE was not told to ignore the -100 mask"
    assert math.isclose(loss.item(), math.log(3), rel_tol=1e-5)


def test_grad_checkpoint_enabled_by_default():
    """The checkpoint config that load_from_checkpoint rebuilds has no
    use_checkpoint field, so SFT must switch it on explicitly."""
    assert SFTConfig().grad_checkpoint is True


def test_load_sft_config_reads_grad_checkpoint(tmp_path):
    cfg_path = tmp_path / "sft.yaml"
    cfg_path.write_text("init_from: base.pt\ngrad_checkpoint: false\n")
    assert load_sft_config(str(cfg_path)).grad_checkpoint is False


def test_apply_grad_checkpoint_toggles_model_and_keeps_backward_working():
    """apply_grad_checkpoint flips the model flag the forward pass reads, and
    the recomputed forward still produces gradients."""
    from tinylm.model import ModelConfig, TinyLM

    model = TinyLM(ModelConfig(
        n_layers=2, d_model=32, n_heads=2, d_latent=16, d_rope=8,
        ffn_hidden=64, ctx=64, vocab_size=32000, tie_weights=True,
        attention="mla",
    ))
    assert model.cfg.use_checkpoint is False        # the silent default that OOMed

    apply_grad_checkpoint(model, True)
    assert model.cfg.use_checkpoint is True

    model.train()
    labels = torch.randint(0, 32000, (2, 16))
    labels[:, :4] = -100
    loss = sft_loss(model(torch.randint(0, 32000, (2, 16))), labels)
    loss.backward()
    assert math.isfinite(loss.item())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_resolve_total_steps_caps_lr_schedule_at_max_steps():
    """The LR schedule must span the steps we can actually reach in the walltime,
    not the 165k-step full-dataset pass — otherwise cosine never anneals."""
    from tinylm.sft import resolve_total_steps
    full = resolve_total_steps(n_rows=1_100_000, batch_size=16, epochs=3, max_steps=0)
    assert full > 100_000                        # 3 full epochs, unreachable in 8h
    capped = resolve_total_steps(n_rows=1_100_000, batch_size=16, epochs=3,
                                 max_steps=20_000)
    assert capped == 20_000


def test_resolve_total_steps_never_exceeds_the_data():
    """max_steps is a cap, not a floor — a tiny dataset must not inflate it."""
    from tinylm.sft import resolve_total_steps
    assert resolve_total_steps(n_rows=100, batch_size=16, epochs=1,
                               max_steps=20_000) == 5


def test_run_sft_stops_at_max_steps_and_always_leaves_last_pt(tmp_path, monkeypatch):
    """End-to-end loop on CPU: training stops at max_steps, and last.pt exists so
    eval_v3_sft_job.sh can find a model even when the walltime kills the job."""
    import sys, types
    import tinylm.sft as sft_mod
    from tinylm.model import ModelConfig, TinyLM

    tiny = TinyLM(ModelConfig(
        n_layers=2, d_model=32, n_heads=2, d_latent=16, d_rope=8, ffn_hidden=64,
        ctx=128, vocab_size=32000, tie_weights=True, attention="mla",
    ))
    monkeypatch.setattr(sft_mod, "load_from_checkpoint", lambda p, d: tiny)

    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}]}] * 64
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: rows

    class _Tok:
        eos_token_id = 2
        def encode(self, s, add_special_tokens=False):
            return [ord(c) % 32000 for c in s]
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **k: None
    fake_wandb.log = lambda *a, **k: None
    fake_wandb.finish = lambda: None

    monkeypatch.setitem(sys.modules, "datasets", fake_ds)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    out = tmp_path / "sft_run"
    sft_mod.run_sft(SFTConfig(
        init_from="ignored.pt", out_dir=str(out), batch_size=2, epochs=3,
        max_steps=4, save_every=2, max_len=128,
    ))

    ckpts = out / "checkpoints"
    assert (ckpts / "last.pt").exists(), "no last.pt — the SFT eval job would find nothing"
    saved = torch.load(ckpts / "last.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 4, f"did not stop at max_steps: step={saved['step']}"
    assert (ckpts / "step_000001.pt").exists(), "periodic checkpoint missing"


def test_env_overrides_inject_cluster_paths(monkeypatch):
    monkeypatch.setenv("TINYLM_SFT_INIT_FROM", "/scratch/base/step_06999.pt")
    monkeypatch.setenv("TINYLM_SFT_OUT", "/scratch/runs/sft")
    cfg = apply_env_overrides(SFTConfig())
    assert cfg.init_from == "/scratch/base/step_06999.pt"
    assert cfg.out_dir == "/scratch/runs/sft"


def test_env_overrides_noop_when_unset(monkeypatch):
    monkeypatch.delenv("TINYLM_SFT_INIT_FROM", raising=False)
    monkeypatch.delenv("TINYLM_SFT_OUT", raising=False)
    cfg = apply_env_overrides(SFTConfig(init_from="cfg.pt", out_dir="cfg_out"))
    assert cfg.init_from == "cfg.pt"
    assert cfg.out_dir == "cfg_out"


# ---------------------------------------------------------------------------
# Shared fake encode: one token per character, ord(c)
# ---------------------------------------------------------------------------

def _encode(text: str) -> list[int]:
    return [ord(c) for c in text]


EOS_ID = 2  # arbitrary non-printable id that won't appear in _encode output


# ---------------------------------------------------------------------------
# T1 — single [user, assistant] turn
# ---------------------------------------------------------------------------

def test_single_user_assistant_masking():
    """User tokens and all template markers are masked; assistant content + eos unmasked."""
    messages = [
        {"role": "user",      "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    input_ids, labels = render_chat(messages, _encode, EOS_ID)

    # lengths must match
    assert len(input_ids) == len(labels)

    # final token must be eos_id and its label must equal eos_id (not -100)
    assert input_ids[-1] == EOS_ID
    assert labels[-1] == EOS_ID

    # Find the assistant content token positions
    assistant_content = "hello"
    assistant_ids = _encode(assistant_content)

    # Collect non-masked positions
    unmasked = [(i, input_ids[i]) for i in range(len(labels)) if labels[i] != -100]
    unmasked_ids = [v for _, v in unmasked]

    # All unmasked tokens must be the assistant content tokens + eos
    assert unmasked_ids == assistant_ids + [EOS_ID]

    # User content must be fully masked
    user_ids_set = set(_encode("hi"))
    # (we can't check by value alone since 'h' ord could appear in template strings too,
    # but we can check that the label for every occurrence NOT in assistant+eos is -100)
    for i, tok in enumerate(input_ids):
        if labels[i] != -100:
            # every unmasked token must be in the assistant content + eos sequence
            pass  # already checked via unmasked_ids above

    # Sanity: there must be some masked tokens (the user content and template text)
    masked_count = labels.count(-100)
    assert masked_count > 0


def test_single_user_assistant_invariant():
    """T4-style: every non -100 label equals input_ids at that index."""
    messages = [
        {"role": "user",      "content": "abc"},
        {"role": "assistant", "content": "xyz"},
    ]
    input_ids, labels = render_chat(messages, _encode, EOS_ID)
    assert len(input_ids) == len(labels)
    for i in range(len(labels)):
        if labels[i] != -100:
            assert labels[i] == input_ids[i], f"label mismatch at position {i}"


# ---------------------------------------------------------------------------
# T2 — [system, user, assistant]: system and user fully masked
# ---------------------------------------------------------------------------

def test_system_user_assistant_system_and_user_masked():
    """System and user content are all -100; only assistant content + eos unmasked."""
    messages = [
        {"role": "system",    "content": "sys"},
        {"role": "user",      "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    input_ids, labels = render_chat(messages, _encode, EOS_ID)

    assert len(input_ids) == len(labels)

    unmasked_ids = [input_ids[i] for i in range(len(labels)) if labels[i] != -100]
    expected = _encode("answer") + [EOS_ID]
    assert unmasked_ids == expected


# ---------------------------------------------------------------------------
# T3 — multi-turn [user, assistant, user, assistant]
# ---------------------------------------------------------------------------

def test_multi_turn_two_assistant_turns_unmasked():
    """Exactly two assistant turns unmasked; both users masked; two eos positions."""
    messages = [
        {"role": "user",      "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user",      "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    input_ids, labels = render_chat(messages, _encode, EOS_ID)

    assert len(input_ids) == len(labels)

    # Count eos positions that are not masked
    eos_unmasked = [i for i, tok in enumerate(input_ids)
                    if tok == EOS_ID and labels[i] != -100]
    assert len(eos_unmasked) == 2, f"expected 2 unmasked eos, got {len(eos_unmasked)}"

    # All unmasked tokens must be a1 + eos + a2 + eos (in order)
    unmasked_ids = [input_ids[i] for i in range(len(labels)) if labels[i] != -100]
    expected = _encode("a1") + [EOS_ID] + _encode("a2") + [EOS_ID]
    assert unmasked_ids == expected


# ---------------------------------------------------------------------------
# T4 — invariant: label coherence across all turns
# ---------------------------------------------------------------------------

def test_label_coherence_invariant():
    """len(labels)==len(input_ids) and every non -100 label equals input_ids[i]."""
    messages = [
        {"role": "system",    "content": "setup"},
        {"role": "user",      "content": "prompt"},
        {"role": "assistant", "content": "response"},
        {"role": "user",      "content": "followup"},
        {"role": "assistant", "content": "reply"},
    ]
    input_ids, labels = render_chat(messages, _encode, EOS_ID)
    assert len(input_ids) == len(labels)
    for i in range(len(labels)):
        if labels[i] != -100:
            assert labels[i] == input_ids[i], f"mismatch at {i}: label={labels[i]}, id={input_ids[i]}"


# ---------------------------------------------------------------------------
# T5 — iter_sft_examples skips examples exceeding max_len
# ---------------------------------------------------------------------------

def test_iter_sft_examples_skips_long_examples():
    """Examples whose len(input_ids) > max_len are skipped; short ones yielded."""
    # short example: user "hi" + assistant "ok"
    # With char-level encode: "<|user|>\n" (9) + "hi" (2) + "\n" (1) +
    #   "<|assistant|>\n" (15) + "ok" (2) + eos (1) = 30 tokens
    short_row = {"messages": [
        {"role": "user",      "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]}
    # long example: user with 200 chars + assistant with 200 chars -> ~430 tokens
    long_content = "x" * 200
    long_row = {"messages": [
        {"role": "user",      "content": long_content},
        {"role": "assistant", "content": long_content},
    ]}

    max_len = 50  # short row (~30 tokens) fits; long row (~430 tokens) does not
    rows = [short_row, long_row, short_row]

    results = list(iter_sft_examples(rows, _encode, EOS_ID, max_len))

    # Only the two short rows should be yielded
    assert len(results) == 2
    for input_ids, labels in results:
        assert len(input_ids) <= max_len
        assert len(input_ids) == len(labels)


def test_iter_sft_examples_all_fit():
    """When every row fits, all are yielded."""
    rows = [
        {"messages": [
            {"role": "user",      "content": "a"},
            {"role": "assistant", "content": "b"},
        ]},
        {"messages": [
            {"role": "user",      "content": "c"},
            {"role": "assistant", "content": "d"},
        ]},
    ]
    results = list(iter_sft_examples(rows, _encode, EOS_ID, max_len=10000))
    assert len(results) == 2


# ---------------------------------------------------------------------------
# T6 — CPU smoke test: sft_loss is finite and backward populates grads
# ---------------------------------------------------------------------------

def test_sft_loss_finite_and_backward():
    """sft_loss produces a finite scalar loss; .backward() populates gradients."""
    from tinylm.model import ModelConfig, TinyLM

    # Tiny model so this runs in seconds on CPU
    cfg = ModelConfig(
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_latent=16,
        d_rope=8,
        ffn_hidden=64,
        ctx=64,
        vocab_size=32000,
        tie_weights=True,
        attention="mla",
    )
    model = TinyLM(cfg)
    model.train()

    B, T = 2, 16
    # Random input tokens, padded labels with some masked positions
    input_ids = torch.randint(0, 32000, (B, T))
    labels = torch.randint(0, 32000, (B, T))
    # Mask first 4 positions (simulate prompt/user tokens)
    labels[:, :4] = -100

    logits = model(input_ids)  # (B, T, vocab_size)

    loss = sft_loss(logits, labels)

    assert loss.ndim == 0, "loss must be a scalar"
    assert math.isfinite(loss.item()), f"loss must be finite, got {loss.item()}"

    loss.backward()

    # Check at least one parameter has a non-None, non-zero gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad, "no parameter received a gradient"


def test_sft_loss_pure_logic():
    """sft_loss with hand-checkable logits: only unmasked positions contribute."""
    # 1 batch, 4 tokens, vocab 3
    # logits: uniform → each token has loss = ln(3)
    # labels: [-100, -100, 1, 2] → 2 unmasked tokens
    logits = torch.zeros(1, 4, 3)  # uniform over vocab=3
    labels = torch.tensor([[-100, -100, 1, 2]])

    loss = sft_loss(logits, labels)

    # causal shift: logits[:, :-1] vs labels[:, 1:]
    # after shift, positions 0..2 pair with labels[1..3] = [-100, 1, 2]
    # unmasked: positions 1 and 2 (labels 1 and 2), each loss = ln(3)
    expected = math.log(3)
    assert math.isclose(loss.item(), expected, rel_tol=1e-5), \
        f"expected {expected:.6f}, got {loss.item():.6f}"


def test_sft_loss_all_masked_raises_or_nan():
    """sft_loss with all labels masked: cross_entropy with no valid tokens
    returns nan (PyTorch behaviour with reduction='mean' and all ignored).
    We accept either nan or a RuntimeError — the key is it doesn't silently
    produce a finite non-zero number."""
    logits = torch.zeros(1, 4, 3)
    labels = torch.full((1, 4), -100, dtype=torch.long)

    try:
        loss = sft_loss(logits, labels)
        # If it returns, it must be nan (not a fake finite value)
        assert math.isnan(loss.item()) or loss.item() == 0.0, \
            f"expected nan/0 for all-masked, got {loss.item()}"
    except (RuntimeError, ValueError):
        pass  # also acceptable
