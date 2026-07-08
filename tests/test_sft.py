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
    iter_sft_examples,
    render_chat,
    sft_loss,
)


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
