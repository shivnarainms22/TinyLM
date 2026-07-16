"""Tests for the SFT sample generator's prompt priming.

The one thing that must be exact is the ChatML priming string: if generation
primes the model with a template even slightly different from the one SFT
trained on, the assistant turn never triggers and output is garbage. So we
pin build_prompt_ids to render_chat's own encoding, segment for segment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tinylm.sft import render_chat, _ROLE_PREFIX  # noqa: E402
from generate_sft_samples import build_prompt_ids  # noqa: E402


# Deterministic, network-free encode: one id per character.
def _fake_encode(s: str) -> list[int]:
    return [ord(c) for c in s]


EOS = 2


def test_prompt_ids_match_render_chat_plus_assistant_open():
    """build_prompt_ids == render_chat(user turn) + encode('<|assistant|>\\n')."""
    prompt = "Explain gravity."
    user_ids, _ = render_chat(
        [{"role": "user", "content": prompt}], _fake_encode, EOS
    )
    expected = user_ids + _fake_encode(_ROLE_PREFIX["assistant"])

    assert build_prompt_ids(prompt, _fake_encode, EOS) == expected


def test_prompt_ends_with_assistant_open_so_the_model_answers():
    ids = build_prompt_ids("hi", _fake_encode, EOS)
    tail = _fake_encode(_ROLE_PREFIX["assistant"])
    assert ids[-len(tail):] == tail


def test_no_bos_or_eos_in_the_prompt():
    """Training used add_special_tokens=False and never closes the assistant
    turn in the prompt — priming must not sneak a BOS in or an EOS."""
    ids = build_prompt_ids("hi", _fake_encode, EOS)
    # First tokens are the literal '<|user|>\n' marker, not a BOS id.
    assert ids[: len(_fake_encode(_ROLE_PREFIX["user"]))] == _fake_encode(
        _ROLE_PREFIX["user"]
    )
    assert EOS not in ids


def test_system_prompt_is_prepended_when_given():
    ids = build_prompt_ids("hi", _fake_encode, EOS, system="Be terse.")
    sys_ids, _ = render_chat(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        _fake_encode,
        EOS,
    )
    assert ids == sys_ids + _fake_encode(_ROLE_PREFIX["assistant"])
