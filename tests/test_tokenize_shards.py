"""Behavioral tests for scripts/tokenize_shards.py.

The shard packer is exercised with an in-memory fake document stream and a
trivial encode fn, so these run on Windows CPU with no network and no gated
tokenizer. The skip-tokens path is what makes v2 (E1) shards provably disjoint
from the Run D 8B prefix on the identical sample-100BT stream.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tokenize_shards.py"
_spec = importlib.util.spec_from_file_location("tokenize_shards", _SCRIPT)
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)


def _encode(text: str):
    """Doc text 'ID:N' -> token id ID repeated N times (eos appended by packer)."""
    tid, n = text.split(":")
    return [int(tid)] * int(n)


def _docs(*pairs):
    return [{"text": f"{tid}:{n}"} for tid, n in pairs]


def _load_shards(out_dir):
    tokens = []
    for p in sorted(Path(out_dir).glob("shard_*.npy")):
        tokens.extend(np.load(p).tolist())
    return tokens


# ── packing without skip (backward compatible with Run D path) ───────────────

def test_pack_no_skip_includes_all_documents(tmp_path):
    out = str(tmp_path)
    stats = ts.pack_token_shards(
        _docs((1, 5), (2, 5), (3, 5)), _encode, out,
        max_shards=2, skip_tokens=0, eos_id=0, shard_size=8,
    )
    toks = _load_shards(out)
    assert 1 in toks and 2 in toks  # nothing skipped
    assert stats["tokens_skipped"] == 0
    assert stats["shards_written"] == 2


def test_pack_writes_uint16_shards_of_exact_size(tmp_path):
    out = str(tmp_path)
    ts.pack_token_shards(
        _docs((1, 100), (2, 100)), _encode, out,
        max_shards=3, skip_tokens=0, eos_id=0, shard_size=50,
    )
    shards = sorted(Path(out).glob("shard_*.npy"))
    assert len(shards) == 3
    for p in shards:
        arr = np.load(p)
        assert arr.dtype == np.uint16
        assert arr.shape == (50,)


# ── skip-tokens: provable non-overlap with the skipped prefix ────────────────

def test_pack_skip_excludes_every_skipped_document_token(tmp_path):
    out = str(tmp_path)
    # Docs 1 and 2 (6 tokens each w/ eos) fill the 12-token skip budget; the
    # doc that crosses the boundary is dropped entirely. Only docs 3,4,5 emit.
    stats = ts.pack_token_shards(
        _docs((1, 5), (2, 5), (3, 5), (4, 5), (5, 5)), _encode, out,
        max_shards=2, skip_tokens=12, eos_id=0, shard_size=8,
    )
    toks = _load_shards(out)
    assert 1 not in toks, "skipped doc 1 leaked into fresh shards"
    assert 2 not in toks, "skipped doc 2 leaked into fresh shards"
    assert 3 in toks and 4 in toks  # fresh docs present
    assert stats["tokens_skipped"] >= 12


def test_pack_skip_zero_is_identical_to_no_skip(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    Path(a).mkdir(); Path(b).mkdir()
    docs = _docs((7, 10), (8, 10), (9, 10))
    ts.pack_token_shards(docs, _encode, a, max_shards=2, skip_tokens=0,
                         eos_id=0, shard_size=8)
    ts.pack_token_shards(docs, _encode, b, max_shards=2, skip_tokens=0,
                         eos_id=0, shard_size=8)
    assert _load_shards(a) == _load_shards(b)


# ── provenance manifest ──────────────────────────────────────────────────────

def test_write_manifest_records_provenance(tmp_path):
    out = str(tmp_path)
    ts.write_manifest(
        out, dataset="HuggingFaceFW/fineweb-edu", split="sample-100BT",
        skip_tokens=8_000_000_000, shards_written=21, tokens_written=2_100_000_000,
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["split"] == "sample-100BT"
    assert data["skip_tokens"] == 8_000_000_000
    assert data["shards_written"] == 21
