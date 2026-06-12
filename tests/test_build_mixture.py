"""Behavioral tests for scripts/build_mixture_shards.py.

The proportional mixer is the one genuinely new piece of E2 tooling: it must
hit target *token* proportions, interleave sources within each shard (not pack
them in blocks), account per-source tokens, and honor a per-source skip (so the
FineWeb-Edu slice stays disjoint from Run D's 8B). All exercised with in-memory
fake sources — no network, no tokenizer.
"""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_mixture_shards.py"
_spec = importlib.util.spec_from_file_location("build_mixture_shards", _SCRIPT)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


def _encode(text: str):
    """'ID:N' -> token id ID repeated N times (eos appended by the builder)."""
    tid, n = text.split(":")
    return [int(tid)] * int(n)


def _source(tid, n_tokens, n_docs):
    return [{"text": f"{tid}:{n_tokens}"} for _ in range(n_docs)]


def _load_shards(out_dir):
    toks = []
    for p in sorted(Path(out_dir).glob("shard_*.npy")):
        toks.extend(np.load(p).tolist())
    return toks


def test_mixture_hits_target_token_proportions(tmp_path):
    sources = {
        "a": _source(1, 9, 2000),   # 10 tokens/doc incl eos
        "b": _source(2, 9, 2000),
        "c": _source(3, 9, 2000),
    }
    weights = {"a": 0.55, "b": 0.30, "c": 0.15}
    stats = bm.build_mixture(
        sources, _encode, str(tmp_path), weights=weights, eos_id=0,
        max_shards=10, shard_size=100,
    )
    props = stats["proportions"]
    assert abs(props["a"] - 0.55) < 0.03, props
    assert abs(props["b"] - 0.30) < 0.03, props
    assert abs(props["c"] - 0.15) < 0.03, props


def test_mixture_interleaves_sources_within_a_shard(tmp_path):
    # Distinct ids per source; a representative mix means shard 0 holds >1 source.
    sources = {"a": _source(1, 9, 500), "b": _source(2, 9, 500)}
    bm.build_mixture(sources, _encode, str(tmp_path),
                     weights={"a": 0.5, "b": 0.5}, eos_id=0,
                     max_shards=1, shard_size=100)
    shard0 = set(np.load(sorted(Path(tmp_path).glob("shard_*.npy"))[0]).tolist())
    assert 1 in shard0 and 2 in shard0, "shard is not an interleaved mix"


def test_mixture_per_source_accounting(tmp_path):
    sources = {"a": _source(1, 9, 1000), "b": _source(2, 9, 1000)}
    stats = bm.build_mixture(sources, _encode, str(tmp_path),
                             weights={"a": 0.5, "b": 0.5}, eos_id=0,
                             max_shards=5, shard_size=100)
    per = stats["per_source_tokens"]
    assert all(v > 0 for v in per.values())                       # both sources used
    assert sum(per.values()) >= stats["tokens_written"]           # accounts for all packed
    assert abs(sum(stats["proportions"].values()) - 1.0) < 1e-9   # proportions normalized


def test_mixture_per_source_skip_excludes_prefix(tmp_path):
    # Source 'a' first emits id=7 docs (to be skipped), then id=1 docs.
    a = _source(7, 9, 5) + _source(1, 9, 1000)   # skip ~50 tokens -> drop the 7s
    sources = {"a": a, "b": _source(2, 9, 1000)}
    bm.build_mixture(sources, _encode, str(tmp_path),
                     weights={"a": 0.5, "b": 0.5}, eos_id=0,
                     max_shards=4, shard_size=100, skip_tokens={"a": 50})
    toks = set(_load_shards(tmp_path))
    assert 7 not in toks, "skipped prefix leaked into mixed shards"
    assert 1 in toks and 2 in toks


def test_source_registry_weights_and_skip():
    """Mix weights sum to 1 and only FineWeb-Edu skips Run D's 8B prefix."""
    weights = {n: spec[3] for n, spec in bm.SOURCES.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-9, weights
    assert bm.SOURCES["fineweb_edu"][4] == 8_000_000_000
    assert all(spec[4] == 0 for n, spec in bm.SOURCES.items() if n != "fineweb_edu")


def test_recipes_e2_and_e3_registered():
    """Both probe recipes exist; e2 is the back-compat default alias of SOURCES."""
    assert bm.RECIPES["e2"] is bm.SOURCES
    assert set(bm.RECIPES) >= {"e2", "e3"}


def test_e3_recipe_adds_teacher_distill_slice():
    """E3 = 45/20/10/10/15 over FWE/web/code/math/distill; weights sum to 1, the
    distill slice is teacher-generated synthetic text (Cosmopedia-v2), and only the
    FineWeb-Edu slice skips Run D's 8B prefix (disjoint from the base model)."""
    e3 = bm.RECIPES["e3"]
    weights = {n: spec[3] for n, spec in e3.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-9, weights
    assert "distill" in e3, "E3 must add a teacher-distilled slice"
    repo, config, _text, weight, skip = e3["distill"]
    assert weight == 0.15 and skip == 0
    assert repo == "HuggingFaceTB/smollm-corpus" and config == "cosmopedia-v2"
    assert e3["fineweb_edu"][4] == 8_000_000_000
    assert all(spec[4] == 0 for n, spec in e3.items() if n != "fineweb_edu")
