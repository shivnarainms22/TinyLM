#!/usr/bin/env python3
"""Build a token-proportional mixed-source corpus into flat .npy shards (E2).

Streams several HuggingFace datasets, tokenizes with the locked Llama-2 tokenizer,
and interleaves them at the document level to hit target *token* proportions
(default 55% FineWeb-Edu / 20% general web / 15% code / 10% math). Every shard is
a representative mix rather than blocks of one source, so the model never sees a
mid-run distribution shift. The FineWeb-Edu slice skips Run D's 8B prefix, keeping
E2 disjoint from the base model's training data.

Reuses the shard size + manifest helpers from tokenize_shards.py.

Usage (HPC):
    python scripts/build_mixture_shards.py \\
        --out-dir /scratch/$USER/tinylm/data_v2/mixed_web_code_math \\
        --max-shards 21
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize_shards import SHARD_SIZE, write_manifest


def _skip_source(doc_iter, encode, eos_id, skip_tokens):
    """Discard whole documents from a source until `skip_tokens` have passed,
    dropping the straddling document (same rule as tokenize_shards, so the
    FineWeb-Edu slice is provably disjoint from Run D's prefix)."""
    skipped = 0
    started = skip_tokens <= 0
    for doc in doc_iter:
        if not started:
            skipped += len(encode(doc["text"])) + 1
            if skipped >= skip_tokens:
                started = True
            continue
        yield doc


def build_mixture(sources, encode, out_dir, *, weights, eos_id,
                  max_shards=None, shard_size=SHARD_SIZE, skip_tokens=None):
    """Interleave `sources` to target token `weights`, packing uint16 shards.

    sources: dict name -> iterable of {"text": str}.
    weights: dict name -> target token fraction (need not pre-normalize).
    skip_tokens: optional dict name -> tokens to discard from that source first.

    At each step the source furthest below its target token share is drawn, so
    proportions converge regardless of document length. Returns
    dict(shards_written, tokens_written, per_source_tokens, proportions).
    """
    skip_tokens = skip_tokens or {}
    wsum = sum(weights.values())
    weights = {n: w / wsum for n, w in weights.items()}
    names = list(sources)

    iters = {}
    for n in names:
        it = iter(sources[n])
        s = skip_tokens.get(n, 0)
        iters[n] = _skip_source(it, encode, eos_id, s) if s > 0 else it

    emitted = {n: 0 for n in names}
    exhausted = {n: False for n in names}
    buffer: list[int] = []
    shard_idx = 0
    total = 0

    while max_shards is None or shard_idx < max_shards:
        live = [n for n in names if not exhausted[n]]
        if not live:
            break
        # Most "behind" source by token deficit vs its target share of the mix.
        pick = max(live, key=lambda m: weights[m] * total - emitted[m])
        try:
            doc = next(iters[pick])
        except StopIteration:
            exhausted[pick] = True
            continue

        ids = encode(doc["text"])
        ids.append(eos_id)
        buffer.extend(ids)
        emitted[pick] += len(ids)
        total += len(ids)

        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            np.save(os.path.join(out_dir, f"shard_{shard_idx:04d}.npy"), shard)
            print(f"  Saved shard_{shard_idx:04d}.npy  ({shard_size / 1e6:.0f}M tokens)")
            buffer = buffer[shard_size:]
            shard_idx += 1
            if max_shards is not None and shard_idx >= max_shards:
                break

    emitted_total = sum(emitted.values()) or 1
    return {
        "shards_written": shard_idx,
        "tokens_written": shard_idx * shard_size,
        "per_source_tokens": emitted,
        "proportions": {n: emitted[n] / emitted_total for n in names},
    }


# ── source registry (ungated, parquet-native — no loader scripts) ────────────
# (repo, config, text_key, weight, skip_tokens). datasets>=4 dropped script-based
# datasets, so every source here must ship parquet/data files, not a *.py loader.
SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-100BT",  "text",    0.55, 8_000_000_000),
    "web":         ("HuggingFaceFW/fineweb",     "sample-100BT",  "text",    0.20, 0),
    "code":        ("codeparrot/codeparrot-clean", None,          "content", 0.15, 0),
    "math":        ("HuggingFaceTB/finemath",    "finemath-3plus", "text",   0.10, 0),
}

_TEXT_KEY_FALLBACKS = ("text", "content", "code", "raw_content")


def _hf_stream(repo, config, text_key):
    """Stream a dataset yielding {"text": ...}. Robust to the text column name:
    prefers `text_key`, else the first non-empty known field."""
    from datasets import load_dataset
    ds = load_dataset(repo, name=config, split="train", streaming=True)
    for sample in ds:
        t = sample.get(text_key)
        if not t:
            t = next((sample[k] for k in _TEXT_KEY_FALLBACKS if sample.get(k)), "")
        if t:
            yield {"text": t}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-shards", type=int, default=21)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    encode = lambda text: tok.encode(text, add_special_tokens=False)

    # Fail fast: confirm every source streams a first document before the long run.
    streams, weights, skips = {}, {}, {}
    for name, (repo, config, text_key, weight, skip) in SOURCES.items():
        print(f"Probing source {name}: {repo} ({config}) ...")
        gen = _hf_stream(repo, config, text_key)
        first = next(gen)
        assert first["text"], f"empty first doc from {name}"

        def chained(first=first, gen=gen):
            yield first
            yield from gen
        streams[name], weights[name], skips[name] = chained(), weight, skip

    stats = build_mixture(streams, encode, args.out_dir, weights=weights,
                          eos_id=tok.eos_token_id, max_shards=args.max_shards,
                          skip_tokens=skips)

    write_manifest(
        args.out_dir,
        kind="mixture",
        sources={n: SOURCES[n][0] for n in SOURCES},
        target_weights={n: SOURCES[n][3] for n in SOURCES},
        tokenizer="meta-llama/Llama-2-7b-hf",
        shard_size=SHARD_SIZE,
        fineweb_edu_skip_tokens=SOURCES["fineweb_edu"][4],
        **stats,
    )
    print(f"Done. {stats['shards_written']} shards. proportions={stats['proportions']}")
    if stats["shards_written"] == 0:
        raise RuntimeError("No shards written — check source streaming.")


if __name__ == "__main__":
    main()
