#!/usr/bin/env python3
"""Pre-tokenize FineWeb-Edu into flat .npy shards for TinyLM training.

Tokenizer: meta-llama/Llama-2-7b-hf (vocab_size=32000, locked in Phase 0).
Shard size: 100M tokens per file (~200MB at uint16).

Usage — toy run (1B tokens = 10 shards):
    python scripts/tokenize_shards.py \\
        --split sample-10BT \\
        --out-dir data/shards \\
        --max-shards 10

Usage — full Phase 4 run (20B tokens, omit --max-shards):
    python scripts/tokenize_shards.py \\
        --split sample-100BT \\
        --out-dir data/shards

Requires (on RunPod):
    pip install transformers datasets
    huggingface-cli login --token $HF_TOKEN  (Llama-2 is gated)
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

SHARD_SIZE = 100_000_000  # 100M uint16 tokens per shard ≈ 200MB


def pack_token_shards(doc_iter, encode, out_dir, *, max_shards=None,
                      skip_tokens=0, eos_id, shard_size=SHARD_SIZE):
    """Encode a document stream and pack tokens into flat uint16 .npy shards.

    `encode(text) -> list[int]`; an `eos_id` is appended after each document.

    When `skip_tokens > 0`, whole documents are consumed and discarded until at
    least that many tokens have passed (the document crossing the boundary is
    dropped entirely). Emitted shards are therefore disjoint from any prefix run
    that consumed the same deterministic stream — this is what makes the v2 E1
    shards provably non-overlapping with the Run D 8B prefix on sample-100BT.

    Returns dict(shards_written, tokens_written, tokens_skipped).
    """
    buffer: list[int] = []
    shard_idx = 0
    tokens_skipped = 0
    started = skip_tokens <= 0

    for sample in doc_iter:
        if max_shards is not None and shard_idx >= max_shards:
            break

        ids = encode(sample["text"])
        ids.append(eos_id)

        if not started:
            tokens_skipped += len(ids)
            if tokens_skipped >= skip_tokens:
                started = True  # drop the straddling doc to guarantee no overlap
            continue

        buffer.extend(ids)
        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npy")
            np.save(path, shard)
            print(f"  Saved {path}  ({shard_size / 1e6:.0f}M tokens)")
            buffer = buffer[shard_size:]
            shard_idx += 1
            if max_shards is not None and shard_idx >= max_shards:
                break

    return {
        "shards_written": shard_idx,
        "tokens_written": shard_idx * shard_size,
        "tokens_skipped": tokens_skipped,
    }


def write_manifest(out_dir, **fields):
    """Write a provenance manifest.json next to the shards (Data Rules: keep
    shard manifests with the run artifacts)."""
    fields["written_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(fields, f, indent=2)


def tokenize_shards(split: str, out_dir: str, max_shards: int | None,
                    skip_tokens: int = 0) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    # Fail fast on permission issues before downloading the tokenizer.
    test_path = os.path.join(out_dir, ".write_test")
    try:
        with open(test_path, "w") as f:
            f.write("")
        os.remove(test_path)
    except OSError as e:
        raise OSError(f"Output directory {out_dir!r} is not writable: {e}") from e

    print(f"Loading tokenizer meta-llama/Llama-2-7b-hf ...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print(f"Streaming HuggingFaceFW/fineweb-edu ({split}) ...")
    if skip_tokens > 0:
        print(f"Skipping first {skip_tokens:,} tokens (non-overlap with prior run) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=split,
        split="train",
        streaming=True,
    )

    stats = pack_token_shards(
        ds,
        lambda text: tok.encode(text, add_special_tokens=False),
        out_dir,
        max_shards=max_shards,
        skip_tokens=skip_tokens,
        eos_id=tok.eos_token_id,
    )

    write_manifest(
        out_dir,
        dataset="HuggingFaceFW/fineweb-edu",
        split=split,
        tokenizer="meta-llama/Llama-2-7b-hf",
        shard_size=SHARD_SIZE,
        skip_tokens=skip_tokens,
        **stats,
    )

    print(f"Done. {stats['shards_written']} shards written to {out_dir!r}.")
    if stats["shards_written"] == 0:
        raise RuntimeError("No shards were written — check dataset name and streaming.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sample-10BT",
                        help="FineWeb-Edu split: sample-10BT or sample-100BT")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write shard_XXXX.npy files")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Stop after this many shards (10 = 1B tokens)")
    parser.add_argument("--skip-tokens", type=int, default=0,
                        help="Discard this many tokens before emitting, so the "
                             "shards are disjoint from a prior run on the same "
                             "stream (e.g. 8_000_000_000 = Run D's 8B prefix)")
    args = parser.parse_args()
    tokenize_shards(args.split, args.out_dir, args.max_shards, args.skip_tokens)
