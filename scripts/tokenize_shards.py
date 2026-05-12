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
import os

import numpy as np

SHARD_SIZE = 100_000_000  # 100M uint16 tokens per shard ≈ 200MB


def tokenize_shards(split: str, out_dir: str, max_shards: int | None) -> None:
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
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=split,
        split="train",
        streaming=True,
    )

    buffer: list[int] = []
    shard_idx = 0

    for sample in ds:
        if max_shards is not None and shard_idx >= max_shards:
            break

        ids = tok.encode(sample["text"], add_special_tokens=False)
        ids.append(tok.eos_token_id)
        buffer.extend(ids)

        while len(buffer) >= SHARD_SIZE:
            shard = np.array(buffer[:SHARD_SIZE], dtype=np.uint16)
            path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npy")
            np.save(path, shard)
            print(f"  Saved {path}  ({SHARD_SIZE / 1e6:.0f}M tokens)")
            buffer = buffer[SHARD_SIZE:]
            shard_idx += 1

        if max_shards is not None and shard_idx >= max_shards:
            break

    print(f"Done. {shard_idx} shards written to {out_dir!r}.")
    if shard_idx == 0:
        raise RuntimeError("No shards were written — check dataset name and streaming.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sample-10BT",
                        help="FineWeb-Edu split: sample-10BT or sample-100BT")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write shard_XXXX.npy files")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Stop after this many shards (10 = 1B tokens)")
    args = parser.parse_args()
    tokenize_shards(args.split, args.out_dir, args.max_shards)
