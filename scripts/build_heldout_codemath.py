#!/usr/bin/env python3
"""Build small, disjoint held-out CODE and MATH shards for the v3 perplexity
diagnostic (Deliverable 2).

Reuses the already-tested `build_mixture` / `_skip_source` from
`build_mixture_shards.py`: each source is streamed as a single-source "mixture"
(weight 1.0) with a large `--skip-tokens` so the held-out slice starts *past*
what the v2 E2/E3 mixtures consumed (they built code/math with skip=0, i.e. from
the stream head). E3-full built ~0.8B code and ~0.8B math tokens, so the default
1B skip keeps the held-out provably disjoint from E3's training data. Run D never
saw any code/math, so this is held-out for both models.

Usage (see build_heldout_codemath.sh for the batch wrapper):
    python scripts/build_heldout_codemath.py --out-root /scratch/$USER/tinylm/data_v3
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

_BM = Path(__file__).resolve().parent / "build_mixture_shards.py"
_spec = importlib.util.spec_from_file_location("build_mixture_shards", _BM)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

# Same sources the E2/E3 mixtures used, so "held-out" means the same distribution.
SOURCES = {
    "code": ("codeparrot/codeparrot-clean", None, "content"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "text"),
}


def build_one(name, repo, config, text_key, out_dir, skip_tokens, target_tokens, tok):
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    gen = bm._hf_stream(repo, config, text_key)
    first = next(gen)  # fail fast on an empty/broken stream before the long skip
    assert first["text"], f"empty first doc from {name}"

    def chained():
        yield first
        yield from gen

    return bm.build_mixture(
        {name: chained()},
        encode,
        out_dir,
        weights={name: 1.0},
        eos_id=tok.eos_token_id,
        max_shards=1,
        shard_size=target_tokens,      # one small held-out shard, not 100M
        skip_tokens={name: skip_tokens},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True, help="Parent dir; writes heldout_{code,math}/")
    p.add_argument("--skip-tokens", type=int, default=1_000_000_000,
                   help="Tokens to skip per source, past the training prefix (default 1B).")
    p.add_argument("--target-tokens", type=int, default=10_000_000,
                   help="Held-out tokens to collect per source (default 10M).")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    for name, (repo, config, text_key) in SOURCES.items():
        out_dir = os.path.join(args.out_root, f"heldout_{name}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{name}] skip {args.skip_tokens:,} then collect {args.target_tokens:,} "
              f"from {repo} ({config}) -> {out_dir}")
        stats = build_one(name, repo, config, text_key, out_dir,
                          args.skip_tokens, args.target_tokens, tok)
        print(f"[{name}] done: {stats}")


if __name__ == "__main__":
    main()
