#!/usr/bin/env python3
"""v3 Deliverable 2 — held-out perplexity diagnostic.

Measures a checkpoint's next-token perplexity over pre-tokenized held-out shards
(same .npy uint16 format the trainer uses). Run for Run D vs E3-full on held-out
CODE and MATH to de-blind the ~30% of E3's data the locked commonsense suite
cannot see. Both models share the Llama-2 tokenizer, so per-token perplexity is
directly comparable.

Usage:
    python scripts/eval_perplexity.py \
        --checkpoint /scratch/$USER/tinylm/runs/.../step_06999.pt \
        --shard-dir  /scratch/$USER/tinylm/data_v3/heldout_code \
        --output     results/v3/ppl_E3full_code.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinylm.loader import load_from_checkpoint
from tinylm.perplexity import nll_sum, perplexity


@torch.no_grad()
def evaluate_shards(model, shard_dir: str, ctx: int, batch_size: int, device: str):
    """Accumulate NLL over every non-overlapping ctx-length window in the shards."""
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npy")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npy in {shard_dir!r}")

    total_nll, total_tokens = 0.0, 0
    for path in shards:
        tokens = torch.from_numpy(np.load(path).astype(np.int64))
        n_windows = len(tokens) // ctx
        if n_windows == 0:
            continue
        windows = tokens[: n_windows * ctx].view(n_windows, ctx)
        for i in range(0, n_windows, batch_size):
            batch = windows[i : i + batch_size].to(device)
            logits = model(batch[:, :-1]).float()   # (B, ctx-1, V)
            targets = batch[:, 1:]                   # (B, ctx-1)
            nll, n = nll_sum(logits, targets)
            total_nll += nll
            total_tokens += n
    return total_nll, total_tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard-dir", required=True, help="Dir of held-out shard_*.npy")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/v3/ppl.json")
    args = p.parse_args()

    model = load_from_checkpoint(args.checkpoint, args.device)
    ctx = model.cfg.ctx
    print(f"Loaded {args.checkpoint} on {args.device}; ctx={ctx}")

    total_nll, n_tokens = evaluate_shards(
        model, args.shard_dir, ctx, args.batch_size, args.device
    )
    ppl = perplexity(total_nll, n_tokens)
    result = {
        "checkpoint": args.checkpoint,
        "shard_dir": args.shard_dir,
        "n_tokens": n_tokens,
        "mean_nll_nats": total_nll / n_tokens,
        "perplexity": ppl,
    }
    print(f"tokens={n_tokens:,}  perplexity={ppl:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
