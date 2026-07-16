#!/usr/bin/env python3
"""lm-eval wrapper for TinyLM .pt checkpoints.

Loads a raw checkpoint, wraps the model in the lm_eval LM interface, and
runs HellaSwag, ARC-Easy, LAMBADA, and Winogrande.

Setup:
    pip install lm-eval transformers

Usage:
    python scripts/eval_tinylm.py \
        --checkpoint checkpoints/step_19999.pt \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --output results/run_D_eval.json

    # smaller batch if OOM
    python scripts/eval_tinylm.py ... --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinylm.loader import load_from_checkpoint
from tinylm.model import TinyLM


def _load_model(ckpt_path: str, device: str) -> TinyLM:
    return load_from_checkpoint(ckpt_path, device)


class TinyLMEval(LM):
    """lm_eval LM wrapper around a raw TinyLM checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        tokenizer_name: str,
        batch_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self._device = device
        self._batch_size = batch_size
        self._model = _load_model(checkpoint, device)
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self._max_len = self._model.cfg.ctx
        self._bos: List[int] = (
            [self._tok.bos_token_id]
            if self._tok.bos_token_id is not None
            else []
        )

    # ---- LM interface properties ----------------------------------------

    @property
    def eot_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def tok_encode(self, string: str) -> List[int]:
        return self._tok.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self._tok.decode(tokens)

    # ---- core batched forward -------------------------------------------

    @torch.no_grad()
    def _score_pairs(
        self,
        pairs: List[Tuple[List[int], List[int]]],  # (ctx_ids, cont_ids)
    ) -> List[Tuple[float, bool]]:
        """Score (context, continuation) pairs. Returns (log_prob, is_greedy)."""
        all_results: List[Tuple[float, bool]] = []

        for i in range(0, len(pairs), self._batch_size):
            chunk = pairs[i : i + self._batch_size]

            # Truncate from left if too long, always keep full continuation.
            seqs: List[List[int]] = []
            cont_starts: List[int] = []
            for ctx_ids, cont_ids in chunk:
                combined = ctx_ids + cont_ids
                if len(combined) > self._max_len:
                    trim = len(combined) - self._max_len
                    ctx_ids = ctx_ids[trim:]
                seqs.append(ctx_ids + cont_ids)
                cont_starts.append(len(ctx_ids))

            max_len = max(len(s) for s in seqs)
            input_ids = torch.zeros(len(chunk), max_len, dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)

            input_ids = input_ids.to(self._device)
            # model takes (B, T) -> (B, T, V); we feed all but the last token
            logits = self._model(input_ids[:, :-1]).float()  # (B, T-1, V)
            targets = input_ids[:, 1:]                        # (B, T-1)
            log_probs = F.log_softmax(logits, dim=-1)

            for j, (seq, cs) in enumerate(zip(seqs, cont_starts)):
                cont_len = len(seq) - cs
                # Continuation tokens in targets are at [cs-1, cs-1+cont_len).
                # cs >= 1 because we always prepend BOS.
                start = cs - 1
                end = start + cont_len
                tgt = targets[j, start:end]               # (cont_len,)
                lp = log_probs[j, start:end]              # (cont_len, V)
                sum_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item()
                greedy = lp.argmax(dim=-1)
                is_greedy = (greedy == tgt).all().item()
                all_results.append((sum_lp, bool(is_greedy)))

        return all_results

    # ---- LM interface methods -------------------------------------------

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pairs = []
        for req in requests:
            ctx_str, cont_str = req.args
            ctx_ids = self._bos + self._tok.encode(ctx_str, add_special_tokens=False)
            cont_ids = self._tok.encode(cont_str, add_special_tokens=False)
            pairs.append((ctx_ids, cont_ids))
        return self._score_pairs(pairs)

    def loglikelihood_rolling(self, requests) -> List[Tuple[float, bool]]:
        # Context = BOS, continuation = full string.
        pairs = []
        for req in requests:
            ids = self._tok.encode(req.args[0], add_special_tokens=False)
            pairs.append((list(self._bos), ids))
        return self._score_pairs(pairs)

    def generate_until(self, requests):
        raise NotImplementedError(
            "generate_until is not required for HellaSwag / ARC-Easy / LAMBADA / Winogrande"
        )


# ---- CLI ----------------------------------------------------------------

TASKS = ["hellaswag", "arc_easy", "lambada_openai", "winogrande"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace tokenizer name or local path",
    )
    p.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task list")
    p.add_argument("--num-fewshot", type=int, default=0,
                   help="Number of in-context few-shot examples (default 0 = zero-shot)")
    p.add_argument("--batch-size", type=int, default=16, help="Eval batch size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/eval.json", help="Path to save results JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = TinyLMEval(
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Model loaded on {args.device}. Running tasks: {args.tasks}")

    print(f"Model loaded on {args.device}. num_fewshot={args.num_fewshot}")
    results = simple_evaluate(
        model=model,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        log_samples=False,
    )

    # Print summary table
    print("\n=== Results ===")
    task_results = results["results"]
    for task, metrics in task_results.items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or metrics.get("acc", "N/A")
        if isinstance(acc, float):
            acc = f"{acc * 100:.2f}%"
        print(f"  {task:<20} {acc}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
