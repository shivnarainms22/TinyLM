"""Baseline eval runner for TinyLM Phase 0.

Runs `lm-eval-harness` on TinyLlama-1.1B against the locked four-task
suite (HellaSwag, ARC-Easy, LAMBADA OpenAI, Winogrande) and writes
results to baseline_results.json at the repo root.

USAGE (Colab/Kaggle notebook, GPU runtime selected):

    !pip install -q lm-eval==0.4.* transformers accelerate
    !python scripts/eval_baseline.py

USAGE (dry-run, validates the command without invoking lm-eval):

    python scripts/eval_baseline.py --dry-run

The script is intentionally a thin wrapper. All eval logic lives in
`lm-eval` itself; this file just pins the exact command so the same
invocation is reproducible in Phase 5 for the four ablation
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASELINE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
TASKS = ["hellaswag", "arc_easy", "lambada_openai", "winogrande"]
OUTPUT_PATH = "baseline_results.json"


def build_command(output_dir: Path, batch_size: int) -> list[str]:
    """Return the exact argv that will be invoked.

    Phase 5 uses an identical command shape against each ablation
    checkpoint, with only `pretrained=` swapped.
    """
    return [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={BASELINE_MODEL}",
        "--tasks", ",".join(TASKS),
        "--device", "cuda",
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
    ]


def find_results_json(output_dir: Path) -> Path:
    """lm-eval writes a nested results JSON; find the canonical one."""
    candidates = sorted(output_dir.rglob("results_*.json"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results JSON produced in {output_dir}. "
            "Check the lm-eval stderr above."
        )
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="lm-eval batch size (16 fits a free Colab T4; bump on A100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline"),
        help="lm-eval working dir (results JSON written here).",
    )
    parser.add_argument(
        "--final-path",
        type=Path,
        default=Path(OUTPUT_PATH),
        help="Destination for the canonical baseline_results.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without invoking lm-eval.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args.output_dir, args.batch_size)

    print("Command:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        print("[dry-run] Skipping lm-eval invocation.", flush=True)
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"lm-eval exited {result.returncode}", file=sys.stderr)
        return result.returncode

    src = find_results_json(args.output_dir)
    payload = json.loads(src.read_text())
    args.final_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote canonical baseline JSON to {args.final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
