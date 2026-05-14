#!/usr/bin/env python3
"""Upload TinyLM checkpoints to HuggingFace Hub as they are written.

Polls the checkpoint directory every 60 seconds and uploads any new .pt files.
Creates the HF repo if it does not exist.

Usage:
    python scripts/upload_checkpoints.py --repo <hf-username/repo-name>

Environment:
    HF_TOKEN — HuggingFace token with write access (repo scope)

Example:
    export HF_TOKEN=hf_...
    python scripts/upload_checkpoints.py --repo shivnarainns22/tinylm-checkpoints
"""

import argparse
import os
import time
import glob

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="HF repo id, e.g. username/tinylm-checkpoints")
    p.add_argument("--ckpt-dir", default="checkpoints", help="Local checkpoint directory")
    p.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    create_repo(args.repo, repo_type="model", exist_ok=True, token=token)
    print(f"Uploading to: https://huggingface.co/{args.repo}")
    print(f"Watching:     {args.ckpt_dir}/  (polling every {args.interval}s)")

    uploaded: set[str] = set()

    while True:
        pattern = os.path.join(args.ckpt_dir, "*.pt")
        for path in sorted(glob.glob(pattern)):
            filename = os.path.basename(path)
            if filename in uploaded:
                continue
            # Skip files still being written (size stable check)
            size_before = os.path.getsize(path)
            time.sleep(2)
            if os.path.getsize(path) != size_before:
                continue  # still writing, pick it up next poll

            print(f"  Uploading {filename} ...", flush=True)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=filename,
                repo_id=args.repo,
                repo_type="model",
            )
            uploaded.add(filename)
            print(f"  Done: {filename}", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
