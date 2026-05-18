#!/usr/bin/env python3
"""Push TinyLM source + model card to Shiv-22/tinylm on HuggingFace.

Creates the repo if it doesn't exist, then uploads:
  hf/README.md        -> README.md
  scripts/generate.py -> generate.py
  src/tinylm/*.py     -> tinylm/*.py

The checkpoint (step_19999.pt) is NOT re-uploaded — the model card links
to Shiv-22/tinylm-checkpoints where it already lives.

Usage:
    pip install huggingface_hub
    huggingface-cli login      # or set HF_TOKEN env var
    python scripts/push_to_hf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    repo_id = "Shiv-22/tinylm"
    api = HfApi()

    print(f"Creating repo {repo_id} (skips if already exists) ...")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    uploads = [
        (ROOT / "hf" / "README.md",           "README.md"),
        (ROOT / "scripts" / "generate.py",     "generate.py"),
        (ROOT / "src" / "tinylm" / "__init__.py", "tinylm/__init__.py"),
        (ROOT / "src" / "tinylm" / "model.py",    "tinylm/model.py"),
        (ROOT / "src" / "tinylm" / "muon.py",     "tinylm/muon.py"),
        (ROOT / "src" / "tinylm" / "train.py",    "tinylm/train.py"),
        (ROOT / "src" / "tinylm" / "data.py",     "tinylm/data.py"),
    ]

    for local_path, repo_path in uploads:
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path}")
            continue
        print(f"  Uploading {local_path.relative_to(ROOT)} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"add {repo_path}",
        )

    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
