#!/usr/bin/env bash
# One-time setup for TinyLM on Northeastern Explorer.
#   bash scripts/setup_hpc.sh
# No CUDA-extension compiles needed (MLA + Muon are pure PyTorch).
set -euo pipefail
SCRATCH="/scratch/${USER}"

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

# An env can exist and still be unusable: ~/.conda is a symlink to /scratch, and
# a scratch purge reaps stdlib .py sources by access time while leaving
# __pycache__ behind (imports read the .pyc, so only the sources go stale).
# Python then dies at startup with "no codec search functions registered".
# Existence is not health — verify the interpreter before trusting the env.
if conda env list | grep -q "^tinylm "; then
    if conda run -n tinylm python -c "import encodings, sys" >/dev/null 2>&1; then
        echo "conda env 'tinylm' exists and its interpreter runs — skipping create."
    else
        echo "conda env 'tinylm' exists but its interpreter is broken — recreating."
        conda env remove -n tinylm -y
        conda create -n tinylm python=3.11 -y
    fi
else
    conda create -n tinylm python=3.11 -y
fi
conda activate tinylm

# Explorer driver 570.x caps at CUDA 12.8 — a cu130 wheel loads but is_available()==False.
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers datasets wandb huggingface_hub pyyaml numpy "lm-eval>=0.4.9" pytest
pip install -q -e "${HOME}/TinyLM"

mkdir -p "${SCRATCH}/tinylm/data" "${SCRATCH}/tinylm/runs" "${SCRATCH}/tinylm/logs" "${SCRATCH}/wandb"

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
echo "Add to ~/.bashrc: export HF_TOKEN=... WANDB_API_KEY=... HF_HUB_REPO_ID=..."
