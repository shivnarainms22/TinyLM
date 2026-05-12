#!/bin/bash
# TinyLM — RunPod instance setup
# Run once after launching the A100 pod.
#
# Prerequisites (set as RunPod environment variables or export before running):
#   WANDB_API_KEY  — from wandb.ai/settings
#   HF_TOKEN       — from huggingface.co/settings/tokens (needs Llama-2 access)
#
# Usage:
#   bash /workspace/tinylm/scripts/setup_runpod.sh
#
# Then train:
#   python -m tinylm.train /workspace/tinylm/configs/run_D_mla_muon.yaml

set -e

REPO=/workspace/tinylm

echo "=== Installing Python dependencies ==="
pip install torch transformers datasets wandb pyyaml numpy --quiet
pip install -e "$REPO" --quiet

echo "=== Logging into WandB ==="
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Export it before running this script."
    exit 1
fi
wandb login "$WANDB_API_KEY"

echo "=== Logging into HuggingFace (required for Llama-2 tokenizer) ==="
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script."
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

echo "=== Creating output directories ==="
mkdir -p "$REPO/data/shards"
mkdir -p "$REPO/checkpoints"

echo "=== Pre-tokenizing 1B tokens from FineWeb-Edu (sample-10BT) ==="
echo "    This takes ~30 minutes. Output: 10 x shard_XXXX.npy in data/shards/"
python "$REPO/scripts/tokenize_shards.py" \
    --split sample-10BT \
    --out-dir "$REPO/data/shards" \
    --max-shards 10

echo ""
echo "=== Setup complete ==="
echo "To start training:"
echo "  python -m tinylm.train $REPO/configs/run_D_mla_muon.yaml"
echo ""
echo "To resume from a checkpoint:"
echo "  Edit resume_from in configs/run_D_mla_muon.yaml, then rerun the same command."
