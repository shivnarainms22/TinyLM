#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=build_heldout_codemath
#SBATCH --output=/scratch/%u/tinylm/logs/build_heldout_codemath_%j.log
#
# v3 Deliverable 2 — build disjoint held-out CODE and MATH shards for the
# perplexity diagnostic. Streams past the 1B-token training prefix (default) then
# collects ~10M held-out tokens per source. CPU job (tokenization only, no GPU).
# The 1B skip is streamed, so this takes a while — submit and walk away:
#   sbatch ~/TinyLM/scripts/build_heldout_codemath.sh
set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinylm

cd ~/TinyLM
python scripts/build_heldout_codemath.py \
    --out-root /scratch/$USER/tinylm/data_v3 \
    --skip-tokens 1000000000 \
    --target-tokens 10000000
echo "=== held-out code/math build done $(date) ==="
