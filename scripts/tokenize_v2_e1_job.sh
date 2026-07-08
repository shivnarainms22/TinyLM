#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=tokenize_v2_e1
#SBATCH --output=/scratch/%u/tinylm/logs/tokenize_v2_e1_%j.log
#
# Tokenize the v2 Run E1 "fresh FineWeb-Edu" shards (21 x 100M = 2.1B tokens).
#
# Non-overlap with Run D: Run D consumed the first 80 shards (8B tokens) of the
# sample-100BT stream (see tokenize_job.sh). E1 streams the identical, ordered
# sample-100BT, --skip-tokens 8000000000 to step past Run D's prefix, then emits
# the next 21 shards. Disjoint document ranges -> provably fresh, same
# distribution. A manifest.json recording this is written beside the shards.
#
# Submit with:
#   sbatch ~/TinyLM/scripts/tokenize_v2_e1_job.sh
set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinylm

cd ~/TinyLM
python scripts/tokenize_shards.py \
    --split sample-100BT \
    --skip-tokens 8000000000 \
    --max-shards 21 \
    --out-dir /scratch/$USER/tinylm/data_v2/fwe_fresh
