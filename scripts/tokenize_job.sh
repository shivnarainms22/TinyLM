#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=tokenize
#SBATCH --output=/scratch/%u/tinylm/logs/tokenize_%j.log
#
# Tokenize 8B unique FineWeb-Edu tokens (80 shards x 100M) for the
# TinyLM HPC re-run. Submit with:
#   sbatch ~/TinyLM/scripts/tokenize_job.sh
set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinylm

cd ~/TinyLM
python scripts/tokenize_shards.py \
    --split sample-100BT \
    --out-dir /scratch/$USER/tinylm/data \
    --max-shards 80
