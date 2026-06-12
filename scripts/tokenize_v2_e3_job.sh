#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=build_v2_e3
#SBATCH --output=/scratch/%u/tinylm/logs/build_v2_e3_%j.log
#
# Build the v2 Run E3 distillation-mixture shards (21 x 100M = 2.1B tokens):
# 45% FineWeb-Edu / 20% general web / 10% code / 10% math / 15% teacher-distilled
# (Cosmopedia-v2, Mixtral-generated synthetic explanations/QA), document-interleaved.
# The FineWeb-Edu slice skips Run D's 8B prefix (provably disjoint from the base
# model). A manifest.json records achieved per-source proportions beside the shards.
#
# Submit with:
#   sbatch ~/TinyLM/scripts/tokenize_v2_e3_job.sh
set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinylm

cd ~/TinyLM
python scripts/build_mixture_shards.py \
    --recipe e3 \
    --out-dir /scratch/$USER/tinylm/data_v2/distill_mix \
    --max-shards 21
