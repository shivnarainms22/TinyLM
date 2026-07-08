#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=build_v2_e3_full
#SBATCH --output=/scratch/%u/tinylm/logs/build_v2_e3_full_%j.log
#
# Build the FULL v2 Run E3 distill-mixture shards (80 x 100M = 8.0B tokens) for the
# full continuation run. Same E3 recipe as the probe (45% FineWeb-Edu / 20% web /
# 10% code / 10% math / 15% Cosmopedia-v2), document-interleaved, FineWeb-Edu slice
# skips Run D's 8B prefix. A larger non-repeating set so the 7.34B-token full run
# never repeats data. Writes to a separate dir from the 2.1B probe shards.
#
# Submit with:
#   sbatch ~/TinyLM/scripts/tokenize_v2_e3_full_job.sh
set -euo pipefail

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tinylm

cd ~/TinyLM
python scripts/build_mixture_shards.py \
    --recipe e3 \
    --out-dir /scratch/$USER/tinylm/data_v2/distill_mix_full \
    --max-shards 80
