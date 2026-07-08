#!/usr/bin/env bash
# v3 Deliverable 3 — evaluate the SFT model on the locked 4-task suite (0-shot),
# to test whether instruction-tuning moves HellaSwag/ARC/LAMBADA/Winogrande vs the
# E3-full base. The SFT checkpoint is saved in the same format eval_tinylm.py loads.
# Submit after SFT training has produced a checkpoint:  sbatch scripts/eval_v3_sft_job.sh
#SBATCH --job-name=eval_v3_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/eval_v3_sft_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
SFT_CKPT="${SCRATCH}/tinylm/runs/phase_v3_sft_smoltalk/checkpoints/last.pt"
mkdir -p "${SCRATCH}/tinylm/logs"

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"

cd "${REPO}"
mkdir -p results/v3

echo "=== 0-shot locked-suite eval of SFT model  $(date) ==="
python scripts/eval_tinylm.py \
    --checkpoint "${SFT_CKPT}" \
    --output "results/v3/run_sft_eval.json" \
    --batch-size 32
echo "=== SFT eval done $(date) ==="
