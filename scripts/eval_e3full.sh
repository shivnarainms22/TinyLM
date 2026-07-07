#!/usr/bin/env bash
# Evaluate the full E3 run's milestone checkpoints on the locked 4-task suite
# (HellaSwag, ARC-Easy, LAMBADA, Winogrande) — the trajectory across
# 1.8B -> 7.3B tokens that answers "does reasoning move with scale, or only LM?".
# Submit and walk away:  sbatch scripts/eval_e3full.sh
# JSONs land in ~/TinyLM/results/v2/run_E3full_step<N>_eval.json.
#SBATCH --job-name=eval_e3full
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/eval_e3full_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
CKPT_DIR="${SCRATCH}/tinylm/runs/phase_v2_E3_distill_mix_full/checkpoints"
mkdir -p "${SCRATCH}/tinylm/logs"

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"

cd "${REPO}"
mkdir -p results/v2

# a100 is pinned above: V100 (sm_70) is unsupported by torch cu128.
for S in 01749 03499 05249 06999; do
    echo "=== evaluating step_${S} $(date) ==="
    python scripts/eval_tinylm.py \
        --checkpoint "${CKPT_DIR}/step_${S}.pt" \
        --output "results/v2/run_E3full_step${S}_eval.json" \
        --batch-size 32 \
        || echo "WARNING: step_${S} eval failed — continuing with the rest."
done
echo "=== all E3-full evals done $(date) ==="
