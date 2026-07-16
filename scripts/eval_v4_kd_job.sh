#!/usr/bin/env bash
# v4 — evaluate the KD probe on the locked 4-task suite (0-shot), to test whether
# logit distillation from TinyLlama-1.1B moves reasoning where E1's cross-entropy
# could not. Submit after the KD run has produced a checkpoint:
#   sbatch scripts/eval_v4_kd_job.sh
#SBATCH --job-name=eval_v4_kd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/eval_v4_kd_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
KD_CKPT="${SCRATCH}/tinylm/runs/phase_v4_KD_tinyllama_fwe/checkpoints/last.pt"
mkdir -p "${SCRATCH}/tinylm/logs"

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"

# sbatch inherits the submitting shell's env. A PYTHONHOME/PYTHONPATH left over
# from another project points this interpreter at the wrong stdlib and it dies
# with "LookupError: no codec search functions registered".
unset PYTHONHOME PYTHONPATH
python -c "import sys; print('interpreter:', sys.executable)"

cd "${REPO}"
mkdir -p results/v4

echo "=== 0-shot locked-suite eval of KD probe  $(date) ==="
python scripts/eval_tinylm.py \
    --checkpoint "${KD_CKPT}" \
    --output "results/v4/run_KD_eval.json" \
    --batch-size 32
echo "=== KD eval done $(date) ==="
