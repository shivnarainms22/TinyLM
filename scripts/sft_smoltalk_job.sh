#!/usr/bin/env bash
# v3 Deliverable 3 — SFT training. Full fine-tunes the E3-full base on SmolTalk.
# Checkpoints every save_every (500) steps to out_dir/checkpoints/, so a walltime
# stop is safe — you evaluate the latest checkpoint (no resume needed).
# Requires HF_TOKEN (Llama-2 tokenizer + SmolTalk are gated) and WANDB_API_KEY in
# the submitting shell (sbatch exports your environment by default).
# Submit:  sbatch scripts/sft_smoltalk_job.sh
#SBATCH --job-name=sft_smoltalk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/sft_smoltalk_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
mkdir -p "${SCRATCH}/tinylm/logs"

for V in HF_TOKEN WANDB_API_KEY; do
    [[ -z "${!V:-}" ]] && echo "WARNING: ${V} not set — SmolTalk/tokenizer download or wandb may fail."
done

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="${SCRATCH}/wandb"

# Inject cluster paths without editing the YAML (apply_env_overrides in sft.py).
export TINYLM_SFT_INIT_FROM="${SCRATCH}/tinylm/runs/phase_v2_E3_distill_mix_full/checkpoints/step_06999.pt"
export TINYLM_SFT_OUT="${SCRATCH}/tinylm/runs/phase_v3_sft_smoltalk"

cd "${REPO}"
python -m tinylm.sft configs/v3/sft_smoltalk.yaml
echo "=== SFT training done $(date) ==="
