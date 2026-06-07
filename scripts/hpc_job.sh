#!/usr/bin/env bash
# 8h SLURM segment for one TinyLM run. Submit via scripts/submit_hpc.sh.
# Pre-emptively chains the next segment; resumes from checkpoints/last.pt.
# Env injected by submit: RUN_NAME, CONFIG, TOTAL_STEPS, optional INIT_FROM/SHARD_DIR
#SBATCH --partition=gpu
#SBATCH --time=7:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:SIGTERM@120
set -euo pipefail
USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
RUN_DIR="${SCRATCH}/tinylm/runs/${RUN_NAME}"; LOG_DIR="${SCRATCH}/tinylm/logs"
CKPT="${RUN_DIR}/checkpoints/last.pt"
mkdir -p "${RUN_DIR}/checkpoints" "${LOG_DIR}"

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate tinylm
export PATH="${HOME}/.conda/envs/tinylm/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="${SCRATCH}/wandb"
export TINYLM_SHARD_DIR="${SHARD_DIR:-${SCRATCH}/tinylm/data}"
if [[ -n "${INIT_FROM:-}" ]]; then
    export TINYLM_INIT_FROM="${INIT_FROM}"
fi

# Early-exit if this run already reached TOTAL_STEPS.
if [[ -f "${CKPT}" ]]; then
    STEP=$(python -c "import torch;print(torch.load('${CKPT}',map_location='cpu',weights_only=True).get('step',0))")
    echo "Resuming ${RUN_NAME} from step ${STEP}/${TOTAL_STEPS}"
    if [[ "${STEP}" -ge "$((TOTAL_STEPS-1))" ]]; then echo "Already complete."; exit 0; fi
    export TINYLM_RESUME="${CKPT}"
else
    echo "No checkpoint — starting ${RUN_NAME} from scratch."
fi

# Pre-emptively queue the next segment (survives a hard SIGKILL at the wall).
NEXT=$(sbatch --dependency=afterany:"${SLURM_JOB_ID}" --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG="${CONFIG}",TOTAL_STEPS="${TOTAL_STEPS}",INIT_FROM="${INIT_FROM:-}",SHARD_DIR="${SHARD_DIR:-}" \
    "${REPO}/scripts/hpc_job.sh" | awk '{print $NF}')
echo "Next segment queued as ${NEXT}."

cd "${RUN_DIR}"   # checkpoints/ is written here, relative to cwd
python -m tinylm.train "${REPO}/${CONFIG}" \
    || echo "Training exited non-zero (SIGTERM save is normal)."
echo "=== ${RUN_NAME} job ${SLURM_JOB_ID} done $(date) ==="
