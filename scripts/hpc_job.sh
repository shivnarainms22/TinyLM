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
# Guard the scratch-purge codec failure (a leftover PYTHONHOME/PYTHONPATH points
# the interpreter at the wrong stdlib -> "no codec search functions registered").
unset PYTHONHOME PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="${SCRATCH}/wandb"
# KD downloads a ~2GB teacher; cache it on scratch, not the quota'd $HOME.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
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
# --signal=B:SIGTERM@120 delivers SIGTERM to THIS batch shell ~120s before the
# wall, not to the python child. Run training in the background and forward the
# signal so python's handler checkpoints last.pt at the next step boundary; the
# next segment then resumes from the true last step instead of redoing the steps
# since the last periodic save. (save_every still guarantees a checkpoint floor.)
# Entry module is configurable so the same rechain/SIGTERM machinery drives both
# plain pretraining (tinylm.train) and the v4 KD probe (tinylm.kd). Defaults to
# tinylm.train, so every existing run is byte-identical.
python -m "${TINYLM_MODULE:-tinylm.train}" "${REPO}/${CONFIG}" &
TRAIN_PID=$!
trap 'echo "[job] wall approaching — forwarding SIGTERM to train pid ${TRAIN_PID}"; kill -TERM "${TRAIN_PID}" 2>/dev/null || true' TERM
# wait is interrupted when the trap fires; loop until the child truly exits so we
# never tear the job down mid-checkpoint-write.
while kill -0 "${TRAIN_PID}" 2>/dev/null; do wait "${TRAIN_PID}" || true; done
echo "=== ${RUN_NAME} job ${SLURM_JOB_ID} done $(date) ==="
