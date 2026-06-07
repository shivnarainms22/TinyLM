#!/usr/bin/env bash
# Submit one TinyLM run chain to Explorer.
#   bash scripts/submit_hpc.sh <run_name> <config_rel_path> <total_steps> [init_from] [shard_dir]
# e.g. bash scripts/submit_hpc.sh run_A_mha_adamw configs/run_A_mha_adamw.yaml 23000
set -euo pipefail
RUN_NAME="${1:?run_name}"; CONFIG="${2:?config path}"; TOTAL_STEPS="${3:?total_steps}"
INIT_FROM="${4:-}"
SHARD_DIR="${5:-}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"; LOG_DIR="${SCRATCH}/tinylm/logs"
mkdir -p "${LOG_DIR}"
for V in HF_TOKEN WANDB_API_KEY HF_HUB_REPO_ID; do
    [[ -z "${!V:-}" ]] && echo "WARNING: ${V} not set."
done
JID=$(sbatch --job-name="${RUN_NAME}" --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",CONFIG="${CONFIG}",TOTAL_STEPS="${TOTAL_STEPS}",INIT_FROM="${INIT_FROM}",SHARD_DIR="${SHARD_DIR}" \
    "${REPO}/scripts/hpc_job.sh" | awk '{print $NF}')
echo "Submitted ${RUN_NAME} as ${JID}. Monitor: squeue -u ${USER} ; tail -f ${LOG_DIR}/${RUN_NAME}_${JID}.log"
