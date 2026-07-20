#!/usr/bin/env bash
# Submit the v4 KD probe. Thin wrapper over submit_hpc.sh that points the shared
# rechain/SIGTERM job (hpc_job.sh) at the tinylm.kd entry module instead of
# tinylm.train. Everything else — init_from, shard_dir, 8h auto-rechain — is reused.
#
#   bash scripts/submit_kd.sh <init_from> [shard_dir]
# e.g.
#   bash scripts/submit_kd.sh \
#     /scratch/$USER/tinylm/runs/run_D_mla_muon_v2/checkpoints/step_22999.pt \
#     /scratch/$USER/tinylm/data_v2/fwe_fresh_shards
set -euo pipefail
INIT_FROM="${1:?init_from checkpoint (Run D step_22999.pt)}"
SHARD_DIR="${2:-}"

REPO="${HOME}/TinyLM"
export TINYLM_MODULE="tinylm.kd"        # propagates via submit_hpc.sh --export=ALL

bash "${REPO}/scripts/submit_hpc.sh" \
    phase_v4_KD_tinyllama_fwe \
    configs/v4/run_KD_tinyllama_fwe.yaml \
    2000 \
    "${INIT_FROM}" \
    "${SHARD_DIR}"
