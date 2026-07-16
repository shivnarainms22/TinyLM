#!/usr/bin/env bash
# v3 Deliverable 1 — few-shot eval. Runs the locked 4-task suite at 5-shot on the
# Run D base and the E3-full checkpoint, to test whether v2's "flat reasoning" is
# partly a zero-shot artifact. Submit and walk away:  sbatch scripts/eval_v3_fewshot_job.sh
# JSONs land in ~/TinyLM/results/v3/run_{D,E3full}_5shot_eval.json.
#SBATCH --job-name=eval_v3_fewshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/eval_v3_fewshot_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
RUNS="${SCRATCH}/tinylm/runs"
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
mkdir -p results/v3

# checkpoint label -> path (a100 pinned above: V100 sm_70 unsupported by torch cu128)
declare -A CKPTS=(
  [D]="${RUNS}/run_D_mla_muon_v2/checkpoints/step_22999.pt"
  [E3full]="${RUNS}/phase_v2_E3_distill_mix_full/checkpoints/step_06999.pt"
)

FAILURES=0
for LABEL in D E3full; do
    echo "=== 5-shot eval: ${LABEL}  $(date) ==="
    python scripts/eval_tinylm.py \
        --checkpoint "${CKPTS[$LABEL]}" \
        --num-fewshot 5 \
        --output "results/v3/run_${LABEL}_5shot_eval.json" \
        --batch-size 16 \
        || { echo "WARNING: ${LABEL} 5-shot eval failed — continuing."
             FAILURES=$((FAILURES + 1)); }
done
echo "=== v3 few-shot evals done $(date) — ${FAILURES} failure(s) ==="
[[ ${FAILURES} -gt 0 ]] && exit 1
exit 0
