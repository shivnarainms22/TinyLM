#!/usr/bin/env bash
# v3 Deliverable 2 — run the held-out perplexity diagnostic for Run D vs E3-full
# on the held-out CODE and MATH sets built by build_heldout_codemath.sh.
# Submit after the held-out build finishes:  sbatch scripts/eval_v3_ppl_job.sh
# JSONs land in ~/TinyLM/results/v3/ppl_{D,E3full}_{code,math}.json.
#SBATCH --job-name=eval_v3_ppl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/tinylm/logs/eval_v3_ppl_%j.log
set -euo pipefail

USER="${USER:-$(whoami)}"; HOME="${HOME:-/home/${USER}}"
SCRATCH="/scratch/${USER}"; REPO="${HOME}/TinyLM"
RUNS="${SCRATCH}/tinylm/runs"; DATA="${SCRATCH}/tinylm/data_v3"
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

declare -A CKPTS=(
  [D]="${RUNS}/run_D_mla_muon_v2/checkpoints/step_22999.pt"
  [E3full]="${RUNS}/phase_v2_E3_distill_mix_full/checkpoints/step_06999.pt"
)

FAILURES=0
for LABEL in D E3full; do
    for KIND in code math; do
        echo "=== perplexity: ${LABEL} on held-out ${KIND}  $(date) ==="
        python scripts/eval_perplexity.py \
            --checkpoint "${CKPTS[$LABEL]}" \
            --shard-dir "${DATA}/heldout_${KIND}" \
            --output "results/v3/ppl_${LABEL}_${KIND}.json" \
            || { echo "WARNING: ${LABEL}/${KIND} perplexity eval failed — continuing."
                 FAILURES=$((FAILURES + 1)); }
    done
done
echo "=== v3 perplexity diagnostic done $(date) — ${FAILURES} failure(s) ==="
[[ ${FAILURES} -gt 0 ]] && exit 1
exit 0
