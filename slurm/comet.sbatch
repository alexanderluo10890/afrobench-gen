#!/bin/bash
# SSA-COMET scoring of every finished FLORES direction in a run (idempotent).
#   sbatch $AFB_SBATCH_ARGS slurm/comet.sbatch <run_dir> [repo_dir]
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
RUN_DIR=$1; REPO=${2:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source slurm/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "== $(date -Is) job=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
"$AFB_VENV_COMET/bin/python" -u -m afbgen.comet_score --run-dir "$RUN_DIR"
