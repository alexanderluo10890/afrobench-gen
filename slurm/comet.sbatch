#!/bin/bash -l
# SSA-COMET scoring of every finished FLORES direction in a run (idempotent).
#   sbatch $AFB_SBATCH_ARGS slurm/comet.sbatch <run_dir> [repo_dir] [stl|mtl]   (default: reported metric = stl)
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
RUN_DIR=$1; REPO=${2:-$SLURM_SUBMIT_DIR}; VARIANT=${3:-}
cd "$REPO"
source slurm/env.sh
# import from the node-local disk, not the shared network disk (slurm/pack_env.sh)
source slurm/local_env.sh venv-comet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
MAX_RESTARTS=${AFB_MAX_RESTARTS:-6}
echo "== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
"$AFB_VENV_COMET/bin/python" -u -m afbgen.comet_score --run-dir "$RUN_DIR" ${VARIANT:+--variant "$VARIANT"}
RC=$?
echo "== $(date -Is) exit $RC"
# Crash (e.g. a file briefly invisible on network storage while importing / loading the
# encoder): requeue; finished directions are skipped on the next attempt.
if (( RC != 0 )) && (( ${SLURM_RESTART_COUNT:-0} < MAX_RESTARTS )); then
  scontrol requeue "$SLURM_JOB_ID" && echo "== requeued after failure"
fi
exit $RC
