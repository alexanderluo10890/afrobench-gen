#!/bin/bash
# Everything on a cluster login node in one command:
#   bash slurm/bootstrap.sh [run_name]
# 1. environment (uv venvs from lock files; writes + commits the lock files the first time)
# 2. Hugging Face login if needed (FLORES is gated: accept its terms on the website first)
# 3. pinned downloads: model, datasets, SSA-COMET; verifies every (task, language) loads
# 4. submits a 20-doc smoke run AND the full run as parallel L40S jobs (both resumable;
#    rerunning this script later only resubmits unfinished work)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-v1.0.0}

bash slurm/setup.sh
source slurm/env.sh

if [[ -n "$(git status --porcelain requirements)" ]]; then
  git add requirements/*.lock
  git -c user.name="${GIT_NAME:-afbgen}" -c user.email="${GIT_EMAIL:-afbgen@localhost}" \
    commit -q -m "Lock Python environments (first install on $AFB_CLUSTER)"
  echo ">>> committed lock files"
fi

"$AFB_VENV/bin/hf" auth whoami >/dev/null 2>&1 || "$AFB_VENV/bin/hf" auth login
"$AFB_VENV/bin/python" -m afbgen.prepare
"$AFB_VENV/bin/python" -m afbgen.prepare --only verify > "$AFB_ROOT/verify.json"
echo ">>> all datasets load (sizes in $AFB_ROOT/verify.json)"

"$AFB_VENV/bin/python" slurm/submit.py --run "$RUN" --limit 20 --submit
"$AFB_VENV/bin/python" slurm/submit.py --run "$RUN" --submit
squeue --me
echo ">>> logs: $AFB_ROOT/logs/$RUN/   results: $AFB_RESULTS/$RUN/"
echo ">>> when jobs finish: $AFB_VENV/bin/python -m afbgen.aggregate --run-dir $AFB_RESULTS/$RUN"
