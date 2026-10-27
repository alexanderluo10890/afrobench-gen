#!/bin/bash -l
# One shard = one benchmark x a few languages on one L40S. Submitted by slurm/submit.py:
#   sbatch $AFB_SBATCH_ARGS slurm/eval.sbatch <run_dir> <task> <langs> [repo_dir]
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append

set -uo pipefail
RUN_DIR=$1; TASK=$2; LANGS=$3; REPO=${4:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source slurm/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAX_RESTARTS=${AFB_MAX_RESTARTS:-6}

echo "== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) task=$TASK langs=$LANGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Per-job JIT caches on node-local disk: parallel jobs sharing one cache in $HOME can
# corrupt each other's compiled kernels.
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}/triton" TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}/inductor"
# Tools installed in the venv (ninja for JIT builds) must be on PATH inside the job.
export PATH="$AFB_VENV/bin:$PATH"
# Sample with vLLM's PyTorch sampler instead of FlashInfer's JIT-compiled kernels:
# no per-job compilation, same sampling parameters (temperature/top-p/top-k/min-p).
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "CUDA_HOME=${CUDA_HOME:-unset} nvcc=$(command -v nvcc || echo none) ninja=$(command -v ninja || echo none)"
"$AFB_VENV/bin/python" -c "import torch, triton, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| triton', triton.__version__, '| vllm', vllm.__version__)"

# 10 min before the 3 h limit: requeue ourselves so the job resumes from the on-disk cache.
requeue () {
  echo "== $(date -Is) USR1: time limit close"
  if (( ${SLURM_RESTART_COUNT:-0} < MAX_RESTARTS )); then
    scontrol requeue "$SLURM_JOB_ID" && echo "== requeued"
  else
    echo "== restart budget exhausted; resubmit with slurm/submit.py (finished work is kept)"
  fi
  kill -TERM "$PID" 2>/dev/null
}
trap requeue USR1

"$AFB_VENV/bin/python" -u -m afbgen.run_eval --run-dir "$RUN_DIR" --task "$TASK" --langs "$LANGS" ${AFB_LIMIT:+--limit $AFB_LIMIT} &
PID=$!
wait "$PID"; RC=$?
# `wait` returns early when the trap fires; wait again for the python process to exit.
kill -0 "$PID" 2>/dev/null && { wait "$PID"; RC=$?; }
echo "== $(date -Is) exit $RC"
exit $RC
