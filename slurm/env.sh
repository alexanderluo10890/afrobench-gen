# Source me:  source slurm/env.sh
# Detects the cluster and sets paths. Everything large lives on scratch.

if [[ -n "${AFB_CLUSTER:-}" ]]; then :;
elif [[ -d /network/scratch ]]; then AFB_CLUSTER=mila;
elif [[ "${CC_CLUSTER:-}" == vulcan || "$(hostname -f 2>/dev/null)" == *vulcan* ]]; then AFB_CLUSTER=vulcan;
else echo "env.sh: unknown cluster; set AFB_CLUSTER=mila|vulcan" >&2; return 1 2>/dev/null || exit 1; fi
export AFB_CLUSTER

: "${SCRATCH:?SCRATCH is not set}"
export AFB_ROOT="${AFB_ROOT:-$SCRATCH/afrobench-gen}"
export AFB_DATA_DIR="$AFB_ROOT/data"
export AFB_RESULTS="$AFB_ROOT/results"
export AFB_VENV="$AFB_ROOT/venv"            # vLLM + lm-eval
export AFB_VENV_COMET="$AFB_ROOT/venv-comet" # unbabel-comet (conflicting pins, kept separate)
export HF_HOME="$AFB_ROOT/hf_home"
export UV_CACHE_DIR="$AFB_ROOT/uv_cache"
export VLLM_CACHE_ROOT="$AFB_ROOT/vllm_cache"
export PATH="$HOME/.local/bin:$PATH"

case "$AFB_CLUSTER" in
  # Mila: `long` partition (no GPU quota limit); L40S only.
  mila)   export AFB_SBATCH_ARGS="--partition=long" ;;
  # Vulcan (Alliance / Amii): allocation account; L40S nodes.
  vulcan) export AFB_SBATCH_ARGS="--account=${AFB_ACCOUNT:-aip-davlan}" ;;
esac

# vLLM JIT-compiles some kernels (FlashInfer/Triton) at startup and needs a CUDA toolkit
# (nvcc + headers) matching the CUDA major version PyTorch was built with.
# Override with AFB_CUDA_MODULE=<module name>.
if [[ -z "${AFB_CUDA_MODULE:-}" ]]; then
  _afb_tc=$("$AFB_VENV/bin/python" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null)
  case "$AFB_CLUSTER:${_afb_tc%%.*}" in
    mila:13)   AFB_CUDA_MODULE=cuda/13.1.0 ;;
    mila:*)    AFB_CUDA_MODULE=cuda/12.6.0 ;;
    vulcan:13) AFB_CUDA_MODULE=cuda/13 ;;
    vulcan:*)  AFB_CUDA_MODULE=cuda/12 ;;
  esac
fi
if command -v module >/dev/null 2>&1; then
  if module load "$AFB_CUDA_MODULE" 2>/tmp/afb-module-$USER.err; then
    echo "env.sh: loaded $AFB_CUDA_MODULE (torch cuda ${_afb_tc:-?})" >&2
  else
    echo "env.sh: could not load $AFB_CUDA_MODULE:" >&2; tail -3 /tmp/afb-module-$USER.err >&2
  fi
else
  echo "env.sh: no 'module' command in this shell" >&2
fi
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

mkdir -p "$AFB_DATA_DIR" "$AFB_RESULTS" "$HF_HOME" "$AFB_ROOT/logs"
