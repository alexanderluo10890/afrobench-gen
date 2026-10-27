#!/bin/bash
# One-time environment setup, run on a LOGIN node (needs internet):
#   bash slurm/setup.sh
# Builds two uv venvs with Python 3.11. If requirements/*.lock exist they are installed
# exactly (uv pip sync); otherwise the top-level requirements are resolved and the lock
# files are written. Commit the lock files so every later install is byte-identical.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

make_env () {  # $1 venv  $2 requirements.in  $3 lock
  [[ -x "$1/bin/python" ]] || uv venv --python 3.11 "$1"
  if [[ -f "$3" ]]; then
    VIRTUAL_ENV="$1" uv pip sync "$3"
  else
    VIRTUAL_ENV="$1" uv pip install -r "$2"
    VIRTUAL_ENV="$1" uv pip freeze > "$3"
    echo ">>> wrote $3 -- commit it"
  fi
}

make_env "$AFB_VENV"       requirements/eval.in  requirements/eval.lock
make_env "$AFB_VENV_COMET" requirements/comet.in requirements/comet.lock
# this repo itself (no deps: they come from the lock files)
VIRTUAL_ENV="$AFB_VENV"       uv pip install --no-deps -e .
VIRTUAL_ENV="$AFB_VENV_COMET" uv pip install --no-deps -e .

"$AFB_VENV/bin/python" -c "import vllm, lm_eval; print('vllm', vllm.__version__)"
"$AFB_VENV_COMET/bin/python" -c "import comet; print('comet ok')"
echo "setup done. next: $AFB_VENV/bin/python -m afbgen.prepare"
