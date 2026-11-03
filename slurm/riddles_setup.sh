#!/bin/bash
# One-time setup on a login node: the authors' CulturalRiddles repo at the pinned commit, with its
# own uv environment (runner + scorer; no torch). Rerun after a pin change.
#   bash slurm/riddles_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
COMMIT=$("$AFB_VENV/bin/python" -c "from afbgen import config; print(config.load('pins')['github']['culturalriddles']['commit'])")
DIR="$AFB_ROOT/CulturalRiddles"
[[ -d "$DIR/.git" ]] || git clone -q https://github.com/Cohere-Labs-Community/CulturalRiddles.git "$DIR"
git -C "$DIR" fetch -q origin "$COMMIT" 2>/dev/null || git -C "$DIR" fetch -q origin
git -C "$DIR" -c advice.detachedHead=false checkout -q "$COMMIT"
# Use a uv-managed Python (in $HOME, visible on every node). A system Python found on the login
# node may not exist on compute nodes ("bad interpreter" in the jobs).
PY=$(readlink -f "$DIR/.venv/bin/python" 2>/dev/null || true)
if [[ -n "$PY" && "$PY" != "$HOME"/* ]]; then echo ">>> rebuilding .venv (interpreter $PY is not uv-managed)"; rm -rf "$DIR/.venv"; fi
(cd "$DIR" && UV_PYTHON_PREFERENCE=only-managed uv sync --frozen -q)
PY=$(readlink -f "$DIR/.venv/bin/python")
[[ "$PY" == "$HOME"/* ]] || { echo "interpreter $PY is outside \$HOME; compute nodes may not see it"; exit 1; }
"$DIR/.venv/bin/riddles-framework" --help >/dev/null
echo ">>> interpreter: $PY"
echo ">>> CulturalRiddles @ ${COMMIT:0:10} ready in $DIR"
