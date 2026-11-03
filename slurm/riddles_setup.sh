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
(cd "$DIR" && uv sync --frozen -q)
"$DIR/.venv/bin/riddles-framework" --help >/dev/null
echo ">>> CulturalRiddles @ ${COMMIT:0:10} ready in $DIR"
