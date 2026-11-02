#!/bin/bash
# Pack the Python environments into single tar files (run on a login node after setup.sh):
#   bash slurm/pack_env.sh
# Jobs extract them to the node-local disk ($SLURM_TMPDIR) and import from there
# (slurm/local_env.sh). Importing transformers opens thousands of small files; on the shared
# network disk some lookups fail on compute nodes (FileNotFoundError for files that exist).
# Each tar records the sha256 of its lock file; a job uses it only if that still matches.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
for pair in "venv:requirements/eval.lock" "venv-comet:requirements/comet.lock"; do
  name=${pair%%:*}; lock=${pair#*:}
  echo ">>> packing $AFB_ROOT/$name -> $AFB_ROOT/$name.tar"
  tar -C "$AFB_ROOT" -cf "$AFB_ROOT/$name.tar.tmp" "$name"
  sha256sum "$lock" | cut -d' ' -f1 > "$AFB_ROOT/$name.tar.lock.tmp"
  mv "$AFB_ROOT/$name.tar.tmp" "$AFB_ROOT/$name.tar"
  mv "$AFB_ROOT/$name.tar.lock.tmp" "$AFB_ROOT/$name.tar.lock"
  du -h "$AFB_ROOT/$name.tar" | cut -f1
done
