# Source me in a job after slurm/env.sh:  source slurm/local_env.sh venv|venv-comet
# Extracts the packed environment (slurm/pack_env.sh) to the node-local disk and points
# AFB_VENV / AFB_VENV_COMET at it. Falls back to the network copy if there is no tar, no
# $SLURM_TMPDIR, or the tar was packed from a different lock file.
_afb_name=${1:-venv}
_afb_lock=requirements/eval.lock; [[ "$_afb_name" == venv-comet ]] && _afb_lock=requirements/comet.lock
_afb_tar="$AFB_ROOT/$_afb_name.tar"
if [[ -n "${SLURM_TMPDIR:-}" && -f "$_afb_tar" && -f "$_afb_tar.lock" ]] \
   && [[ "$(cat "$_afb_tar.lock")" == "$(sha256sum "$_afb_lock" | cut -d' ' -f1)" ]]; then
  _afb_t0=$SECONDS
  if tar -xf "$_afb_tar" -C "$SLURM_TMPDIR"; then
    if [[ "$_afb_name" == venv ]]; then export AFB_VENV="$SLURM_TMPDIR/venv"; else export AFB_VENV_COMET="$SLURM_TMPDIR/venv-comet"; fi
    echo "local_env.sh: using node-local $_afb_name ($((SECONDS - _afb_t0))s to extract)"
  else
    echo "local_env.sh: extracting $_afb_tar failed; using the network copy" >&2
  fi
else
  echo "local_env.sh: no matching $_afb_name.tar (run slurm/pack_env.sh); using the network copy" >&2
fi
unset _afb_name _afb_lock _afb_tar _afb_t0
