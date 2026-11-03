#!/bin/bash -l
# CulturalRiddles for one model: vLLM server on this node + the authors' runner + their scorer.
#   sbatch $AFB_SBATCH_ARGS slurm/riddles.sbatch <run_dir> <model> [repo_dir]
# Smoke test: AFB_LIMIT=20 sbatch ... (first 20 riddles; use a separate run_dir)
# Resumable: the runner caches every answer in <run_dir>/culturalriddles/cache.
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600

set -uo pipefail
RUN_DIR=$1; MODEL=$2; REPO=${3:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source slurm/env.sh
# import from the node-local disk, not the shared network disk (slurm/pack_env.sh)
source slurm/local_env.sh venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY=EMPTY
export PATH="$AFB_VENV/bin:$PATH"
MAX_RESTARTS=${AFB_MAX_RESTARTS:-6}
OUT="$RUN_DIR/culturalriddles"; mkdir -p "$OUT/raw" "$OUT/cache"
RD="$AFB_ROOT/CulturalRiddles"
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
echo "== $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) model=$MODEL port=$PORT"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "not an L40S"; exit 1; }

finish() {  # requeue on time limit or failure; finished answers are cached
  [[ -n "${SERVER:-}" ]] && kill "$SERVER" 2>/dev/null
  if (( ${SLURM_RESTART_COUNT:-0} < MAX_RESTARTS )); then scontrol requeue "$SLURM_JOB_ID" && echo "== requeued"; fi
  exit 1
}
trap finish USR1

read -r MDIR TOK LMO MAXLEN < <("$AFB_VENV/bin/python" -c "
from afbgen import models; m = models.get('$MODEL')
print(models.model_dir('$MODEL'), models.tokenizer_path('$MODEL'), int(bool(m.get('language_model_only'))), min(8192, m.get('max_model_len') or 8192))")
"$AFB_VENV/bin/python" -c "from afbgen import models, run_eval; run_eval.check_model('$MODEL')" || exit 1

vllm serve "$MDIR" --served-model-name "$MODEL" --tokenizer "$TOK" --dtype bfloat16 \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 --seed 1234 --host 127.0.0.1 --port "$PORT" \
  $([[ $LMO == 1 ]] && echo --language-model-only) > "$OUT/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "vLLM server died"; tail -20 "$OUT/vllm-$SLURM_JOB_ID.log"; finish; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM server did not start"; finish; }
echo "== $(date -Is) server up"

"$AFB_VENV/bin/python" -m afbgen.riddles config --model "$MODEL" --port "$PORT" --cache "$OUT/cache" --out "$OUT/config.yaml"
(cd "$RD" && "$RD/.venv/bin/riddles-framework" --config "$OUT/config.yaml" --output-root "$OUT/raw" ${AFB_LIMIT:+--limit $AFB_LIMIT}) &
wait $! || finish
kill "$SERVER" 2>/dev/null

RES=$(ls -t "$OUT"/raw/*/results.jsonl | head -1)
SCORED="$RD/data/runs/afbgen/$(basename "$RUN_DIR")-$MODEL/scored.parquet"; mkdir -p "$(dirname "$SCORED")"
(cd "$RD" && .venv/bin/python -m analysis.scoring --responses "$RES" --model "$MODEL" --out "$SCORED") || finish
cp "$SCORED" "$OUT/scored.parquet"
"$AFB_VENV/bin/python" -m afbgen.riddles summary --scored "$OUT/scored.parquet" --model "$MODEL" --out "$OUT/summary.json"
echo "== $(date -Is) done"
