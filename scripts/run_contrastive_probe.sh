#!/usr/bin/env bash
# Contrastive-probe experiment, end to end, on the VM.
# Every step is skipped when its output already exists, so rerunning resumes
# after a crash. Results land in output/<date>_<model>_<name>/.
#
# Usage (inside tmux):
#   bash scripts/run_contrastive_probe.sh openbmb/MiniCPM5-2B minicpm5-2b [name] [dataset]
# Set MAX_NEW=16384 (etc.) to change the reply token limit; the default is 4096.
# Set GEN_ARGS to pass sampling flags to generate, e.g.
#   GEN_ARGS="--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5"
set -euo pipefail
cd "$(dirname "$0")/.."

M=$1
SLUG=$2                      # must be a key in plots.SLUGS
NAME=${3:-contrastive-probe}
DS=${4:-data/exp_data.json}
RUN=${RUN:-output/$(date +%F)_${M##*/}_${NAME}}
D=$RUN/local/$SLUG           # plots expects <root>/local/<slug>/results
mkdir -p "$D/results" "$RUN/logs"
exec > >(tee -a "$RUN/logs/run.log") 2>&1

export PATH=/nvme/venv/bin:$PATH HF_HOME=/nvme/hf-cache TORCH_HOME=/nvme/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoids fragmentation OOM on long rows
PY="python -m evalaware.cli"
step() { echo "== $(date +%T) $*"; }
# HF model loading (the probe step) needs accelerate; the /nvme venv loses it on every stop.
python -c "import accelerate" 2>/dev/null || UV_CACHE_DIR=/nvme/.cache/uv uv pip install --python "$(command -v python)" accelerate

[ -f "$RUN/run_info.json" ] || cat > "$RUN/run_info.json" <<EOF
{"model": "$M", "slug": "$SLUG", "dataset": "$DS", "run_dir": "$RUN", "max_new_tokens": ${MAX_NEW:-4096}, "gen_args": "${GEN_ARGS:-}",
 "git": "$(git describe --always --dirty)", "started": "$(date -Is)"}
EOF
cp -n "$DS" "$RUN/" || true

if [ ! -f "$D/.generate_done" ]; then
  step "generate (vLLM, all rows batched) + activations"
  $PY generate --backend vllm --model-path "$M" --dataset "$DS" --out-dir "$D" --max-new-tokens "${MAX_NEW:-4096}" ${GEN_ARGS:-}
  touch "$D/.generate_done"
fi

[ -f "$D/probe.pt" ] || { step probe; $PY probe --model-path "$M" --out "$D/probe.pt"; }

for R in answer prompt; do
  ACT=$D/activations; SC=$D/scores.npz; SUF=""
  [ $R = prompt ] && { ACT=$D/activations_prompt; SC=$D/scores_prompt.npz; SUF=_prompt; }
  [ -f "$SC" ] || { step "score $R"; $PY score --probe "$D/probe.pt" --activations-dir "$ACT" --dataset "$DS" --out "$SC"; }
  for P in last mean; do
    OUT=$D/results/internal_auroc${SUF}_$P.json
    [ -f "$OUT" ] || { step "auroc $R $P"; $PY auroc --scores "$SC" --pooling $P --out "$OUT"; }
    OUT=$D/results/nulls_${R}_$P.json
    [ -f "$OUT" ] || { step "nulls $R $P"; $PY nulls --scores "$SC" --activations-dir "$ACT" \
        --traces "$D/reasoning_traces.json" --dataset "$DS" --pooling $P --out "$OUT"; }
  done
done

step plots
$PY plots --output-dir "$RUN"

date -Is > "$RUN/.done"
step "done: $RUN"
