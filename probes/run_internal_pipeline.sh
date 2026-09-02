#!/usr/bin/env bash
# Experiment 1 (plan2.md): internal-representation pipeline for one model.
# compute_probe.py -> run_inference.py (WITH hidden states) -> score_activations.py
# -> classify_auroc.py (both last-token and mean-token pooling).
#
# Usage: run_internal_pipeline.sh <model-path> <out-slug> [--dtype bfloat16]
set -euo pipefail

MODEL_PATH="$1"
SLUG="$2"
shift 2
EXTRA_ARGS=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT_DIR="$ROOT/output/local/$SLUG"
PY=/nvme/venvs/interp/bin/python

mkdir -p "$OUT_DIR/results"

echo "=== [$SLUG] compute_probe.py ==="
"$PY" "$HERE/compute_probe.py" \
  --model-path "$MODEL_PATH" \
  --out "$OUT_DIR/probe.pt" \
  "${EXTRA_ARGS[@]}"

echo "=== [$SLUG] run_inference.py (with hidden states) ==="
cd "$ROOT/inference"
"$PY" run_inference.py \
  --dataset "$ROOT/data/dataset.json" \
  --out-dir "$OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --enable-thinking \
  "${EXTRA_ARGS[@]}"

echo "=== [$SLUG] score_activations.py ==="
"$PY" "$HERE/score_activations.py" \
  --probe "$OUT_DIR/probe.pt" \
  --activations-dir "$OUT_DIR/activations" \
  --dataset "$ROOT/data/dataset.json" \
  --out "$OUT_DIR/scores.npz" \
  --model-key "$MODEL_PATH"

echo "=== [$SLUG] classify_auroc.py (last-token) ==="
"$PY" "$ROOT/analysis/classify_auroc.py" \
  --scores "$OUT_DIR/scores.npz" \
  --pooling last \
  --out "$OUT_DIR/results/internal_auroc_last.json"

echo "=== [$SLUG] classify_auroc.py (mean-token) ==="
"$PY" "$ROOT/analysis/classify_auroc.py" \
  --scores "$OUT_DIR/scores.npz" \
  --pooling mean \
  --out "$OUT_DIR/results/internal_auroc_mean.json"

echo "=== [$SLUG] DONE ==="
