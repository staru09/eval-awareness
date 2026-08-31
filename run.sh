#!/usr/bin/env bash
# Orchestrates the full Qwen3-8B evaluation-awareness pipeline (plan.md
# steps 0-5), mirroring
# ../evaluation-awareness/cls_corr_mi/reproduce.sh's structure.
#
# Usage:
#   ./run.sh [--out-root output] [--model-path Qwen/Qwen3-8B] [--dtype bfloat16] [--skip-dataset]
#
# Requires a CUDA GPU with enough VRAM for Qwen3-8B (bf16, ~16GB) — see
# inference/model_utils.py. Uses uv for all package management, per this
# workspace's convention (uv sync / uv run), not raw pip.

set -euo pipefail

OUT_ROOT="output"
MODEL_PATH="Qwen/Qwen3-8B"
DTYPE="bfloat16"
SKIP_DATASET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT/results"

if [[ "$SKIP_DATASET" -eq 0 ]]; then
  echo "== Step 0: build_dataset =="
  ( cd data && uv run python build_dataset.py )
fi

echo "== Step 1: compute_probe =="
( cd probes && uv run python compute_probe.py \
    --model-path "$MODEL_PATH" --dtype "$DTYPE" \
    --out "../$OUT_ROOT/probe.pt" )

echo "== Step (inference): run_inference =="
( cd inference && uv run python run_inference.py \
    --dataset ../data/dataset.json --out-dir "../$OUT_ROOT" \
    --model-path "$MODEL_PATH" --dtype "$DTYPE" )

echo "== Step 2: score_activations =="
( cd probes && uv run python score_activations.py \
    --probe "../$OUT_ROOT/probe.pt" \
    --activations-dir "../$OUT_ROOT/activations" \
    --dataset ../data/dataset.json \
    --out "../$OUT_ROOT/scores.npz" )

echo "== Step 3: classify_auroc (internal probe) =="
( cd analysis && uv run python classify_auroc.py \
    --scores "../$OUT_ROOT/scores.npz" --pooling last \
    --out "../$OUT_ROOT/results/internal_auroc.json" )

echo "== Step 4: verbalize_probe =="
( cd probes && uv run python verbalize_probe.py \
    --reasoning-traces "../$OUT_ROOT/reasoning_traces.jsonl" \
    --probe-questions probe_questions.yaml \
    --model-path "$MODEL_PATH" --dtype "$DTYPE" \
    --out-csv "../$OUT_ROOT/verbalized_responses.csv" \
    --out-json "../$OUT_ROOT/results/verbalized_auroc.json" )

echo "== Step 5: compare_internal_vs_verbal =="
( cd analysis && uv run python compare_internal_vs_verbal.py \
    --internal-auroc "../$OUT_ROOT/results/internal_auroc.json" \
    --verbalized-auroc "../$OUT_ROOT/results/verbalized_auroc.json" \
    --scores "../$OUT_ROOT/scores.npz" \
    --verbalized-csv "../$OUT_ROOT/verbalized_responses.csv" \
    --pooling last \
    --out "../$OUT_ROOT/COMPARISON_REPORT.md" )

echo "Done. See $OUT_ROOT/COMPARISON_REPORT.md"
