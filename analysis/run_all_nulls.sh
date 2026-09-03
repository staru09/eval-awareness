#!/usr/bin/env bash
for SLUG in qwen3-8b qwen3.6-27b-obliterated granite-4.2-30b; do
for READ in answer prompt; do
  if [ "$READ" = "answer" ]; then SF=scores.npz; AD=activations; else SF=scores_prompt.npz; AD=activations_prompt; fi
  for POOL in last mean; do
    python probe_null_controls.py --scores "../output/local/$SLUG/$SF" --dataset ../data/dataset.json \
      --activations-dir "../output/local/$SLUG/$AD" --reasoning-traces "../output/local/$SLUG/reasoning_traces.json" \
      --pooling $POOL --n-perm 5000 --band-width 3 --n-fresh-random 1000 \
      --out "../output/local/$SLUG/results/null_controls_${READ}_${POOL}.json" 2>&1 | grep -E "^scores|BAND|MCQ"
  done
done
done
echo ALL_CELLS_DONE
