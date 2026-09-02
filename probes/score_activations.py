#!/usr/bin/env python3
"""Project run_inference.py's saved per-transcript activations onto the
probe direction from compute_probe.py (plan.md step 2).

Reuses the activations already captured during generation
(../output/activations/<id>.npz, produced by run_inference.py's
teacher-forced forward pass over prompt+response) instead of re-running
inference — this script is pure linear algebra + I/O, no model load needed.

Layer-index note: run_inference.py's `pool_hidden_states` drops the
embedding layer (hidden_states[0]) before saving, so activation array index
`i` (0-based) is the output of decoder layer `i + 1`. compute_probe.py's
probe directions are keyed 0..num_hidden_layers (0 = embeddings), so the
matching probe key for activation index `i` is `i + 1`.

Output: a single scores.npz with the same schema as
../../evaluation-awareness/cls_corr_mi's score_prompts.py output, so
../analysis/classify_auroc.py can be reused with minimal changes:

    scores_last_real    float32[N, L]
    scores_mean_real    float32[N, L]
    scores_last_random  float32[N, L, R]
    scores_mean_random  float32[N, L, R]
    labels              int32[N]              # 0=deploy, 1=eval
    source              str[N]
    prompt_ids          int64[N]
    n_layers, n_random  scalars

Usage:
    python score_activations.py \
      --probe ../output/probe.pt \
      --activations-dir ../output/activations \
      --dataset ../data/dataset.json \
      --out ../output/scores.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--activations-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-key", default=None, help="label for output metadata; defaults to probe's model_path")
    args = ap.parse_args()

    probe = torch.load(args.probe, map_location="cpu")
    real_dirs = probe["real"]  # {probe_layer_key(int) -> tensor[D]}, keys 0..num_hidden_layers
    rand_dirs = probe["random"]  # {probe_layer_key(int) -> tensor[R, D]}
    n_random = int(probe["metadata"]["n_random"])

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))

    n_layers_act = None  # = model.config.num_hidden_layers, set from the first .npz
    rows_last_real, rows_mean_real = [], []
    rows_last_rand, rows_mean_rand = [], []
    labels, sources, prompt_ids = [], [], []

    for row in dataset:
        npz_path = args.activations_dir / f"{row['id']}.npz"
        if not npz_path.exists():
            raise SystemExit(
                f"Missing activations for {row['id']!r} at {npz_path}. "
                "Run run_inference.py first (without --no-hidden-states)."
            )
        with np.load(npz_path) as z:
            last_h = z["last_token"]  # [n_layers_act, d_model]
            mean_h = z["mean_token"]  # [n_layers_act, d_model]

        if n_layers_act is None:
            n_layers_act = last_h.shape[0]

        last_real_row = np.zeros(n_layers_act, dtype=np.float32)
        mean_real_row = np.zeros(n_layers_act, dtype=np.float32)
        last_rand_row = np.zeros((n_layers_act, n_random), dtype=np.float32)
        mean_rand_row = np.zeros((n_layers_act, n_random), dtype=np.float32)

        for i in range(n_layers_act):
            probe_key = i + 1  # see module docstring: activation i = decoder layer i+1
            v_real = real_dirs[probe_key].numpy()  # (D,)
            v_rand = rand_dirs[probe_key].numpy()  # (R, D)

            last_real_row[i] = float(np.dot(last_h[i], v_real))
            mean_real_row[i] = float(np.dot(mean_h[i], v_real))
            last_rand_row[i] = v_rand @ last_h[i]
            mean_rand_row[i] = v_rand @ mean_h[i]

        rows_last_real.append(last_real_row)
        rows_mean_real.append(mean_real_row)
        rows_last_rand.append(last_rand_row)
        rows_mean_rand.append(mean_rand_row)
        labels.append(int(row["label"]))
        sources.append(row["source"])
        prompt_ids.append(int(row["prompt_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        scores_last_real=np.stack(rows_last_real),
        scores_mean_real=np.stack(rows_mean_real),
        scores_last_random=np.stack(rows_last_rand),
        scores_mean_random=np.stack(rows_mean_rand),
        labels=np.array(labels, dtype=np.int32),
        source=np.array(sources),
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        n_layers=n_layers_act,
        n_random=n_random,
        model_key=args.model_key or probe["metadata"]["model_path"],
    )
    print(f"Wrote {len(dataset)} rows x {n_layers_act} layers to {args.out}")


if __name__ == "__main__":
    main()
