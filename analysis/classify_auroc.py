#!/usr/bin/env python3
"""Layerwise internal-probe AUROC for the 20-transcript Qwen3-8B dataset
(plan.md step 3).

Adapted from ../../evaluation-awareness/cls_corr_mi/classify_auroc.py. At
n=20 there is no meaningful held-out train/test split (per plan.md's
explicit exploratory-limitation note), so this version:

  1) Reports AUROC directly on the full 20-sample set per layer, for the
     real probe direction and for each of the R random-control directions
     (max/mean across R) — AUROC is rank-based and needs no threshold, so
     this is unbiased regardless of split.
  2) Reports leave-one-out (LOO) accuracy: for each held-out transcript,
     calibrates Youden's J threshold on the other 19, predicts the held-out
     label, and aggregates accuracy across all 20 folds — avoiding the
     threshold being fit and evaluated on the same points.

AUROC is reported raw (not max(AUROC, 1-AUROC)), preserving probe
directionality, matching cls_corr_mi's convention.

Usage:
    python classify_auroc.py \
      --scores ../output/scores.npz \
      --pooling last \
      --out ../output/results/internal_auroc.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

LOG = logging.getLogger("classify_auroc")


def _youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    s = np.concatenate([pos, neg])
    thresholds = np.unique(s)
    best_j, best_t = -1.0, 0.0
    for t in thresholds:
        tp = (pos >= t).sum(); fn = len(pos) - tp
        tn = (neg < t).sum(); fp = len(neg) - tn
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _loo_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out accuracy: threshold calibrated on the other N-1 points."""
    n = len(scores)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        s_rest, y_rest = scores[mask], labels[mask]
        pos = s_rest[y_rest == 1]
        neg = s_rest[y_rest == 0]
        t = _youden_threshold(pos, neg)
        pred = int(scores[i] >= t)
        correct += int(pred == labels[i])
    return correct / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    with np.load(args.scores, allow_pickle=True) as z:
        real = z[f"scores_{args.pooling}_real"]      # [N, L]
        rand = z[f"scores_{args.pooling}_random"]     # [N, L, R]
        labels = z["labels"]                          # [N]
        n_layers = int(z["n_layers"])
        n_random = int(z["n_random"])

    layer_results: Dict[str, Dict] = {}
    for l in range(n_layers):
        real_auroc = _auroc(real[:, l], labels)
        real_loo_acc = _loo_accuracy(real[:, l], labels)

        rand_aurocs = [_auroc(rand[:, l, r], labels) for r in range(n_random)]
        rand_arr = np.array(rand_aurocs) if rand_aurocs else np.array([0.5])

        layer_results[str(l)] = {
            "real": {
                "auroc": real_auroc,
                "loo_accuracy": real_loo_acc,
            },
            "random": {
                "max_auroc_across_R": float(rand_arr.max()),
                "mean_auroc_across_R": float(rand_arr.mean()),
            },
        }

    peak_layer = max(layer_results, key=lambda k: layer_results[k]["real"]["auroc"])
    summary = {
        "model_key": "qwen3_8b",
        "pooling": args.pooling,
        "n_samples": int(len(labels)),
        "n_layers": n_layers,
        "n_random_controls": n_random,
        "peak_layer": int(peak_layer),
        "peak_auroc": layer_results[peak_layer]["real"]["auroc"],
        "peak_loo_accuracy": layer_results[peak_layer]["real"]["loo_accuracy"],
        "note": (
            "n=20: AUROC computed on the full set (rank-based, threshold-free); "
            "accuracy is leave-one-out. Exploratory effect sizes, not "
            "significance tests (cf. EXPERIMENT_SUMMARY.md)."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "layer_results": layer_results}, indent=2))
    LOG.info("Wrote %s", args.out)
    LOG.info(
        "Peak layer %s: AUROC=%.3f, LOO accuracy=%.3f",
        peak_layer,
        summary["peak_auroc"],
        summary["peak_loo_accuracy"],
    )


if __name__ == "__main__":
    main()
