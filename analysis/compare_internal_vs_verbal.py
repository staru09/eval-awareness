#!/usr/bin/env python3
"""Experiment 2 (plan2.md): internal-vs-verbalized coupling.

Pairs each transcript's per-layer internal probe score (from
score_activations.py's scores.npz) with its verbalized eval_signal (from
run_inference_api.py's verbalized_responses.csv), per layer, per probe
question, and computes Spearman rank correlation and Kraskov k-NN mutual
information between the two -- replicating the paper's Appendix G
methodology (arXiv:2608.21766). A random-direction baseline (the same R
random directions already saved in scores.npz by compute_probe.py) is
computed the same way, and the max across R is reported per layer as a
conservative baseline, matching the paper's own convention.

Pure analysis -- no model load, no GPU needed.

Usage:
    python compare_internal_vs_verbal.py \
      --scores ../output/local/<model>/scores.npz \
      --verbalized-csv ../output/local/<model>/verbalized_responses.csv \
      --pooling last \
      --out ../output/local/<model>/results/coupling.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

PARSE_FAILURE_VALUE = -1.0


def load_verbalized(csv_path: Path) -> dict[str, dict[int, float]]:
    """probe_id -> {prompt_id: eval_signal}, excluding parse failures."""
    out: dict[str, dict[int, float]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sig = float(r["eval_signal"])
            if sig == PARSE_FAILURE_VALUE:
                continue
            out.setdefault(r["probe_id"], {})[int(r["prompt_id"])] = sig
    return out


def _mi(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Kraskov-style (KSG) continuous mutual information via sklearn's k-NN
    estimator, matching the paper's Appendix G.2 tooling. Returns nats.
    """
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0
    mi = mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False, random_state=seed)
    return float(mi[0])


def coupling_for_layer(
    internal_scores: np.ndarray, verbal_signal: dict[int, float], prompt_ids: np.ndarray, seed: int
) -> tuple[float | None, float, bool]:
    """Spearman rho, MI, and a `degenerate` flag between one layer's internal
    scores and one probe's verbalized eval_signal, joined by prompt_id
    (robust to the two pipelines processing dataset.json in different
    orders). `degenerate=True` means the verbalized signal had zero
    variance (e.g. the model gave the identical answer on every transcript)
    -- correlation/MI are mathematically undefined in that case, not a
    measured "no coupling" result, so callers must not report 0.0 as if it
    were a real null finding.
    """
    pairs = [
        (internal_scores[i], verbal_signal[pid])
        for i, pid in enumerate(prompt_ids)
        if int(pid) in verbal_signal
    ]
    if len(pairs) < 3:
        return None, 0.0, True
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if len(np.unique(y)) < 2:
        return None, 0.0, True
    rho, _ = spearmanr(x, y)
    mi = _mi(x, y, seed)
    return (float(rho) if rho is not None and not np.isnan(rho) else 0.0), mi, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--verbalized-csv", required=True, type=Path)
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with np.load(args.scores, allow_pickle=True) as z:
        real = z[f"scores_{args.pooling}_real"]       # [N, L]
        rand = z[f"scores_{args.pooling}_random"]      # [N, L, R]
        prompt_ids = z["prompt_ids"]                    # [N]
        n_layers = int(z["n_layers"])
        n_random = int(z["n_random"])
        model_key = str(z["model_key"]) if "model_key" in z else "unknown"

    verbal_by_probe = load_verbalized(args.verbalized_csv)
    probe_ids = sorted(verbal_by_probe)

    def compute_layers(target_signal: dict[int, float]) -> dict:
        layer_rows = []
        any_degenerate = False
        for layer in range(n_layers):
            rho, mi, degenerate = coupling_for_layer(real[:, layer], target_signal, prompt_ids, args.seed)
            any_degenerate = any_degenerate or degenerate
            rand_rhos, rand_mis = [], []
            for r in range(n_random):
                rr, rm, _ = coupling_for_layer(rand[:, layer, r], target_signal, prompt_ids, args.seed + r + 1)
                rand_rhos.append(abs(rr) if rr is not None else 0.0)
                rand_mis.append(rm)
            layer_rows.append({
                "layer": layer,
                "spearman_rho": rho,
                "mutual_info_nats": mi,
                "random_max_abs_rho": float(max(rand_rhos)) if rand_rhos else 0.0,
                "random_max_mi_nats": float(max(rand_mis)) if rand_mis else 0.0,
            })
        if any_degenerate:
            # The verbalized signal had zero variance (e.g. the model gave
            # an identical answer on every transcript) -- correlation/MI
            # are undefined, not a measured "no coupling" result. Surface
            # this explicitly rather than reporting misleading 0.0 peaks.
            return {
                "layers": layer_rows,
                "degenerate": True,
                "degenerate_reason": (
                    "verbalized eval_signal has zero variance across transcripts "
                    "(e.g. the model answered identically every time) -- "
                    "correlation/MI are mathematically undefined here, not zero"
                ),
                "peak_abs_rho_layer": None,
                "peak_abs_rho": None,
                "peak_mi_layer": None,
                "peak_mi_nats": None,
            }
        abs_rhos = [row["spearman_rho"] for row in layer_rows]
        peak_rho_idx = int(np.argmax([abs(r) for r in abs_rhos]))
        peak_mi_idx = int(np.argmax([row["mutual_info_nats"] for row in layer_rows]))
        return {
            "layers": layer_rows,
            "degenerate": False,
            "peak_abs_rho_layer": peak_rho_idx,
            "peak_abs_rho": abs(layer_rows[peak_rho_idx]["spearman_rho"]),
            "peak_mi_layer": peak_mi_idx,
            "peak_mi_nats": layer_rows[peak_mi_idx]["mutual_info_nats"],
        }

    per_probe = {pid_str: compute_layers(verbal_by_probe[pid_str]) for pid_str in probe_ids}

    # Ensemble: mean eval_signal across all 5 probes per transcript (same
    # convention as run_inference_api.py's ensemble_auroc), coupled against
    # the same per-layer internal scores.
    all_prompt_ids = sorted({pid for pmap in verbal_by_probe.values() for pid in pmap})
    ensemble_signal = {}
    for pid in all_prompt_ids:
        vals = [pmap[pid] for pmap in verbal_by_probe.values() if pid in pmap]
        if vals:
            ensemble_signal[pid] = float(np.mean(vals))

    result = {
        "model_key": model_key,
        "pooling": args.pooling,
        "n_layers": n_layers,
        "n_random_controls": n_random,
        "n_probes": len(probe_ids),
        "per_probe": per_probe,
        "ensemble": compute_layers(ensemble_signal),
        "note": (
            "n=20: exploratory effect sizes, not significance tests. "
            "random_max_* is the max |rho|/MI across the same R random "
            "directions used for classify_auroc.py's control band, matching "
            "the paper's Appendix G.2 baseline convention."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for pid_str, info in per_probe.items():
        if info["degenerate"]:
            print(f"  [{pid_str}] degenerate: {info['degenerate_reason']}")
        else:
            print(
                f"  [{pid_str}] peak |rho|={info['peak_abs_rho']:.3f} @L{info['peak_abs_rho_layer']}, "
                f"peak MI={info['peak_mi_nats']:.4f} @L{info['peak_mi_layer']}"
            )


if __name__ == "__main__":
    main()
