#!/usr/bin/env python3
"""Honest null models and confound baselines for the internal-probe AUROC.

classify_auroc.py reports `peak_auroc = max over layers` but compares it to
a random-control band computed *at that same layer* (classify_auroc.py:102,
:116). That comparison is mis-specified: the real probe gets to pick its
best of 36-64 layers, the control does not. At n=20 the selection is worth
a lot, so the reported "beats the random band" can be an artifact of the
maximum, not evidence of signal.

This script computes the comparisons that are actually valid, all as pure
post-processing on files that already exist (no GPU, no model load):

  1. max-over-layers random null. Each of the R random directions gets the
     same privilege as the real probe -- its own best layer. The real
     peak is then scored against that distribution as an empirical
     p-value, (#{random >= real} + 1) / (R + 1).

  2. label-permutation null. Shuffle the 20 eval/deploy labels `--n-perm`
     times and recompute the real probe's own max-over-layers AUROC each
     time. This is the stronger test: it holds the probe, the activations
     and the layer-selection procedure fixed and varies only the thing
     being predicted.

  3. cosine-normalised probe score. score_activations.py:91 stores an
     unnormalised dot product h.v_hat, so the activation *magnitude*
     |h| leaks into the score. Dividing it out asks whether the probe
     separates by direction or merely by norm.

  4. two confound baselines that use no probe at all:
       - |h| alone, per layer (does activation magnitude classify?)
       - prompt token count alone (do eval prompts just run longer?)
     A probe that does not clearly beat these is not measuring what the
     experiment claims to measure.

  5. format-vs-evaluation split. Our eval class is mostly multiple-choice
     (hellaswag/arc/mmlu) while the deploy class is all conversational, so
     "detects a benchmark-formatted prompt" and "detects an evaluation"
     make near-identical predictions. Reported here as: AUROC for the MCQ
     split, and AUROC restricted to the non-MCQ transcripts (the eval
     prompts that are *not* formatted like a benchmark item, vs. deploy),
     which is the only format-controlled contrast this dataset supports.

Usage:
    python probe_null_controls.py \
      --scores ../output/local/qwen3-8b/scores_prompt.npz \
      --activations-dir ../output/local/qwen3-8b/activations_prompt \
      --dataset ../data/dataset.json \
      --reasoning-traces ../output/local/qwen3-8b/reasoning_traces.json \
      --pooling last \
      --out ../output/local/qwen3-8b/results/null_controls_prompt_last.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Eval sources whose prompts are literally formatted as multiple-choice
# items. AgentHarm is the only eval source that is not, which is what makes
# the format-controlled contrast in (5) possible at all -- on 2 positives.
MCQ_SOURCES = {"hellaswag", "arc_challenge", "mmlu"}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def peak_over_layers(scores_2d: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """(best AUROC, its layer) for a [N, L] score matrix, one-sided like
    classify_auroc.py:116 -- max of the raw AUROC, not max |AUROC - 0.5|."""
    per_layer = np.array([auroc(scores_2d[:, l], labels) for l in range(scores_2d.shape[1])])
    l = int(np.argmax(per_layer))
    return float(per_layer[l]), l


def _layer_ranks(scores_2d: np.ndarray) -> np.ndarray:
    """[N, L] of average ranks within each layer, for the fast AUROC below."""
    return np.column_stack([rankdata(scores_2d[:, l]) for l in range(scores_2d.shape[1])])


def peaks_from_ranks(ranks: np.ndarray, label_matrix: np.ndarray) -> np.ndarray:
    """Max-over-layers AUROC for many label vectors at once.

    Uses the rank-sum identity AUROC = (sum of positive ranks - n_pos(n_pos+1)/2)
    / (n_pos * n_neg), which matches sklearn exactly including tie handling
    (average ranks). Permuting labels only re-selects which ranks count as
    positive, so the ranks are computed once and 2000 permutations cost one
    matrix product instead of 2000 * n_layers sklearn calls.

    ranks:        [N, L]
    label_matrix: [P, N] of 0/1, one row per permutation
    returns:      [P] of max-over-layers AUROC
    """
    n_pos = label_matrix.sum(axis=1)                      # [P]
    n_neg = label_matrix.shape[1] - n_pos
    rank_sums = label_matrix @ ranks                      # [P, L]
    aurocs = (rank_sums - (n_pos * (n_pos + 1) / 2)[:, None]) / (n_pos * n_neg)[:, None]
    return aurocs.max(axis=1)


def activation_norms(act_dir: Path, ids: list[str], pooling: str) -> np.ndarray | None:
    """[N, L] of ||h|| per transcript per layer, or None if unavailable."""
    key = "last_token" if pooling == "last" else "mean_token"
    rows = []
    for tid in ids:
        p = act_dir / f"{tid}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            rows.append(np.linalg.norm(z[key], axis=1))  # [L]
    return np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--activations-dir", type=Path, default=None,
                    help="the dir the --scores came from; enables the cosine and |h| controls")
    ap.add_argument("--reasoning-traces", type=Path, default=None,
                    help="enables the prompt-length baseline (reads n_prompt_tokens)")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with np.load(args.scores, allow_pickle=True) as z:
        real = z[f"scores_{args.pooling}_real"]        # [N, L]
        rand = z[f"scores_{args.pooling}_random"]      # [N, L, R]
        labels = z["labels"].astype(int)               # [N]
        prompt_ids = z["prompt_ids"]
        n_layers = int(z["n_layers"])
        n_random = int(z["n_random"])
        model_key = str(z["model_key"]) if "model_key" in z else "unknown"

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    by_pid = {int(r["prompt_id"]): r for r in dataset}
    # score_activations.py writes rows in dataset order, but join on
    # prompt_id anyway so this stays correct if that ever changes.
    ids = [by_pid[int(p)]["id"] for p in prompt_ids]
    sources = np.array([by_pid[int(p)]["source"] for p in prompt_ids])

    out: dict = {
        "summary": {
            "model_key": model_key,
            "pooling": args.pooling,
            "scores_file": str(args.scores),
            "n_samples": int(len(labels)),
            "n_layers": n_layers,
            "n_random_controls": n_random,
            "n_permutations": args.n_perm,
        }
    }

    # --- 0) the headline the existing pipeline reports -------------------
    real_peak, real_peak_layer = peak_over_layers(real, labels)
    out["real"] = {"peak_auroc": real_peak, "peak_layer": real_peak_layer}

    # --- 1) max-over-layers random null ----------------------------------
    rand_peaks = np.array([peak_over_layers(rand[:, :, r], labels)[0] for r in range(n_random)])
    p_rand = (int((rand_peaks >= real_peak).sum()) + 1) / (n_random + 1)
    out["random_direction_null"] = {
        "note": "each random direction gets its own best layer, same as the real probe",
        "peaks": [float(v) for v in rand_peaks],
        "mean": float(rand_peaks.mean()),
        "max": float(rand_peaks.max()),
        "n_beating_real": int((rand_peaks >= real_peak).sum()),
        "empirical_p": float(p_rand),
    }

    # --- 2) label-permutation null ---------------------------------------
    ranks = _layer_ranks(real)
    perm_labels = np.stack([rng.permutation(labels) for _ in range(args.n_perm)]).astype(float)
    perm_peaks = peaks_from_ranks(ranks, perm_labels)
    p_perm = (int((perm_peaks >= real_peak).sum()) + 1) / (args.n_perm + 1)
    out["label_permutation_null"] = {
        "note": "same probe, same activations, same argmax-over-layers; only the labels are shuffled",
        "mean": float(perm_peaks.mean()),
        "p95": float(np.percentile(perm_peaks, 95)),
        "p99": float(np.percentile(perm_peaks, 99)),
        "max": float(perm_peaks.max()),
        "n_beating_real": int((perm_peaks >= real_peak).sum()),
        "empirical_p": float(p_perm),
    }

    # --- 3/4) magnitude controls -----------------------------------------
    norms = activation_norms(args.activations_dir, ids, args.pooling) if args.activations_dir else None
    if norms is not None:
        cos = real / np.maximum(norms, 1e-8)
        cos_peak, cos_layer = peak_over_layers(cos, labels)
        norm_peak, norm_layer = peak_over_layers(norms, labels)
        out["cosine_normalised"] = {
            "note": "probe score divided by ||h||: does the probe separate by direction, or by magnitude?",
            "peak_auroc": cos_peak, "peak_layer": cos_layer,
        }
        out["norm_only_baseline"] = {
            "note": "||h|| alone, no probe direction at all",
            "peak_auroc": norm_peak, "peak_layer": norm_layer,
        }
    else:
        out["cosine_normalised"] = None
        out["norm_only_baseline"] = None

    if args.reasoning_traces is not None:
        traces = json.loads(args.reasoning_traces.read_text(encoding="utf-8"))
        tok_by_pid = {int(t["prompt_id"]): t.get("n_prompt_tokens") for t in traces}
        lens = np.array([tok_by_pid.get(int(p)) or 0 for p in prompt_ids], dtype=float)
        out["prompt_length_baseline"] = {
            "note": "prompt token count alone, no model internals at all",
            "auroc": auroc(lens, labels),
        }
    else:
        out["prompt_length_baseline"] = None

    # --- 5) format vs evaluation -----------------------------------------
    is_mcq = np.array([s in MCQ_SOURCES for s in sources]).astype(int)
    mcq_peak, mcq_layer = peak_over_layers(real, is_mcq)
    keep = is_mcq == 0
    ctrl_labels = labels[keep]
    if np.unique(ctrl_labels).size > 1:
        ctrl_peak, ctrl_layer = peak_over_layers(real[keep], ctrl_labels)
        ctrl = {
            "peak_auroc": ctrl_peak, "peak_layer": ctrl_layer,
            "n_eval": int((ctrl_labels == 1).sum()), "n_deploy": int((ctrl_labels == 0).sum()),
        }
    else:
        ctrl = None
    out["format_confound"] = {
        "note": (
            "our eval class is mostly MCQ and our deploy class is entirely "
            "conversational, so 'detects a benchmark-formatted prompt' predicts "
            "almost the same labels as 'detects an evaluation'"
        ),
        "mcq_split_peak_auroc": mcq_peak,
        "mcq_split_peak_layer": mcq_layer,
        "n_mcq": int(is_mcq.sum()),
        "format_controlled": ctrl,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{args.scores.name} [{args.pooling}]  real peak AUROC = {real_peak:.3f} @L{real_peak_layer}")
    print(f"  random-direction null (own best layer): mean={rand_peaks.mean():.3f} "
          f"max={rand_peaks.max():.3f}  p={p_rand:.3f}")
    print(f"  label-permutation null:                 mean={perm_peaks.mean():.3f} "
          f"p95={np.percentile(perm_peaks, 95):.3f}  p={p_perm:.4f}")
    if norms is not None:
        print(f"  cosine-normalised peak: {out['cosine_normalised']['peak_auroc']:.3f}"
              f"   ||h||-only baseline: {out['norm_only_baseline']['peak_auroc']:.3f}")
    if out["prompt_length_baseline"]:
        print(f"  prompt-length-only baseline AUROC: {out['prompt_length_baseline']['auroc']:.3f}")
    print(f"  MCQ-split peak AUROC: {mcq_peak:.3f}"
          + (f"   format-controlled (n={ctrl['n_eval']}v{ctrl['n_deploy']}): {ctrl['peak_auroc']:.3f}"
             if ctrl else ""))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
