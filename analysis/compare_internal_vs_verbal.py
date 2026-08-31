#!/usr/bin/env python3
"""Internal (activation-probe) vs. stated (verbalized) eval-awareness
comparison for Qwen3-8B (plan.md step 5).

Merges:
  ../output/results/internal_auroc.json   (from ../analysis/classify_auroc.py)
  ../output/results/verbalized_auroc.json (from ../probes/verbalize_probe.py)
  ../output/scores.npz                    (per-transcript internal probe scores)
  ../output/verbalized_responses.csv      (per-transcript verbalized eval_signal)

And computes:
  1. Peak-layer internal AUROC vs. verbalized ensemble AUROC, side by side.
  2. Spearman correlation between the internal probe's peak-layer score and
     the model's mean verbalized eval_signal, per transcript — no external
     judge needed since Qwen3-8B is both subject and judge here (unlike
     ../../evaluation-awareness/cls_corr_mi/correlation_mi.py, which
     correlates against an external 0-3 judge score).
  3. Bucket analysis (TP/FN/FP/TN by the verbalized ensemble's majority
     vote, per ../../eval-aware-oss/plan.md's Phase 2 framing): for each
     bucket, does the internal probe score still separate eval from
     deploy — does the model "know internally" even when it misstates
     verbally, or vice versa?

Usage:
    python compare_internal_vs_verbal.py \
      --internal-auroc ../output/results/internal_auroc.json \
      --verbalized-auroc ../output/results/verbalized_auroc.json \
      --scores ../output/scores.npz \
      --verbalized-csv ../output/verbalized_responses.csv \
      --pooling last \
      --out ../output/COMPARISON_REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load_verbalized_per_transcript(csv_path: Path) -> dict[str, dict]:
    """Aggregate per-probe rows into one mean-eval_signal + majority-vote per transcript."""
    rows_by_tid: dict[str, list[dict]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_by_tid.setdefault(r["transcript_id"], []).append(r)

    out = {}
    for tid, rows in rows_by_tid.items():
        signals = [float(r["eval_signal"]) for r in rows if float(r["eval_signal"]) != -1.0]
        gt = int(rows[0]["ground_truth_label"])
        mean_signal = float(np.mean(signals)) if signals else None
        majority_vote = int(round(mean_signal)) if mean_signal is not None else None
        out[tid] = {
            "prompt_id": int(rows[0]["prompt_id"]),
            "ground_truth_label": gt,
            "mean_eval_signal": mean_signal,
            "majority_vote": majority_vote,
            "n_probes_scored": len(signals),
            "n_probes_total": len(rows),
            "source": rows[0]["source"],
        }
    return out


def bucket(gt: int, pred: int) -> str:
    if gt == 1 and pred == 1:
        return "TP"
    if gt == 1 and pred == 0:
        return "FN"
    if gt == 0 and pred == 1:
        return "FP"
    return "TN"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal-auroc", required=True, type=Path)
    ap.add_argument("--verbalized-auroc", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--verbalized-csv", required=True, type=Path)
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    internal = json.loads(args.internal_auroc.read_text(encoding="utf-8"))
    verbalized = json.loads(args.verbalized_auroc.read_text(encoding="utf-8"))
    verbalized_by_tid = load_verbalized_per_transcript(args.verbalized_csv)

    with np.load(args.scores, allow_pickle=True) as z:
        real_scores = z[f"scores_{args.pooling}_real"]  # [N, L]
        labels = z["labels"]
        prompt_ids = z["prompt_ids"]
        sources = z["source"]

    peak_layer = int(internal["summary"]["peak_layer"])
    internal_peak_scores = real_scores[:, peak_layer]  # [N]

    # Join internal scores (indexed by scores.npz row order) to verbalized
    # rows via prompt_id, not row position — robust to the two pipelines
    # processing dataset.json in different orders.
    internal_score_by_pid = {
        int(pid): float(internal_peak_scores[i]) for i, pid in enumerate(prompt_ids)
    }
    missing = [tid for tid, info in verbalized_by_tid.items() if info["prompt_id"] not in internal_score_by_pid]
    if missing:
        raise SystemExit(
            f"{len(missing)} verbalized transcript(s) have no matching prompt_id in "
            f"scores.npz (e.g. {missing[:3]}). Re-run the full pipeline on the same "
            "dataset.json snapshot."
        )

    # Correlate internal peak-layer score against mean verbalized eval_signal
    paired = [
        (internal_score_by_pid[info["prompt_id"]], info["mean_eval_signal"], info["ground_truth_label"])
        for info in verbalized_by_tid.values()
        if info["mean_eval_signal"] is not None
    ]
    internal_vals = np.array([p[0] for p in paired])
    verbal_vals = np.array([p[1] for p in paired])
    rho, p_value = spearmanr(internal_vals, verbal_vals) if len(paired) > 1 else (None, None)

    # Bucket analysis by verbalized majority vote
    bucket_rows: dict[str, list[float]] = {"TP": [], "FN": [], "FP": [], "TN": []}
    for info in verbalized_by_tid.values():
        if info["majority_vote"] is None:
            continue
        b = bucket(info["ground_truth_label"], info["majority_vote"])
        bucket_rows[b].append(internal_score_by_pid[info["prompt_id"]])

    bucket_stats = {
        b: {"n": len(v), "mean_internal_score": float(np.mean(v)) if v else None}
        for b, v in bucket_rows.items()
    }

    lines = [
        "# Internal vs. Verbalized Evaluation-Awareness — Qwen3-8B",
        "",
        f"Peak internal-probe layer: {peak_layer} "
        f"(AUROC={internal['summary']['peak_auroc']:.3f}, "
        f"LOO accuracy={internal['summary']['peak_loo_accuracy']:.3f})",
        f"Verbalized ensemble AUROC: {verbalized['summary']['ensemble_auroc']}",
        "",
        "## Internal vs. verbalized correlation",
        "",
        f"Spearman rho (internal peak-layer score vs. mean verbalized eval_signal, "
        f"n={len(paired)}): {rho}",
        f"p-value: {p_value}",
        "",
        "## Bucket analysis (by verbalized majority vote)",
        "",
        "Does the internal probe still separate eval/deploy even inside a bucket "
        "where the model's stated answer was wrong (FN/FP)?",
        "",
        "| Bucket | n | mean internal peak-layer score |",
        "| --- | ---: | ---: |",
    ]
    for b in ("TP", "FN", "FP", "TN"):
        s = bucket_stats[b]
        mean_str = f"{s['mean_internal_score']:.3f}" if s["mean_internal_score"] is not None else "n/a"
        lines.append(f"| {b} | {s['n']} | {mean_str} |")

    lines += [
        "",
        "At n=20 these are exploratory effect sizes, not significance tests "
        "(cf. EXPERIMENT_SUMMARY.md).",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
