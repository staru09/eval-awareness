import argparse
import json
from pathlib import Path

import numpy as np

from evalaware import metrics, scoring, verbalize

AGGS = {"argmax": 1, "band": 3}


def group_scores(scores, n_eval_aware, group, stat=1):
    """[n, rollouts, positions, candidates, 2] -> [n, positions].

    Max over rollouts and over candidates within the group, matching the post's
    statistic. stat=1 selects the length-normalised mean log-prob; stat=0 is the
    raw sum, which favours short candidates.
    """
    part = scores[..., :n_eval_aware, stat] if group == "eval_aware" else scores[..., n_eval_aware:, stat]
    with np.errstate(invalid="ignore"):
        pooled = np.nanmax(part, axis=(1, 3))
    return np.nan_to_num(pooled, nan=np.nanmin(pooled))


def summarise(real, labels, n_perm, seed):
    per_position = metrics.per_layer_auroc(real, labels)
    out = {"per_position_auroc": [float(v) for v in per_position]}
    for name, width in AGGS.items():
        value, start = metrics.band(per_position, width)
        null = metrics.permutation_null(real, labels, n_perm, width, seed)
        out[name] = {"auroc": value, "start_position": start, "p": metrics.empirical_p(null, value)}
    out["mean_over_positions"] = {"auroc": metrics.auroc(real.mean(axis=1), labels)}
    return out


def rollout_curve(scores, n_eval_aware, labels):
    return [
        {"rollouts": k,
         "argmax_auroc": metrics.band(
             metrics.per_layer_auroc(group_scores(scores[:, :k], n_eval_aware, "eval_aware"), labels), 1)[0]}
        for k in range(1, scores.shape[1] + 1)
    ]


def couple(x, y_map, prompt_ids, seed):
    rho, mi, degenerate = metrics.coupling(x, y_map, prompt_ids, seed)
    return {"spearman_rho": rho, "mutual_info_nats": mi, "degenerate": degenerate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbalized-csv", type=Path)
    ap.add_argument("--probe-scores", type=Path)
    ap.add_argument("--pooling", default="mean", choices=["last", "mean"])
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.scores, allow_pickle=True)
    scores, labels = z["scores"], z["labels"].astype(int)
    prompt_ids, n_eval_aware = z["prompt_ids"], int(z["n_eval_aware"])

    payload = {
        "model_key": str(z["model_key"]),
        "n_transcripts": int(scores.shape[0]),
        "n_rollouts": int(scores.shape[1]),
        "n_positions": int(scores.shape[2]),
        "rollout_curve": rollout_curve(scores, n_eval_aware, labels),
    }
    for group in ("eval_aware", "neutral"):
        payload[group] = summarise(group_scores(scores, n_eval_aware, group), labels, args.n_perm, args.seed)

    monitor = group_scores(scores, n_eval_aware, "eval_aware").max(axis=1)
    if args.verbalized_csv:
        verbal = verbalize.ensemble_signal(
            verbalize.signals_by_probe(verbalize.load_csv(args.verbalized_csv)))
        payload["monitor_vs_verbalized"] = couple(monitor, verbal, prompt_ids, args.seed)

        if args.probe_scores:
            s = scoring.load_scores(args.probe_scores, args.pooling)
            layer = int(np.argmax(metrics.per_layer_auroc(s["real"], s["labels"])))
            probe = s["real"][:, layer]
            monitor_map = {int(p): float(m) for p, m in zip(prompt_ids, monitor)}
            payload["probe_vs_monitor"] = couple(probe, monitor_map, s["prompt_ids"], args.seed)
            payload["probe_vs_monitor"]["probe_layer"] = layer
            payload["probe_vs_verbalized"] = couple(probe, verbal, s["prompt_ids"], args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ea, nt = payload["eval_aware"]["band"], payload["neutral"]["band"]
    print(f"band AUROC: eval-aware {ea['auroc']:.3f} (p={ea['p']:.4f}), neutral {nt['auroc']:.3f} (p={nt['p']:.4f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
