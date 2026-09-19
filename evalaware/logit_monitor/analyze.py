import argparse
import json
from pathlib import Path

import numpy as np

from evalaware import metrics, scoring, verbalize

AGGS = {"argmax": 1, "band": 3}


def _candidates(scores, n_eval_aware, group, stat=1):
    """[n, rollouts, positions, candidates, 2] -> [n, rollouts, positions].

    stat=1 is the length-normalised mean log-prob; stat=0 the raw sum, which
    favours short candidates. Padded positions stay NaN.
    """
    part = scores[..., :n_eval_aware, stat] if group == "eval_aware" else scores[..., n_eval_aware:, stat]
    with np.errstate(invalid="ignore"):
        return np.nanmax(part, axis=3)


def pooled(scores, n_eval_aware, group, stat=1):
    """One score per transcript: max over rollouts, positions and candidates.

    This is the post's statistic. Positions are ragged because every rollout
    reasons for a different number of sentences, so there is no fixed grid to
    take a band over; the max is defined regardless.
    """
    per_position = _candidates(scores, n_eval_aware, group, stat)
    with np.errstate(invalid="ignore"):
        out = np.nanmax(per_position, axis=(1, 2))
    return np.nan_to_num(out, nan=np.nanmin(out))


def depth_bins(scores, n_eval_aware, group, n_bins=8, stat=1):
    """[n, n_bins] by relative depth through the trace.

    Absolute position index is not comparable across rollouts of different
    lengths, so positions are bucketed by fraction of the way through. This is
    what lets the argmax-versus-band selection check run on a rectangular
    matrix without discarding any boundary.
    """
    per_position = _candidates(scores, n_eval_aware, group, stat)
    n = per_position.shape[0]
    out = np.full((n, n_bins), np.nan, dtype=np.float64)
    for i in range(n):
        flat = per_position[i]
        real = ~np.isnan(flat)
        for r in range(flat.shape[0]):
            idx = np.flatnonzero(real[r])
            if idx.size == 0:
                continue
            edges = np.linspace(0, idx.size, n_bins + 1).astype(int)
            for b in range(n_bins):
                chunk = flat[r, idx[edges[b]:edges[b + 1]]] if edges[b + 1] > edges[b] else []
                if len(chunk):
                    with np.errstate(invalid="ignore"):
                        value = np.nanmax(chunk)
                    if not np.isnan(value):
                        out[i, b] = value if np.isnan(out[i, b]) else max(out[i, b], value)
    column_min = np.nanmin(out, axis=0)
    return np.where(np.isnan(out), column_min[None, :], out)


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
         "auroc": metrics.auroc(pooled(scores[:, :k], n_eval_aware, "eval_aware"), labels)}
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
        one = pooled(scores, n_eval_aware, group)
        payload[group] = {
            "auroc": metrics.auroc(one, labels),
            "p": metrics.empirical_p(
                metrics.permutation_null(one[:, None], labels, args.n_perm, 1, args.seed),
                metrics.auroc(one, labels)),
            "by_depth": summarise(depth_bins(scores, n_eval_aware, group), labels, args.n_perm, args.seed),
        }

    monitor = pooled(scores, n_eval_aware, "eval_aware")
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
    ea, nt = payload["eval_aware"], payload["neutral"]
    print(f"monitor AUROC: eval-aware {ea['auroc']:.3f} (p={ea['p']:.4f}), "
          f"neutral control {nt['auroc']:.3f} (p={nt['p']:.4f})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
