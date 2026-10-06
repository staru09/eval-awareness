"""Rank eleven benchmarks by how evaluation-like the probe finds them.

Every group is scored against the same 50 real user messages, at the same layer,
with the same probe, and all groups are length-matched. So the numbers are
comparable to each other, which is the thing the first run could not do.

Each group also gets a random-direction floor: the best of the 10 random
directions on that same contrast. A group whose effect sits under its own floor
is not distinguishable from noise.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalaware.metrics import auroc

N_PERM = 2000


def vs_deployment(scores, groups, name):
    mask = (groups == name) | (groups == "deployment")
    y = (groups[mask] == name).astype(int)
    return float(auroc(scores[mask], y))


def main():
    scores_path, dataset_path, out_path = map(Path, sys.argv[1:4])
    z = np.load(scores_path, allow_pickle=True)
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    order = {int(p): i for i, p in enumerate(z["prompt_ids"])}
    rows = sorted(rows, key=lambda r: order[r["prompt_id"]])
    groups = np.array([r["source"] for r in rows])
    family = {r["source"]: r["category"] for r in rows}
    labels = z["labels"]

    out = {"scores": str(scores_path), "n": len(rows)}
    for pooling in ("last", "mean"):
        real, rand = z[f"scores_{pooling}_real"], z[f"scores_{pooling}_random"]
        curve = np.array([auroc(real[:, l], labels) for l in range(real.shape[1])])
        peak = int(np.argmax(np.abs(curve - 0.5)))

        # label-permutation null on the pooled eval-vs-deployment split
        rng = np.random.default_rng(0)
        y = labels.copy()
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            rng.shuffle(y)
            null[i] = np.abs(np.array([auroc(real[:, l], y) for l in range(real.shape[1])]) - 0.5).max()

        per_group = {}
        for name in sorted(set(groups) - {"deployment"}):
            a = vs_deployment(real[:, peak], groups, name)
            floor = max(abs(vs_deployment(rand[:, peak, r], groups, name) - 0.5)
                        for r in range(rand.shape[2]))
            per_group[name] = {"family": family[name], "n": int((groups == name).sum()),
                              "auroc": a, "effect": abs(a - 0.5), "random_floor": float(floor),
                              "beats_floor": bool(abs(a - 0.5) > floor),
                              "mean_score": float(real[groups == name, peak].mean())}

        fam = {}
        for f in ("technical", "safety"):
            members = [g for g in per_group if per_group[g]["family"] == f]
            mask = np.isin(groups, members) | (groups == "deployment")
            yy = (~np.isin(groups[mask], ["deployment"])).astype(int)
            a = float(auroc(real[mask, peak], yy))
            fam[f] = {"groups": len(members), "n": int(np.isin(groups, members).sum()),
                      "auroc": a, "effect": abs(a - 0.5)}

        out[pooling] = {"peak_layer": peak, "overall_auroc": float(curve[peak]),
                        "perm_null_p95": float(np.quantile(null, 0.95)),
                        "deployment_mean": float(real[groups == "deployment", peak].mean()),
                        "by_family": fam, "by_group": per_group,
                        "auroc_by_layer": [float(v) for v in curve]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for pooling in ("last", "mean"):
        b = out[pooling]
        print(f"\n[{pooling}] peak layer {b['peak_layer']}, pooled AUROC {b['overall_auroc']:.3f}, "
              f"permutation p95 {b['perm_null_p95']:.3f}")
        print(f"   technical {b['by_family']['technical']['effect']:.3f}   "
              f"safety {b['by_family']['safety']['effect']:.3f}")
        ranked = sorted(b["by_group"].items(), key=lambda kv: -kv[1]["effect"])
        for name, g in ranked:
            mark = "yes" if g["beats_floor"] else " no"
            print(f"   {name:20s} {g['family']:9s} n={g['n']:2d}  auroc {g['auroc']:.3f}  "
                  f"effect {g['effect']:.3f}  floor {g['random_floor']:.3f}  beats floor: {mark}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
