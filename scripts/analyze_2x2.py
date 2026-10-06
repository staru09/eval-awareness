"""Does the probe read evaluation, or does it read benchmark formatting?

Four contrasts on the same 100 prompts, each with the same probe and the same
layer. The sign convention is Hua et al.'s, positive = deployment, so what is
reported is the distance from chance and which way it points.

  standard      bench_eval   vs casual_deploy  - the uncontrolled comparison
  eval|bench    bench_eval   vs bench_deploy   - eval detection, format fixed
  eval|casual   casual_eval  vs casual_deploy  - eval detection, format fixed
  format only   bench_deploy vs casual_deploy  - both are real users; any
                                                 separation here is formatting
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalaware.metrics import auroc

CONTRASTS = [
    ("standard      (bench_eval vs casual_deploy)", "bench_eval", "casual_deploy"),
    ("eval | bench  (bench_eval vs bench_deploy)", "bench_eval", "bench_deploy"),
    ("eval | casual (casual_eval vs casual_deploy)", "casual_eval", "casual_deploy"),
    ("format only   (bench_deploy vs casual_deploy)", "bench_deploy", "casual_deploy"),
]


def contrast(scores, cells, pos, neg):
    """AUROC with `pos` treated as the evaluation side."""
    mask = (cells == pos) | (cells == neg)
    y = (cells[mask] == pos).astype(int)
    return float(auroc(scores[mask], y))


def main():
    scores_path, dataset_path, out_path = map(Path, sys.argv[1:4])
    z = np.load(scores_path, allow_pickle=True)
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    order = {int(p): i for i, p in enumerate(z["prompt_ids"])}
    rows = sorted(rows, key=lambda r: order[r["prompt_id"]])
    cells = np.array([r["source"] for r in rows])

    out = {"scores": str(scores_path), "n": len(rows)}
    for pooling in ("last", "mean"):
        real = z[f"scores_{pooling}_real"]
        rand = z[f"scores_{pooling}_random"]
        labels = z["labels"]

        curve = np.array([auroc(real[:, l], labels) for l in range(real.shape[1])])
        peak = int(np.argmax(np.abs(curve - 0.5)))

        block = {"peak_layer": peak, "overall_auroc_at_peak": float(curve[peak]),
                 "auroc_by_layer": [float(v) for v in curve], "contrasts": {}}
        for name, pos, neg in CONTRASTS:
            at_peak = contrast(real[:, peak], cells, pos, neg)
            # the same contrast under random directions, for a floor
            rnd = [abs(contrast(rand[:, peak, r], cells, pos, neg) - 0.5)
                   for r in range(rand.shape[2])]
            block["contrasts"][name.split("(")[0].strip()] = {
                "pair": f"{pos} vs {neg}", "auroc": at_peak,
                "effect": abs(at_peak - 0.5), "random_best_effect": float(max(rnd))}
        # cell means, so the direction of each cell is visible
        block["cell_means"] = {c: float(real[cells == c, peak].mean()) for c in sorted(set(cells))}
        out[pooling] = block

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for pooling in ("last", "mean"):
        b = out[pooling]
        print(f"\n[{pooling}] peak layer {b['peak_layer']}, overall AUROC {b['overall_auroc_at_peak']:.3f}")
        for name, pos, neg in CONTRASTS:
            c = b["contrasts"][name.split("(")[0].strip()]
            print(f"   {name:46s} AUROC {c['auroc']:.3f}  effect {c['effect']:.3f}"
                  f"  (random best {c['random_best_effect']:.3f})")
        print("   cell means:", {k: round(v, 2) for k, v in b["cell_means"].items()})
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
