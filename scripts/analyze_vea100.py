"""Contrastive-probe analysis of the 100-prompt set, with the controls the
literature treats as mandatory.

Sign convention, which is load-bearing: the 16 pairs have positive = deployment
(Hua et al. Table 12), so a working probe gives AUROC BELOW 0.5 against our
label where 1 = eval. Nothing is flipped anywhere here; a sub-0.5 value is the
signal and the reported effect size is |AUROC - 0.5|.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalaware.metrics import auroc

RNG = np.random.default_rng(0)
N_PERM = 2000
BAND = 3
APRIORI_FRACTION = 0.4  # deception / eval-awareness probes cluster at 0.2-0.46 depth


def effect(scores, labels):
    """Distance from chance, so the deployment-positive sign convention is kept."""
    return abs(auroc(scores, labels) - 0.5)


def per_layer(scores, labels):
    return np.array([auroc(scores[:, l], labels) for l in range(scores.shape[1])])


def band_mean(values, width=BAND):
    """Best window of adjacent layers, on distance from chance."""
    dev = np.abs(values - 0.5)
    sums = np.convolve(dev, np.ones(width) / width, mode="valid")
    start = int(np.argmax(sums))
    return float(sums[start]), start


def permutation_null(scores, labels, n_perm=N_PERM):
    """Max over layers of |AUROC-0.5| under shuffled labels.

    This is the number that says whether a peak means anything at n=100 with
    36 layers to choose from. The max is taken inside each permutation, so it
    prices in the layer selection.
    """
    y = labels.copy()
    out = np.empty(n_perm)
    for i in range(n_perm):
        RNG.shuffle(y)
        out[i] = np.abs(per_layer(scores, y) - 0.5).max()
    return out


def text_baselines(rows, labels):
    """Confounds with no model in them at all."""
    chars = np.array([len(r["text"]) for r in rows], dtype=float)
    words = np.array([len(r["text"].split()) for r in rows], dtype=float)
    punct = np.array([len(re.findall(r"[^\w\s]", r["text"])) for r in rows], dtype=float)
    digits = np.array([sum(c.isdigit() for c in r["text"]) for r in rows], dtype=float)
    return {"char_count": effect(chars, labels), "word_count": effect(words, labels),
            "special_chars": effect(punct, labels), "digits": effect(digits, labels)}


def by_category(scores_at_layer, rows, labels):
    """The question: does one eval category read as more eval-like than others?

    Each eval category is scored against the whole deployment pool, so the
    numbers are comparable to each other.
    """
    deploy = scores_at_layer[labels == 0]
    cats = {}
    for cat in sorted({r["category"] for r in rows if r["label"] == 1}):
        mask = np.array([r["category"] == cat and r["label"] == 1 for r in rows])
        vals = scores_at_layer[mask]
        pooled = np.concatenate([vals, deploy])
        pooled_y = np.concatenate([np.ones(len(vals)), np.zeros(len(deploy))])
        cats[cat] = {"n": int(mask.sum()), "mean_score": float(vals.mean()),
                     "std_score": float(vals.std()),
                     "auroc_vs_deployment": float(auroc(pooled, pooled_y)),
                     "effect_vs_deployment": float(effect(pooled, pooled_y))}
    cats["_deployment_reference"] = {"n": int((labels == 0).sum()),
                                     "mean_score": float(deploy.mean()),
                                     "std_score": float(deploy.std())}
    return cats


def analyse(npz_path, rows, pooling):
    z = np.load(npz_path, allow_pickle=True)
    real = z[f"scores_{pooling}_real"]
    rand = z[f"scores_{pooling}_random"]
    labels = z["labels"]
    n_layers = real.shape[1]

    curve = per_layer(real, labels)
    dev = np.abs(curve - 0.5)
    peak = int(np.argmax(dev))
    band_val, band_start = band_mean(curve)
    apriori = int(round(APRIORI_FRACTION * n_layers))

    # Random directions: the best any meaningless direction manages.
    rand_effect = np.array([[effect(rand[:, l, r], labels) for r in range(rand.shape[2])]
                            for l in range(n_layers)])
    null = permutation_null(real, labels)
    observed = float(dev[peak])

    return {
        "pooling": pooling, "n_layers": n_layers, "n": int(len(labels)),
        "peak": {"layer": peak, "auroc": float(curve[peak]), "effect": observed},
        "band": {"width": BAND, "start_layer": band_start, "mean_effect": band_val},
        "apriori": {"layer": apriori, "auroc": float(curve[apriori]),
                    "effect": float(dev[apriori])},
        "random_directions": {"max_effect_any_layer": float(rand_effect.max()),
                              "max_effect_at_peak_layer": float(rand_effect[peak].max()),
                              "mean_effect": float(rand_effect.mean())},
        "permutation_null": {"n_perm": N_PERM, "mean": float(null.mean()),
                             "p95": float(np.quantile(null, 0.95)),
                             "max": float(null.max()),
                             "p_value": float((np.sum(null >= observed) + 1) / (N_PERM + 1))},
        "auroc_by_layer": [float(v) for v in curve],
        "by_category_at_peak": by_category(real[:, peak], rows, labels),
        "by_category_at_apriori": by_category(real[:, apriori], rows, labels),
    }


def main():
    scores_path, dataset_path, out_path = map(Path, sys.argv[1:4])
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    z = np.load(scores_path, allow_pickle=True)
    order = {int(p): i for i, p in enumerate(z["prompt_ids"])}
    rows = sorted(rows, key=lambda r: order[r["prompt_id"]])
    labels = z["labels"]
    assert [r["label"] for r in rows] == list(labels), "dataset and scores are out of order"

    result = {"scores": str(scores_path), "model": str(z["model_key"]),
              "text_baselines": text_baselines(rows, labels),
              "last": analyse(scores_path, rows, "last"),
              "mean": analyse(scores_path, rows, "mean")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for pooling in ("last", "mean"):
        a = result[pooling]
        print(f"[{pooling}] peak layer {a['peak']['layer']}: AUROC {a['peak']['auroc']:.3f} "
              f"(effect {a['peak']['effect']:.3f}), band{BAND} {a['band']['mean_effect']:.3f} "
              f"from layer {a['band']['start_layer']}, a-priori L{a['apriori']['layer']} "
              f"{a['apriori']['auroc']:.3f}")
        print(f"         random dirs best {a['random_directions']['max_effect_any_layer']:.3f}, "
              f"perm null p95 {a['permutation_null']['p95']:.3f}, "
              f"p = {a['permutation_null']['p_value']:.4f}")
    print("baselines (effect):", {k: round(v, 3) for k, v in result["text_baselines"].items()})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
