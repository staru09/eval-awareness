"""Experiment A charts: does the internal probe signal exist BEFORE the
model generates, or only after?

Compares two readouts of the same probe direction on the same 20
transcripts, per model, per layer:

  * prompt+answer -- results/internal_auroc_<pooling>.json, activations
    pooled over `prompt + the model's own generated answer`
    (run_inference.py's second forward pass). Its last token is the last
    token the *model wrote*, which for benchmark transcripts is often the
    end of a LaTeX-boxed multiple-choice answer -- so a probe read there
    is partly reading the model's own output format back to itself.
  * prompt-only -- results/internal_auroc_prompt_<pooling>.json, from
    capture_prompt_activations.py. Read before any answer token exists;
    the readout the source paper and cls_corr_mi/score_prompts.py use.

Chart 13 adds the honest null models from probe_null_controls.py, because
the peak AUROC alone is not interpretable: the real probe picks its best
of 36-64 layers, and at n=20 that selection is worth a lot. The reference
lines are the 95th percentile of the label-permutation null (same probe,
same activations, same argmax, shuffled labels) and the two confound
baselines that use no probe direction at all.

Usage:
    python plot_prompt_vs_answer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
LOCAL_DIR = HERE.parent / "output" / "local"
OUT_DIR = HERE.parent / "output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Qwen3-8B": "#2A78D6",
    "granite-4.2-30b": "#4A3AA7",
    "Qwen3.6-27B-OBLITERATED": "#8C6D1F",
}
SLUG_LABELS = {
    "qwen3-8b": "Qwen3-8B",
    "granite-4.2-30b": "granite-4.2-30b",
    "qwen3.6-27b-obliterated": "Qwen3.6-27B-OBLITERATED",
}

INK = "#0B0B0B"
INK_2 = "#52514E"
MUTED = "#898781"
GRID = "#E1E0D9"
SURFACE = "#FCFCFB"
WARN = "#B4451F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_results(pooling: str) -> dict[str, dict[str, dict]]:
    """slug -> {"answer": auroc, "prompt": auroc, "answer_null": ..., "prompt_null": ...}."""
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(LOCAL_DIR.glob("*")):
        if d.name not in SLUG_LABELS:
            continue
        files = {
            "answer": f"internal_auroc_{pooling}.json",
            "prompt": f"internal_auroc_prompt_{pooling}.json",
            "answer_null": f"null_controls_answer_{pooling}.json",
            "prompt_null": f"null_controls_prompt_{pooling}.json",
        }
        got = {}
        for key, fname in files.items():
            p = d / "results" / fname
            if p.exists():
                got[key] = json.loads(p.read_text(encoding="utf-8"))
        if "answer" in got and "prompt" in got:
            out[d.name] = got
        elif got:
            print(f"  skipping {d.name}: only has {sorted(got)} for pooling={pooling}")
    return out


def series(data: dict):
    lr = data["layer_results"]
    n_layers = data["summary"]["n_layers"]
    layers = sorted(int(k) for k in lr)
    depth = np.array([l / max(n_layers - 1, 1) for l in layers])
    real = np.array([lr[str(l)]["real"]["auroc"] for l in layers])
    rand_max = np.array([lr[str(l)]["random"]["max_auroc_across_R"] for l in layers])
    return depth, real, rand_max


def plot_layers(results: dict[str, dict[str, dict]], pooling: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 4.9), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, got) in zip(axes, results.items()):
        label = SLUG_LABELS[slug]
        color = COLORS.get(label, "#666666")

        d_a, real_a, rmax_a = series(got["answer"])
        d_p, real_p, rmax_p = series(got["prompt"])

        # Shared per-layer control envelope: compute_probe.py seeds the same
        # 10 random directions per layer for both runs, so this is the same
        # noise floor for both curves. Note it is the *per-layer* band, not
        # the max-over-layers null -- chart 13 carries that one.
        band = np.maximum(rmax_a, rmax_p) if len(rmax_a) == len(rmax_p) else rmax_a
        ax.fill_between(d_a, 0, band, color=GRID, alpha=0.6,
                        label="random direction, per-layer max", zorder=1)
        ax.axhline(0.5, color=GRID, linewidth=1, zorder=0)

        # The bar that actually matters: 95th percentile of the
        # label-permutation null for this readout.
        null = got.get("prompt_null")
        if null:
            p95 = null["label_permutation_null"]["p95"]
            ax.axhline(p95, color=WARN, linewidth=1.3, linestyle=":", zorder=2,
                       label=f"label-permutation p95 ({p95:.2f})")

        ax.plot(d_a, real_a, color=color, linewidth=2.2, alpha=0.42, linestyle="--",
                label="prompt + own answer", zorder=3)
        ax.plot(d_p, real_p, color=color, linewidth=2.6, marker="o", markersize=3.4,
                label="prompt only (before generating)", zorder=4)

        sa, sp = got["answer"]["summary"], got["prompt"]["summary"]
        ax.scatter([sp["peak_layer"] / max(sp["n_layers"] - 1, 1)], [sp["peak_auroc"]],
                   s=95, facecolor="none", edgecolor=color, linewidth=2, zorder=5)

        p_txt = ""
        if null:
            p_txt = f", perm p={null['label_permutation_null']['empirical_p']:.3f}"
        ax.set_title(
            f"{label}\nprompt-only peak L{sp['peak_layer']}/{sp['n_layers']}: "
            f"{sp['peak_auroc']:.3f}{p_txt}\nprompt+answer peak: {sa['peak_auroc']:.3f}",
            loc="left", fontsize=10, fontweight="bold", color=INK,
        )
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("AUROC (eval vs. deploy)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    pool_label = "last-token" if pooling == "last" else "mean-token"
    fig.suptitle(
        f"Probe AUROC before vs. after generation ({pool_label} readout)",
        x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / f"11_prompt_vs_answer_layers_{pooling}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote 11_prompt_vs_answer_layers_{pooling}.png")


def plot_peaks(by_pooling: dict[str, dict[str, dict[str, dict]]]) -> None:
    slugs = sorted({s for r in by_pooling.values() for s in r})
    if not slugs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.85 * len(slugs) + 2.6), sharex=True)
    for ax, pooling in zip(axes, ("last", "mean")):
        results = by_pooling.get(pooling, {})
        labels = [SLUG_LABELS[s] for s in slugs]
        colors = [COLORS.get(l, "#666666") for l in labels]
        ans = [results.get(s, {}).get("answer", {}).get("summary", {}).get("peak_auroc") for s in slugs]
        pro = [results.get(s, {}).get("prompt", {}).get("summary", {}).get("peak_auroc") for s in slugs]

        y = np.arange(len(slugs))
        h = 0.34
        ax.barh(y + h / 2, [v or 0 for v in pro], height=h, color=colors, zorder=3,
                label="prompt only")
        ax.barh(y - h / 2, [v or 0 for v in ans], height=h, color=colors, alpha=0.4, zorder=3,
                label="prompt + own answer")

        # Per-model permutation-null p95 as a tick on each bar group.
        for yi, s in zip(y, slugs):
            null = results.get(s, {}).get("prompt_null")
            if null:
                p95 = null["label_permutation_null"]["p95"]
                ax.plot([p95, p95], [yi - 0.42, yi + 0.42], color=WARN, linewidth=1.6,
                        linestyle=":", zorder=5)

        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Peak-layer AUROC")
        ax.set_title(f"{pooling}-token readout", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for yi, v in zip(y + h / 2, pro):
            if v is not None:
                ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9.5,
                        fontweight="bold", color=INK)
        for yi, v in zip(y - h / 2, ans):
            if v is not None:
                ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9, color=INK_2)

    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "Peak internal-probe AUROC: before vs. after the model answers\n"
        "dotted line = 95th percentile of the label-permutation null (prompt-only)",
        x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "12_prompt_vs_answer_peaks.png", dpi=200)
    plt.close(fig)
    print("  wrote 12_prompt_vs_answer_peaks.png")


def plot_null_controls(by_pooling: dict[str, dict[str, dict[str, dict]]]) -> None:
    """Chart 11: peak AUROC against every null and confound baseline."""
    rows = []
    for pooling, results in by_pooling.items():
        for slug, got in results.items():
            for readout, key in (("prompt only", "prompt_null"), ("prompt + answer", "answer_null")):
                null = got.get(key)
                if not null:
                    continue
                rows.append({
                    "label": f"{SLUG_LABELS[slug]}\n{readout}, {pooling}-token",
                    "color": COLORS.get(SLUG_LABELS[slug], "#666666"),
                    "real": null["real"]["peak_auroc"],
                    "perm_p95": null["label_permutation_null"]["p95"],
                    "perm_p": null["label_permutation_null"]["empirical_p"],
                    "rand_mean": null["random_direction_null"]["mean"],
                    "norm": (null.get("norm_only_baseline") or {}).get("peak_auroc"),
                    "length": (null.get("prompt_length_baseline") or {}).get("auroc"),
                    "cos": (null.get("cosine_normalised") or {}).get("peak_auroc"),
                })
    if not rows:
        return
    rows.sort(key=lambda r: r["real"], reverse=True)

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["real"] for r in rows], height=0.55,
            color=[r["color"] for r in rows], zorder=3, label="real probe, peak layer")
    ax.scatter([r["cos"] for r in rows], y, marker="D", s=26, color=INK,
               zorder=5, label="cosine-normalised (magnitude removed)")
    ax.scatter([r["rand_mean"] for r in rows], y, marker="s", s=30, color=MUTED,
               zorder=5, label="random directions, mean of their own peaks")
    ax.scatter([r["norm"] for r in rows], y, marker="^", s=34, color="#7A7A55",
               zorder=5, label="||h|| alone, no probe")
    for yi, r in zip(y, rows):
        ax.plot([r["perm_p95"], r["perm_p95"]], [yi - 0.33, yi + 0.33], color=WARN,
                linewidth=2, zorder=6)
        ax.text(1.015, yi, f"p={r['perm_p']:.3f}", va="center", fontsize=8.5,
                color=WARN if r["perm_p"] < 0.05 else MUTED,
                fontweight="bold" if r["perm_p"] < 0.05 else "normal")
    ax.plot([], [], color=WARN, linewidth=2, label="label-permutation null, p95")
    if rows[0]["length"] is not None:
        ax.axvline(rows[0]["length"], color=INK_2, linewidth=1, linestyle="--", zorder=2)
        ax.text(rows[0]["length"] - 0.008, y[0] + 0.85,
                f"prompt length alone: {rows[0]['length']:.2f}",
                fontsize=8.5, color=INK_2, ha="right", va="bottom")

    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("AUROC (eval vs. deploy)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=8.5, ncol=2, columnspacing=1.4, handletextpad=0.5)
    fig.suptitle(
        "Internal-probe peak AUROC against its actual null models\n"
        "a bar only counts as signal if it clears the orange permutation bar AND the baselines",
        x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "13_probe_null_controls.png", dpi=200)
    plt.close(fig)
    print("  wrote 13_probe_null_controls.png")


def plot_band_statistic(by_pooling: dict[str, dict[str, dict[str, dict]]]) -> None:
    """Chart 14: the argmax statistic vs the band statistic, with both nulls.

    Chart 13 shows every cell failing the random-direction null. The reason is
    the statistic, not the data: at n=20 a random direction's per-layer AUROC
    has |AUROC-0.5| ~ 0.16-0.23 and adjacent layers are near-independent
    (measured correlation 0.00-0.03), so argmax over 36-64 layers reaches ~0.9
    under the null. Requiring a coherent 3-layer band -- which a real
    representation should produce and a random direction should not -- recovers
    signal in three cells against 1000 fresh random directions.
    """
    rows = []
    for pooling, results in by_pooling.items():
        for slug, got in results.items():
            for readout, key in (("prompt only", "prompt_null"), ("prompt + answer", "answer_null")):
                null = got.get(key)
                if not null or not null.get("band_statistic"):
                    continue
                b = null["band_statistic"]
                fr = b["fresh_random_directions"]
                if not fr:
                    continue
                rows.append({
                    "label": f"{SLUG_LABELS[slug]}\n{readout}, {pooling}-token",
                    "color": COLORS.get(SLUG_LABELS[slug], "#666666"),
                    "argmax": null["real"]["peak_auroc"],
                    "band": b["real"]["band_auroc"],
                    "band_layer": b["real"]["band_start_layer"],
                    "rand_mean": fr["mean"],
                    "rand_p": fr["empirical_p"],
                    "perm_p95": b["label_permutation"]["p95"],
                    "perm_p": b["label_permutation"]["empirical_p"],
                })
    if not rows:
        return
    rows.sort(key=lambda r: r["perm_p"])

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.4))
    y = np.arange(len(rows))[::-1]
    h = 0.34
    ax.barh(y + h / 2, [r["band"] for r in rows], height=h,
            color=[r["color"] for r in rows], zorder=3, label="band statistic (3-layer)")
    ax.barh(y - h / 2, [r["argmax"] for r in rows], height=h,
            color=[r["color"] for r in rows], alpha=0.35, zorder=3, label="argmax statistic (single layer)")
    ax.scatter([r["rand_mean"] for r in rows], y + h / 2, marker="s", s=32, color=MUTED,
               zorder=5, label="1000 random directions, mean band")
    for yi, r in zip(y, rows):
        ax.plot([r["perm_p95"], r["perm_p95"]], [yi + h / 2 - 0.17, yi + h / 2 + 0.17],
                color=WARN, linewidth=2, zorder=6)
        passes = r["perm_p"] < 0.05 and r["rand_p"] < 0.05
        ax.text(1.015, yi, f"perm {r['perm_p']:.3f} / rand {r['rand_p']:.3f}", va="center",
                fontsize=8, color=WARN if passes else MUTED,
                fontweight="bold" if passes else "normal")
    ax.plot([], [], color=WARN, linewidth=2, label="label-permutation null, p95 (band)")

    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.28)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.7)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("AUROC (eval vs. deploy)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=8.5, ncol=2, columnspacing=1.4, handletextpad=0.5)
    fig.suptitle(
        "Requiring a coherent 3-layer band instead of a single best layer\n"
        "three cells clear both nulls; the argmax statistic cleared neither",
        x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "14_band_statistic.png", dpi=200)
    plt.close(fig)
    print("  wrote 14_band_statistic.png")


def main() -> None:
    by_pooling = {}
    for pooling in ("last", "mean"):
        results = load_results(pooling)
        print(f"pooling={pooling}: {len(results)} model(s) with both readouts: {list(results)}")
        if results:
            by_pooling[pooling] = results
            plot_layers(results, pooling)
    if not by_pooling:
        raise SystemExit(
            f"No model under {LOCAL_DIR} has both internal_auroc_<pooling>.json and "
            "internal_auroc_prompt_<pooling>.json -- run capture_prompt_activations.py, "
            "score_activations.py and classify_auroc.py first."
        )
    plot_peaks(by_pooling)
    plot_null_controls(by_pooling)
    plot_band_statistic(by_pooling)
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
