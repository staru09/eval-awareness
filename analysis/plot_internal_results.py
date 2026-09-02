"""Charts for the internal-representation probe experiment (plan2.md
Experiment 1): AUROC per layer at classifying eval-vs-deploy transcripts,
for each model with a completed output/local/<model>/results/internal_auroc_*.json.

Mirrors the paper's Figure 2 (AUROC vs. normalized layer depth, real probe
vs. random-direction control band).

Usage:
    python plot_internal_results.py
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

# Same palette as plot_results.py, keyed by the same slugs used there.
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


def load_results() -> dict[str, dict[str, dict]]:
    """slug -> {"last": {...}, "mean": {...}} for each model with results."""
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(LOCAL_DIR.glob("*")):
        if d.name not in SLUG_LABELS:
            continue
        pooled = {}
        for pooling in ("last", "mean"):
            p = d / "results" / f"internal_auroc_{pooling}.json"
            if p.exists():
                pooled[pooling] = json.loads(p.read_text(encoding="utf-8"))
        if pooled:
            out[d.name] = pooled
    return out


def series_for(data: dict, pooling: str):
    lr = data[pooling]["layer_results"]
    n_layers = data[pooling]["summary"]["n_layers"]
    layers = sorted(int(k) for k in lr)
    depth = np.array([l / max(n_layers - 1, 1) for l in layers])
    real = np.array([lr[str(l)]["real"]["auroc"] for l in layers])
    rand_mean = np.array([lr[str(l)]["random"]["mean_auroc_across_R"] for l in layers])
    rand_max = np.array([lr[str(l)]["random"]["max_auroc_across_R"] for l in layers])
    return depth, real, rand_mean, rand_max


def plot_layer_auroc(results: dict[str, dict[str, dict]], pooling: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (slug, data) in zip(axes, results.items()):
        if pooling not in data:
            ax.set_visible(False)
            continue
        label = SLUG_LABELS[slug]
        color = COLORS.get(label, "#666666")
        depth, real, rand_mean, rand_max = series_for(data, pooling)

        ax.fill_between(depth, 0, rand_max, color=GRID, alpha=0.6, label="random-direction max", zorder=1)
        ax.plot(depth, rand_mean, color=MUTED, linewidth=1.2, linestyle="--", label="random-direction mean", zorder=2)
        ax.plot(depth, real, color=color, linewidth=2.4, marker="o", markersize=3.5, label="real probe", zorder=3)
        ax.axhline(0.5, color=GRID, linewidth=1, zorder=0)

        peak = data[pooling]["summary"]
        ax.scatter(
            [peak["peak_layer"] / max(peak["n_layers"] - 1, 1)], [peak["peak_auroc"]],
            s=90, facecolor="none", edgecolor=color, linewidth=2, zorder=4,
        )
        ax.set_title(
            f"{label}\npeak L{peak['peak_layer']}/{peak['n_layers']}: "
            f"AUROC={peak['peak_auroc']:.3f}, LOO acc={peak['peak_loo_accuracy']:.3f}",
            loc="left", fontsize=11, fontweight="bold", color=INK,
        )
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylim(0.15, 1.02)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("AUROC (eval vs. deploy)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    pool_label = "last-token" if pooling == "last" else "mean-token"
    fig.suptitle(
        f"Internal linear-probe AUROC per layer ({pool_label} readout)",
        x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / f"05_internal_auroc_layers_{pooling}.png", dpi=200)
    plt.close(fig)


def plot_peak_summary(results: dict[str, dict[str, dict]]) -> None:
    labels, last_vals, mean_vals, colors = [], [], [], []
    for slug, data in results.items():
        label = SLUG_LABELS[slug]
        labels.append(label)
        colors.append(COLORS.get(label, "#666666"))
        last_vals.append(data.get("last", {}).get("summary", {}).get("peak_auroc"))
        mean_vals.append(data.get("mean", {}).get("summary", {}).get("peak_auroc"))

    order = np.argsort([v if v is not None else -1 for v in last_vals])
    labels = [labels[i] for i in order]
    last_vals = [last_vals[i] for i in order]
    mean_vals = [mean_vals[i] for i in order]
    colors = [colors[i] for i in order]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(labels) + 1.4))
    h = 0.32
    ax.barh(y + h / 2, [v or 0 for v in last_vals], height=h, color=colors, zorder=3, label="last-token")
    ax.barh(y - h / 2, [v or 0 for v in mean_vals], height=h, color=colors, alpha=0.45, zorder=3, label="mean-token")
    ax.axvline(0.7, color=INK_2, linewidth=1, linestyle=":", zorder=2)
    ax.text(0.712, y[-1] + 0.44, "paper's 0.7 benchmark", fontsize=8.5, color=INK_2, va="bottom")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, y[-1] + 0.75)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Peak-layer AUROC")
    ax.set_title("Peak internal-probe AUROC per model", loc="left", fontsize=13, fontweight="bold", color=INK, pad=34)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, v in zip(y + h / 2, last_vals):
        if v is not None:
            ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for yi, v in zip(y - h / 2, mean_vals):
        if v is not None:
            ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.96),
        frameon=False, fontsize=9, ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT_DIR / "06_internal_auroc_peak_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results = load_results()
    print(f"Loaded internal-probe results for {len(results)} models: {list(results)}")
    if not results:
        raise SystemExit(f"No results found under {LOCAL_DIR}/*/results/internal_auroc_*.json")
    plot_layer_auroc(results, "last")
    plot_layer_auroc(results, "mean")
    plot_peak_summary(results)
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
