"""Charts for Experiment 3 (plan2.md): activation steering, the causal
test. Reads output/local/<model>/results/steering_results.json (written
by probes/steer.py) and plots the shift in verbalized eval_signal vs. the
unsteered baseline, per layer, one line per steering strength alpha --
mirroring the paper's Figure 5 style.

Usage:
    python plot_steering.py
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
N_LAYERS = {"qwen3-8b": 36, "granite-4.2-30b": 64, "qwen3.6-27b-obliterated": 65}

# Diverging: negative alpha (toward eval-like, expected to raise verbalization)
# in cool blue-purples; positive alpha (toward deploy-like) in warm oranges.
ALPHA_COLORS = {
    -1.0: "#1B3E8C", -0.5: "#5F86D6",
    0.5: "#E0954D", 1.0: "#B6501F",
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


def load_results() -> dict[str, dict]:
    out = {}
    for d in sorted(LOCAL_DIR.glob("*")):
        if d.name not in SLUG_LABELS:
            continue
        p = d / "results" / "steering_results.json"
        if p.exists():
            out[d.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def plot_steering_effect(results: dict[str, dict]) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (slug, data) in zip(axes, results.items()):
        label = SLUG_LABELS[slug]
        n_layers = N_LAYERS[slug]
        layers = sorted(set(c["layer"] for c in data["cells"]))
        alphas = data["alphas"]

        for alpha in alphas:
            depth, delta = [], []
            for layer in layers:
                cell = next((c for c in data["cells"] if c["layer"] == layer and c["alpha"] == alpha), None)
                if cell is None or cell["delta_vs_baseline"] is None:
                    continue
                depth.append(layer / max(n_layers - 1, 1))
                delta.append(cell["delta_vs_baseline"])
            if not depth:
                continue
            order = np.argsort(depth)
            depth_arr, delta_arr = np.array(depth)[order], np.array(delta)[order]
            ax.plot(
                depth_arr, delta_arr, marker="o", markersize=5, linewidth=2.2,
                color=ALPHA_COLORS.get(alpha, "#666666"), label=f"alpha={alpha:+.1f}",
            )

        ax.axhline(0, color=GRID, linewidth=1.2, zorder=0)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_xlabel("Normalized layer depth")
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Delta mean eval_signal vs. unsteered baseline")
    axes[0].legend(loc="best", frameon=False, fontsize=9)

    fig.suptitle(
        "Activation steering: shift in verbalized eval-awareness (direct_a probe)",
        x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "09_steering_effect.png", dpi=200)
    plt.close(fig)


def plot_peak_summary(results: dict[str, dict]) -> None:
    labels, neg_peak, pos_peak, colors = [], [], [], []
    for slug, data in results.items():
        label = SLUG_LABELS[slug]
        deltas = [c["delta_vs_baseline"] for c in data["cells"] if c["delta_vs_baseline"] is not None]
        neg_deltas = [c["delta_vs_baseline"] for c in data["cells"] if c["alpha"] < 0 and c["delta_vs_baseline"] is not None]
        pos_deltas = [c["delta_vs_baseline"] for c in data["cells"] if c["alpha"] > 0 and c["delta_vs_baseline"] is not None]
        labels.append(label)
        colors.append(COLORS.get(label, "#666666"))
        neg_peak.append(max(neg_deltas) if neg_deltas else 0.0)   # most positive shift from eval-ward steering
        pos_peak.append(min(pos_deltas) if pos_deltas else 0.0)   # most negative shift from deploy-ward steering

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(labels) + 1.6))
    h = 0.32
    bars1 = ax.barh(y + h / 2, neg_peak, height=h, color=colors, zorder=3, label="peak +delta (eval-ward alpha<0)")
    bars2 = ax.barh(y - h / 2, pos_peak, height=h, color=colors, alpha=0.45, zorder=3, label="peak -delta (deploy-ward alpha>0)")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Delta mean eval_signal vs. baseline")
    ax.set_title("Peak steering effect per model, by direction", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, v in zip(bars1, neg_peak):
        ax.text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9.5, fontweight="bold", color=INK)
    for bar, v in zip(bars2, pos_peak):
        ax.text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK_2)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_DIR / "10_steering_peak_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results = load_results()
    print(f"Loaded steering results for {len(results)} models: {list(results)}")
    if not results:
        raise SystemExit(f"No results found under {LOCAL_DIR}/*/results/steering_results.json")
    plot_steering_effect(results)
    plot_peak_summary(results)
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
