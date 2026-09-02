"""Charts for Experiment 2 (plan2.md): internal-vs-verbalized coupling.
Reads output/local/<model>/results/coupling_{last,mean}.json (written by
compare_internal_vs_verbal.py) and plots the *ensemble* coupling (mean
eval_signal across all 5 verbalized probes) per model, mirroring the
paper's Figure 4 / Appendix G style: |Spearman rho| and mutual information
vs. normalized layer depth, real probe vs. random-direction control band.

A model whose verbalized answers never varied (e.g. it said "no" on every
single transcript) has coupling that is mathematically undefined, not
zero -- compare_internal_vs_verbal.py marks this as `degenerate: true`,
and this script draws those panels as an explicit "undefined" placeholder
instead of a misleading flat line at 0.

Usage:
    python plot_coupling.py
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
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(LOCAL_DIR.glob("*")):
        if d.name not in SLUG_LABELS:
            continue
        pooled = {}
        for pooling in ("last", "mean"):
            p = d / "results" / f"coupling_{pooling}.json"
            if p.exists():
                pooled[pooling] = json.loads(p.read_text(encoding="utf-8"))
        if pooled:
            out[d.name] = pooled
    return out


def plot_layer_coupling(results: dict[str, dict[str, dict]], pooling: str, metric: str) -> None:
    """metric: 'rho' (abs Spearman) or 'mi' (mutual information, nats)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), sharey=(metric == "rho"))
    if n == 1:
        axes = [axes]

    for ax, (slug, data) in zip(axes, results.items()):
        if pooling not in data:
            ax.set_visible(False)
            continue
        label = SLUG_LABELS[slug]
        color = COLORS.get(label, "#666666")
        ens = data[pooling]["ensemble"]
        n_layers = data[pooling]["n_layers"]

        if ens.get("degenerate"):
            ax.text(
                0.5, 0.5, "undefined\n(verbalized answer\nnever varied)",
                ha="center", va="center", fontsize=11, color=MUTED, transform=ax.transAxes,
            )
            ax.set_title(label, loc="left", fontsize=11, fontweight="bold", color=INK)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        layers = ens["layers"]
        depth = np.array([row["layer"] / max(n_layers - 1, 1) for row in layers])
        if metric == "rho":
            real = np.array([abs(row["spearman_rho"]) for row in layers])
            rand = np.array([row["random_max_abs_rho"] for row in layers])
            ylabel = "|Spearman rho|"
            peak_layer, peak_val = ens["peak_abs_rho_layer"], ens["peak_abs_rho"]
        else:
            real = np.array([row["mutual_info_nats"] for row in layers])
            rand = np.array([row["random_max_mi_nats"] for row in layers])
            ylabel = "Mutual information (nats)"
            peak_layer, peak_val = ens["peak_mi_layer"], ens["peak_mi_nats"]

        ax.fill_between(depth, 0, rand, color=GRID, alpha=0.6, label="random-direction max", zorder=1)
        ax.plot(depth, real, color=color, linewidth=2.4, marker="o", markersize=3.5, label="real probe", zorder=3)
        ax.scatter(
            [peak_layer / max(n_layers - 1, 1)], [peak_val],
            s=90, facecolor="none", edgecolor=color, linewidth=2, zorder=4,
        )
        ax.set_title(
            f"{label}\npeak {ylabel} = {peak_val:.3f} @ L{peak_layer}/{n_layers}",
            loc="left", fontsize=11, fontweight="bold", color=INK,
        )
        ax.set_xlabel("Normalized layer depth")
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel({"rho": "|Spearman rho|", "mi": "Mutual information (nats)"}[metric])
    for ax in axes:
        if ax.get_visible() and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", frameon=False, fontsize=8.5)
            break

    metric_label = "Spearman correlation" if metric == "rho" else "mutual information"
    pool_label = "last-token" if pooling == "last" else "mean-token"
    fig.suptitle(
        f"Internal-vs-verbalized coupling: {metric_label} per layer ({pool_label} readout, ensemble probe)",
        x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / f"07_coupling_layers_{pooling}_{metric}.png", dpi=200)
    plt.close(fig)


def plot_peak_summary(results: dict[str, dict[str, dict]]) -> None:
    labels, rho_vals, mi_vals, colors, degenerate = [], [], [], [], []
    for slug, data in results.items():
        label = SLUG_LABELS[slug]
        ens = data.get("last", data.get("mean", {})).get("ensemble", {})
        labels.append(label)
        colors.append(COLORS.get(label, "#666666"))
        is_degenerate = ens.get("degenerate", False)
        degenerate.append(is_degenerate)
        rho_vals.append(None if is_degenerate else ens.get("peak_abs_rho"))
        mi_vals.append(None if is_degenerate else ens.get("peak_mi_nats"))

    y = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.7 * len(labels) + 1.6))

    for ax, vals, xlabel in ((ax1, rho_vals, "Peak |Spearman rho|"), (ax2, mi_vals, "Peak mutual info (nats)")):
        for yi, v, c, deg in zip(y, vals, colors, degenerate):
            if deg:
                ax.barh([yi], [ax.get_xlim()[1] or 1], height=0.55, color=GRID, zorder=2)
                ax.text(0.02, yi, "undefined (constant answer)", va="center", fontsize=9, color=MUTED)
            else:
                ax.barh([yi], [v], height=0.55, color=c, zorder=3)
                ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    ax1.set_xlim(0, 1.0)

    fig.suptitle(
        "Peak internal-vs-verbalized coupling per model (ensemble probe, last-token)",
        x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT_DIR / "08_coupling_peak_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results = load_results()
    print(f"Loaded coupling results for {len(results)} models: {list(results)}")
    if not results:
        raise SystemExit(f"No results found under {LOCAL_DIR}/*/results/coupling_*.json")
    for pooling in ("last", "mean"):
        plot_layer_coupling(results, pooling, "rho")
        plot_layer_coupling(results, pooling, "mi")
    plot_peak_summary(results)
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
