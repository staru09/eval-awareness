"""Generate summary charts for the verbalized eval-awareness benchmark
(output/api/*/results/verbalized_auroc.json) and save them as PNGs under
output/charts/, embedded in README.md's Results section.

Usage:
    python plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent / "output" / "api"
OUT_DIR = HERE.parent / "output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# validated categorical palette (dataviz skill reference instance), fixed
# per-model assignment matching report.tex / dashboard
COLORS = {
    "GPT-5.6-terra": "#EB6834",
    "Qwen3-8B": "#2A78D6",
    "Claude Sonnet 5": "#1BAF7A",
    "Mistral-Small-24B": "#EDA100",
    "Llama-3.1-8B": "#E87BA4",
    "granite-4.2-30b": "#4A3AA7",
    "qwen3-4b-OBLITERATED": "#1B8A9E",
    "Gemma-4-12B-OBLITERATED": "#C23B3B",
    "Qwen3.6-27B-OBLITERATED": "#8C6D1F",
}
MODEL_LABELS = {
    "openai__gpt-5.6-terra": "GPT-5.6-terra",
    "vllm__Qwen-Qwen3-8B": "Qwen3-8B",
    "anthropic__claude-sonnet-5": "Claude Sonnet 5",
    "vllm__mistralai-Mistral-Small-24B-Instruct-2501": "Mistral-Small-24B",
    "vllm__meta-llama-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "vllm__ibm-granite-granite-4.2-30b": "granite-4.2-30b",
    "vllm__OBLITERATUS-qwen3-4b-structured-output-merged-stage-a-OBLITERATED": "qwen3-4b-OBLITERATED",
    "local__gemma-4-12b-obliterated": "Gemma-4-12B-OBLITERATED",
    "local__qwen3.6-27b-obliterated": "Qwen3.6-27B-OBLITERATED",
}
PROBE_LABELS = {
    "direct_a": "Direct",
    "direct_vn_a": "Direct v2",
    "h_purpose_analysis_a": "Purpose",
    "h_intent_and_style_a": "Intent/Style",
    "h_user_or_control_set_a": "User/Control",
}
PROBE_ORDER = list(PROBE_LABELS)

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
    for d in sorted(API_DIR.glob("*")):
        key = d.name
        if key not in MODEL_LABELS:
            continue
        p = d / "results" / "verbalized_auroc.json"
        if not p.exists():
            continue
        out[MODEL_LABELS[key]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def plot_ensemble_bar(results: dict[str, dict]) -> None:
    ranked = sorted(results.items(), key=lambda kv: kv[1]["summary"]["ensemble_auroc"])
    labels = [k for k, _ in ranked]
    values = [v["summary"]["ensemble_auroc"] for _, v in ranked]
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.6, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Ensemble AUROC")
    ax.set_title("Ensemble AUROC (5-probe average) per model", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_ensemble_auroc_bar.png", dpi=200)
    plt.close(fig)


def plot_probe_heatmap(results: dict[str, dict]) -> None:
    models = sorted(results.keys(), key=lambda k: -results[k]["summary"]["ensemble_auroc"])
    matrix = np.array([
        [results[m]["per_probe"][p]["auroc"] for p in PROBE_ORDER]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PROBE_ORDER)), [PROBE_LABELS[p] for p in PROBE_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(models)), models)
    ax.set_title("AUROC per model per probe phrasing", loc="left", fontsize=13, fontweight="bold", color=INK)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val >= 0.72 else INK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("AUROC", color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_probe_heatmap.png", dpi=200)
    plt.close(fig)


def plot_probe_lines(results: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = np.arange(len(PROBE_ORDER))
    for model, data in sorted(results.items(), key=lambda kv: -kv[1]["summary"]["ensemble_auroc"]):
        y = [data["per_probe"][p]["auroc"] for p in PROBE_ORDER]
        ax.plot(x, y, marker="o", markersize=5, linewidth=2, color=COLORS[model], label=model)
    ax.set_xticks(x, [PROBE_LABELS[p] for p in PROBE_ORDER])
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-probe AUROC by model — stability under rewording", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_probe_lines.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results = load_results()
    print(f"Loaded results for {len(results)} models: {list(results)}")
    if not results:
        raise SystemExit(f"No results found under {API_DIR}")
    plot_ensemble_bar(results)
    plot_probe_heatmap(results)
    plot_probe_lines(results)
    print(f"Wrote 3 charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
