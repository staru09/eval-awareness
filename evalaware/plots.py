import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

COLORS = {
    "GPT-5.6-terra": "#EB6834",
    "Qwen3-8B": "#2A78D6",
    "Claude Sonnet 5": "#1BAF7A",
    "Mistral-Small-24B": "#EDA100",
    "granite-4.2-30b": "#4A3AA7",
    "Qwen3.6-27B-OBLITERATED": "#8C6D1F",
    "Gemma-4-12B-OBLITERATED": "#B4451F",
    "Llama-3.1-8B": "#7A7A55",
    "OBLITERATUS-qwen3-4b": "#898781",
}
SLUGS = {
    "qwen3-8b": "Qwen3-8B",
    "granite-4.2-30b": "granite-4.2-30b",
    "qwen3.6-27b-obliterated": "Qwen3.6-27B-OBLITERATED",
}
INK, INK2, MUTED, GRID, SURFACE, WARN = "#0B0B0B", "#52514E", "#898781", "#E1E0D9", "#FCFCFB", "#B4451F"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _finish(fig, ax_list, out: Path, title: str, rect=(0, 0, 1, 0.93)) -> None:
    for ax in ax_list:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=rect)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _series(data: dict):
    layers = sorted(int(k) for k in data["layer_results"])
    n = data["summary"]["n_layers"]
    depth = np.array([l / max(n - 1, 1) for l in layers])
    real = np.array([data["layer_results"][str(l)]["real"]["auroc"] for l in layers])
    rand = np.array([data["layer_results"][str(l)]["random"]["max_auroc_across_R"] for l in layers])
    return depth, real, rand


def verbalized_bar(models: dict[str, float], out: Path) -> None:
    items = sorted(models.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1.8))
    y = np.arange(len(items))
    ax.barh(y, values, color=[COLORS.get(l, "#666666") for l in labels], zorder=3)
    for yi, v in zip(y, values):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Ensemble AUROC")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    _finish(fig, [ax], out, "Ensemble AUROC (5-probe average) per model")


def probe_heatmap(matrix: np.ndarray, models: list[str], probes: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.5 * len(probes) + 4, 0.42 * len(models) + 2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(probes)), probes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(probes)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _finish(fig, [ax], out, "AUROC per model per probe phrasing")


def probe_lines(matrix: np.ndarray, models: list[str], probes: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(probes))
    for i, name in enumerate(models):
        ax.plot(x, matrix[i], marker="o", markersize=4, linewidth=1.8,
                color=COLORS.get(name, "#666666"), label=name)
    ax.set_xticks(x, probes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _finish(fig, [ax], out, "Per-probe AUROC by model: stability under rewording")


def coupling_layers(results: dict[str, dict], metric: str, pooling: str, out: Path) -> None:
    key = "spearman_rho" if metric == "rho" else "mutual_info_nats"
    rand_key = "random_max_abs_rho" if metric == "rho" else "random_max_mi_nats"
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (slug, data) in zip(axes, results.items()):
        rows = data["ensemble"]["layers"]
        n = len(rows)
        depth = np.array([r["layer"] / max(n - 1, 1) for r in rows])
        real = np.array([abs(r[key]) if r[key] is not None else 0.0 for r in rows])
        rand = np.array([r[rand_key] for r in rows])
        color = COLORS.get(SLUGS[slug], "#666666")
        ax.fill_between(depth, 0, rand, color=GRID, alpha=0.6, label="random-direction max", zorder=1)
        ax.plot(depth, real, color=color, linewidth=2.2, marker="o", markersize=3, zorder=3,
                label="real probe")
        ax.set_title(SLUGS[slug], loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("Normalized layer depth")
        ax.grid(axis="y", color=GRID, zorder=0)
    axes[0].set_ylabel("|Spearman rho|" if metric == "rho" else "Mutual information (nats)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    name = {"rho": "|Spearman rho|", "mi": "Mutual information"}[metric]
    _finish(fig, list(axes), out, f"Internal-vs-verbalized coupling: {name} ({pooling}-token)")


def steering_effect(results: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (slug, data) in zip(axes, results.items()):
        n = max(data["layers"]) or 1
        for alpha in data["alphas"]:
            cells = [c for c in data["cells"] if c["alpha"] == alpha and c["delta_vs_baseline"] is not None]
            if not cells:
                continue
            ax.plot([c["layer"] / n for c in cells], [c["delta_vs_baseline"] for c in cells],
                    marker="o", markersize=4, linewidth=1.8, label=f"alpha={alpha:+.1f}")
        ax.axhline(0, color=GRID, linewidth=1, zorder=0)
        ax.set_title(SLUGS[slug], loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("Normalized layer depth")
        ax.grid(axis="y", color=GRID, zorder=0)
    axes[0].set_ylabel("Delta mean eval_signal vs. baseline")
    axes[0].legend(frameon=False, fontsize=8.5)
    _finish(fig, list(axes), out, "Steering effect on verbalized eval_signal")


def steering_peaks(results: dict[str, dict], out: Path) -> None:
    labels, neg, pos = [], [], []
    for slug, data in results.items():
        labels.append(SLUGS[slug])
        deltas = [(c["alpha"], c["delta_vs_baseline"]) for c in data["cells"]
                  if c["delta_vs_baseline"] is not None]
        neg.append(max([d for a, d in deltas if a < 0], default=0.0))
        pos.append(max([d for a, d in deltas if a > 0], default=0.0))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y + 0.18, neg, height=0.34, color=[COLORS.get(l, "#666") for l in labels],
            zorder=3, label="best at alpha < 0")
    ax.barh(y - 0.18, pos, height=0.34, color=[COLORS.get(l, "#666") for l in labels],
            alpha=0.45, zorder=3, label="best at alpha > 0")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Delta mean eval_signal vs. baseline")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    _finish(fig, [ax], out, "Peak steering effect per model, by direction")


def readout_layers(results: dict[str, dict], pooling: str, out: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (slug, got) in zip(axes, results.items()):
        color = COLORS.get(SLUGS[slug], "#666666")
        d_a, real_a, rand_a = _series(got["answer"])
        d_p, real_p, rand_p = _series(got["prompt"])
        band = np.maximum(rand_a, rand_p) if len(rand_a) == len(rand_p) else rand_a
        ax.fill_between(d_a, 0, band, color=GRID, alpha=0.6, label="random direction, per-layer max")
        ax.axhline(0.5, color=GRID, linewidth=1, zorder=0)
        null = got.get("prompt_null")
        if null:
            p95 = null["permutation"]["p95"]
            ax.axhline(p95, color=WARN, linewidth=1.3, linestyle=":", label=f"permutation p95 ({p95:.2f})")
        ax.plot(d_a, real_a, color=color, linewidth=2.1, alpha=0.42, linestyle="--",
                label="prompt + own answer")
        ax.plot(d_p, real_p, color=color, linewidth=2.5, marker="o", markersize=3.2,
                label="prompt only")
        sa, sp = got["answer"]["summary"], got["prompt"]["summary"]
        ax.set_title(f"{SLUGS[slug]}\nprompt-only {sp['peak_auroc']:.3f} @L{sp['peak_layer']}"
                     f"  |  prompt+answer {sa['peak_auroc']:.3f}",
                     loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.set_xlabel("Normalized layer depth")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, zorder=0)
    axes[0].set_ylabel("AUROC (eval vs. deploy)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    _finish(fig, list(axes), out, f"Probe AUROC before vs. after generation ({pooling}-token)")


def readout_peaks(by_pooling: dict[str, dict], out: Path) -> None:
    slugs = sorted({s for r in by_pooling.values() for s in r})
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.85 * len(slugs) + 2.6), sharex=True)
    for ax, pooling in zip(axes, ("last", "mean")):
        results = by_pooling.get(pooling, {})
        labels = [SLUGS[s] for s in slugs]
        colors = [COLORS.get(l, "#666666") for l in labels]
        answer = [results.get(s, {}).get("answer", {}).get("summary", {}).get("peak_auroc") for s in slugs]
        prompt = [results.get(s, {}).get("prompt", {}).get("summary", {}).get("peak_auroc") for s in slugs]
        y = np.arange(len(slugs))
        ax.barh(y + 0.17, [v or 0 for v in prompt], height=0.34, color=colors, zorder=3,
                label="prompt only")
        ax.barh(y - 0.17, [v or 0 for v in answer], height=0.34, color=colors, alpha=0.4, zorder=3,
                label="prompt + own answer")
        for yi, slug in zip(y, slugs):
            null = results.get(slug, {}).get("prompt_null")
            if null:
                p95 = null["permutation"]["p95"]
                ax.plot([p95, p95], [yi - 0.42, yi + 0.42], color=WARN, linewidth=1.6, linestyle=":")
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Peak-layer AUROC")
        ax.set_title(f"{pooling}-token readout", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines["left"].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    _finish(fig, list(axes), out,
            "Peak internal-probe AUROC: before vs. after the model answers", rect=(0, 0, 1, 0.9))


def null_controls(rows: list[dict], out: Path, use_band: bool) -> None:
    rows = sorted(rows, key=lambda r: r["perm_p"])
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.2))
    stat = "band" if use_band else "argmax"
    ax.barh(y, [r[stat] for r in rows], height=0.55, color=[r["color"] for r in rows], zorder=3,
            label=f"{stat} statistic")
    ax.scatter([r["rand_mean"] for r in rows], y, marker="s", s=30, color=MUTED, zorder=5,
               label="random directions, mean")
    for yi, r in zip(y, rows):
        ax.plot([r["perm_p95"], r["perm_p95"]], [yi - 0.3, yi + 0.3], color=WARN, linewidth=2, zorder=6)
        strong = r["perm_p"] < 0.05 and r["rand_p"] < 0.05
        ax.text(1.02, yi, f"perm {r['perm_p']:.3f} / rand {r['rand_p']:.3f}", va="center",
                fontsize=8, color=WARN if strong else MUTED,
                fontweight="bold" if strong else "normal")
    ax.plot([], [], color=WARN, linewidth=2, label="label-permutation null, p95")
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.7)
    ax.set_xlabel("AUROC (eval vs. deploy)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    title = ("Band statistic against 1000 fresh random directions" if use_band
             else "Peak AUROC against its actual null models")
    _finish(fig, [ax], out, title, rect=(0, 0, 1, 0.94))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
