import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import activations, api, datasets, metrics, plots, probe, scoring, steering, verbalize
from .models import ModelSpec, load_model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _spec(args) -> ModelSpec:
    return ModelSpec(hf_id=args.model_path, chat_template=True,
                     enable_thinking=not getattr(args, "no_thinking", False))


def cmd_generate(args) -> None:
    model, tok = load_model(args.model_path, args.dtype)
    rows = datasets.load_transcripts(args.dataset)
    entries = activations.generate_and_capture(
        model, tok, _spec(args), rows, args.out_dir,
        max_new_tokens=args.max_new_tokens, capture=not args.no_activations)
    print(f"wrote {len(entries)} transcripts to {args.out_dir}")


def cmd_capture_prompt(args) -> None:
    model, tok = load_model(args.model_path, args.dtype)
    traces = datasets.load_traces(args.traces)
    n = activations.capture_prompt_only(model, tok, traces, args.out_dir, args.model_path)
    print(f"wrote {n} prompt-only activation files to {args.out_dir}")


def cmd_probe(args) -> None:
    model, tok = load_model(args.model_path, args.dtype)
    payload = probe.build(model, tok, datasets.load_pairs(), args.out, args.model_path,
                          n_random=args.n_random, seed=args.seed)
    meta = payload["metadata"]
    print(f"probe: {meta['num_layers']} layers, d_model={meta['d_model']} -> {args.out}")


def cmd_score(args) -> None:
    info = scoring.score(probe.load(args.probe), args.activations_dir,
                         datasets.load_transcripts(args.dataset), args.out, args.model_key)
    print(f"scored {info['n_rows']} rows x {info['n_layers']} layers -> {args.out}")


def cmd_auroc(args) -> None:
    s = scoring.load_scores(args.scores, args.pooling)
    table = metrics.layer_auroc_table(s["real"], s["random"], s["labels"])
    peak = max(table, key=lambda k: table[k]["real"]["auroc"])
    payload = {
        "summary": {
            "model_key": s["model_key"], "pooling": args.pooling,
            "n_samples": len(s["labels"]), "n_layers": s["n_layers"],
            "n_random_controls": s["n_random"], "peak_layer": int(peak),
            "peak_auroc": table[peak]["real"]["auroc"],
            "peak_loo_accuracy": table[peak]["real"]["loo_accuracy"],
        },
        "layer_results": table,
    }
    _write_json(args.out, payload)
    print(f"peak layer {peak}: AUROC={payload['summary']['peak_auroc']:.3f} -> {args.out}")


def cmd_verbalize(args) -> None:
    model, tok = load_model(args.model_path, args.dtype)
    questions = datasets.load_probe_questions()
    rows = verbalize.run(model, tok, _spec(args), datasets.load_traces(args.traces),
                         questions, args.out_csv, max_new_tokens=args.max_new_tokens)
    _write_json(args.out_json, {
        "summary": {"model": args.model_path, "n_transcripts": len({r["transcript_id"] for r in rows}),
                    "n_probes": len(questions), "ensemble_auroc": metrics.ensemble_auroc(rows)},
        "per_probe": metrics.per_probe_auroc(rows),
    })
    print(f"ensemble AUROC: {metrics.ensemble_auroc(rows)}")


def cmd_api(args) -> None:
    client = api.make_client(args.provider, args.base_url)
    rows = datasets.load_transcripts(args.dataset)
    traces = api.generate_traces(client, args.provider, args.model, rows, args.out_dir,
                                 args.max_tokens, args.temperature, args.enable_thinking,
                                 args.reasoning_effort)
    questions = datasets.load_probe_questions()
    verbal = api.verbalize_traces(client, args.provider, args.model, traces, questions,
                                  args.out_dir / "verbalized_responses.csv",
                                  args.max_tokens, args.temperature)
    _write_json(args.out_dir / "results" / "verbalized_auroc.json", {
        "summary": {"model": args.model, "n_transcripts": len(traces), "n_probes": len(questions),
                    "ensemble_auroc": metrics.ensemble_auroc(verbal)},
        "per_probe": metrics.per_probe_auroc(verbal),
    })
    print(f"ensemble AUROC: {metrics.ensemble_auroc(verbal)}")


def _coupling_block(real, prompt_ids, signal, n_random, rand, seed) -> dict:
    layers = []
    degenerate = False
    for layer in range(real.shape[1]):
        rho, mi, deg = metrics.coupling(real[:, layer], signal, prompt_ids, seed)
        degenerate = degenerate or deg
        rand_rho, rand_mi = [], []
        for r in range(n_random):
            rr, rm, _ = metrics.coupling(rand[:, layer, r], signal, prompt_ids, seed + r + 1)
            rand_rho.append(abs(rr) if rr is not None else 0.0)
            rand_mi.append(rm)
        layers.append({
            "layer": layer, "spearman_rho": rho, "mutual_info_nats": mi,
            "random_max_abs_rho": float(max(rand_rho)) if rand_rho else 0.0,
            "random_max_mi_nats": float(max(rand_mi)) if rand_mi else 0.0,
        })
    if degenerate:
        return {"layers": layers, "degenerate": True, "peak_abs_rho": None, "peak_mi_nats": None}
    rho_idx = int(np.argmax([abs(r["spearman_rho"]) for r in layers]))
    mi_idx = int(np.argmax([r["mutual_info_nats"] for r in layers]))
    return {
        "layers": layers, "degenerate": False,
        "peak_abs_rho_layer": rho_idx, "peak_abs_rho": abs(layers[rho_idx]["spearman_rho"]),
        "peak_mi_layer": mi_idx, "peak_mi_nats": layers[mi_idx]["mutual_info_nats"],
    }


def cmd_coupling(args) -> None:
    s = scoring.load_scores(args.scores, args.pooling)
    by_probe = verbalize.signals_by_probe(verbalize.load_csv(args.verbalized_csv))
    payload = {
        "model_key": s["model_key"], "pooling": args.pooling, "n_layers": s["n_layers"],
        "n_random_controls": s["n_random"], "n_probes": len(by_probe),
        "per_probe": {
            pid: _coupling_block(s["real"], s["prompt_ids"], sig, s["n_random"], s["random"], args.seed)
            for pid, sig in sorted(by_probe.items())
        },
        "ensemble": _coupling_block(s["real"], s["prompt_ids"], verbalize.ensemble_signal(by_probe),
                                    s["n_random"], s["random"], args.seed),
    }
    _write_json(args.out, payload)
    peak = payload["ensemble"].get("peak_abs_rho")
    print(f"ensemble peak |rho| = {peak} -> {args.out}")


def cmd_steer(args) -> None:
    model, tok = load_model(args.model_path, args.dtype)
    questions = datasets.load_probe_questions()
    question = next((q for q in questions if q["id"] == args.probe_id), None)
    if question is None:
        raise SystemExit(f"unknown probe id {args.probe_id!r}")
    traces = datasets.load_traces(args.traces)
    ids = args.transcript_ids.split(",") if args.transcript_ids else [t["id"] for t in traces[:6]]
    payload = steering.run(
        model, tok, ModelSpec(hf_id=args.model_path, chat_template=True, enable_thinking=False),
        probe.load(args.probe), traces, question, ids,
        [int(x) for x in args.layers.split(",")],
        [float(x) for x in args.alphas.split(",")],
        args.activations_dir, max_new_tokens=args.max_new_tokens)
    _write_json(args.out, payload)
    print(f"baseline {payload['baseline']['mean_eval_signal']} -> {args.out}")


def _baselines(real, hidden, labels, prompt_ids, traces_path, width) -> dict:
    out = {}
    if hidden is not None:
        norms = np.linalg.norm(hidden, axis=2)
        out["cosine"] = metrics.band(
            metrics.per_layer_auroc(real / np.maximum(norms, 1e-8), labels), width)[0]
        out["norm_only"] = metrics.band(metrics.per_layer_auroc(norms, labels), width)[0]
    if traces_path:
        tokens = {int(t["prompt_id"]): t.get("n_prompt_tokens") or 0
                  for t in datasets.load_traces(traces_path)}
        lengths = np.array([tokens.get(int(p), 0) for p in prompt_ids], dtype=float)
        out["prompt_length"] = metrics.auroc(lengths, labels)
    return out


def _format_split(real, labels, sources, width) -> dict:
    is_mcq = np.array([s in datasets.MCQ_SOURCES for s in sources]).astype(int)
    keep = is_mcq == 0
    controlled = None
    if np.unique(labels[keep]).size > 1:
        controlled = metrics.band(metrics.per_layer_auroc(real[keep], labels[keep]), width)[0]
    return {
        "mcq_auroc": metrics.band(metrics.per_layer_auroc(real, is_mcq), width)[0],
        "controlled_auroc": controlled,
        "n_mcq": int(is_mcq.sum()),
    }


def _stack_activations(act_dir: Path, ids: list[str], pooling: str) -> np.ndarray | None:
    key = "last_token" if pooling == "last" else "mean_token"
    rows = []
    for tid in ids:
        path = act_dir / f"{tid}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            rows.append(z[key])
    return np.stack(rows)


def cmd_nulls(args) -> None:
    s = scoring.load_scores(args.scores, args.pooling)
    by_pid = {int(r["prompt_id"]): r for r in datasets.load_transcripts(args.dataset)}
    ids = [by_pid[int(p)]["id"] for p in s["prompt_ids"]]
    sources = [by_pid[int(p)]["source"] for p in s["prompt_ids"]]
    real, labels = s["real"], s["labels"]

    per_layer = metrics.per_layer_auroc(real, labels)
    argmax_auroc, argmax_layer = metrics.band(per_layer, 1)
    band_auroc, band_layer = metrics.band(per_layer, args.band_width)

    perm = metrics.permutation_null(real, labels, args.n_perm, args.band_width, args.seed)
    hidden = _stack_activations(args.activations_dir, ids, args.pooling) if args.activations_dir else None
    rand = None
    if hidden is not None and args.n_random:
        rand = metrics.random_direction_null(hidden, labels, args.band_width, args.n_random,
                                             args.seed + 991)

    payload = {
        "summary": {"model_key": s["model_key"], "pooling": args.pooling,
                    "scores_file": str(args.scores), "n_layers": s["n_layers"],
                    "band_width": args.band_width, "n_permutations": args.n_perm},
        "argmax": {"auroc": argmax_auroc, "layer": argmax_layer},
        "band": {"auroc": band_auroc, "start_layer": band_layer},
        "permutation": {"mean": float(perm.mean()), "p95": float(np.percentile(perm, 95)),
                        "p": metrics.empirical_p(perm, band_auroc)},
        "random_directions": None if rand is None else {
            "n": args.n_random, "mean": float(rand.mean()),
            "p": metrics.empirical_p(rand, band_auroc)},
        "baselines": _baselines(real, hidden, labels, s["prompt_ids"], args.traces, args.band_width),
        "format": _format_split(real, labels, np.array(sources), args.band_width),
    }
    _write_json(args.out, payload)
    print(f"band {band_auroc:.3f} @L{band_layer}  perm p={payload['permutation']['p']:.4f}"
          + (f"  rand p={payload['random_directions']['p']:.4f}" if rand is not None else ""))


def _plot_verbalized(root: Path, charts: Path) -> None:
    results = {}
    for d in sorted((root / "api").glob("*")):
        path = d / "results" / "verbalized_auroc.json"
        if path.exists():
            results[d.name] = plots.load_json(path)
    if not results:
        return
    names = list(results)
    ensemble = {n: results[n]["summary"]["ensemble_auroc"] for n in names
                if results[n]["summary"]["ensemble_auroc"] is not None}
    plots.verbalized_bar(ensemble, charts / "01_ensemble_auroc_bar.png")
    probes = sorted({p for r in results.values() for p in r["per_probe"]})
    matrix = np.array([[results[n]["per_probe"].get(p, {}).get("auroc") or np.nan for p in probes]
                       for n in names])
    plots.probe_heatmap(matrix, names, probes, charts / "02_probe_heatmap.png")
    plots.probe_lines(matrix, names, probes, charts / "03_probe_lines.png")


def _plot_coupling(root: Path, charts: Path) -> None:
    for pooling in ("last", "mean"):
        found = {}
        for slug in plots.SLUGS:
            path = root / "local" / slug / "results" / f"coupling_{pooling}.json"
            if path.exists():
                found[slug] = plots.load_json(path)
        for metric in ("rho", "mi"):
            if found:
                plots.coupling_layers(found, metric, pooling,
                                      charts / f"07_coupling_layers_{pooling}_{metric}.png")


def _plot_steering(root: Path, charts: Path) -> None:
    found = {}
    for slug in plots.SLUGS:
        path = root / "local" / slug / "results" / "steering_results.json"
        if path.exists():
            found[slug] = plots.load_json(path)
    if found:
        plots.steering_effect(found, charts / "09_steering_effect.png")
        plots.steering_peaks(found, charts / "10_steering_peak_summary.png")


def _null_row(slug: str, pooling: str, readout: str, null: dict) -> dict:
    return {
        "label": f"{plots.SLUGS[slug]}\n{readout}, {pooling}-token",
        "color": plots.COLORS.get(plots.SLUGS[slug], "#666666"),
        "argmax": null["argmax"]["auroc"], "band": null["band"]["auroc"],
        "perm_p": null["permutation"]["p"], "perm_p95": null["permutation"]["p95"],
        "rand_p": null["random_directions"]["p"], "rand_mean": null["random_directions"]["mean"],
    }


def _load_readout(base: Path, pooling: str) -> dict:
    names = {"answer": f"internal_auroc_{pooling}.json",
             "prompt": f"internal_auroc_prompt_{pooling}.json",
             "answer_null": f"nulls_answer_{pooling}.json",
             "prompt_null": f"nulls_prompt_{pooling}.json"}
    return {key: plots.load_json(base / name) for key, name in names.items()
            if (base / name).exists()}


def _plot_probe(root: Path, charts: Path) -> None:
    rows, by_pooling = [], {}
    for pooling in ("last", "mean"):
        readouts = {}
        for slug in plots.SLUGS:
            got = _load_readout(root / "local" / slug / "results", pooling)
            if "answer" in got and "prompt" in got:
                readouts[slug] = got
            for readout, key in (("prompt only", "prompt_null"), ("prompt + answer", "answer_null")):
                null = got.get(key)
                if null and null.get("random_directions"):
                    rows.append(_null_row(slug, pooling, readout, null))
        if readouts:
            plots.readout_layers(readouts, pooling,
                                 charts / f"11_prompt_vs_answer_layers_{pooling}.png")
            by_pooling[pooling] = readouts
    if by_pooling:
        plots.readout_peaks(by_pooling, charts / "12_prompt_vs_answer_peaks.png")
    if rows:
        plots.null_controls(rows, charts / "13_probe_null_controls.png", use_band=False)
        plots.null_controls(rows, charts / "14_band_statistic.png", use_band=True)


def cmd_plots(args) -> None:
    charts = args.output_dir / "charts"
    _plot_verbalized(args.output_dir, charts)
    _plot_coupling(args.output_dir, charts)
    _plot_steering(args.output_dir, charts)
    _plot_probe(args.output_dir, charts)
    print(f"charts written to {charts}")


def _add_model_args(sub, name, fn, dtype=True):
    s = sub.add_parser(name)
    s.set_defaults(func=fn)
    s.add_argument("--model-path", required=True)
    s.add_argument("--no-thinking", action="store_true")
    if dtype:
        s.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return s


def _add_generate(sub) -> None:
    g = _add_model_args(sub, "generate", cmd_generate)
    g.add_argument("--dataset", type=Path, default=None)
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--max-new-tokens", type=int, default=4096)
    g.add_argument("--no-activations", action="store_true")

    c = _add_model_args(sub, "capture-prompt", cmd_capture_prompt)
    c.add_argument("--traces", type=Path, required=True)
    c.add_argument("--out-dir", type=Path, required=True)

    p = _add_model_args(sub, "probe", cmd_probe)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)


def _add_analysis(sub) -> None:
    sc = sub.add_parser("score")
    sc.set_defaults(func=cmd_score)
    sc.add_argument("--probe", type=Path, required=True)
    sc.add_argument("--activations-dir", type=Path, required=True)
    sc.add_argument("--dataset", type=Path, default=None)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--model-key", default=None)

    au = sub.add_parser("auroc")
    au.set_defaults(func=cmd_auroc)
    au.add_argument("--scores", type=Path, required=True)
    au.add_argument("--pooling", default="last", choices=["last", "mean"])
    au.add_argument("--out", type=Path, required=True)

    co = sub.add_parser("coupling")
    co.set_defaults(func=cmd_coupling)
    co.add_argument("--scores", type=Path, required=True)
    co.add_argument("--verbalized-csv", type=Path, required=True)
    co.add_argument("--pooling", default="last", choices=["last", "mean"])
    co.add_argument("--seed", type=int, default=0)
    co.add_argument("--out", type=Path, required=True)

    nu = sub.add_parser("nulls")
    nu.set_defaults(func=cmd_nulls)
    nu.add_argument("--scores", type=Path, required=True)
    nu.add_argument("--dataset", type=Path, default=None)
    nu.add_argument("--activations-dir", type=Path, default=None)
    nu.add_argument("--traces", type=Path, default=None)
    nu.add_argument("--pooling", default="last", choices=["last", "mean"])
    nu.add_argument("--band-width", type=int, default=3)
    nu.add_argument("--n-perm", type=int, default=5000)
    nu.add_argument("--n-random", type=int, default=1000)
    nu.add_argument("--seed", type=int, default=0)
    nu.add_argument("--out", type=Path, required=True)

    pl = sub.add_parser("plots")
    pl.set_defaults(func=cmd_plots)
    pl.add_argument("--output-dir", type=Path, default=Path("output"))


def _add_probing(sub) -> None:
    vb = _add_model_args(sub, "verbalize", cmd_verbalize)
    vb.add_argument("--traces", type=Path, required=True)
    vb.add_argument("--out-csv", type=Path, required=True)
    vb.add_argument("--out-json", type=Path, required=True)
    vb.add_argument("--max-new-tokens", type=int, default=1024)

    st = _add_model_args(sub, "steer", cmd_steer)
    st.add_argument("--probe", type=Path, required=True)
    st.add_argument("--traces", type=Path, required=True)
    st.add_argument("--activations-dir", type=Path, required=True)
    st.add_argument("--probe-id", default="direct_a")
    st.add_argument("--layers", required=True)
    st.add_argument("--alphas", default="-1.0,-0.5,0.5,1.0")
    st.add_argument("--transcript-ids", default=None)
    st.add_argument("--max-new-tokens", type=int, default=512)
    st.add_argument("--out", type=Path, required=True)

    ap = sub.add_parser("api")
    ap.set_defaults(func=cmd_api)
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default="medium")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalaware")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate(sub)
    _add_probing(sub)
    _add_analysis(sub)
    return parser


def _glue_negatives(argv: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--alphas" and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"--alphas={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(_glue_negatives(argv if argv is not None else sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
