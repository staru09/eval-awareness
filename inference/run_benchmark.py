#!/usr/bin/env python3
"""Runner: benchmark multiple OpenAI/Anthropic models against the
eval-awareness dataset, recording per-model logs and an aggregated
metrics summary.

Wraps run_inference_api.py (one subprocess per model, so a crash/API error
on one model doesn't take down the rest) and:
  - tees each model's stdout/stderr to both the console and a per-model
    timestamped log file under --log-dir
  - after all models finish, reads each model's
    output/api/<provider>__<model_slug>/results/verbalized_auroc.json and
    writes one aggregated comparison table (JSON + Markdown), analogous to
    ../EXPERIMENT_SUMMARY.md's six-model table

Usage:
    python run_benchmark.py \
      --models openai:gpt-4o anthropic:claude-sonnet-4-6 \
      --limit 5

    # Full run with extended thinking on the Anthropic model:
    python run_benchmark.py \
      --models openai:gpt-4o openai:gpt-4o-mini anthropic:claude-sonnet-4-6 \
      --enable-thinking anthropic:claude-sonnet-4-6

    # Re-run only models whose output is missing/incomplete:
    python run_benchmark.py --models openai:gpt-4o anthropic:claude-sonnet-4-6 --resume
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "data" / "dataset.json"
DEFAULT_PROBE_QUESTIONS = HERE.parent / "probes" / "probe_questions.yaml"
DEFAULT_OUT_ROOT = HERE.parent / "output" / "api"
DEFAULT_LOG_DIR = HERE.parent / "output" / "logs"


def slugify(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model)


def parse_model_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"Invalid --models entry {spec!r}, expected 'provider:model' (e.g. openai:gpt-4o)")
    provider, model = spec.split(":", 1)
    if provider not in ("openai", "anthropic"):
        raise SystemExit(f"Unknown provider {provider!r} in {spec!r}; expected 'openai' or 'anthropic'")
    return provider, model


def run_one_model(
    provider: str,
    model: str,
    *,
    dataset: Path,
    probe_questions: Path,
    out_root: Path,
    log_dir: Path,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    limit: int | None,
) -> dict:
    """Runs run_inference_api.py for one model as a subprocess, tees output
    to console + a timestamped log file. Returns a status dict.
    """
    out_dir = out_root / f"{provider}__{slugify(model)}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{provider}__{slugify(model)}__{timestamp}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(HERE / "run_inference_api.py"),
        "--provider", provider,
        "--model", model,
        "--dataset", str(dataset),
        "--probe-questions", str(probe_questions),
        "--out-dir", str(out_dir),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
    ]
    if enable_thinking:
        cmd.append("--enable-thinking")
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n=== [{provider}:{model}] starting, logging to {log_path} ===")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# command: {' '.join(cmd)}\n# started: {timestamp}\n\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"[{provider}:{model}] {line}")
            log_f.write(line)
        proc.wait()
    elapsed = time.time() - t0

    status = {
        "provider": provider,
        "model": model,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log_path": str(log_path),
        "out_dir": str(out_dir),
    }
    ok = proc.returncode == 0
    print(f"=== [{provider}:{model}] {'done' if ok else 'FAILED'} in {elapsed:.1f}s (see {log_path}) ===")
    return status


def already_has_results(out_root: Path, provider: str, model: str) -> bool:
    results_path = out_root / f"{provider}__{slugify(model)}" / "results" / "verbalized_auroc.json"
    return results_path.exists()


def load_model_results(out_root: Path, provider: str, model: str) -> dict | None:
    results_path = out_root / f"{provider}__{slugify(model)}" / "results" / "verbalized_auroc.json"
    if not results_path.exists():
        return None
    return json.loads(results_path.read_text(encoding="utf-8"))


def build_summary(out_root: Path, run_statuses: list[dict]) -> dict:
    rows = []
    for s in run_statuses:
        result = load_model_results(out_root, s["provider"], s["model"])
        rows.append(
            {
                "provider": s["provider"],
                "model": s["model"],
                "run_ok": s["returncode"] == 0,
                "elapsed_s": s["elapsed_s"],
                "ensemble_auroc": (result or {}).get("summary", {}).get("ensemble_auroc"),
                "n_transcripts": (result or {}).get("summary", {}).get("n_transcripts"),
                "per_probe": (result or {}).get("per_probe"),
                "log_path": s["log_path"],
            }
        )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "models": rows}


def write_summary_markdown(summary: dict, out_path: Path) -> None:
    lines = [
        "# Benchmark Summary — OpenAI / Anthropic Models",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Provider | Model | Ran OK | n | Ensemble AUROC | Elapsed (s) |",
        "| --- | --- | :---: | ---: | ---: | ---: |",
    ]
    for r in summary["models"]:
        auroc = f"{r['ensemble_auroc']:.3f}" if r["ensemble_auroc"] is not None else "n/a"
        lines.append(
            f"| {r['provider']} | {r['model']} | {'yes' if r['run_ok'] else '**no**'} | "
            f"{r['n_transcripts'] or '-'} | {auroc} | {r['elapsed_s']} |"
        )
    lines += ["", "## Per-probe breakdown", ""]
    for r in summary["models"]:
        lines.append(f"### {r['provider']}:{r['model']}")
        if not r["per_probe"]:
            lines.append("(no results)")
            lines.append("")
            continue
        lines.append("| Probe | AUROC | Accuracy | n scored | parse failures |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for pid, m in r["per_probe"].items():
            auroc = f"{m['auroc']:.3f}" if m["auroc"] is not None else "n/a"
            acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "n/a"
            lines.append(f"| {pid} | {auroc} | {acc} | {m['n_scored']} | {m['n_parse_failures']} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+",
        help="One or more 'provider:model' specs, e.g. openai:gpt-4o anthropic:claude-sonnet-4-6",
    )
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), type=Path)
    ap.add_argument("--probe-questions", default=str(DEFAULT_PROBE_QUESTIONS), type=Path)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), type=Path)
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), type=Path)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--enable-thinking", nargs="*", default=None,
        help="Enable extended thinking for these 'provider:model' specs only "
             "(no args = enable for all models in --models).",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N transcripts per model (smoke test)")
    ap.add_argument("--resume", action="store_true", help="skip models that already have results/verbalized_auroc.json")
    ap.add_argument("--summary-json", default=None, type=Path)
    ap.add_argument("--summary-md", default=None, type=Path)
    args = ap.parse_args()

    model_specs = [parse_model_spec(m) for m in args.models]
    thinking_set = None
    if args.enable_thinking is not None:
        thinking_set = set(args.enable_thinking) if args.enable_thinking else {m for m in args.models}

    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    summary_json = args.summary_json or (out_root / "benchmark_summary.json")
    summary_md = args.summary_md or (out_root / "BENCHMARK_SUMMARY.md")

    run_statuses = []
    for spec_str, (provider, model) in zip(args.models, model_specs):
        if args.resume and already_has_results(out_root, provider, model):
            print(f"--resume: skipping {provider}:{model} (results already present)")
            continue
        enable_thinking = thinking_set is not None and spec_str in thinking_set
        status = run_one_model(
            provider, model,
            dataset=Path(args.dataset), probe_questions=Path(args.probe_questions),
            out_root=out_root, log_dir=log_dir,
            max_tokens=args.max_tokens, temperature=args.temperature,
            enable_thinking=enable_thinking, limit=args.limit,
        )
        run_statuses.append(status)

    # Include models skipped by --resume in the summary too (they still have results).
    if args.resume:
        ran = {(s["provider"], s["model"]) for s in run_statuses}
        for provider, model in model_specs:
            if (provider, model) not in ran and already_has_results(out_root, provider, model):
                run_statuses.append(
                    {"provider": provider, "model": model, "returncode": 0, "elapsed_s": 0.0,
                     "log_path": "(resumed, not re-run)", "out_dir": str(out_root / f"{provider}__{slugify(model)}")}
                )

    summary = build_summary(out_root, run_statuses)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(summary, summary_md)

    n_ok = sum(1 for s in run_statuses if s["returncode"] == 0)
    print(f"\nDone: {n_ok}/{len(run_statuses)} model runs succeeded.")
    print(f"Summary: {summary_json}")
    print(f"Summary: {summary_md}")

    if any(s["returncode"] != 0 for s in run_statuses):
        sys.exit(1)


if __name__ == "__main__":
    main()
