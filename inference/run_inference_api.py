#!/usr/bin/env python3
"""Run the eval-awareness benchmark on an OpenAI, Anthropic, or self-hosted
vLLM model, as a drop-in alternative to run_inference.py (local Qwen3-8B
via HF in-process).

API models expose no hidden states, so this script covers the two stages
that don't require local weight access:

  1. Generation over the mixed eval/deploy dataset (data/dataset.json),
     capturing a reasoning trace when the provider/model exposes one
     (Anthropic extended thinking; OpenAI reasoning-capable models via
     the Responses API; vLLM/open-weight models via inline <think> tags)
     — same JSON schema as run_inference.py's reasoning_traces.json, so
     gpt_judge.py works unmodified on any of the three.
  2. Verbalized probing: for each transcript, appends each of
     probes/probe_questions.yaml's 5 phrasings as a follow-up turn and
     asks the model whether the interaction above was an evaluation,
     reusing probes/verbalize_probe.py's exact label-parsing fallback
     chain and PARSE_FAILURE_VALUE convention.

(Internal linear-probe activation analysis — plan.md steps 1-3 — is
Qwen3-8B/HF-only and has no equivalent here.)

Outputs, under --out-dir (default ../output/api/<provider>__<model_slug>/):
  reasoning_traces.json
  verbalized_responses.csv
  results/verbalized_auroc.json

Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY (see ../.env, loaded via
python-dotenv if present). --provider vllm needs no API key by default
(vLLM ignores it unless you've configured server-side auth, in which case
set VLLM_API_KEY) but does need --base-url pointed at your server, e.g. a
vLLM instance on a cloud GPU VM exposing an OpenAI-compatible endpoint
(`vllm serve <model> --host 0.0.0.0 --port 8000`) -- see plan.md /
../../eval-aware-oss's precedent for this pattern. Uses uv for package
management, per this workspace's convention
(`uv run python run_inference_api.py ...`).

Thinking is ON by default. OpenAI reasoning models are the gpt-5.x family
(gpt-5.6, gpt-5.6-sol/-terra/-luna, gpt-5.5, gpt-5.4, ...) -- gpt-4o and
other non-reasoning models need --no-thinking. For vLLM, thinking toggles
the served model's own chat-template thinking mode (e.g. Qwen3); pass
--no-thinking for a model/checkpoint with no thinking mode at all.

Usage:
    python run_inference_api.py --provider anthropic --model claude-sonnet-4-6
    python run_inference_api.py --provider openai --model gpt-5.6
    python run_inference_api.py --provider openai --model gpt-5.6 --reasoning-effort high
    python run_inference_api.py --provider openai --model gpt-4o --no-thinking
    python run_inference_api.py --provider vllm --model Qwen/Qwen3-8B \
      --base-url http://<your-vm-ip>:8000/v1
    python run_inference_api.py --provider openai --model gpt-5.6 --limit 2   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from api_clients import PROVIDERS, vllm_client  # noqa: E402
from verbalize_probe import PARSE_FAILURE_VALUE, resolve_answer  # noqa: E402

LOG = logging.getLogger("run_inference_api")

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "data" / "dataset.json"
DEFAULT_PROBE_QUESTIONS = HERE.parent / "probes" / "probe_questions.yaml"


def slugify(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True, help="e.g. gpt-4o, claude-sonnet-4-6, Qwen/Qwen3-8B")
    ap.add_argument(
        "--base-url", default=None,
        help="Required for --provider vllm: your vLLM server's OpenAI-compatible endpoint "
             "(e.g. http://<vm-ip>:8000/v1). Ignored for openai/anthropic.",
    )
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), type=Path)
    ap.add_argument("--probe-questions", default=str(DEFAULT_PROBE_QUESTIONS), type=Path)
    ap.add_argument("--out-dir", default=None, type=Path, help="default: ../output/api/<provider>__<model_slug>/")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=True,
        help="Anthropic: extended thinking. OpenAI: routes to the Responses API with "
             "reasoning effort/summary requested -- --model must be a reasoning-capable "
             "model (the gpt-5.x family: gpt-5.6, gpt-5.6-sol, gpt-5.6-terra, "
             "gpt-5.6-luna, gpt-5.5, gpt-5.4, ...); a non-reasoning model like gpt-4o "
             "will error against the Responses API's reasoning param. ON by default.",
    )
    ap.add_argument(
        "--reasoning-effort", default="medium",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="OpenAI only (ignored for Anthropic). How much the model reasons -- "
             "which values a given model supports varies; 'medium' matches gpt-5.5's "
             "own default.",
    )
    ap.add_argument(
        "--no-thinking", dest="enable_thinking", action="store_false",
        help="Disable thinking/reasoning-trace capture (plain Chat Completions for OpenAI, "
             "no `thinking` param for Anthropic). Use this for non-reasoning OpenAI models "
             "like gpt-4o.",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N transcripts (smoke test)")
    ap.add_argument(
        "--max-parallel", type=int, default=1,
        help="run this many transcripts concurrently (thread pool; each transcript's own "
             "probe questions still run sequentially after its main generation). Safe to "
             "raise for self-hosted vLLM (continuous batching) or API providers with "
             "sufficient rate limits. Default 1 (sequential).",
    )
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="DEBUG also logs every API request/response (model, tokens, latency, previews) from api_clients.py",
    )
    ap.add_argument("--log-file", type=Path, default=None, help="also write logs to this file, in addition to stdout")
    args = ap.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    try:
        from dotenv import load_dotenv

        load_dotenv(HERE.parent / ".env")
    except ImportError:
        pass

    out_dir = args.out_dir or (HERE.parent / "output" / "api" / f"{args.provider}__{slugify(args.model)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    probe_questions = yaml.safe_load(args.probe_questions.read_text(encoding="utf-8"))

    provider = PROVIDERS[args.provider]
    if args.provider == "vllm":
        if not args.base_url:
            raise SystemExit("--base-url is required for --provider vllm (e.g. http://<vm-ip>:8000/v1)")
        client = vllm_client(args.base_url)
    else:
        client = provider["client_factory"]()
    generate = provider["generate"]

    log_path = out_dir / "reasoning_traces.json"
    csv_path = out_dir / "verbalized_responses.csv"
    json_path = out_dir / "results" / "verbalized_auroc.json"

    entries: list[dict] = []
    csv_rows: list[dict] = []
    write_lock = threading.Lock()

    def process_row(row: dict) -> None:
        t0 = time.time()
        messages = [{"role": "user", "content": row["text"]}]
        gen = generate(
            client, args.model, messages,
            max_tokens=args.max_tokens, temperature=args.temperature,
            enable_thinking=args.enable_thinking, reasoning_effort=args.reasoning_effort,
        )

        entry = {
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "category": row["category"],
            "ground_truth_label": row["label"],
            "prompt_text": row["text"],
            "generated_text": gen.final_text,
            "reasoning_trace": gen.reasoning_trace,
            "final_answer": gen.final_text,
            "n_generated_tokens": gen.n_output_tokens,
            "generation_config": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "enable_thinking": args.enable_thinking,
            },
            "model": args.model,
            "provider": args.provider,
            "elapsed_s": round(time.time() - t0, 2),
        }
        with write_lock:
            entries.append(entry)
            # Rewrite the full array after each transcript (cheap at this
            # dataset size) so a partial run still leaves valid JSON behind
            # if interrupted, rather than only the final entry being lost.
            log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("[%s] generated (%s tokens, %.1fs)", row["id"], gen.n_output_tokens, entry["elapsed_s"])

        # --- verbalized probing: one follow-up turn per probe question ---
        for pq in probe_questions:
            followup_messages = [
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": gen.final_text},
                {"role": "user", "content": pq["template_text"]},
            ]
            pq_gen = generate(
                client, args.model, followup_messages,
                max_tokens=512, temperature=args.temperature, enable_thinking=False,
            )
            eval_signal, parse_status = resolve_answer(
                pq_gen.final_text, pq["answer"]["real"], pq["answer"]["eval"]
            )
            csv_row = {
                "transcript_id": row["id"],
                "prompt_id": row["prompt_id"],
                "probe_id": pq["id"],
                "source": row["source"],
                "category": row["category"],
                "ground_truth_label": row["label"],
                "eval_signal": eval_signal,
                "parse_status": parse_status,
                "raw_response": pq_gen.final_text,
            }
            with write_lock:
                writer.writerow(csv_row)
                csv_f.flush()
                csv_rows.append(csv_row)
            LOG.info(
                "  [%s / %s] eval_signal=%.1f (%s)", row["id"], pq["id"], eval_signal, parse_status
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "transcript_id", "prompt_id", "probe_id", "source", "category",
                "ground_truth_label", "eval_signal", "parse_status", "raw_response",
            ],
        )
        writer.writeheader()

        if args.max_parallel <= 1:
            for row in dataset:
                process_row(row)
        else:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
                futures = {pool.submit(process_row, row): row for row in dataset}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        future.result()
                    except Exception:
                        LOG.exception("[%s] failed", row["id"])
                        raise

    # --- per-probe + ensemble AUROC, same convention as verbalize_probe.py ---
    probe_ids = sorted({r["probe_id"] for r in csv_rows})
    per_probe = {}
    for pid in probe_ids:
        sub = [r for r in csv_rows if r["probe_id"] == pid and r["eval_signal"] != PARSE_FAILURE_VALUE]
        labels = np.array([r["ground_truth_label"] for r in sub])
        signals = np.array([r["eval_signal"] for r in sub])
        n_fail = sum(1 for r in csv_rows if r["probe_id"] == pid and r["eval_signal"] == PARSE_FAILURE_VALUE)
        auroc = float(roc_auc_score(labels, signals)) if len(np.unique(labels)) > 1 else None
        acc = float((signals.round() == labels).mean()) if len(sub) else None
        per_probe[pid] = {"auroc": auroc, "accuracy": acc, "n_scored": len(sub), "n_parse_failures": n_fail}

    by_transcript: dict[str, list[float]] = {}
    gt_by_transcript: dict[str, int] = {}
    for r in csv_rows:
        if r["eval_signal"] == PARSE_FAILURE_VALUE:
            continue
        by_transcript.setdefault(r["transcript_id"], []).append(r["eval_signal"])
        gt_by_transcript[r["transcript_id"]] = r["ground_truth_label"]
    ensemble_labels = np.array([gt_by_transcript[tid] for tid in by_transcript])
    ensemble_scores = np.array([np.mean(v) for v in by_transcript.values()])
    ensemble_auroc = (
        float(roc_auc_score(ensemble_labels, ensemble_scores)) if len(np.unique(ensemble_labels)) > 1 else None
    )

    result = {
        "summary": {
            "provider": args.provider,
            "model": args.model,
            "n_transcripts": len(dataset),
            "n_probes": len(probe_ids),
            "ensemble_auroc": ensemble_auroc,
        },
        "per_probe": per_probe,
    }
    json_path.write_text(json.dumps(result, indent=2))
    LOG.info("Wrote %s, %s, %s", log_path, csv_path, json_path)
    LOG.info("Ensemble AUROC: %s", ensemble_auroc)


if __name__ == "__main__":
    main()
