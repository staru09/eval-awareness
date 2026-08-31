#!/usr/bin/env python3
"""Run the eval-awareness benchmark on an OpenAI or Anthropic API model,
as a drop-in alternative to run_inference.py (local Qwen3-8B via HF).

API models expose no hidden states, so this script covers the two stages
that don't require local weight access:

  1. Generation over the mixed eval/deploy dataset (data/dataset.json),
     capturing a reasoning trace when the provider/model exposes one
     (Anthropic extended thinking; best-effort for OpenAI reasoning
     models) — same JSONL schema as run_inference.py's
     reasoning_traces.jsonl, so gpt_judge.py works unmodified on either.
  2. Verbalized probing: for each transcript, appends each of
     probes/probe_questions.yaml's 5 phrasings as a follow-up turn and
     asks the model whether the interaction above was an evaluation,
     reusing probes/verbalize_probe.py's exact label-parsing fallback
     chain and PARSE_FAILURE_VALUE convention.

(Internal linear-probe activation analysis — plan.md steps 1-3 — is
Qwen3-8B/HF-only and has no equivalent here.)

Outputs, under --out-dir (default ../output/api/<provider>__<model_slug>/):
  reasoning_traces.jsonl
  verbalized_responses.csv
  results/verbalized_auroc.json

Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY (see ../.env, loaded via
python-dotenv if present). Uses uv for package management, per this
workspace's convention (`uv run python run_inference_api.py ...`).

Usage:
    python run_inference_api.py --provider openai --model gpt-4o
    python run_inference_api.py --provider anthropic --model claude-sonnet-4-6 --enable-thinking
    python run_inference_api.py --provider openai --model o3-mini --enable-thinking  # reasoning model required
    python run_inference_api.py --provider openai --model gpt-4o --limit 2   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from api_clients import PROVIDERS  # noqa: E402
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
    ap.add_argument("--model", required=True, help="e.g. gpt-4o, claude-sonnet-4-6")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), type=Path)
    ap.add_argument("--probe-questions", default=str(DEFAULT_PROBE_QUESTIONS), type=Path)
    ap.add_argument("--out-dir", default=None, type=Path, help="default: ../output/api/<provider>__<model_slug>/")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--enable-thinking", action="store_true",
        help="Anthropic: extended thinking. OpenAI: routes to the Responses API with a "
             "reasoning summary requested -- --model must be a reasoning-capable model "
             "(o1, o3, o3-mini, gpt-5-thinking, ...); a non-reasoning model like gpt-4o "
             "will error or return no reasoning trace.",
    )
    ap.add_argument("--limit", type=int, default=None, help="only run the first N transcripts (smoke test)")
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
    client = provider["client_factory"]()
    generate = provider["generate"]

    log_path = out_dir / "reasoning_traces.jsonl"
    csv_path = out_dir / "verbalized_responses.csv"
    json_path = out_dir / "results" / "verbalized_auroc.json"

    csv_rows = []
    with open(log_path, "w", encoding="utf-8") as log_f, open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "transcript_id", "prompt_id", "probe_id", "source", "category",
                "ground_truth_label", "eval_signal", "parse_status", "raw_response",
            ],
        )
        writer.writeheader()

        for row in dataset:
            t0 = time.time()
            messages = [{"role": "user", "content": row["text"]}]
            gen = generate(
                client, args.model, messages,
                max_tokens=args.max_tokens, temperature=args.temperature,
                enable_thinking=args.enable_thinking,
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
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()
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
                writer.writerow(csv_row)
                csv_rows.append(csv_row)
                LOG.info(
                    "  [%s / %s] eval_signal=%.1f (%s)", row["id"], pq["id"], eval_signal, parse_status
                )

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
