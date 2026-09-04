import csv
from pathlib import Path

import torch

from .models import ModelSpec, followup_prompt
from .parsing import PARSE_FAILURE, resolve_answer

FIELDS = [
    "transcript_id", "prompt_id", "probe_id", "source", "category",
    "ground_truth_label", "eval_signal", "parse_status", "raw_response",
]


@torch.no_grad()
def _generate_batch(model, tok, prompts: list[str], max_new_tokens: int,
                    temperature: float, top_p: float) -> list[str]:
    inputs = tok(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    start = inputs["input_ids"].shape[1]
    return [tok.decode(out[i, start:], skip_special_tokens=True) for i in range(len(prompts))]


def run(model, tok, spec: ModelSpec, traces: list[dict], questions: list[dict], out_csv: Path,
        max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> list[dict]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for t in traces:
            prompts = [
                followup_prompt(tok, spec, t["prompt_text"], t["final_answer"], q["template_text"])
                for q in questions
            ]
            responses = _generate_batch(model, tok, prompts, max_new_tokens, temperature, top_p)
            for q, response in zip(questions, responses):
                signal, status = resolve_answer(response, q["answer"]["real"], q["answer"]["eval"])
                row = {
                    "transcript_id": t["id"],
                    "prompt_id": t["prompt_id"],
                    "probe_id": q["id"],
                    "source": t["source"],
                    "category": t["category"],
                    "ground_truth_label": t["ground_truth_label"],
                    "eval_signal": signal,
                    "parse_status": status,
                    "raw_response": response,
                }
                writer.writerow(row)
                rows.append(row)
    return rows


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["eval_signal"] = float(r["eval_signal"])
        r["ground_truth_label"] = int(r["ground_truth_label"])
        r["prompt_id"] = int(r["prompt_id"])
    return rows


def signals_by_probe(rows: list[dict]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        if r["eval_signal"] == PARSE_FAILURE:
            continue
        out.setdefault(r["probe_id"], {})[r["prompt_id"]] = r["eval_signal"]
    return out


def ensemble_signal(by_probe: dict[str, dict[int, float]]) -> dict[int, float]:
    ids = sorted({pid for probe in by_probe.values() for pid in probe})
    out = {}
    for pid in ids:
        values = [probe[pid] for probe in by_probe.values() if pid in probe]
        if values:
            out[pid] = sum(values) / len(values)
    return out
