import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from evalaware.datasets import load_traces
from evalaware.models import ModelSpec, load_model, prompt_only
from evalaware.parsing import split_reasoning
from .prefixes import cot_prefix, load_sentences, sentence_cuts

MAX_NEW_TOKENS = 16384
HEADER_RE = re.compile(r"^\s*to=\w+", re.IGNORECASE)


@torch.no_grad()
def force_logprob(model, tok, prefix, candidate):
    """Log-probability the model assigns to `candidate` continuing `prefix`.

    Tokenised as one string: tokenising the candidate alone would use different
    leading-space tokens. Returns (sum, mean) so length normalisation is a
    choice made at analysis time rather than baked in here.
    """
    pre = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
    full = tok(prefix + candidate, return_tensors="pt", add_special_tokens=False).input_ids
    k = full.shape[1] - pre.shape[1]
    if k <= 0:
        return float("nan"), float("nan")
    logits = model(full.to(model.device)).logits
    logp = torch.log_softmax(logits[0, -k - 1:-1].float(), dim=-1)
    chosen = logp.gather(-1, full[0, -k:, None].to(logp.device))[:, 0]
    return float(chosen.sum()), float(chosen.mean())


@torch.no_grad()
def rollout(model, tok, spec, text, gen_cfg):
    prompt = prompt_only(tok, text, spec)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", MAX_NEW_TOKENS),
        do_sample=True,
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    generated = tok.decode(gen_ids, skip_special_tokens=True)
    reasoning, _ = split_reasoning(generated, gen_ids=gen_ids, tok=tok)
    if reasoning is None and gen_ids.shape[0] >= gen_cfg.get("max_new_tokens", MAX_NEW_TOKENS):
        reasoning = HEADER_RE.sub("", generated, count=1).strip()
    return prompt, generated, reasoning


def echo_end(reasoning, prompt_text):
    """Where the model's restatement of the question stops.

    Glimmer copies the whole conversation into the top of its reasoning before
    it starts thinking. Boundaries inside that copy are positions in the
    question, not in any thought, so scoring them measures nothing.

    Matched line by line, not as one string: the copy often drops or truncates
    turns, and a whole-prompt match then fails and skips nothing. That put the
    top score of every deploy rollout in the smoke run inside the copy.
    """
    end = 0
    for line in re.finditer(r"[^\n]+", reasoning):
        if line.group().strip() not in prompt_text:
            break
        end = line.end()
    return end


def score_trace(model, tok, prompt, generated, reasoning, candidates, n_positions, skip=0):
    cuts = [c for c in sentence_cuts(reasoning, n_positions) if c > skip] or [len(reasoning)]
    scores = np.full((len(cuts), len(candidates), 2), np.nan, dtype=np.float32)
    for i, cut in enumerate(cuts):
        prefix = cot_prefix(prompt, generated, reasoning, cut)
        for j, candidate in enumerate(candidates):
            scores[i, j] = force_logprob(model, tok, prefix, " " + candidate)
    return cuts, scores


def _append(dump, record):
    """One JSON line per rollout, flushed immediately.

    Written as it goes rather than at the end: the npz only lands once every
    rollout is done, so a crash at the last one used to lose the whole run, and
    the generated text was never kept at all.
    """
    with dump.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(model, tok, spec, traces, candidates, n_positions, n_rollouts, gen_cfg, dump):
    cells, sentence_counts, kept_total = {}, [], 0
    for t, trace in enumerate(traces):
        kept = 0
        for r in range(n_rollouts):
            prompt, generated, reasoning = rollout(model, tok, spec, trace["prompt_text"], gen_cfg)
            record = {"id": trace["id"], "prompt_id": trace["prompt_id"],
                      "label": trace["ground_truth_label"], "rollout": r,
                      "prompt": prompt, "generated_text": generated,
                      "reasoning_trace": reasoning}
            if not reasoning or reasoning not in generated:
                record["dropped"] = "no chain of thought found in generation"
                _append(dump, record)
                continue
            skip = echo_end(reasoning, trace["prompt_text"])
            cuts, block = score_trace(model, tok, prompt, generated, reasoning, candidates,
                                      n_positions, skip=skip)
            cells[(t, r)] = block
            sentence_counts.append(block.shape[0])
            kept += 1
            record.update(echo_skip=skip, cuts=cuts, scores=block.tolist())
            _append(dump, record)
        kept_total += kept
        print(f"  {trace['id']}: {kept}/{n_rollouts} rollouts scored", flush=True)

    if not cells:
        raise SystemExit("no rollout produced a chain of thought; check split_reasoning for this model")
    if kept_total < len(traces) * n_rollouts / 2:
        raise SystemExit(f"only {kept_total} of {len(traces) * n_rollouts} rollouts had a chain of thought")

    width = max(b.shape[0] for b in cells.values())
    out = np.full((len(traces), n_rollouts, width, len(candidates), 2), np.nan, dtype=np.float32)
    for (t, r), block in cells.items():
        out[t, r, :block.shape[0]] = block
    counts = sorted(sentence_counts)
    print(f"sentence boundaries per rollout: min {counts[0]} median "
          f"{counts[len(counts) // 2]} max {counts[-1]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sentences", type=Path)
    ap.add_argument("--dump", type=Path,
                    help="per-rollout JSONL of generations and scores; defaults next to --out")
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--positions", type=int, default=0,
                    help="0 scores every sentence boundary; a number subsamples to a fixed grid")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    eval_aware, neutral = load_sentences(args.sentences)
    candidates = eval_aware + neutral
    traces = load_traces(args.traces)
    gen_cfg = traces[0].get("generation_config") or {}
    model, tok = load_model(args.model_path, args.dtype)
    spec = ModelSpec(hf_id=args.model_path, chat_template=True, enable_thinking=not args.no_thinking)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump = args.dump or args.out.with_suffix(".rollouts.jsonl")
    dump.unlink(missing_ok=True)
    scores = run(model, tok, spec, traces, candidates, args.positions, args.rollouts,
                 gen_cfg, dump=dump)

    np.savez(
        args.out,
        scores=scores,
        labels=np.array([int(t["ground_truth_label"]) for t in traces], dtype=np.int32),
        prompt_ids=np.array([int(t["prompt_id"]) for t in traces], dtype=np.int64),
        transcript_ids=np.array([t["id"] for t in traces]),
        candidates=np.array(candidates),
        n_eval_aware=len(eval_aware),
        model_key=args.model_path,
    )
    print(f"wrote {scores.shape} -> {args.out}")
    print(f"wrote per-rollout generations -> {dump}")


if __name__ == "__main__":
    main()
