import argparse
import re
from pathlib import Path

import numpy as np
import torch

from evalaware.datasets import load_traces
from evalaware.models import ModelSpec, load_model, prompt_only
from evalaware.parsing import split_reasoning
from .prefixes import cot_prefix, load_sentences, sentence_cuts

# High enough that the model stops on its own token rather than being cut off.
# Not unlimited: degenerate repetition loops are a real failure mode in this
# project, and an unbounded generate would hang on one.
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
    logits = model(full.to(model.device), output_hidden_states=False).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    target = full[0, 1:].to(logp.device)
    chosen = logp.gather(-1, target[:, None])[:, 0][-k:]
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
        # Ran out of room before closing the reasoning channel. The text is
        # still thinking, so keep it rather than discarding the whole rollout.
        reasoning = HEADER_RE.sub("", generated, count=1).strip()
    return prompt, generated, reasoning


def echo_end(reasoning, prompt_text):
    """Where the model's restatement of the question stops.

    Glimmer copies the whole conversation into the top of its reasoning before
    it starts thinking. Boundaries inside that copy are positions in the
    question, not in any thought, so scoring them measures nothing.
    """
    if not prompt_text:
        return 0
    found = reasoning.find(prompt_text)
    return found + len(prompt_text) if found != -1 else 0


def score_trace(model, tok, prompt, generated, reasoning, candidates, n_positions, skip=0):
    # ponytail: one forward per (position, candidate). Cache the prefix KV once
    # per position and reuse it across candidates if this is ever too slow.
    cuts = [c for c in sentence_cuts(reasoning, n_positions) if c > skip] or [len(reasoning)]
    scores = np.full((len(cuts), len(candidates), 2), np.nan, dtype=np.float32)
    for i, cut in enumerate(cuts):
        prefix = cot_prefix(prompt, generated, reasoning, cut)
        for j, candidate in enumerate(candidates):
            scores[i, j] = force_logprob(model, tok, prefix, " " + candidate)
    return scores


def run(model, tok, spec, traces, candidates, n_positions, n_rollouts, gen_cfg):
    cells, sentence_counts, kept_total = {}, [], 0
    for t, trace in enumerate(traces):
        kept = 0
        for r in range(n_rollouts):
            prompt, generated, reasoning = rollout(model, tok, spec, trace["prompt_text"], gen_cfg)
            if not reasoning or reasoning not in generated:
                continue
            block = score_trace(model, tok, prompt, generated, reasoning, candidates,
                                n_positions, skip=echo_end(reasoning, trace["prompt_text"]))
            cells[(t, r)] = block
            sentence_counts.append(block.shape[0])
            kept += 1
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

    scores = run(model, tok, spec, traces, candidates, args.positions, args.rollouts, gen_cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
