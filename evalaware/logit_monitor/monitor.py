import argparse
from pathlib import Path

import numpy as np
import torch

from evalaware.datasets import load_traces
from evalaware.models import ModelSpec, load_model, prompt_only
from evalaware.parsing import split_reasoning
from .prefixes import cot_prefix, load_sentences, sentence_cuts


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
        max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
        do_sample=True,
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    generated = tok.decode(gen_ids, skip_special_tokens=True)
    reasoning, _ = split_reasoning(generated, gen_ids=gen_ids, tok=tok)
    return prompt, generated, reasoning


def score_trace(model, tok, prompt, generated, reasoning, candidates, n_positions):
    # ponytail: one forward per (position, candidate). Cache the prefix KV once
    # per position and reuse it across candidates if this is ever too slow.
    scores = np.full((n_positions, len(candidates), 2), np.nan, dtype=np.float32)
    for i, cut in enumerate(sentence_cuts(reasoning, n_positions)):
        prefix = cot_prefix(prompt, generated, reasoning, cut)
        for j, candidate in enumerate(candidates):
            scores[i, j] = force_logprob(model, tok, prefix, " " + candidate)
    return scores


def run(model, tok, spec, traces, candidates, n_positions, n_rollouts, gen_cfg):
    shape = (len(traces), n_rollouts, n_positions, len(candidates), 2)
    out = np.full(shape, np.nan, dtype=np.float32)
    for t, trace in enumerate(traces):
        kept = 0
        for r in range(n_rollouts):
            prompt, generated, reasoning = rollout(model, tok, spec, trace["prompt_text"], gen_cfg)
            if not reasoning or reasoning not in generated:
                continue
            out[t, r] = score_trace(model, tok, prompt, generated, reasoning, candidates, n_positions)
            kept += 1
        print(f"  {trace['id']}: {kept}/{n_rollouts} rollouts scored", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sentences", type=Path)
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--positions", type=int, default=16)
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
