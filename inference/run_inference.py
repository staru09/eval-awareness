"""Run Qwen3-8B inference over the mixed eval/deploy dataset via plain HF
transformers, capturing:

  - the full generation (including the model's <think>...</think> reasoning
    trace, when thinking mode emits one) as a JSONL log
    (`output/reasoning_traces.jsonl`)
  - per-layer hidden-state activations (last-token + mean-pooled over the
    full prompt+response), one .npz per transcript, under
    `output/activations/`
  - a one-time model/weights summary (`output/model_summary.json`)

Single HF code path for generation and hidden-state extraction, matching
../../evaluation-awareness/cls_corr_mi's conventions (device_map="auto",
left-padding, chat-template + enable_thinking passthrough) so the resulting
activations stay directly comparable with that pipeline. No inspect_ai
dependency (see plan.md step 1-2 / 4).

Usage:
    python run_inference.py --dataset ../data/dataset.json --out-dir ../output
    python run_inference.py --limit 2 --no-hidden-states   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from model_utils import ModelSpec, QWEN3_8B, apply_chat_template, load_hf_model, model_weight_summary

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_reasoning(generated_text: str) -> tuple[str | None, str]:
    """Split a thinking-mode response into (reasoning_trace, final_answer).

    Returns (None, generated_text) when no <think>...</think> block is
    present (e.g. thinking disabled, or the model didn't emit one).
    """
    m = THINK_RE.search(generated_text)
    if not m:
        return None, generated_text.strip()
    reasoning = m.group(1).strip()
    final_answer = generated_text[m.end():].strip()
    return reasoning, final_answer


def pool_hidden_states(hidden_states, attn_mask):
    """Reduce a forward pass's hidden states to per-layer summary vectors.

    hidden_states: tuple of (n_layers + 1) tensors, each [1, seq_len, d_model]
    (index 0 = embedding output). attn_mask: [1, seq_len].

    Returns (last_token, mean_token), each float32 numpy [n_layers, d_model],
    skipping the embedding layer so layer indices match cls_corr_mi's 1..N.
    """
    mask = attn_mask[0].bool()
    last_idx = int(mask.nonzero()[-1].item())
    last_list, mean_list = [], []
    for layer_h in hidden_states[1:]:
        h = layer_h[0]  # [seq_len, d_model]
        last_list.append(h[last_idx].float().cpu().numpy())
        mean_list.append(h[mask].float().mean(dim=0).cpu().numpy())
    return np.stack(last_list), np.stack(mean_list)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../data/dataset.json")
    ap.add_argument("--out-dir", default="../output")
    ap.add_argument("--model-path", default=QWEN3_8B.hf_id)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N transcripts (smoke test)")
    ap.add_argument("--no-hidden-states", action="store_true", help="skip activation capture (generation-only, faster)")
    ap.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    spec = ModelSpec(hf_id=args.model_path, chat_template=True, enable_thinking=args.enable_thinking)
    print(f"Loading {args.model_path} (dtype={args.dtype}) ...")
    model, tok = load_hf_model(args.model_path, dtype=args.dtype)

    # One-time weight/config summary (not per-sample): architecture + a
    # cheap per-parameter L2-norm/shape table, so "inspect the weights" is
    # documented without duplicating multi-GB checkpoint data on disk.
    summary_path = out_dir / "model_summary.json"
    if not summary_path.exists():
        print("Writing model_summary.json ...")
        summary_path.write_text(json.dumps(model_weight_summary(model), indent=2), encoding="utf-8")

    log_path = out_dir / "reasoning_traces.jsonl"
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    with open(log_path, "w", encoding="utf-8") as log_f:
        for row in dataset:
            t0 = time.time()
            prompt_text = apply_chat_template(tok, row["text"], spec)
            inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            gen_ids = gen_out[0, inputs["input_ids"].shape[1]:]
            generated_text = tok.decode(gen_ids, skip_special_tokens=True)
            reasoning_trace, final_answer = split_reasoning(generated_text)

            entry = {
                "id": row["id"],
                "prompt_id": row["prompt_id"],
                "source": row["source"],
                "category": row["category"],
                "ground_truth_label": row["label"],
                "prompt_text": row["text"],
                "templated_prompt": prompt_text,
                "generated_text": generated_text,
                "reasoning_trace": reasoning_trace,
                "final_answer": final_answer,
                "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
                "n_generated_tokens": int(gen_ids.shape[0]),
                "generation_config": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "enable_thinking": args.enable_thinking,
                },
                "model": args.model_path,
                "elapsed_s": round(time.time() - t0, 2),
            }
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()

            if not args.no_hidden_states:
                full_ids = gen_out  # [1, prompt_len + gen_len]
                attn_mask = torch.ones_like(full_ids)
                with torch.no_grad():
                    fwd_out = model(full_ids, attention_mask=attn_mask, output_hidden_states=True)
                last_h, mean_h = pool_hidden_states(fwd_out.hidden_states, attn_mask)
                np.savez(
                    act_dir / f"{row['id']}.npz",
                    last_token=last_h.astype(np.float32),  # [n_layers, d_model]
                    mean_token=mean_h.astype(np.float32),  # [n_layers, d_model]
                    label=row["label"],
                    prompt_id=row["prompt_id"],
                    source=row["source"],
                    n_layers=n_layers,
                    d_model=d_model,
                )

            print(f"[{row['id']}] {entry['n_generated_tokens']} tokens in {entry['elapsed_s']}s")

    print(f"Wrote {len(dataset)} entries to {log_path}")
    if not args.no_hidden_states:
        print(f"Wrote per-transcript activations to {act_dir}/")


if __name__ == "__main__":
    main()
