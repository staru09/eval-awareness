"""Build the traces file capture-prompt needs, without generating anything.

capture_prompt_only only reads `templated_prompt`, `ground_truth_label`,
`prompt_id` and `source`. The first of those is a pure function of the text and
the tokenizer, so running `generate` just to produce it wastes a GPU pass and
samples text nobody reads.

This matters more on the jjpn2 data than it did before: those transcripts
already contain the assistant's reply, so there is no "prompt only" versus
"prompt plus answer" distinction left. There is one readout, over the whole
transcript.

Loads the tokenizer only, never the weights.

Usage:
    python data/make_traces.py --dataset data/dataset_jjpn2.json \
        --model-path Qwen/Qwen3-8B --out output/local/qwen3-8b-jjpn2/reasoning_traces.json
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def templated(tok, text: str, mode: str) -> str:
    """Render one transcript for the forward pass.

    raw  : the transcript exactly as written. The last token is then the last
           real token of the conversation, which is the position the probing
           literature reads (Geometry of Truth reads the closing period).
    chat : wrapped as a user turn. Keeps the model in its normal operating
           mode, but the sequence then ends in template boilerplate that is
           byte-identical across every transcript, so a last-token readout
           lands on a position carrying no class-specific content. This cost us
           0.05 AUROC the last time we did it.
    """
    if mode == "raw":
        return text
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", default="raw", choices=["raw", "chat"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))

    traces, lengths = [], []
    for row in rows:
        prompt = templated(tok, row["text"], args.template)
        n_tokens = len(tok(prompt, add_special_tokens=False).input_ids)
        lengths.append(n_tokens)
        traces.append({
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "category": row["category"],
            "ground_truth_label": row["label"],
            "prompt_text": row["text"],
            "templated_prompt": prompt,
            "n_prompt_tokens": n_tokens,
            "model": args.model_path,
            "template_mode": args.template,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    lengths.sort()
    print(
        f"wrote {len(traces)} traces ({args.template}) to {args.out}\n"
        f"  tokens min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}"
    )


if __name__ == "__main__":
    main()
