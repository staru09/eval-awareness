import json
from pathlib import Path

import numpy as np
import torch

from .models import ModelSpec, followup_prompt, layer_dims, prompt_only
from .parsing import split_reasoning


def pool(hidden_states, attn_mask) -> tuple[np.ndarray, np.ndarray]:
    mask = attn_mask[0].bool()
    last_idx = int(mask.nonzero()[-1].item())
    last, mean = [], []
    for layer in hidden_states[1:]:
        h = layer[0]
        last.append(h[last_idx].float().cpu().numpy())
        mean.append(h[mask].float().mean(dim=0).cpu().numpy())
    return np.stack(last), np.stack(mean)


def _save(path: Path, last, mean, row: dict, n_layers: int, d_model: int, extra: dict) -> None:
    np.savez(
        path,
        last_token=last.astype(np.float32),
        mean_token=mean.astype(np.float32),
        label=row["label"],
        prompt_id=row["prompt_id"],
        source=row["source"],
        n_layers=n_layers,
        d_model=d_model,
        **extra,
    )


@torch.no_grad()
def _forward(model, tok, text: str):
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**inputs, output_hidden_states=True)
    return inputs, out.hidden_states


def generate_and_capture(model, tok, spec: ModelSpec, rows: list[dict], out_dir: Path,
                         max_new_tokens: int = 4096, temperature: float = 0.7,
                         top_p: float = 0.9, capture: bool = True) -> list[dict]:
    n_layers, d_model = layer_dims(model)
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    if capture:
        act_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for row in rows:
        prompt = prompt_only(tok, row["text"], spec)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = gen[0, inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        reasoning, answer = split_reasoning(text, gen_ids=gen_ids, tok=tok)

        entries.append({
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "category": row["category"],
            "ground_truth_label": row["label"],
            "prompt_text": row["text"],
            "templated_prompt": prompt,
            "generated_text": text,
            "reasoning_trace": reasoning,
            "final_answer": answer,
            "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
            "n_generated_tokens": int(gen_ids.shape[0]),
            "model": spec.hf_id,
        })
        (out_dir / "reasoning_traces.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

        if capture:
            full = gen
            attn = torch.ones_like(full)
            with torch.no_grad():
                hidden = model(full, attention_mask=attn, output_hidden_states=True).hidden_states
            last, mean = pool(hidden, attn)
            _save(act_dir / f"{row['id']}.npz", last, mean, row, n_layers, d_model, {})
    return entries


def capture_prompt_only(model, tok, traces: list[dict], out_dir: Path, model_path: str) -> int:
    trace_model = next((t.get("model") for t in traces if t.get("model")), None)
    if trace_model and trace_model != model_path:
        raise SystemExit(f"traces came from {trace_model!r}, not {model_path!r}")

    n_layers, d_model = layer_dims(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in traces:
        prompt = t.get("templated_prompt")
        if not prompt:
            raise SystemExit(f"{t['id']} has no templated_prompt")
        inputs, hidden = _forward(model, tok, prompt)
        n_tok = int(inputs["input_ids"].shape[1])
        if t.get("n_prompt_tokens") not in (None, n_tok):
            raise SystemExit(f"{t['id']}: retokenised to {n_tok}, run recorded {t['n_prompt_tokens']}")
        last, mean = pool(hidden, inputs["attention_mask"])
        row = {"label": t["ground_truth_label"], "prompt_id": t["prompt_id"], "source": t["source"]}
        _save(out_dir / f"{t['id']}.npz", last, mean, row, n_layers, d_model,
              {"n_prompt_tokens": n_tok, "readout": "prompt_only"})
    return len(traces)
