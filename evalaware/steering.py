from pathlib import Path

import numpy as np
import torch

from .models import ModelSpec, followup_prompt
from .parsing import PARSE_FAILURE, resolve_answer


class Steerer:
    def __init__(self, model, decoder_idx: int):
        layers = model.model.layers
        if not 0 <= decoder_idx < len(layers):
            raise SystemExit(f"layer {decoder_idx} out of range 0..{len(layers) - 1}")
        self.module = layers[decoder_idx]
        self.handle = None
        self.vec = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + self.vec.to(output[0].dtype).to(output[0].device),) + output[1:]
        return output + self.vec.to(output.dtype).to(output.device)

    def apply(self, vec: torch.Tensor) -> None:
        self.remove()
        self.vec = vec
        self.handle = self.module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def mean_activation_norms(act_dir: Path, ids: list[str], n_layers: int) -> np.ndarray:
    totals, seen = np.zeros(n_layers), 0
    for tid in ids:
        path = act_dir / f"{tid}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            totals += np.linalg.norm(z["mean_token"], axis=1)
        seen += 1
    if not seen:
        raise SystemExit(f"no activations under {act_dir}")
    return totals / seen


@torch.no_grad()
def _answer(model, tok, spec, trace, question, max_new_tokens, temperature, top_p) -> str:
    prompt = followup_prompt(tok, spec, trace["prompt_text"], trace["final_answer"],
                             question["template_text"])
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _score_batch(model, tok, spec, traces, question, ids, gen_kwargs) -> dict:
    per_transcript = {}
    for tid in ids:
        raw = _answer(model, tok, spec, traces[tid], question, **gen_kwargs)
        signal, status = resolve_answer(raw, question["answer"]["real"], question["answer"]["eval"])
        per_transcript[tid] = {
            "eval_signal": signal,
            "parse_status": status,
            "ground_truth_label": traces[tid]["ground_truth_label"],
            "raw_response": raw,
        }
    scored = [v["eval_signal"] for v in per_transcript.values() if v["eval_signal"] != PARSE_FAILURE]
    return {
        "per_transcript": per_transcript,
        "mean_eval_signal": float(np.mean(scored)) if scored else None,
        "n_scored": len(scored),
    }


def run(model, tok, spec: ModelSpec, probe: dict, traces: list[dict], question: dict,
        transcript_ids: list[str], layers: list[int], alphas: list[float], act_dir: Path,
        max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> dict:
    by_id = {t["id"]: t for t in traces}
    n_layers = int(probe["metadata"]["num_layers"]) - 1
    norms = mean_activation_norms(act_dir, [t["id"] for t in traces], n_layers)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}

    baseline = _score_batch(model, tok, spec, by_id, question, transcript_ids, gen_kwargs)
    cells = []
    for layer in layers:
        if not 1 <= layer <= n_layers:
            raise SystemExit(f"--layers value {layer} out of range 1..{n_layers}")
        direction = probe["real"][layer] * float(norms[layer - 1])
        steerer = Steerer(model, layer - 1)
        for alpha in alphas:
            steerer.apply(alpha * direction)
            try:
                result = _score_batch(model, tok, spec, by_id, question, transcript_ids, gen_kwargs)
            finally:
                steerer.remove()
            delta = None
            if result["mean_eval_signal"] is not None and baseline["mean_eval_signal"] is not None:
                delta = result["mean_eval_signal"] - baseline["mean_eval_signal"]
            cells.append({"layer": layer, "alpha": alpha, "delta_vs_baseline": delta, **result})

    return {
        "model_key": spec.hf_id,
        "probe_id": question["id"],
        "transcript_ids": transcript_ids,
        "layers": layers,
        "alphas": alphas,
        "baseline": baseline,
        "cells": cells,
    }
