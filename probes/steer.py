#!/usr/bin/env python3
"""Experiment 3 (plan2.md): activation steering, the causal test.

Contrastive Activation Addition (paper's Sec. 2/4, arXiv:2608.21766): at a
chosen layer L, add alpha * c_L to the residual stream at every token
position while the model answers a verbalized eval-awareness probe
question, where

    c_L = mean_activation_norm(L) * v_hat_L

v_hat_L is the unit probe direction from compute_probe.py (Experiment 1),
and mean_activation_norm(L) is the average ||mean-pooled activation||_2 at
layer L across the model's own 20 dataset transcripts (its own
run_inference.py activations/*.npz), matching the paper's calibration
convention of scaling the direction to the model's own typical activation
magnitude at that layer.

Steering is applied via a forward hook on the decoder block, so it affects
every token position during both the prefill of the (transcript + its own
answer + follow-up probe question) context and every subsequently
generated token -- one of the two valid modes the paper describes (the
other being last-token-only during generation).

Sign convention (carried over from Experiment 1, from how the direction is
constructed): v = mean(deploy-framed activations) - mean(eval-framed
activations), so *in theory* higher probe-direction score = more
deployment-like, and negative alpha (subtracting the direction) steers
toward eval-like. In practice, per a real run against Qwen3-8B, don't
expect this to produce a clean, symmetric bidirectional effect --
empirically both signs of alpha increased mean eval_signal above baseline
at moderate magnitude, with output breaking down (unparseable) at the
extremes, especially alpha=-1.0. This is not unique to us: the paper
itself reports the same asymmetry for its own models ("positive values of
alpha produce smaller ∆J(α,ℓ) values, and do not consistently reduce the
judge score below baseline"). Report the actual per-model, per-layer
sign/magnitude pattern observed, don't assume the naive
negative-increases/positive-decreases story holds.

Simplification vs. the paper: the paper scores steered generations with a
separate 0-3 LLM-judge call. We reuse our own rule-based
verbalize_probe.resolve_answer() on the (possibly steered) model's own
answer to the probe question -- a binary eval/deploy signal, not a
separate model call. See plan2.md's Experiment 3 section for the
rationale.

Usage:
    python steer.py \
      --model-path Qwen/Qwen3-8B \
      --probe ../output/local/qwen3-8b/probe.pt \
      --reasoning-traces ../output/local/qwen3-8b/reasoning_traces.json \
      --activations-dir ../output/local/qwen3-8b/activations \
      --probe-id direct_a \
      --layers 15,19,23,27 \
      --alphas -1.0,-0.5,0.5,1.0 \
      --transcript-ids hellaswag_0,hellaswag_1,arc_challenge_0,everyday_chat_0,everyday_chat_1 \
      --out ../output/local/qwen3-8b/results/steering_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from model_utils import ModelSpec, QWEN3_8B, load_hf_model  # noqa: E402
from verbalize_probe import build_followup_prompt, resolve_answer  # noqa: E402

LOG = logging.getLogger("steer")


def mean_activation_norms(activations_dir: Path, transcript_ids: list[str], n_layers_act: int) -> np.ndarray:
    """m_l for l in activation-array index space (0..n_layers_act-1):
    average ||mean_token[l]||_2 across the given transcripts' saved
    activations/*.npz (produced by run_inference.py with hidden states
    enabled). Uses mean-token pooling for the norm estimate regardless of
    which readout (last/mean) the probe direction itself uses, matching
    the paper's own mℓ definition (built from a_mean_ℓ).
    """
    sums = np.zeros(n_layers_act, dtype=np.float64)
    n = 0
    for tid in transcript_ids:
        p = activations_dir / f"{tid}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            mean_h = z["mean_token"]  # [n_layers_act, d_model]
        sums += np.linalg.norm(mean_h, axis=1)
        n += 1
    if n == 0:
        raise SystemExit(f"No activation files found under {activations_dir} for the requested transcripts")
    return sums / n


class Steerer:
    """Registers/removes a forward hook on one decoder block that adds a
    fixed vector to its output hidden state at every token position.
    """

    def __init__(self, model, decoder_layer_idx: int):
        self.layers = model.model.layers  # standard Llama-family HF layout
        if not (0 <= decoder_layer_idx < len(self.layers)):
            raise SystemExit(f"decoder layer index {decoder_layer_idx} out of range (0..{len(self.layers) - 1})")
        self.module = self.layers[decoder_layer_idx]
        self._handle = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0] + self.vec.to(output[0].dtype).to(output[0].device)
            return (hs,) + output[1:]
        return output + self.vec.to(output.dtype).to(output.device)

    def __enter__(self):
        return self

    def apply(self, vec: torch.Tensor):
        self.remove()
        self.vec = vec
        self._handle = self.module.register_forward_hook(self._hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def generate_followup(model, tok, spec, prompt_text, assistant_reply, pq_template_text, max_new_tokens, temperature, top_p):
    followup_prompt = build_followup_prompt(tok, spec, prompt_text, assistant_reply, pq_template_text)
    inputs = tok(followup_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = gen_out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=QWEN3_8B.hf_id)
    ap.add_argument("--probe", required=True, type=Path, help="probe.pt from compute_probe.py")
    ap.add_argument("--reasoning-traces", required=True, type=Path)
    ap.add_argument("--activations-dir", required=True, type=Path)
    ap.add_argument("--probe-questions", default="probe_questions.yaml", type=Path)
    ap.add_argument("--probe-id", default="direct_a")
    ap.add_argument("--layers", required=True, help="comma-separated probe-direction layer keys (1..num_hidden_layers)")
    ap.add_argument("--alphas", default="-1.0,-0.5,0.5,1.0", help="comma-separated, nonzero (alpha=0 baseline is always run once)")
    ap.add_argument("--transcript-ids", default=None, help="comma-separated transcript ids; default: first --n-transcripts of the dataset, balanced eval/deploy")
    ap.add_argument("--n-transcripts", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    transcripts = json.loads(args.reasoning_traces.read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in transcripts}

    if args.transcript_ids:
        transcript_ids = args.transcript_ids.split(",")
    else:
        evals = [t["id"] for t in transcripts if t["ground_truth_label"] == 1]
        deploys = [t["id"] for t in transcripts if t["ground_truth_label"] == 0]
        half = args.n_transcripts // 2
        transcript_ids = evals[:half] + deploys[:args.n_transcripts - half]
    LOG.info("Steering probe transcripts: %s", transcript_ids)

    probe_questions = yaml.safe_load(args.probe_questions.read_text(encoding="utf-8"))
    pq = next((p for p in probe_questions if p["id"] == args.probe_id), None)
    if pq is None:
        raise SystemExit(f"probe id {args.probe_id!r} not found in {args.probe_questions}")

    probe = torch.load(args.probe, map_location="cpu")
    real_dirs = probe["real"]  # {probe_key(int) -> unit tensor[D]}
    # probe.pt's "real" dict has keys 0..num_layers-1 (key 0 = embeddings,
    # per compute_probe.py's enumerate over the full hidden_states tuple),
    # so the highest valid decoder-layer key is num_layers-1, not
    # num_layers -- confirmed via a real KeyError before this bound was
    # tightened.
    n_layers_probe = int(probe["metadata"]["num_layers"])
    layers = [int(x) for x in args.layers.split(",")]
    for l in layers:
        if not (1 <= l <= n_layers_probe - 1):
            raise SystemExit(f"--layers value {l} out of range 1..{n_layers_probe - 1}")
    alphas = [float(x) for x in args.alphas.split(",")]

    LOG.info("Computing reference activation norms from %s ...", args.activations_dir)
    # probe.pt's key space is 0..n_layers_probe-1 (key 0 = embeddings), but
    # run_inference.py's saved activations/*.npz only cover the n_layers_probe-1
    # decoder-layer outputs (embedding layer is dropped before saving) --
    # activation array index i corresponds to probe key i+1 (see
    # score_activations.py's identical convention). Confirmed via a real
    # crash: preallocating this array at n_layers_probe instead of
    # n_layers_probe-1 raised a numpy broadcast ValueError (37 vs 36).
    n_layers_act = n_layers_probe - 1
    m = mean_activation_norms(args.activations_dir, [t["id"] for t in transcripts], n_layers_act=n_layers_act)

    spec = ModelSpec(hf_id=args.model_path, chat_template=True, enable_thinking=False)
    LOG.info("Loading %s ...", args.model_path)
    model, tok = load_hf_model(args.model_path, dtype=args.dtype)

    def run_batch(label: str) -> dict:
        per_tx = {}
        for tid in transcript_ids:
            t = by_id[tid]
            raw = generate_followup(
                model, tok, spec, t["prompt_text"], t["final_answer"], pq["template_text"],
                args.max_new_tokens, args.temperature, args.top_p,
            )
            eval_signal, parse_status = resolve_answer(raw, pq["answer"]["real"], pq["answer"]["eval"])
            per_tx[tid] = {
                "eval_signal": eval_signal,
                "parse_status": parse_status,
                "ground_truth_label": t["ground_truth_label"],
                # Retain the raw generation. Without it, a later fix to
                # resolve_answer() cannot be applied retroactively to this
                # experiment, which is exactly what happened once already:
                # the label-parsing bug was re-derivable for every other
                # experiment from saved text, but not for this one.
                "raw_response": raw,
            }
            LOG.info("  [%s] %s: eval_signal=%.1f (%s)", label, tid, eval_signal, parse_status)
        scored = [v["eval_signal"] for v in per_tx.values() if v["eval_signal"] != -1.0]
        mean_signal = float(np.mean(scored)) if scored else None
        return {"per_transcript": per_tx, "mean_eval_signal": mean_signal, "n_scored": len(scored)}

    LOG.info("=== baseline (alpha=0, unsteered) ===")
    baseline = run_batch("baseline")

    cells = []
    for layer in layers:
        act_idx = layer - 1  # probe key -> activation-array index (score_activations.py convention)
        decoder_idx = layer - 1  # probe key -> 0-indexed decoder block (same convention: probe key = decoder-block output index + 1)
        v_hat = real_dirs[layer]
        c_l = v_hat * float(m[act_idx])
        steerer_layer = Steerer(model, decoder_layer_idx=decoder_idx)
        for alpha in alphas:
            steerer_layer.apply(alpha * c_l)
            LOG.info("=== layer=%d alpha=%+.2f ===", layer, alpha)
            try:
                result = run_batch(f"L{layer} a{alpha:+.2f}")
            finally:
                steerer_layer.remove()
            delta = None
            if result["mean_eval_signal"] is not None and baseline["mean_eval_signal"] is not None:
                delta = result["mean_eval_signal"] - baseline["mean_eval_signal"]
            cells.append({
                "layer": layer,
                "alpha": alpha,
                "mean_eval_signal": result["mean_eval_signal"],
                "delta_vs_baseline": delta,
                "n_scored": result["n_scored"],
                "per_transcript": result["per_transcript"],
            })

    out = {
        "model_key": args.model_path,
        "probe_id": args.probe_id,
        "transcript_ids": transcript_ids,
        "layers": layers,
        "alphas": alphas,
        "baseline": baseline,
        "cells": cells,
        "note": (
            "eval_signal is our binary resolve_answer() proxy for the paper's "
            "0-3 LLM-judge verbalization score. Sign convention: negative alpha "
            "steers toward eval-like (expected to raise mean_eval_signal); "
            "positive alpha steers toward deploy-like (expected to lower it)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    LOG.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
