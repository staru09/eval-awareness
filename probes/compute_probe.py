#!/usr/bin/env python3
"""Compute the 16-pair evaluation-awareness probe direction for Qwen3-8B.

Adapted from ../../evaluation-awareness/cls_corr_mi/compute_probe.py, scoped
to a single model (no --model-key registry — this folder is Qwen3-8B only)
per plan.md step 1.

For each transformer layer l, takes the hidden state at the last token of
each contrastive phrase (left-padding makes -1 the last real token), then::

    v_l = mean(h_pos_l) - mean(h_neg_l)
    v_hat_l = v_l / ||v_l||_2

Also saves R random unit-norm directions per layer for the random-control
baseline used by ../analysis/classify_auroc.py.

Hidden-state indexing: layer 0 = embeddings, layer l+1 = output of decoder
layer l. Every emitted hidden-state slot is scored so layer indices match
exactly between this script and score_activations.py.

Usage:
    python compute_probe.py --out ../output/probe.pt
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from model_utils import QWEN3_8B, load_hf_model  # noqa: E402

LOG = logging.getLogger("compute_probe")

HERE = Path(__file__).resolve().parent
PAIRS_FILE = HERE.parent / "data" / "contrastive_pairs_16.txt"


@dataclass(frozen=True)
class ContrastivePair:
    positive: str
    negative: str


def _ensure_colon(text: str) -> str:
    t = text.strip()
    return t if t.endswith(":") else f"{t}:"


def load_pairs(path: Path = PAIRS_FILE) -> List[ContrastivePair]:
    pairs: List[ContrastivePair] = []
    pos: str | None = None
    neg: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pair_"):
            if pos is not None and neg is not None:
                pairs.append(ContrastivePair(_ensure_colon(pos), _ensure_colon(neg)))
            pos, neg = None, None
            continue
        if line.startswith("positive_example:"):
            pos = line.split(":", 1)[1].strip()
        elif line.startswith("negative_example:"):
            neg = line.split(":", 1)[1].strip()
    if pos is not None and neg is not None:
        pairs.append(ContrastivePair(_ensure_colon(pos), _ensure_colon(neg)))
    if not pairs:
        raise ValueError(f"No pairs loaded from {path}")
    return pairs


def _layer_seed(layer: int, base_seed: int, i: int) -> int:
    h = hashlib.sha256(f"qwen3_8b|L{layer}|seed{base_seed}|r{i}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


@torch.inference_mode()
def compute_directions(model, tok, pairs: List[ContrastivePair], batch_size: int = 8) -> Dict[int, torch.Tensor]:
    """Mean-difference direction at every hidden-state layer (last-token readout)."""
    device = next(model.parameters()).device
    pos = [p.positive for p in pairs]
    neg = [p.negative for p in pairs]

    def _last_token_hs(phrases: List[str]) -> List[torch.Tensor]:
        out_per_layer = None
        n_done = 0
        for s in range(0, len(phrases), batch_size):
            batch = phrases[s : s + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (B, S, D) per layer
            last = [h[:, -1, :].float().cpu() for h in hs]
            if out_per_layer is None:
                out_per_layer = [t.sum(dim=0) for t in last]
            else:
                for i, t in enumerate(last):
                    out_per_layer[i] += t.sum(dim=0)
            n_done += len(batch)
        return [t / n_done for t in out_per_layer]

    LOG.info("Encoding %d positive phrases...", len(pos))
    pos_means = _last_token_hs(pos)
    LOG.info("Encoding %d negative phrases...", len(neg))
    neg_means = _last_token_hs(neg)

    directions: Dict[int, torch.Tensor] = {}
    for layer_idx, (p_mean, n_mean) in enumerate(zip(pos_means, neg_means)):
        v = (p_mean - n_mean).float()
        norm = v.norm()
        if float(norm) > 0:
            v = v / norm
        directions[layer_idx] = v
    return directions


def build_random_controls(d_model: int, n_layers: int, n_random: int, base_seed: int) -> Dict[int, torch.Tensor]:
    randoms: Dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        rows = []
        for i in range(n_random):
            g = torch.Generator()
            g.manual_seed(_layer_seed(layer, base_seed, i))
            v = torch.randn(d_model, generator=g)
            v = v / v.norm()
            rows.append(v)
        randoms[layer] = torch.stack(rows, dim=0)  # (R, D)
    return randoms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=QWEN3_8B.hf_id)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    pairs = load_pairs()
    LOG.info("Loaded %d contrastive pairs", len(pairs))

    LOG.info("Loading %s ...", args.model_path)
    model, tok = load_hf_model(args.model_path, dtype=args.dtype)
    tok.padding_side = "left"  # last-token readout at position -1

    directions = compute_directions(model, tok, pairs, batch_size=args.batch_size)
    n_layers = len(directions)
    d_model = directions[0].shape[0]
    LOG.info("Extracted directions for %d layers, d_model=%d", n_layers, d_model)

    randoms = build_random_controls(
        d_model=d_model, n_layers=n_layers, n_random=args.n_random, base_seed=args.random_seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "real": directions,
            "random": randoms,
            "metadata": {
                "model_path": args.model_path,
                "num_layers": n_layers,
                "d_model": d_model,
                "n_pairs": len(pairs),
                "n_random": args.n_random,
                "random_seed": args.random_seed,
            },
        },
        args.out,
    )
    LOG.info("Saved probe to %s", args.out)


if __name__ == "__main__":
    main()
