import hashlib
from pathlib import Path

import torch

from .datasets import Pair

SEED_KEY = "qwen3_8b"


def _layer_seed(layer: int, base_seed: int, index: int, key: str = SEED_KEY) -> int:
    digest = hashlib.sha256(f"{key}|L{layer}|seed{base_seed}|r{index}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


@torch.inference_mode()
def _mean_last_token(model, tok, phrases: list[str], batch_size: int) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    totals, seen = None, 0
    for start in range(0, len(phrases), batch_size):
        batch = phrases[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc, output_hidden_states=True).hidden_states
        last = [h[:, -1, :].float().cpu() for h in hidden]
        if totals is None:
            totals = [t.sum(dim=0) for t in last]
        else:
            for i, t in enumerate(last):
                totals[i] += t.sum(dim=0)
        seen += len(batch)
    return [t / seen for t in totals]


def compute_directions(model, tok, pairs: list[Pair], batch_size: int = 8) -> dict[int, torch.Tensor]:
    pos = _mean_last_token(model, tok, [p.positive for p in pairs], batch_size)
    neg = _mean_last_token(model, tok, [p.negative for p in pairs], batch_size)
    directions = {}
    for layer, (p, n) in enumerate(zip(pos, neg)):
        v = (p - n).float()
        norm = v.norm()
        directions[layer] = v / norm if float(norm) > 0 else v
    return directions


def random_controls(d_model: int, n_layers: int, n_random: int, base_seed: int,
                    key: str = SEED_KEY) -> dict[int, torch.Tensor]:
    controls = {}
    for layer in range(n_layers):
        rows = []
        for i in range(n_random):
            gen = torch.Generator().manual_seed(_layer_seed(layer, base_seed, i, key))
            v = torch.randn(d_model, generator=gen)
            rows.append(v / v.norm())
        controls[layer] = torch.stack(rows)
    return controls


def build(model, tok, pairs: list[Pair], out: Path, model_path: str,
          n_random: int = 10, seed: int = 42, batch_size: int = 8) -> dict:
    directions = compute_directions(model, tok, pairs, batch_size)
    n_layers = len(directions)
    d_model = directions[0].shape[0]
    payload = {
        "real": directions,
        "random": random_controls(d_model, n_layers, n_random, seed),
        "metadata": {
            "model_path": model_path,
            "num_layers": n_layers,
            "d_model": d_model,
            "n_pairs": len(pairs),
            "n_random": n_random,
            "random_seed": seed,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return payload


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
