"""Jacobian lens, through the reference implementation (jlens).

    lens_l(h) = unembed( J_l @ h ),   J_l = E[ d h_final / d h_l ]

J_l is the model's input-output Jacobian averaged over generic web text; the
readout is the tokens a residual vector at block l is disposed to make the
model say. jlens computes it exactly with autograd, from the last block's
pre-norm output, averaged over source positions (the first 16, attention
sinks, skipped) with the effect summed over later positions, and `unembed`
applies the model's own final norm once. Pre-fitted lenses for many open
models are on the Hub, so no fitting data is needed for those.

Install: uv pip install -e ../jacobian-lens   (or put it on PYTHONPATH)
"""

import torch

LENS_REPO = "neuronpedia/jacobian-lens"
PREFITTED = {  # HF model id -> file in LENS_REPO (fitted on Salesforce/wikitext)
    "Qwen/Qwen3-8B": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    "meta-llama/Llama-3.1-8B-Instruct":
        "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
}


def load(hf_model, tok, model_path: str, lens_path: str | None = None):
    """Wrap a loaded HF model for jlens and load its lens (pre-fitted unless lens_path).

    jlens.from_hf freezes the model's parameters in place.
    """
    import jlens

    lm = jlens.from_hf(hf_model, tok)
    if lens_path:
        lens = jlens.JacobianLens.load(lens_path)
    elif model_path in PREFITTED:
        lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=PREFITTED[model_path])
    else:
        raise SystemExit(f"no pre-fitted lens for {model_path}; fit one with jlens.fit and pass lens_path")
    return lm, lens


@torch.no_grad()
def direction_logits(lm, lens, direction: torch.Tensor, layer: int, use_jacobian: bool = True):
    """Vocabulary logits for a direction in the residual stream after block `layer`.

    use_jacobian=False is the plain logit lens, the baseline the J-lens should beat.
    """
    v = direction.float().to(lm.input_device)
    if use_jacobian:
        v = lens.transport(v, layer)
    return lm.unembed(v).float().cpu()


@torch.no_grad()
def prompt_logits(lm, lens, prompt: str, layers, position: int = -1,
                  use_jacobian: bool = True, max_seq_len: int = 2048):
    """Lens logits at one position of a prompt, per layer, plus the model's own logits."""
    per_layer, model_logits, _ = lens.apply(lm, prompt, layers=list(layers), positions=[position],
                                            max_seq_len=max_seq_len, use_jacobian=use_jacobian)
    return {l: x[0] for l, x in per_layer.items()}, model_logits[0]


def top_tokens(tok, logits: torch.Tensor, k: int = 10) -> list[dict]:
    probs = logits.float().softmax(-1)
    values, indices = logits.float().topk(k)
    return [{"token": tok.decode([int(i)]), "token_id": int(i), "logit": float(v), "prob": float(probs[i])}
            for i, v in zip(indices, values)]


def ranks(logits: torch.Tensor, token_ids: list[int]) -> list[int]:
    """0-based rank of each token over the whole vocabulary (0 = top)."""
    order = logits.float().argsort(descending=True)
    where = torch.empty_like(order)
    where[order] = torch.arange(len(order))
    return [int(where[i]) for i in token_ids]
