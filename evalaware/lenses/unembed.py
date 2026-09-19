import torch


def head(model):
    """Unembedding matrix as [vocab, d_model]."""
    out = model.get_output_embeddings()
    if out is None:
        raise ValueError("model exposes no output embeddings")
    return out.weight


def final_norm(model):
    """The norm applied after the last block, before unembedding."""
    inner = getattr(model, "model", model)
    for attr in ("norm", "final_layernorm", "ln_f"):
        found = getattr(inner, attr, None)
        if found is not None:
            return found
    found = getattr(getattr(model, "transformer", None), "ln_f", None)
    if found is None:
        raise ValueError("could not locate the final norm; add this architecture to final_norm()")
    return found


def to_logits(model, residual):
    """Apply the model's own final norm and unembedding to a residual vector."""
    normed = final_norm(model)(residual.to(head(model).dtype))
    return normed @ head(model).T


def top_tokens(tok, logits, k=10):
    values, indices = logits.float().topk(k)
    probs = logits.float().softmax(-1)[indices]
    return [
        {"token": tok.decode([int(i)]), "token_id": int(i), "logit": float(v), "prob": float(p)}
        for i, v, p in zip(indices, values, probs)
    ]


@torch.no_grad()
def logit_lens(model, tok, residual, k=10):
    """Vocabulary projection with no correction: the M = I baseline.

    Works in late layers, degrades early because the residual basis has not yet
    rotated into the one the unembedding expects. That failure is the reason the
    Jacobian lens exists, so this is the comparison it has to beat.
    """
    return top_tokens(tok, to_logits(model, residual), k)
