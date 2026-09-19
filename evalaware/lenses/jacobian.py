import torch

from ..steering import Steerer
from .unembed import to_logits, top_tokens


def contexts_from_texts(tok, texts, device):
    """Text -> the list of model inputs the lens functions consume.

    Kept separate so the lens maths never touches a tokenizer, which makes it
    testable without one.
    """
    return [tok(t, return_tensors="pt", add_special_tokens=False).to(device) for t in texts]


def _final_residual(model, inputs, position=-1):
    out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, position].float()


@torch.no_grad()
def directional_derivative(model, inputs, block, direction, eps=1e-2, position=-1):
    """Estimate J @ direction at one context, where J = d h_L / d h_block.

    Central finite difference rather than an exact JVP. The lens is already a
    first-order approximation, so a second-order-accurate difference costs it
    nothing, and it works through the existing forward hook on any architecture
    instead of needing torch.func to trace a HuggingFace model.

    ponytail: two forward passes per context. Swap in torch.func.jvp for one
    pass if this ever dominates, and only if the model traces cleanly.
    """
    direction = direction.float()
    step = eps / direction.norm().clamp(min=1e-8)
    steerer = Steerer(model, block)
    try:
        steerer.apply(direction * step)
        plus = _final_residual(model, inputs, position)
        steerer.apply(direction * -step)
        minus = _final_residual(model, inputs, position)
    finally:
        steerer.remove()
    return (plus - minus) / (2 * step)


def averaged_readout(model, contexts, block, direction, eps=1e-2, position=-1):
    """The targeted J-lens: where `direction` at `block` pushes the output.

    h_l J_l = E[J h_l] by linearity, so the full averaged matrix is never needed
    when only one direction is being read. One pair of forward passes per
    context, averaged.

    Averaging over contexts is what turns a context-specific Jacobian into a
    dispositional statement about the direction itself.
    """
    total = None
    for inputs in contexts:
        delta = directional_derivative(model, inputs, block, direction, eps, position)
        total = delta if total is None else total + delta
    return total / len(contexts)


def jacobian_lens(model, tok, texts, block, direction, k=10, eps=1e-2, position=-1):
    """Top tokens the direction is disposed to make the model say."""
    contexts = contexts_from_texts(tok, texts, model.device)
    readout = averaged_readout(model, contexts, block, direction, eps, position)
    return top_tokens(tok, to_logits(model, readout), k)


def averaged_jacobian(model, contexts, block, d_model, eps=1e-2, position=-1, basis=None):
    """The full averaged J_l as [rows, d_model]. Expensive on purpose.

    One row per basis vector, so 2 * d_model forward passes per context. At
    d_model=4096 over a thousand contexts that is millions of passes. Pass a
    reduced `basis` ([r, d_model]) for an r-dimensional sketch, which is
    usually what is actually wanted.
    """
    vectors = torch.eye(d_model) if basis is None else basis.float()
    return torch.stack([averaged_readout(model, contexts, block, v, eps, position) for v in vectors])


@torch.no_grad()
def linearity_check(model, contexts, block, direction, alphas=(0.5, 1.0, 2.0), eps=1e-2, position=-1):
    """How far the first-order approximation holds for finite steps.

    Predicts the change in the final residual as alpha * (J @ direction) and
    compares it against the change a real perturbation of alpha * direction
    produces. The cosine and relative error say whether a J-lens readout over
    this layer range means anything. METHODS.md flags exactly this assumption,
    and reading from layer 19 to 36 is a long way for it to hold.
    """
    jv = averaged_readout(model, contexts, block, direction, eps, position)
    direction = direction.float()
    rows = []
    for alpha in alphas:
        predicted = jv * alpha
        actual = None
        for inputs in contexts:
            base = _final_residual(model, inputs, position)
            steerer = Steerer(model, block)
            try:
                steerer.apply(direction * alpha)
                moved = _final_residual(model, inputs, position) - base
            finally:
                steerer.remove()
            actual = moved if actual is None else actual + moved
        actual = actual / len(contexts)
        rows.append({
            "alpha": float(alpha),
            "cosine": float(torch.nn.functional.cosine_similarity(predicted, actual, dim=0)),
            "relative_error": float((predicted - actual).norm() / actual.norm().clamp(min=1e-8)),
        })
    return rows
