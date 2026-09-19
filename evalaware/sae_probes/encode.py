import torch

from ..linear_probes.base import as_tensor


def load_sae(release, sae_id, device="cuda"):
    """Load a pretrained SAE. SAELens v6 returns a single object, not a tuple."""
    from sae_lens import SAE

    return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


def sae_d_in(sae):
    d_in = getattr(sae.cfg, "d_in", None)
    return int(d_in) if d_in is not None else int(sae.W_enc.shape[0])


def sae_d_sae(sae):
    d_sae = getattr(sae.cfg, "d_sae", None)
    return int(d_sae) if d_sae is not None else int(sae.W_enc.shape[1])


def encode(sae, acts, batch_size=256):
    """Residual activations -> sparse SAE features.

    acts is [n, d_in] or [n, seq, d_in]; the SAE is applied to the last axis.
    Dimension mismatch is the failure mode that matters here: an SAE trained on
    a different model produces plausible garbage rather than an error, so it is
    checked explicitly.
    """
    acts = as_tensor(acts)
    expected = sae_d_in(sae)
    if acts.shape[-1] != expected:
        raise ValueError(
            f"activation width {acts.shape[-1]} != SAE d_in {expected}. "
            "The SAE was trained on a different model or hook point."
        )
    device = next(sae.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(acts), batch_size):
            chunk = acts[start:start + batch_size].to(device)
            out.append(sae.encode(chunk).cpu())
    return torch.cat(out)


def feature_stats(features):
    """Per-feature firing rate and mean active magnitude, for triage."""
    active = features > 0
    rate = active.float().mean(0)
    magnitude = torch.where(active, features, torch.zeros_like(features)).sum(0) / active.sum(0).clamp(min=1)
    return {"firing_rate": rate, "mean_active": magnitude, "n_dead": int((rate == 0).sum())}
