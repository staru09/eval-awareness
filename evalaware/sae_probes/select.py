import torch

from ..linear_probes.base import as_tensor


def _auroc(scores, labels):
    order = scores.argsort(dim=0)
    ranks = torch.zeros_like(scores)
    idx = torch.arange(1, len(scores) + 1, dtype=scores.dtype).unsqueeze(1)
    ranks.scatter_(0, order, idx.expand_as(scores))
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.full((scores.shape[1],), 0.5)
    rank_sum = (ranks * labels.unsqueeze(1)).sum(0)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_features(features, labels, method="auroc"):
    """Per-feature separation score, oriented so larger means more class-1."""
    features, labels = as_tensor(features), as_tensor(labels)
    if method == "auroc":
        return (_auroc(features, labels) - 0.5).abs() * 2
    if method == "mean_diff":
        pos, neg = features[labels == 1], features[labels == 0]
        return (pos.mean(0) - neg.mean(0)).abs()
    raise ValueError(f"unknown selection method {method!r}")


def top_features(features, labels, k, method="auroc", min_firing_rate=0.0):
    """Indices of the k most separating features, ignoring near-dead ones.

    Selection uses labels, so it must be fit on train data only. Selecting on
    the full set and then reporting accuracy on it is the standard way to get a
    fake result out of an SAE probe.
    """
    scores = score_features(features, labels, method)
    if min_firing_rate > 0:
        rate = (as_tensor(features) > 0).float().mean(0)
        scores = torch.where(rate >= min_firing_rate, scores, torch.zeros_like(scores))
    k = min(k, scores.numel())
    return scores.topk(k).indices.sort().values
