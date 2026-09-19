import torch
from torch import nn

from .base import Probe, as_tensor, unit


class PCAProbe(Probe):
    """Unsupervised contrast-pair probe: first principal component of the
    difference vectors.

    Emmons showed PC1 of the contrast-pair differences recovers roughly 97% of
    CCS accuracy without the CCS loss, so that is what this implements. PCA sign
    is arbitrary, so the direction is oriented using the pairs it was built from.
    """

    supervised = False

    def __init__(self, direction, bias):
        super().__init__()
        self.register_buffer("_direction", as_tensor(direction))
        self.register_buffer("_bias", as_tensor(bias))
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, acts):
        centred = as_tensor(acts).to(self._direction.device) - self._bias
        return torch.sigmoid(centred @ self._direction)

    @property
    def direction(self):
        return unit(self._direction)

    @classmethod
    def from_pairs(cls, pos_acts, neg_acts):
        pos, neg = as_tensor(pos_acts), as_tensor(neg_acts)
        diffs = pos - neg
        # Uncentred on purpose. The contrast direction is the *mean* of the
        # difference vectors, so subtracting that mean removes exactly the
        # signal and leaves PC1 fitting noise.
        _, _, v = torch.pca_lowrank(diffs, q=min(8, diffs.shape[1], diffs.shape[0]), center=False)
        direction = v[:, 0]
        bias = 0.5 * (pos.mean(0) + neg.mean(0))
        if ((pos - bias) @ direction).mean() < ((neg - bias) @ direction).mean():
            direction = -direction
        return cls(direction, bias)

    @classmethod
    def from_data(cls, acts, labels, **kwargs):
        """Pair positives with negatives by order, then defer to from_pairs."""
        acts, labels = as_tensor(acts), as_tensor(labels)
        pos, neg = acts[labels == 1], acts[labels == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            raise ValueError("need both classes to form contrast pairs")
        return cls.from_pairs(pos[:n], neg[:n])
