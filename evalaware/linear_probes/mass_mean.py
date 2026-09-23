import torch
from torch import nn

from .base import Probe, as_tensor, unit


class MMProbe(Probe):
    """Mass-mean probe, as in Geometry of Truth (probes.py).

    Direction = mean(label 1) - mean(label 0); score = sigmoid(x . direction)
    on mean-centred activations. The paper finds this direction more causally
    implicated than logistic regression at similar accuracy.

    iid=True first applies the pseudo-inverse of the pooled within-class
    covariance (divided by n). With far fewer rows than d_model that
    covariance is rank-limited, so treat iid results with care.
    """

    def __init__(self, direction, inv=None):
        super().__init__()
        self.register_buffer("_direction", as_tensor(direction))
        self.register_buffer("_inv", None if inv is None else as_tensor(inv))
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)  # gives the probe a .device

    def forward(self, acts, iid=False):
        x = as_tensor(acts).to(self._direction.device)
        if iid:
            if self._inv is None:
                raise ValueError("probe was built with iid=False, so it has no inverse covariance")
            return torch.sigmoid(x @ self._inv @ self._direction)
        return torch.sigmoid(x @ self._direction)

    @property
    def direction(self):
        return unit(self._direction)

    @classmethod
    def from_data(cls, acts, labels, iid=False, atol=1e-3, device="cpu"):
        acts, labels = as_tensor(acts).to(device), as_tensor(labels).to(device)
        pos, neg = acts[labels == 1], acts[labels == 0]
        mu_pos, mu_neg = pos.mean(0), neg.mean(0)
        inv = None
        if iid:
            centred = torch.cat([pos - mu_pos, neg - mu_neg])
            inv = torch.linalg.pinv(centred.T @ centred / acts.shape[0], hermitian=True, atol=atol)
        return cls(mu_pos - mu_neg, inv).to(device)
