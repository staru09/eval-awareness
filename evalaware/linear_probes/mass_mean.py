import torch
from torch import nn

from .base import Probe, as_tensor, unit


class MMProbe(Probe):
    """Difference-of-means probe (Geometry of Truth).

    The direction is simply mean(true) - mean(false). The paper finds this is
    more causally implicated in the model's truth computation than logistic
    regression, despite similar classification accuracy.
    """

    def __init__(self, direction, bias, covariance=None):
        super().__init__()
        self.register_buffer("_direction", as_tensor(direction))
        self.register_buffer("_bias", as_tensor(bias))
        inv = torch.linalg.pinv(as_tensor(covariance)) if covariance is not None else None
        self.register_buffer("_inv_covariance", inv)
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, acts, iid=False):
        centred = as_tensor(acts).to(self._direction.device) - self._bias
        if iid:
            if self._inv_covariance is None:
                raise ValueError("probe was built without a covariance matrix")
            return torch.sigmoid(centred @ self._inv_covariance @ self._direction)
        return torch.sigmoid(centred @ self._direction)

    @property
    def direction(self):
        return unit(self._direction)

    @classmethod
    def from_data(cls, acts, labels, covariance=True):
        acts, labels = as_tensor(acts), as_tensor(labels)
        pos, neg = acts[labels == 1], acts[labels == 0]
        mu_pos, mu_neg = pos.mean(0), neg.mean(0)
        cov = None
        if covariance:
            centred = torch.cat([pos - mu_pos, neg - mu_neg])
            cov = centred.T @ centred / max(len(centred) - 2, 1)
        return cls(mu_pos - mu_neg, 0.5 * (mu_pos + mu_neg), cov)
